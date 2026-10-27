# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Digest-only capture: a declared argument field never reaches a store in clear.

Without a capture policy the trail, the review queue copy and the approval
request hold the arguments exactly as the host sent them. With one, each
declared field is kept as a keyed digest and a reference, on all three, on
every surface that writes them (pipeline, Claude Code hook, MCP proxy, shell
proxy). The key stays outside the trail, so a holder of the trail alone cannot
confirm a guessed value; the operator, who holds the key, can.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import re
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from vaara import _capture
from vaara.audit.review_queue import ReviewQueue
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline

PATIENT = "010170-123A"
NOTE = "chest pain since monday"
TOOL = "mcp__ehr__lookup"

POLICY = {
    "version": 1,
    "rules": [
        {"tool": "mcp__ehr__*", "fields": ["patient.id", "note"], "reference": "ehr"},
        {"tool": "lookup", "fields": ["patient.id", "note"], "reference": "ehr"},
        {"tool": "shell.exec", "fields": ["command"]},
    ],
}


def _args() -> dict:
    return {"patient": {"id": PATIENT, "ward": "B2"}, "note": NOTE, "limit": 5}


@pytest.fixture
def policy_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(POLICY), encoding="utf-8")
    monkeypatch.setenv("VAARA_CAPTURE_POLICY", str(path))
    return path


def _key(policy: Path) -> bytes:
    return bytes.fromhex((policy.parent / "keys" / "capture-hmac.key").read_text().strip())


def _independent_digest(key: bytes, path: str, value) -> str:
    """The digest recomputed with the standard library only (plain JSON values)."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    message = b"vaara.capture.v1\n" + path.encode() + b"\n" + canonical.encode()
    return "hmac-sha256:" + hmac.new(key, message, hashlib.sha256).hexdigest()


def _db_bytes(db: Path) -> bytes:
    blob = b""
    for suffix in ("", "-wal", "-shm"):
        f = db.parent / (db.name + suffix)
        if f.exists():
            blob += f.read_bytes()
    return blob


def _no_clear(blob: bytes) -> None:
    assert PATIENT.encode() not in blob
    assert NOTE.encode() not in blob


# ---- the masking itself -------------------------------------------------


def test_without_a_policy_the_arguments_come_back_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_CAPTURE_POLICY", str(tmp_path / "absent.json"))
    args = _args()
    assert _capture.mask(TOOL, args) is args
    assert not (tmp_path / "keys").exists()


def test_declared_fields_become_a_digest_and_a_reference(policy_file):
    masked = _capture.mask(TOOL, _args())
    assert masked["limit"] == 5
    assert masked["patient"]["ward"] == "B2"
    for field in (masked["patient"]["id"], masked["note"]):
        assert field["captured"] == "digest-only"
        assert re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", field["digest"])
        assert field["reference"] == "ehr"
    assert PATIENT not in json.dumps(masked) and NOTE not in json.dumps(masked)


def test_the_caller_keeps_its_own_arguments(policy_file):
    args = _args()
    _capture.mask(TOOL, args)
    assert args == _args()


def test_the_digest_is_recomputable_from_the_key_and_the_value(policy_file):
    masked = _capture.mask(TOOL, _args())
    key = _key(policy_file)
    assert masked["patient"]["id"]["digest"] == _independent_digest(key, "patient.id", PATIENT)
    assert masked["note"]["digest"] == _independent_digest(key, "note", NOTE)


def test_a_bare_hash_of_the_value_does_not_match_the_digest(policy_file):
    """Someone holding only the trail cannot confirm a guessed identifier."""
    masked = _capture.mask(TOOL, _args())
    bare = hashlib.sha256(json.dumps(PATIENT).encode()).hexdigest()
    assert bare not in masked["patient"]["id"]["digest"]
    plain = hashlib.sha256(PATIENT.encode()).hexdigest()
    assert plain not in masked["patient"]["id"]["digest"]


def test_the_same_value_gives_the_same_digest_every_time(policy_file):
    assert _capture.mask(TOOL, _args()) == _capture.mask(TOOL, _args())


def test_the_same_value_in_another_field_gives_another_digest(policy_file):
    masked = _capture.mask(TOOL, {"patient": {"id": NOTE}, "note": NOTE})
    assert masked["patient"]["id"]["digest"] != masked["note"]["digest"]


def test_another_key_gives_another_digest(policy_file, tmp_path):
    first = _capture.mask(TOOL, _args())["note"]["digest"]
    (policy_file.parent / "keys" / "capture-hmac.key").write_text("ab" * 32)
    assert _capture.mask(TOOL, _args())["note"]["digest"] != first


def test_the_key_is_created_once_with_mode_0600(policy_file):
    _capture.mask(TOOL, _args())
    key_file = policy_file.parent / "keys" / "capture-hmac.key"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    before = key_file.read_text()
    _capture.mask(TOOL, _args())
    assert key_file.read_text() == before


def test_a_tool_the_policy_does_not_name_is_left_alone(policy_file):
    args = _args()
    assert _capture.mask("mcp__other__lookup", args) == args


def test_the_default_policy_file_is_in_the_vaara_home(tmp_path, monkeypatch):
    monkeypatch.delenv("VAARA_CAPTURE_POLICY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".vaara").mkdir()
    (tmp_path / ".vaara" / "capture.json").write_text(json.dumps(POLICY))
    assert _capture.mask(TOOL, _args())["note"]["captured"] == "digest-only"
    assert (tmp_path / ".vaara" / "keys" / "capture-hmac.key").exists()


def test_a_wildcard_segment_reaches_every_item_of_a_list(tmp_path, monkeypatch):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps({"version": 1, "rules": [
        {"tool": "*", "fields": ["rows.*.ssn"]}]}))
    monkeypatch.setenv("VAARA_CAPTURE_POLICY", str(path))
    masked = _capture.mask("t", {"rows": [{"ssn": "1", "n": 1}, {"ssn": "2", "n": 2}]})
    assert [r["n"] for r in masked["rows"]] == [1, 2]
    assert all(r["ssn"]["captured"] == "digest-only" for r in masked["rows"])
    # the digest names the concrete position, so two rows never share one
    assert masked["rows"][0]["ssn"]["digest"] != masked["rows"][1]["ssn"]["digest"]


def test_a_lone_wildcard_keeps_the_keys_and_drops_every_value(tmp_path, monkeypatch):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps({"version": 1, "rules": [{"tool": "*", "fields": ["*"]}]}))
    monkeypatch.setenv("VAARA_CAPTURE_POLICY", str(path))
    masked = _capture.mask("t", _args())
    assert set(masked) == {"patient", "note", "limit"}
    assert all(v["captured"] == "digest-only" for v in masked.values())
    assert PATIENT not in json.dumps(masked) and "B2" not in json.dumps(masked)


def test_a_value_shaped_like_a_marker_is_masked_like_any_other(policy_file):
    """An agent cannot keep a declared field in clear by wrapping it in a marker."""
    args = {"note": {"captured": "digest-only", "digest": "x", "leak": NOTE}}
    masked = _capture.mask(TOOL, args)
    assert NOTE not in json.dumps(masked)
    assert masked["note"]["digest"] != "x"


@pytest.mark.parametrize("body", [
    "{not json",
    json.dumps({"version": 2, "rules": []}),
    json.dumps({"version": 1, "rules": "all"}),
    json.dumps({"version": 1, "rules": [{"tool": "*", "fields": "note"}]}),
    json.dumps({"version": 1, "rules": [{"tool": "*", "fields": [""]}]}),
    json.dumps([1, 2]),
])
def test_a_policy_it_cannot_read_withholds_everything(tmp_path, monkeypatch, body):
    path = tmp_path / "capture.json"
    path.write_text(body)
    monkeypatch.setenv("VAARA_CAPTURE_POLICY", str(path))
    masked = _capture.mask(TOOL, _args())
    assert set(masked) == {"patient", "note", "limit"}
    assert all(v == {"captured": "withheld"} for v in masked.values())


def test_a_key_it_cannot_read_withholds_the_value_and_writes_no_digest(policy_file):
    (policy_file.parent / "keys").mkdir()
    (policy_file.parent / "keys" / "capture-hmac.key").mkdir()
    masked = _capture.mask(TOOL, _args())
    assert masked["note"] == {"captured": "withheld", "reference": "ehr"}
    assert PATIENT not in json.dumps(masked)


def test_a_value_it_cannot_canonicalise_is_withheld(policy_file):
    masked = _capture.mask(TOOL, {"note": object(), "limit": 1})
    assert masked["note"] == {"captured": "withheld", "reference": "ehr"}


def test_an_edited_policy_applies_to_the_next_call(policy_file):
    assert _capture.mask(TOOL, _args())["limit"] == 5
    policy_file.write_text(json.dumps({"version": 1, "rules": [
        {"tool": "*", "fields": ["limit"]}]}))
    masked = _capture.mask(TOOL, _args())
    assert masked["limit"]["captured"] == "digest-only"
    assert masked["note"] == NOTE


# ---- the trail ----------------------------------------------------------


def test_the_trail_row_holds_no_clear_value_and_the_chain_still_verifies(policy_file, tmp_path):
    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    pipe = InterceptionPipeline(trail=backend.load_trail())
    args = _args()
    pipe.intercept(agent_id="a", tool_name=TOOL, parameters=args)
    verdict = pipe.trail.verify_chain_verdict()
    request = [r for r in pipe.trail._records if r.event_type == EventType.ACTION_REQUESTED][-1]
    backend.close()

    assert args == _args(), "the engine and the caller still hold the real arguments"
    assert verdict.error is None
    assert request.data["parameters"]["limit"] == 5
    assert request.data["parameters"]["note"]["captured"] == "digest-only"
    _no_clear(_db_bytes(db))


def test_the_receipt_commitment_covers_the_arguments_as_recorded(policy_file):
    from vaara.audit import _arguments_commitment as ac

    pipe = InterceptionPipeline()
    pipe.intercept(agent_id="a", tool_name=TOOL, parameters=_args())
    request = [r for r in pipe.trail._records if r.event_type == EventType.ACTION_REQUESTED][-1]
    assert ac.verify(request.data["arguments_commitment"], request.data["parameters"],
                     request.data["arguments_salt"])


def test_a_modified_proposal_is_masked_too(policy_file):
    pipe = InterceptionPipeline()
    pipe.trail.record_decision(
        action_id="act-1", agent_id="a", tool_name=TOOL, decision="deny",
        reason="redacted by policy", risk_score=0.1, decision_detail="modify",
        modified_parameters={"note": NOTE, "limit": 1},
    )
    record = [r for r in pipe.trail._records if "modified_parameters" in r.data][-1]
    assert record.data["modified_parameters"]["limit"] == 1
    assert record.data["modified_parameters"]["note"]["captured"] == "digest-only"
    assert NOTE not in json.dumps(record.data)


def test_erasing_a_masked_record_still_works(policy_file, tmp_path):
    from vaara.audit.erase import erase_record

    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    pipe = InterceptionPipeline(trail=backend.load_trail())
    for _ in range(2):
        pipe.intercept(agent_id="a", tool_name=TOOL, parameters=_args())
    target = [r for r in pipe.trail._records if r.event_type == EventType.ACTION_REQUESTED][0]
    backend.close()

    erase_record(db, target.record_id, reason="data subject request")

    backend = SQLiteAuditBackend(db)
    try:
        verdict = backend.load_trail().verify_chain_verdict()
    finally:
        backend.close()
    assert verdict.error is None and verdict.state == "erased"


# ---- the review queue copy ---------------------------------------------


def test_the_review_queue_copy_is_masked_and_agrees_with_the_trail(policy_file, tmp_path):
    db = tmp_path / "queue.db"
    q = ReviewQueue(db)
    qid = q.enqueue(
        action_id="act-1", agent_id="a", tool_name=TOOL, risk_score=0.6,
        conformal_lower=0.5, conformal_upper=0.7, parameters=_args(),
    )
    stored = q.get(qid).parameters
    q.close()
    assert stored == _capture.mask(TOOL, _args())
    assert stored["note"]["captured"] == "digest-only"
    _no_clear(_db_bytes(db))


def test_an_escalation_through_the_pipeline_queues_masked_arguments(policy_file, tmp_path):
    class _Escalate:
        def evaluate(self, ctx):
            return {
                "action": "escalate", "reason": "fixed for test",
                "raw_result": {
                    "point_estimate": 0.6, "conformal_interval": (0.55, 0.71),
                    "signals": {}, "calibration_size": 50, "effective_alpha": 0.1,
                    "bucket_category": "filesystem",
                },
            }

    db = tmp_path / "queue.db"
    queue = ReviewQueue(db)
    pipe = InterceptionPipeline(scorer=_Escalate(), review_queue=queue)
    result = pipe.intercept(agent_id="a", tool_name=TOOL, parameters=_args())
    (item,) = queue.list_items()
    queue.close()
    assert result.decision == "escalate"
    assert item.parameters["limit"] == 5
    assert item.parameters["patient"]["id"]["captured"] == "digest-only"
    _no_clear(_db_bytes(db))


# ---- the approval request file -----------------------------------------


def test_the_approval_request_file_is_masked_and_the_answer_binds_that_view(policy_file, tmp_path):
    from vaara.approvals import parameters_digest, request_approval, write_decision

    approvals = tmp_path / "approvals"
    seen: dict = {}
    raw: list[str] = []

    def human() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            files = list(approvals.glob("*.request.json"))
            if files:
                raw.append(files[0].read_text(encoding="utf-8"))
                seen.update(json.loads(raw[0]))
                write_decision(files[0].name.removesuffix(".request.json"), "approve",
                               approvals_dir=approvals)
                return
            time.sleep(0.02)

    thread = threading.Thread(target=human, daemon=True)
    thread.start()
    answer = request_approval("act-9", TOOL, "risk 0.6", approvals_dir=approvals,
                              timeout=30, parameters=_args())
    thread.join(timeout=30)

    assert answer == "approve"
    assert PATIENT not in raw[0] and NOTE not in raw[0]
    assert seen["parameters"]["limit"] == 5
    assert seen["parameters"]["note"]["captured"] == "digest-only"
    assert seen["parameters_sha256"] == parameters_digest(seen["parameters"])
    assert answer.parameters_sha256 == seen["parameters_sha256"]


# ---- every surface ------------------------------------------------------


def _isolate_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from vaara.integrations import claude_code_hooks as hooks

    db = tmp_path / "hook-audit.db"
    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(db))
    monkeypatch.setattr(hooks, "CONFIG_PATH", tmp_path / "absent-config.json")
    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "0")
    monkeypatch.setenv("VAARA_PLUGIN_APPROVALS", "0")
    monkeypatch.setenv("VAARA_PLUGIN_NOTIFY", "0")
    monkeypatch.delenv("VAARA_PLUGIN_FAIL_OPEN", raising=False)
    monkeypatch.delenv("VAARA_PLUGIN_DISABLE", raising=False)
    return db


def test_hook_path_records_no_clear_value(policy_file, tmp_path, monkeypatch):
    from vaara.integrations import claude_code_hooks as hooks

    db = _isolate_hook(tmp_path, monkeypatch)
    event = {"tool_name": TOOL, "tool_input": _args(), "session_id": "s1"}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    hooks.run_pre_tool_use()
    blob = _db_bytes(db)
    assert b"mcp__ehr__lookup" in blob, "the call must have been recorded"
    _no_clear(blob)


def test_mcp_proxy_path_records_no_clear_value(policy_file, tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from vaara.integrations import mcp_proxy

    db = tmp_path / "proxy-audit.db"
    backend = SQLiteAuditBackend(db)
    pipe = InterceptionPipeline(trail=backend.load_trail())
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    proxy = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipe)
    upstream = MagicMock()
    upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    proxy._upstream = upstream
    proxy._handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "lookup", "arguments": _args()},
    })
    # the perimeter path (resource reads, prompt gets) writes the request itself
    proxy._record_perimeter_audit(
        "agent", TOOL, _args(), decision="allow", reason="read", enforcement="recorded_only",
    )
    backend.close()
    blob = _db_bytes(db)
    assert b"lookup" in blob, "the call must have been recorded"
    _no_clear(blob)


def test_shell_proxy_path_records_no_clear_value(policy_file, tmp_path):
    from vaara.integrations.shell_proxy import _build_pipeline, _run_one

    db = tmp_path / "shell-audit.db"
    pipe = _build_pipeline(db)
    command = f"true  # {PATIENT} {NOTE}"
    _run_one(pipe, command, "/bin/sh", "shell")
    pipe.trail._backend.close()
    blob = _db_bytes(db)
    assert b"shell.exec" in blob, "the call must have been recorded"
    _no_clear(blob)
