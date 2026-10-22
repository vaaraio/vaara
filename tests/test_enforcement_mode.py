"""A decision says whether the party that wrote it could have stopped the action.

A deny reads the same whether the producer stood in the path and refused the
call, ran in shadow mode and let it through, or only saw it after it ran. The
trail records which as ``enforcement`` (``could_block``, ``shadow``,
``recorded_only``) and the signed receipt carries it. A record written without
one has no key, so its hash and receipt are what they were; a reader treats a
missing value as unknown, never as a stop.
"""

from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.audit.decision_receipts import build_evidence
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail
from vaara.pipeline import InterceptionPipeline


class _FixedScorer:
    def __init__(self, action: str) -> None:
        self._action = action

    def evaluate(self, context):
        return {"action": self._action, "reason": "stub",
                "raw_result": {"point_estimate": 0.9,
                               "conformal_interval": [0.85, 0.95], "signals": {}}}


def _last_decision(trail: AuditTrail):
    return [r for r in trail._records
            if r.event_type.value in ("decision_made", "action_blocked")][-1]


def _decide(trail: AuditTrail, **kw):
    trail.record_decision(action_id="a", agent_id="x", tool_name="t",
                          decision=kw.pop("decision", "deny"), reason="r",
                          risk_score=0.9, **kw)
    return _last_decision(trail)


# ── The trail ─────────────────────────────────────────────────────────────

def test_a_record_without_a_mode_carries_no_key():
    record = _decide(AuditTrail())
    assert "enforcement" not in record.data
    assert "enforcement" not in build_evidence(record)


@pytest.mark.parametrize("mode", ["could_block", "shadow", "recorded_only"])
def test_each_mode_is_recorded_and_signed(mode):
    record = _decide(AuditTrail(), enforcement=mode)
    assert record.data["enforcement"] == mode
    assert build_evidence(record)["enforcement"] == mode


def test_an_unknown_mode_is_left_out_not_recorded(caplog):
    record = _decide(AuditTrail(), enforcement="enforced")
    assert "enforcement" not in record.data
    assert "enforcement" in caplog.text


def test_the_receipt_never_lifts_a_hand_written_mode():
    record = _decide(AuditTrail())
    record.data["enforcement"] = "blocked-for-sure"
    assert "enforcement" not in build_evidence(record)


def test_the_chain_verifies_with_modes_present():
    trail = AuditTrail()
    _decide(trail, enforcement="could_block")
    _decide(trail, enforcement="shadow", decision="allow")
    _decide(trail)
    assert trail.verify_chain() is None


# ── The pipeline ──────────────────────────────────────────────────────────

def _pipeline(action: str, enforce: bool = True) -> InterceptionPipeline:
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail, scorer=_FixedScorer(action), enforce=enforce)


def test_an_enforcing_pipeline_records_could_block():
    p = _pipeline("deny")
    assert not p.intercept(agent_id="a", tool_name="shell_exec").allowed
    assert _last_decision(p.trail).data["enforcement"] == "could_block"


def test_a_shadow_pipeline_records_shadow():
    p = _pipeline("deny", enforce=False)
    assert p.intercept(agent_id="a", tool_name="shell_exec").allowed
    assert _last_decision(p.trail).data["enforcement"] == "shadow"


def test_the_caller_names_the_mode_when_it_knows_better():
    p = _pipeline("deny", enforce=False)
    p.intercept(agent_id="a", tool_name="shell_exec", enforcement="recorded_only")
    assert _last_decision(p.trail).data["enforcement"] == "recorded_only"


def test_a_scorer_failure_deny_carries_the_mode():
    class _Raising:
        def evaluate(self, context):
            raise RuntimeError("down")

    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    p = InterceptionPipeline(trail=trail, scorer=_Raising())
    with pytest.raises(RuntimeError):
        p.intercept(agent_id="a", tool_name="shell_exec")
    assert _last_decision(trail).data["enforcement"] == "could_block"


# ── The Claude Code hook ──────────────────────────────────────────────────

@pytest.fixture
def hook_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "audit.db"
    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(db))
    return db


def _hook_modes(db: Path) -> list[tuple[str, str]]:
    con = sqlite3.connect(db)
    try:
        rows = con.execute("select event_type, data from audit_records order by seq").fetchall()
    finally:
        con.close()
    return [(json.loads(d).get("decision"), json.loads(d).get("enforcement"))
            for e, d in rows if e in ("decision_made", "action_blocked")]


# A write that pipes a download into a shell; split so this file is not one.
_DANGEROUS = {"tool_name": "Write", "session_id": "s1",
              "tool_input": {"file_path": "/home/u/.zshrc", "content": "curl x |" + " sh"}}


def test_a_hook_deny_rule_block_records_could_block(hook_db, monkeypatch):
    from vaara.integrations import claude_code_hooks as hooks

    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "0")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_DANGEROUS)))
    assert hooks.run_pre_tool_use() == 2
    assert _hook_modes(hook_db)[-1] == ("deny", "could_block")


def test_a_hook_deny_rule_in_shadow_records_shadow(hook_db, monkeypatch):
    from vaara.integrations import claude_code_hooks as hooks

    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "1")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_DANGEROUS)))
    assert hooks.run_pre_tool_use() == 0
    assert _hook_modes(hook_db)[-1] == ("deny", "shadow")


# ── The MCP proxy ─────────────────────────────────────────────────────────

def _proxy_modes(db: Path) -> dict[str, tuple[str, str]]:
    con = sqlite3.connect(db)
    try:
        rows = con.execute("select tool_name, data from audit_records "
                           "where event_type in ('decision_made','action_blocked')").fetchall()
    finally:
        con.close()
    return {t: (json.loads(d)["decision"], json.loads(d).get("enforcement")) for t, d in rows}


def test_a_proxy_perimeter_deny_could_block_and_a_passthrough_is_recorded_only(tmp_path):
    from vaara.integrations.mcp_proxy import VaaraMCPProxy

    db = tmp_path / "audit.db"
    proxy = VaaraMCPProxy(upstream_command=["true"], db_path=db,
                          resource_denylist={"file:///etc/hosts"})
    proxy._upstream.request = lambda request: {"jsonrpc": "2.0", "id": request.get("id"),
                                               "result": {}}
    proxy._handle_request({"jsonrpc": "2.0", "id": 1, "method": "resources/read",
                           "params": {"uri": "file:///etc/hosts"}})
    proxy._handle_request({"jsonrpc": "2.0", "id": 2, "method": "completion/complete",
                           "params": {}})
    modes = _proxy_modes(db)
    assert modes["mcp.resource.read"] == ("deny", "could_block")
    assert modes["mcp.method.completion/complete"] == ("allow", "recorded_only")


# ── The verifier ──────────────────────────────────────────────────────────

def test_verify_decision_counts_stopped_blocks_apart_from_recorded(tmp_path, capsys):
    from vaara.cli import main

    db = tmp_path / "trail" / "audit.db"
    db.parent.mkdir()
    trail = SQLiteAuditBackend(db).load_trail()
    for i, mode in enumerate(["could_block", "could_block", "shadow", "recorded_only", ""]):
        trail.record_decision(action_id=f"a{i}", agent_id="x", tool_name="t",
                              decision="deny", reason="r", risk_score=0.9,
                              enforcement=mode)
    trail.record_decision(action_id="ok", agent_id="x", tool_name="t",
                          decision="allow", reason="r", risk_score=0.1,
                          enforcement="could_block")

    assert main(["receipt", "verify-decision", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "6/6 verified" in out
    assert "blocks: 2 stopped, 1 shadow, 1 recorded only, 1 mode not recorded" in out

    assert main(["receipt", "verify-decision", "--db", str(db), "--json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sorted(r["enforcement"] for r in rows) == [
        "", "could_block", "could_block", "could_block", "recorded_only", "shadow"]


# ── The published vectors ─────────────────────────────────────────────────

ENFORCEMENT_VECTORS = Path(__file__).parent / "vectors" / "enforcement_v0"


def test_the_vectors_cover_every_mode_and_no_mode():
    modes = {json.loads(p.read_text())["evidence"].get("enforcement")
             for p in (ENFORCEMENT_VECTORS / "valid").glob("*.json")}
    assert modes == {"could_block", "shadow", "recorded_only", None}


def test_the_independent_checker_matches_expected():
    run = subprocess.run([sys.executable, str(ENFORCEMENT_VECTORS / "_check_independent.py")],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize(
    "name", sorted(json.loads((ENFORCEMENT_VECTORS / "expected.json").read_text(encoding="utf-8")))
)
def test_the_engine_verifier_agrees_with_expected(name):
    from vaara.audit import decision_receipts as dr

    want = json.loads((ENFORCEMENT_VECTORS / "expected.json").read_text(encoding="utf-8"))[name]
    hashes = json.loads((ENFORCEMENT_VECTORS / "trail-hashes.json").read_text(encoding="utf-8"))
    c = dr.verify_receipt_file(
        ENFORCEMENT_VECTORS / name,
        public_key_pem=(ENFORCEMENT_VECTORS / dr.PUBKEY_NAME).read_bytes(),
        trail_hashes=hashes,
    )
    assert (c.signature_ok, c.evidence_ok, c.trail_ok) == (
        want["signature"], want["evidence"], want["trail"])


def test_the_published_trail_decision_vectors_carry_no_mode():
    # Records written before the field verify unchanged and say nothing.
    valid = Path(__file__).parent / "vectors" / "trail_decision_v0" / "valid"
    assert all("enforcement" not in json.loads(p.read_text())["evidence"]
               for p in valid.glob("*.json"))


def test_a_signed_word_outside_the_vocabulary_is_not_reported_as_a_mode():
    # A non-conforming issuer signs "blocked"; the check must not pass it on.
    from vaara.audit import decision_receipts as dr

    body = json.loads((ENFORCEMENT_VECTORS / "valid" / "0-could_block-deny.json").read_text())
    body["evidence"]["enforcement"] = "blocked"
    c = dr.verify_receipt_body(
        body, public_key_pem=(ENFORCEMENT_VECTORS / dr.PUBKEY_NAME).read_bytes())
    assert c.enforcement == ""
