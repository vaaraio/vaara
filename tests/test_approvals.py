"""Tests for the file-based approval handshake (vaara.approvals).

The protocol: the gate writes
``<action_id>.request.json`` into the approvals directory and polls for
``<action_id>.decision.json``; whatever fronts the human writes the
decision. Plain files,
zero dependencies, and the gate cleans up its own files whatever happens.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from vaara.approvals import approval_key_path, decision_mac, request_approval, write_decision


def _respond(approvals_dir: Path, decision: str, captured: dict) -> threading.Thread:
    """Play the watcher: wait for a request file, record it, write the decision."""

    def responder() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            requests = list(approvals_dir.glob("*.request.json"))
            if requests:
                captured.update(json.loads(requests[0].read_text()))
                action_id = requests[0].name.removesuffix(".request.json")
                write_decision(action_id, decision, approvals_dir=approvals_dir)
                return
            time.sleep(0.02)

    thread = threading.Thread(target=responder, daemon=True)
    thread.start()
    return thread


def test_approve_returns_approve_and_request_matches_app_schema(tmp_path):
    captured: dict = {}
    thread = _respond(tmp_path, "approve", captured)
    result = request_approval(
        "act-123", "mcp__pay__transfer", "risk 0.55: novel counterparty",
        approvals_dir=tmp_path, timeout=30,
    )
    thread.join(timeout=30)
    assert result == "approve"
    # The exact keys a watcher reads; requested_at is a unix timestamp.
    assert captured["action_id"] == "act-123"
    assert captured["tool_name"] == "mcp__pay__transfer"
    assert captured["reason"] == "risk 0.55: novel counterparty"
    assert abs(captured["requested_at"] - time.time()) < 60


def test_deny_returns_deny(tmp_path):
    thread = _respond(tmp_path, "deny", {})
    result = request_approval(
        "act-456", "mcp__fs__rm", "risk 0.71", approvals_dir=tmp_path, timeout=30,
    )
    thread.join(timeout=30)
    assert result == "deny"


def test_timeout_returns_timeout_and_cleans_up(tmp_path):
    start = time.monotonic()
    result = request_approval(
        "act-789", "mcp__x__y", "r", approvals_dir=tmp_path, timeout=0.3,
    )
    assert result == "timeout"
    assert time.monotonic() - start < 3
    assert list(tmp_path.iterdir()) == []


def test_files_cleaned_up_after_decision(tmp_path):
    thread = _respond(tmp_path, "approve", {})
    request_approval("act-clean", "t", "r", approvals_dir=tmp_path, timeout=30)
    thread.join(timeout=30)
    assert list(tmp_path.iterdir()) == []


def test_garbage_decision_file_is_ignored_until_timeout(tmp_path):
    # A malformed or foreign decision value must not be taken as approval.
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "act-bad.decision.json").write_text("{not json")
    result = request_approval(
        "act-bad", "t", "r", approvals_dir=tmp_path, timeout=0.3,
    )
    assert result == "timeout"
    assert list(tmp_path.iterdir()) == []


def test_creates_missing_approvals_dir(tmp_path):
    nested = tmp_path / "not" / "yet" / "there"
    result = request_approval("a", "t", "r", approvals_dir=nested, timeout=0.2)
    assert result == "timeout"
    assert nested.is_dir()


def test_watcher_never_sees_a_half_written_request(tmp_path):
    """The request file appears with its content already in it.

    ``Path.write_text`` truncates and then writes, so a watcher polling the
    directory could find the file and read nothing: 66 of 300 requests on a
    warm filesystem before this was fixed. A watcher that parses what it
    finds raises on the empty string, and if that ends its poll loop the
    request is never answered and the gate blocks for its whole timeout.
    This is what made the tests above fail roughly one run in three, and it
    would do the same to any real approval surface.
    """
    empty_reads = 0
    for i in range(40):
        run_dir = tmp_path / f"run-{i}"
        run_dir.mkdir()
        seen: list[str] = []

        def watch(d=run_dir, seen=seen) -> None:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                for req in d.glob("*.request.json"):
                    seen.append(req.read_text())
                    action_id = req.name.removesuffix(".request.json")
                    write_decision(action_id, "approve", approvals_dir=d)
                    return

        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        result = request_approval(
            f"act-{i}", "t", "r",
            approvals_dir=run_dir, timeout=10, poll_interval=0.01,
        )
        thread.join(timeout=10)
        assert result == "approve"
        empty_reads += sum(1 for raw in seen if raw == "")

    assert empty_reads == 0, (
        f"a watcher read an empty request file {empty_reads} time(s) in 40 requests"
    )


def _answer_with(approvals_dir: Path, make) -> threading.Thread:
    """A watcher that writes whatever ``make(action_id, request)`` returns."""

    def responder() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for req in approvals_dir.glob("*.request.json"):
                action_id = req.name.removesuffix(".request.json")
                body = make(action_id, json.loads(req.read_text()))
                (approvals_dir / f"{action_id}.decision.json").write_text(json.dumps(body))
                return
            time.sleep(0.02)

    thread = threading.Thread(target=responder, daemon=True)
    thread.start()
    return thread


def test_unsigned_approval_is_not_consent(tmp_path):
    """Any process of the user can write a decision file; that alone is not an approval."""
    approvals = tmp_path / "approvals"
    _answer_with(approvals, lambda a, r: {"decision": "approve", "decided_at": time.time()})
    assert request_approval("act-u", "t", "r", approvals_dir=approvals, timeout=0.6) == "timeout"


def test_approval_signed_with_another_key_is_not_consent(tmp_path):
    approvals = tmp_path / "approvals"
    other = bytes(32)
    _answer_with(approvals, lambda a, r: {
        "decision": "approve", "mac": decision_mac(other, a, r["nonce"], "approve")})
    assert request_approval("act-k", "t", "r", approvals_dir=approvals, timeout=0.6) == "timeout"


def test_a_signed_decision_does_not_answer_a_later_request(tmp_path):
    """The nonce binds a decision to one request: an old approval cannot be replayed."""
    approvals = tmp_path / "approvals"
    request_approval("warmup", "t", "r", approvals_dir=approvals, timeout=0.05)
    key = bytes.fromhex(approval_key_path(approvals).read_text())
    _answer_with(approvals, lambda a, r: {
        "decision": "approve", "mac": decision_mac(key, a, "an-earlier-nonce", "approve")})
    assert request_approval("act-r", "t", "r", approvals_dir=approvals, timeout=0.6) == "timeout"


def test_a_signed_approve_cannot_be_turned_into_another_decision(tmp_path):
    approvals = tmp_path / "approvals"
    request_approval("warmup", "t", "r", approvals_dir=approvals, timeout=0.05)
    key = bytes.fromhex(approval_key_path(approvals).read_text())
    _answer_with(approvals, lambda a, r: {
        "decision": "approve", "mac": decision_mac(key, a, r["nonce"], "deny")})
    assert request_approval("act-s", "t", "r", approvals_dir=approvals, timeout=0.6) == "timeout"


def test_key_is_created_private_beside_the_approvals_dir(tmp_path):
    approvals = tmp_path / "approvals"
    request_approval("a", "t", "r", approvals_dir=approvals, timeout=0.05)
    key = approval_key_path(approvals)
    assert key == tmp_path / "keys" / "approval-hmac.key"
    assert len(bytes.fromhex(key.read_text())) == 32
    assert key.stat().st_mode & 0o777 == 0o600


def test_write_decision_refuses_without_a_request(tmp_path):
    approvals = tmp_path / "approvals"
    request_approval("a", "t", "r", approvals_dir=approvals, timeout=0.05)
    assert write_decision("nothing-pending", "approve", approvals_dir=approvals) is False


def test_mac_matches_the_shared_vectors():
    """The app signs with its own code (ApprovalSigning.swift) against these same vectors."""
    doc = json.loads((Path(__file__).parent / "fixtures" / "approval_v1" / "vectors.json").read_text())
    key = bytes.fromhex(doc["key_hex"])
    assert len(doc["cases"]) >= 4
    for case in doc["cases"]:
        assert decision_mac(key, case["action_id"], case["nonce"], case["decision"]) == case["mac"]
