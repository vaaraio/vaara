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

from vaara.approvals import request_approval


def _respond(approvals_dir: Path, decision: str, captured: dict) -> threading.Thread:
    """Play the watcher: wait for a request file, record it, write the decision."""

    def responder() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            requests = list(approvals_dir.glob("*.request.json"))
            if requests:
                captured.update(json.loads(requests[0].read_text()))
                action_id = requests[0].name.removesuffix(".request.json")
                (approvals_dir / f"{action_id}.decision.json").write_text(
                    json.dumps({"decision": decision, "decided_at": time.time()})
                )
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
