"""The menu-bar app shows what the hook did.

The hook, the trail and the Mac app each passed their own tests for a year
while every call a deny rule blocked was written as ``decision_made: allow``
and the light stayed green (fixed in #787). Nothing checked the seam. This
test drives an allow, a deny-rule block and an escalation through the real
hook entry point (the plugin's ``hooks/run.sh``, which execs ``vaara hook``),
then reads the resulting trail with the app's own SQL and verdict mapping
from clients/macos/Sources/VaaraMenuBar/Model.swift, and fails if the app
would show anything other than what the hook decided.

CI has no Swift on Linux, so the queries are copied verbatim and the
mapping is ported line for line. ``test_model_swift_still_says_this``
fails the moment either changes in Model.swift; update both together.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

REPO = Path(__file__).resolve().parents[1]
MODEL_SWIFT = REPO / "clients" / "macos" / "Sources" / "VaaraMenuBar" / "Model.swift"
RUN_SH = REPO / "plugins" / "claude-code-vaara-governance" / "hooks" / "run.sh"

# --- Model.swift, verbatim -------------------------------------------------

# eventColumns + newDecisions (the live feed and its notifications)
NEW_DECISIONS_SQL = """SELECT seq, event_type, tool_name, timestamp, data FROM audit_records
 WHERE seq > ? AND event_type IN
  ('action_blocked', 'decision_made', 'escalation_sent')
 ORDER BY seq"""

# eventColumns + history (the register)
HISTORY_SQL = """SELECT seq, event_type, tool_name, timestamp, data FROM audit_records
 WHERE event_type IN
  ('action_blocked', 'decision_made', 'escalation_sent')
 ORDER BY seq DESC LIMIT 1000"""

# overallState (the light)
OVERALL_STATE_SQL = """SELECT event_type, data, timestamp FROM audit_records
WHERE timestamp >= ? AND event_type IN
  ('action_blocked', 'escalation_sent', 'decision_made')
ORDER BY seq DESC"""

# agentSummaries (per-agent allowed / escalated / denied counters)
AGENT_SUMMARIES_SQL = """SELECT agent_id, event_type, data, timestamp
FROM audit_records
WHERE timestamp >= ?
ORDER BY seq DESC"""

# The Swift each port below mirrors.
SWIFT_MAPPINGS = [
    # decisionEvent
    """if eventType == "action_blocked" || decision == "deny" {
            verdict = "deny"
        } else if decision == "escalate" {
            verdict = "escalate"
        } else if includeAllows && eventType == "decision_made" && decision == "allow" {
            verdict = "allow"
        } else {
            return nil
        }""",
    # overallState
    """if eventType == "action_blocked" { return (ts, .red) }
                    if eventType == "escalation_sent" { return (ts, .yellow) }
                    switch (parseData(sqlite3_column_text(stmt, 1))["decision"] as? String) ?? "" {
                    case "deny":     return (ts, .red)
                    case "escalate": return (ts, .yellow)
                    case "allow":    return (ts, .green)
                    default:         continue
                    }""",
    # agentSummaries
    """switch eventType {
                    case "action_blocked":  verdict = .red
                    case "decision_made":
                        switch (parseData(sqlite3_column_text(stmt, 2))["decision"] as? String) ?? "" {
                        case "deny":     verdict = .red
                        case "escalate": verdict = .yellow
                        case "allow":    verdict = .green
                        default:         verdict = nil
                        }
                    default: verdict = nil
                    }""",
]


def _decision_event(event_type: str, data: dict, include_allows: bool = False):
    decision = data.get("decision") or ""
    if event_type == "action_blocked" or decision == "deny":
        return "deny"
    if decision == "escalate":
        return "escalate"
    if include_allows and event_type == "decision_made" and decision == "allow":
        return "allow"
    return None


def _overall_state(db: Path) -> str:
    for event_type, data, _ts in _query(db, OVERALL_STATE_SQL, (0,)):
        if event_type == "action_blocked":
            return "red"
        if event_type == "escalation_sent":
            return "yellow"
        decision = json.loads(data or "{}").get("decision") or ""
        if decision in ("deny", "escalate", "allow"):
            return {"deny": "red", "escalate": "yellow", "allow": "green"}[decision]
    return "green"


def _agent_counts(db: Path) -> dict[str, dict[str, int]]:
    acc: dict[str, dict[str, int]] = {}
    for agent, event_type, data, _ts in _query(db, AGENT_SUMMARIES_SQL, (0,)):
        counts = acc.setdefault(agent, {"allowed": 0, "escalated": 0, "denied": 0})
        verdict = None
        if event_type == "action_blocked":
            verdict = "denied"
        elif event_type == "decision_made":
            verdict = {"deny": "denied", "escalate": "escalated",
                       "allow": "allowed"}.get(json.loads(data or "{}").get("decision") or "")
        if verdict:
            counts[verdict] += 1
    return acc


def _events(db: Path, sql: str, params: tuple, include_allows: bool):
    out = []
    for _seq, event_type, tool, _ts, data in _query(db, sql, params):
        verdict = _decision_event(event_type, json.loads(data or "{}"), include_allows)
        if verdict:
            out.append((verdict, tool))
    return out


def _query(db: Path, sql: str, params: tuple):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


# --- the real hook ---------------------------------------------------------

def _hook(home: Path, event: dict) -> int:
    """One PreToolUse call through the plugin's run.sh, as Claude Code makes it."""
    bindir = home / "bin"
    shim = bindir / "vaara"
    if not shim.exists():
        bindir.mkdir(parents=True)
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" -c '
            '"import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))" "$@"\n')
        shim.chmod(0o755)
        cfg = home / ".vaara" / "claude-code" / "config.json"
        cfg.parent.mkdir(parents=True)
        # Only the MCP scorer escalates; these thresholds make any MCP call
        # land between them. No approval surface runs here, so the escalate
        # is held and answered by nobody, as on an unattended machine.
        cfg.write_text(json.dumps({"thresholds": {"escalate": 0.0, "deny": 0.99},
                                   "notifications": False}))
    env = {"HOME": str(home), "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
           "PYTHONPATH": os.pathsep.join(sys.path), "VAARA_PLUGIN_APPROVALS": "0"}
    proc = subprocess.run(["sh", str(RUN_SH), "pre-tool-use"],
                          input=json.dumps({"session_id": "s", **event}),
                          capture_output=True, text=True, env=env, timeout=120)
    return proc.returncode


def _allow(home: Path) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": "ls"}}


def _rule_block(home: Path) -> dict:
    return {"tool_name": "Write",
            "tool_input": {"file_path": str(home / ".claude" / "settings.json"),
                           "content": "{}"}}


def _escalate(home: Path) -> dict:
    return {"tool_name": "mcp__files__read_file", "tool_input": {"path": "README.md"}}


def _trail(home: Path) -> Path:
    return home / ".vaara" / "trail" / "audit.db"


@pytest.mark.parametrize("call, exit_code, light", [
    (_allow, 0, "green"),
    (_rule_block, 2, "red"),
    (_escalate, 2, "yellow"),
])
def test_the_light_shows_the_hooks_decision(tmp_path, call, exit_code, light):
    assert _hook(tmp_path, call(tmp_path)) == exit_code
    assert _overall_state(_trail(tmp_path)) == light


def test_each_call_is_shown_once_with_the_hooks_verdict(tmp_path):
    for call, exit_code in ((_allow, 0), (_rule_block, 2), (_escalate, 2)):
        assert _hook(tmp_path, call(tmp_path)) == exit_code
    db = _trail(tmp_path)

    assert _events(db, NEW_DECISIONS_SQL, (-1,), include_allows=False) == [
        ("deny", "Write"), ("escalate", "mcp__files__read_file")]
    assert _events(db, HISTORY_SQL, (), include_allows=True) == [
        ("escalate", "mcp__files__read_file"), ("deny", "Write"), ("allow", "Bash")]
    assert _agent_counts(db) == {
        "claude-code": {"allowed": 1, "escalated": 1, "denied": 1}}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_model_swift_still_says_this():
    swift = _squash(MODEL_SWIFT.read_text())
    columns = "SELECT seq, event_type, tool_name, timestamp, data FROM audit_records"
    assert f'"{columns}"' in swift, "Model.swift eventColumns changed"
    for sql in (NEW_DECISIONS_SQL, HISTORY_SQL, OVERALL_STATE_SQL, AGENT_SUMMARIES_SQL):
        # newDecisions and history are eventColumns + their own tail.
        assert _squash(sql).removeprefix(columns + " ") in swift, (
            f"Model.swift query changed:\n{sql}")
    for mapping in SWIFT_MAPPINGS:
        assert _squash(mapping) in swift, f"Model.swift mapping changed:\n{mapping}"
