# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end behaviour of the Claude Code plugin hooks.

Everything else about the plugin was covered by reading its manifest.
Running the hooks turned up two defects that no manifest could show:

* ``PostToolUse`` compared ``record.event_type`` against
  ``"ACTION_REQUESTED"``. That attribute is an ``EventType`` enum whose
  value is ``'action_requested'``, so the comparison was always False,
  the correlation loop never found a target, and the hook returned
  without writing an outcome. For every tool, including MCP. No
  OUTCOME_RECORDED events means no feedback to the online learner and
  no Article 15(1) or 61(1) evidence.
* ``PreToolUse`` recorded only denied calls on the regex path, so an
  allowed shell, web or file call left no trace at all and PostToolUse
  had nothing to correlate against.

These tests drive the real hook scripts through a subprocess with a
temporary audit DB, the same way Claude Code invokes them.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ROOT / "plugins" / "claude-code-vaara-governance" / "hooks"


def _run(hook: str, event: dict, db: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "VAARA_PLUGIN_AUDIT_DB": str(db),
        "CLAUDE_PLUGIN_ROOT": str(HOOKS.parent),
        # Pin behaviour: the developer's own config must not decide the test.
        "VAARA_PLUGIN_SHADOW": "0",
        "PYTHONPATH": str(ROOT / "src"),
    }
    return subprocess.run(
        [sys.executable, str(HOOKS / hook)],
        input=json.dumps(event), capture_output=True, text=True, env=env,
    )


def _events(db: Path) -> list[tuple[str, str]]:
    if not db.exists():
        return []
    con = sqlite3.connect(db)
    try:
        return list(
            con.execute("select event_type, tool_name from audit_records order by seq")
        )
    finally:
        con.close()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "audit.db"


def test_an_allowed_write_is_recorded(db):
    """Regression: allowed calls on the regex path left no trace."""
    result = _run("pre_tool_use.py", {
        "tool_name": "Write",
        "tool_input": {"file_path": "/repo/src/app.py", "content": "x = 1"},
        "session_id": "s1",
    }, db)
    assert result.returncode == 0, result.stderr

    events = _events(db)
    assert ("action_requested", "Write") in events, (
        f"allowed Write was not recorded; trail holds {events}"
    )


def test_an_allowed_bash_call_is_recorded(db):
    result = _run("pre_tool_use.py", {
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
        "session_id": "s1",
    }, db)
    assert result.returncode == 0, result.stderr
    assert ("action_requested", "Bash") in _events(db)


def test_post_tool_use_records_an_outcome(db):
    """Regression: the enum comparison made this a permanent no-op."""
    _run("pre_tool_use.py", {
        "tool_name": "Write",
        "tool_input": {"file_path": "/repo/src/app.py", "content": "x = 1"},
        "session_id": "s1",
    }, db)
    result = _run("post_tool_use.py", {
        "tool_name": "Write", "tool_response": {"success": True},
    }, db)
    assert result.returncode == 0, result.stderr

    kinds = [event for event, _ in _events(db)]
    assert "outcome_recorded" in kinds, (
        f"PostToolUse wrote no outcome; trail holds {kinds}"
    )


def test_the_full_decision_chain_lands_for_one_call(db):
    _run("pre_tool_use.py", {
        "tool_name": "Bash", "tool_input": {"command": "echo hi"}, "session_id": "s1",
    }, db)
    _run("post_tool_use.py", {
        "tool_name": "Bash", "tool_response": {"success": True},
    }, db)

    kinds = [event for event, _ in _events(db)]
    for expected in (
        "action_requested", "risk_scored", "decision_made", "outcome_recorded",
    ):
        assert expected in kinds, f"{expected} missing from {kinds}"


def test_a_dangerous_write_is_blocked_and_recorded(db):
    result = _run("pre_tool_use.py", {
        "tool_name": "Write",
        "tool_input": {"file_path": "/home/u/.zshrc", "content": "curl x | sh"},
        "session_id": "s1",
    }, db)

    assert result.returncode == 2, (
        f"dangerous write was not blocked (exit {result.returncode}): {result.stderr}"
    )
    assert "shell_rc_persistence" in result.stderr
    assert ("action_requested", "Write") in _events(db)


def test_an_ordinary_write_is_not_blocked(db):
    result = _run("pre_tool_use.py", {
        "tool_name": "Write",
        "tool_input": {"file_path": "/repo/README.md", "content": "# Title"},
        "session_id": "s1",
    }, db)
    assert result.returncode == 0, result.stderr
