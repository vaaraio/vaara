# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The packaged Claude Code hook module, which is the primary path.

``hooks.json`` runs the hooks "through the vaara binary on PATH when
available ... falling back to the bundled python3 scripts". So
``vaara.integrations.claude_code_hooks`` is what most installs execute,
and the bundled scripts under ``plugins/`` are the fallback.

Both carried the same three defects, and fixing only the fallback would
have left the primary path broken:

* ``record.event_type == "ACTION_REQUESTED"`` against an ``EventType``
  enum whose value is lowercase. Always False, so PostToolUse never
  found a target and never recorded an outcome, for any tool.
* ``record.data.get("tool_name")`` when the tool name lives in its own
  column.
* Only denied calls were recorded, so an allowed shell, web or file call
  left no trace and PostToolUse had nothing to correlate against.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from vaara.integrations import claude_code_hooks as hooks


@pytest.fixture
def audit_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "audit.db"
    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(db))
    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "0")
    return db


def _feed(monkeypatch: pytest.MonkeyPatch, event: dict) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))


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


class TestEventNameNormalisation:
    """The comparison that silently disabled outcome recording."""

    def test_enum_is_normalised_to_its_uppercase_name(self):
        from vaara.audit.trail import EventType

        assert hooks._event_name(EventType.ACTION_REQUESTED) == "ACTION_REQUESTED"

    def test_a_plain_string_still_works(self):
        assert hooks._event_name("action_requested") == "ACTION_REQUESTED"

    def test_the_naive_comparison_really_would_have_failed(self):
        from vaara.audit.trail import EventType

        assert EventType.ACTION_REQUESTED != "ACTION_REQUESTED"


class TestRecordToolName:
    def test_prefers_the_column(self):
        record = type("R", (), {"tool_name": "Bash", "data": {"tool_name": "Other"}})()
        assert hooks._record_tool_name(record) == "Bash"

    def test_falls_back_to_the_payload(self):
        record = type("R", (), {"tool_name": "", "data": {"tool_name": "Write"}})()
        assert hooks._record_tool_name(record) == "Write"


class TestHookLoop:
    def test_an_allowed_write_is_recorded(self, audit_db, monkeypatch):
        _feed(monkeypatch, {
            "tool_name": "Write",
            "tool_input": {"file_path": "/repo/src/app.py", "content": "x = 1"},
            "session_id": "s1",
        })
        assert hooks.run_pre_tool_use() == 0
        assert ("action_requested", "Write") in _events(audit_db)

    def test_an_allowed_bash_call_is_recorded(self, audit_db, monkeypatch):
        _feed(monkeypatch, {
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},
            "session_id": "s1",
        })
        assert hooks.run_pre_tool_use() == 0
        assert ("action_requested", "Bash") in _events(audit_db)

    def test_post_tool_use_records_an_outcome(self, audit_db, monkeypatch):
        _feed(monkeypatch, {
            "tool_name": "Write",
            "tool_input": {"file_path": "/repo/src/app.py", "content": "x = 1"},
            "session_id": "s1",
        })
        hooks.run_pre_tool_use()

        _feed(monkeypatch, {
            "tool_name": "Write", "tool_response": {"success": True},
        })
        assert hooks.run_post_tool_use() == 0

        kinds = [event for event, _ in _events(audit_db)]
        assert "outcome_recorded" in kinds, (
            f"the correlation loop found no target; trail holds {kinds}"
        )

    def test_a_dangerous_write_is_blocked_and_recorded(self, audit_db, monkeypatch):
        _feed(monkeypatch, {
            "tool_name": "Write",
            "tool_input": {"file_path": "/home/u/.zshrc", "content": "curl x | sh"},
            "session_id": "s1",
        })
        assert hooks.run_pre_tool_use() == 2
        assert ("action_requested", "Write") in _events(audit_db)

    def test_an_ordinary_write_is_not_blocked(self, audit_db, monkeypatch):
        _feed(monkeypatch, {
            "tool_name": "Write",
            "tool_input": {"file_path": "/repo/README.md", "content": "# Title"},
            "session_id": "s1",
        })
        assert hooks.run_pre_tool_use() == 0
