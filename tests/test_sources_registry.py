# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The engine lists every trail it writes in ~/.vaara/sources.json."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from vaara.audit import sources
from vaara.audit.sqlite_backend import SQLiteAuditBackend


def _entries(home):
    return json.loads((home / ".vaara" / "sources.json").read_text())["sources"]


def test_a_written_trail_is_listed(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_HOME", str(tmp_path / "home"))
    db = tmp_path / "somewhere" / "audit.db"
    db.parent.mkdir()
    trail = SQLiteAuditBackend(db).load_trail()
    assert not (tmp_path / "home" / ".vaara" / "sources.json").exists()  # opening writes nothing
    trail.record_decision(action_id="a", agent_id="x", tool_name="t",
                          decision="allow", reason="r", risk_score=0.1)
    [entry] = _entries(tmp_path / "home")
    assert entry["trail"] == str(db.resolve())
    assert entry["receipts"] == str(db.resolve().parent / "receipts")


def test_two_trails_and_no_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_HOME", str(tmp_path / "home" / ".vaara"))
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    for db in (a, b, a):
        SQLiteAuditBackend(db).load_trail().record_decision(
            action_id="a", agent_id="x", tool_name="t", decision="allow", reason="r",
            risk_score=0.1)
    assert sorted(e["trail"] for e in _entries(tmp_path / "home")) == sorted(
        [str(a.resolve()), str(b.resolve())])


def test_last_write_refreshes_hourly_and_gone_trails_drop(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_HOME", str(tmp_path))
    live, gone = tmp_path / "live.db", tmp_path / "gone.db"
    live.write_text(""); gone.write_text("")
    t0 = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
    assert sources.register(live, now=t0)
    assert sources.register(gone, now=t0)
    assert not sources.register(live, now=t0 + timedelta(minutes=30))
    gone.unlink()
    assert sources.register(live, now=t0 + timedelta(hours=2))
    [entry] = _entries(tmp_path)
    assert entry["last_write"] == "2026-09-25T03:00:00Z"
    assert entry["first_write"] == "2026-09-25T01:00:00Z"


def test_memory_trails_and_a_broken_file_are_harmless(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_HOME", str(tmp_path))
    assert not sources.register(":memory:")
    (tmp_path / ".vaara").mkdir()
    (tmp_path / ".vaara" / "sources.json").write_text("{not json")
    db = tmp_path / "t.db"
    db.write_text("")
    assert sources.register(db)
    assert [e["trail"] for e in _entries(tmp_path)] == [str(db.resolve())]
