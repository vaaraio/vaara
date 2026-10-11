# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A `vaara check` nobody followed with `vaara outcome` does not stay pending forever.

`vaara check` stores a pending outcome so `vaara outcome` in another process
can close it. When no outcome ever comes (a crash, an abandoned run), the
row stayed pending for good: the only cleanup was a purge method nothing
called, and it deleted rows silently. Now `vaara check` starts by closing
every pending outcome older than a day as `orphaned`, on the trail, the way
the LLM proxy already closes its own. The proxy's `llm.prompt` rows are left
to it.
"""

from __future__ import annotations

import json
import sqlite3
import time

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.cli import main
from vaara.pipeline import InterceptionPipeline

DAY = 86400.0


def _pending(db, tool="fs.read", age=2 * DAY):
    """A checked action with a pending outcome created ``age`` seconds ago."""
    backend = SQLiteAuditBackend(db)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    result = pipeline.intercept(agent_id="agent-1", tool_name=tool, parameters={"p": "1"})
    backend.close()
    conn = sqlite3.connect(db)
    conn.execute("UPDATE pending_outcomes SET created_at=? WHERE action_id=?",
                 (time.time() - age, result.action_id))
    conn.commit()
    conn.close()
    return result.action_id


def _rows(db):
    conn = sqlite3.connect(db)
    rows = {r[0]: r[1] for r in conn.execute("SELECT action_id, tool_name FROM pending_outcomes")}
    conn.close()
    return rows


def _outcomes(db):
    trail = SQLiteAuditBackend(db).load_trail()
    return {r.action_id: r for r in trail._records if r.event_type == EventType.OUTCOME_RECORDED}


def test_check_closes_a_day_old_pending_outcome_as_orphaned_on_the_trail(tmp_path, capsys):
    db = str(tmp_path / "audit.db")
    old = _pending(db, age=2 * DAY)

    rc = main(["check", "--tool", "fs.read", "--agent", "agent-2", "--db", db])
    out = capsys.readouterr().out

    answer = json.loads(out)
    assert rc in (0, 1) and answer["decision"] in ("allow", "escalate", "deny")
    assert old not in _rows(db), "the orphaned row is still pending"
    recorded = _outcomes(db)
    assert old in recorded, "no outcome record was written for the orphan"
    assert recorded[old].data["description"].startswith("orphaned")
    assert recorded[old].data["outcome_severity"] == 0.5


def test_a_fresh_pending_outcome_is_left_for_vaara_outcome(tmp_path, capsys):
    db = str(tmp_path / "audit.db")
    fresh = _pending(db, age=3600)
    main(["check", "--tool", "fs.read", "--agent", "agent-2", "--db", db])
    capsys.readouterr()
    assert fresh in _rows(db)
    assert fresh not in _outcomes(db)


def test_the_proxys_llm_prompt_rows_are_left_to_the_proxy(tmp_path, capsys):
    db = str(tmp_path / "audit.db")
    prompt = _pending(db, tool="llm.prompt", age=3 * DAY)
    main(["check", "--tool", "fs.read", "--agent", "agent-2", "--db", db])
    capsys.readouterr()
    assert prompt in _rows(db)


def test_the_checks_own_outcome_stays_pending_for_vaara_outcome(tmp_path, capsys):
    db = str(tmp_path / "audit.db")
    main(["check", "--tool", "fs.read", "--agent", "agent-2", "--db", db])
    answer = json.loads(capsys.readouterr().out)
    if answer["allowed"]:
        assert answer["action_id"] in _rows(db)
        rc = main(["outcome", "--action-id", answer["action_id"], "--severity", "0.1", "--db", db])
        assert rc == 0
        assert answer["action_id"] not in _rows(db)
        assert answer["action_id"] in _outcomes(db)


def test_the_pipeline_method_reports_what_it_closed(tmp_path):
    db = str(tmp_path / "audit.db")
    old_a = _pending(db, age=2 * DAY)
    old_b = _pending(db, tool="shell.exec", age=5 * DAY)
    _pending(db, age=60)
    backend = SQLiteAuditBackend(db)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    closed = pipeline.close_orphaned_outcomes(max_age_seconds=DAY)
    assert sorted(closed) == sorted([old_a, old_b])
    assert pipeline.close_orphaned_outcomes(max_age_seconds=DAY) == []
