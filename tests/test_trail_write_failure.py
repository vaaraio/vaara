# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The trail stopped recording for thirteen days and nothing said so.

Found 2026-09-04: ``vaara hook pre-tool-use`` had raised
``sqlite3.DatabaseError: database disk image is malformed`` on every insert
since 2026-08-22, printed a traceback, and exited 0. These tests pin the two
things that made that invisible — a per-record traceback nobody reads, and a
failure counter that lived in a process which exits after every tool call —
and they pin that fail-open behaviour is unchanged.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from vaara.audit import write_failure as wf
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail, EventType
from vaara.integrations import claude_code_hooks as hooks
from vaara.taxonomy.actions import UNKNOWN_ACTION, ActionRequest


def _request(tool_name: str = "Bash") -> ActionRequest:
    return ActionRequest(
        agent_id="claude-code", tool_name=tool_name,
        action_type=UNKNOWN_ACTION, parameters={"command": "ls " * 20},
    )


def _populated_trail(tmp_path: Path, records: int = 300) -> Path:
    """A real audit.db with enough records to span several SQLite pages."""
    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    for _ in range(records):
        trail.record_action_requested(_request())
    backend.checkpoint_wal()
    backend.close()
    return db


def _corrupt_a_later_page(db: Path) -> None:
    """Zero one B-tree page, leaving the header readable.

    This is the shape of the 2026-08-22 damage: not a file that is obviously
    not a database, but a real trail with a hole in it.
    """
    data = bytearray(db.read_bytes())
    assert len(data) > 12288, "need a trail with more than three pages to damage one"
    data[8192:12288] = b"\x00" * 4096
    db.write_bytes(bytes(data))


# ── the marker itself ─────────────────────────────────────────────────────

class TestMarkerPath:
    def test_sits_next_to_the_database(self, tmp_path: Path):
        assert wf.marker_path(tmp_path / "audit.db") == (
            tmp_path / "audit.db.write-failure.json"
        )

    @pytest.mark.parametrize("db", [None, "", ":memory:", "file::memory:?cache=shared"])
    def test_no_marker_where_there_is_no_file(self, db):
        assert wf.marker_path(db) is None

    def test_an_in_memory_trail_reports_nowhere_and_does_not_raise(self):
        assert wf.record_failure(":memory:", RuntimeError("boom")) is None
        assert wf.active_failure(":memory:") is None
        assert wf.mark_recovered(":memory:") is None
        assert wf.quick_check(":memory:") is None


class TestRecordFailure:
    def test_the_first_failure_writes_a_marker(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        state = wf.record_failure(db, sqlite3.DatabaseError("database disk image is malformed"))

        assert state["count"] == 1
        assert state["notify"] is True
        assert state["resolved"] is False
        assert "malformed" in state["error"]
        on_disk = json.loads((tmp_path / "audit.db.write-failure.json").read_text())
        assert on_disk["count"] == 1
        # `notify` is a decision for the caller, not persisted state.
        assert "notify" not in on_disk

    def test_the_count_survives_the_process_that_wrote_it(self, tmp_path: Path):
        """The defect in one line: the old counter could never reach 2.

        ``AuditTrail._persistence_failures`` is per-process and the hook is a
        fresh process per tool call, so nothing accumulated. The marker is
        read back off disk, so it does.
        """
        db = tmp_path / "audit.db"
        for _ in range(5):
            state = wf.record_failure(db, RuntimeError("still broken"))
        assert state["count"] == 5

    def test_the_clock_starts_at_the_first_failure_not_the_last(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        first = wf.record_failure(db, RuntimeError("x"))
        time.sleep(0.01)
        later = wf.record_failure(db, RuntimeError("x"))

        assert later["first_failure"] == first["first_failure"]
        assert later["last_failure"] > later["first_failure"]

    def test_repeat_failures_stay_quiet_until_the_interval_passes(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        assert wf.record_failure(db, RuntimeError("x"))["notify"] is True
        assert wf.record_failure(db, RuntimeError("x"))["notify"] is False
        assert wf.record_failure(db, RuntimeError("x"), notify_interval=0)["notify"] is True

    def test_a_new_outage_after_a_recovery_starts_its_own_count(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        wf.record_failure(db, RuntimeError("x"))
        wf.record_failure(db, RuntimeError("x"))
        wf.mark_recovered(db)

        fresh = wf.record_failure(db, RuntimeError("x"))
        assert fresh["count"] == 1, "a resolved outage must not inflate the next one"
        assert fresh["notify"] is True

    def test_an_unwritable_directory_does_not_raise(self, tmp_path: Path):
        blocked = tmp_path / "nodir"
        blocked.write_text("i am a file, not a directory")
        assert wf.record_failure(blocked / "audit.db", RuntimeError("x")) is not None

    def test_a_corrupt_marker_is_treated_as_no_marker(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        (tmp_path / "audit.db.write-failure.json").write_text("{not json")
        assert wf.read_marker(db) is None
        assert wf.record_failure(db, RuntimeError("x"))["count"] == 1


class TestRecovery:
    def test_recovery_clears_the_active_signal(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        wf.record_failure(db, RuntimeError("x"))
        assert wf.active_failure(db) is not None

        resolved = wf.mark_recovered(db)
        assert resolved["resolved"] is True
        assert wf.active_failure(db) is None

    def test_recovery_keeps_the_record_of_the_outage(self, tmp_path: Path):
        """Deleting it would be the same mistake the product argues against."""
        db = tmp_path / "audit.db"
        wf.record_failure(db, RuntimeError("x"))
        wf.mark_recovered(db)

        marker = tmp_path / "audit.db.write-failure.json"
        assert marker.exists()
        assert json.loads(marker.read_text())["count"] == 1

    def test_recovering_twice_is_a_no_op(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        wf.record_failure(db, RuntimeError("x"))
        assert wf.mark_recovered(db) is not None
        assert wf.mark_recovered(db) is None


class TestQuickCheck:
    def test_a_healthy_trail_reads_clean(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        backend = SQLiteAuditBackend(db)
        backend.close()
        assert wf.quick_check(db) is None

    def test_a_file_that_is_not_a_database_is_reported(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        db.write_bytes(b"\x00" * 8192)
        assert wf.quick_check(db) is not None

    def test_a_real_trail_with_a_damaged_page_is_reported(self, tmp_path: Path):
        db = _populated_trail(tmp_path)
        assert wf.quick_check(db) is None
        _corrupt_a_later_page(db)
        problem = wf.quick_check(db)
        assert problem is not None
        assert "page" in problem

    def test_a_missing_file_is_not_a_failure(self, tmp_path: Path):
        assert wf.quick_check(tmp_path / "absent.db") is None

    def test_a_large_trail_is_skipped_rather_than_stalling_startup(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        db.write_bytes(b"\x00" * 8192)
        assert wf.quick_check(db, max_bytes=16) is None


# ── the trail's own write path ────────────────────────────────────────────

class _BrokenBackend:
    """A store whose inserts fail exactly the way 2026-08-22's did."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def append_record(self, record, stamp):
        stamp("")
        raise sqlite3.DatabaseError("database disk image is malformed")

    def write_record(self, record):  # pragma: no cover - the trail calls append_record
        self.append_record(record, lambda _prev: None)


def _trail_on(backend) -> AuditTrail:
    trail = AuditTrail(on_record=backend.write_record)
    trail._backend = backend
    return trail


class TestTrailMarksItsOwnFailures:
    def test_a_failed_append_leaves_a_marker(self, tmp_path: Path):
        trail = _trail_on(_BrokenBackend(tmp_path / "audit.db"))
        trail.record_action_requested(_request())
        state = wf.active_failure(tmp_path / "audit.db")
        assert state is not None
        assert "malformed" in state["error"]
        assert state["stage"] == "append_record"

    def test_the_trail_still_fails_open(self, tmp_path: Path):
        """Fail-open is the right default and this change does not touch it."""
        trail = _trail_on(_BrokenBackend(tmp_path / "audit.db"))
        for i in range(3):
            trail.record_action_requested(_request())
        assert len(trail._records) == 3
        assert wf.active_failure(tmp_path / "audit.db")["count"] == 3

    def test_only_the_first_failure_carries_a_traceback(self, tmp_path: Path, caplog):
        """A traceback per tool call is what made the outage read as normal."""
        trail = _trail_on(_BrokenBackend(tmp_path / "audit.db"))
        with caplog.at_level(logging.ERROR, logger="vaara.audit.trail"):
            for i in range(4):
                trail.record_action_requested(_request())

        with_traceback = [r for r in caplog.records if r.exc_info]
        assert len(with_traceback) == 1, "one diagnosis, not one per call"
        assert len(caplog.records) == 4, "every failure is still reported"
        assert "NOT RECORDING" in caplog.records[-1].getMessage()

    def test_a_working_trail_writes_no_marker(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        backend = SQLiteAuditBackend(db)
        trail = backend.load_trail()
        trail._on_record = backend.write_record
        trail.record_action_requested(_request())
        backend.close()
        assert wf.read_marker(db) is None

    def test_the_next_good_write_clears_the_marker(self, tmp_path: Path):
        db = tmp_path / "audit.db"
        wf.record_failure(db, sqlite3.DatabaseError("database disk image is malformed"))

        backend = SQLiteAuditBackend(db)
        trail = backend.load_trail()
        trail._on_record = backend.write_record
        trail.record_action_requested(_request())
        backend.close()

        assert wf.active_failure(db) is None
        assert wf.read_marker(db)["resolved"] is True

    def test_an_in_memory_trail_is_unaffected(self):
        trail = AuditTrail()
        trail.record_action_requested(_request())
        assert trail._persistence_failures == 0
        assert len(trail._records) == 1


# ── the hook, which is where the operator actually looks ──────────────────

def _feed(monkeypatch: pytest.MonkeyPatch, event: dict) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))


@pytest.fixture
def audit_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "audit.db"
    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(db))
    monkeypatch.setenv("VAARA_PLUGIN_NOTIFY", "0")
    return db


class TestHookSurfacesTheOutage:
    def test_a_dead_trail_does_not_block_the_call(self, audit_db, monkeypatch, capsys):
        audit_db.write_bytes(b"\x00" * 8192)
        _feed(monkeypatch, {
            "tool_name": "Bash", "tool_input": {"command": "ls"}, "session_id": "s1",
        })
        assert hooks.run_pre_tool_use() == 0

    def test_a_dead_trail_says_so_on_stderr(self, audit_db, monkeypatch, capsys):
        audit_db.write_bytes(b"\x00" * 8192)
        _feed(monkeypatch, {
            "tool_name": "Bash", "tool_input": {"command": "ls"}, "session_id": "s1",
        })
        hooks.run_pre_tool_use()

        err = capsys.readouterr().err
        assert "NOT RECORDING" in err
        assert "Traceback" not in err

    def test_an_mcp_call_passes_through_a_dead_trail(self, audit_db, monkeypatch):
        """This path used to take the hook down with an unhandled exception."""
        audit_db.write_bytes(b"\x00" * 8192)
        _feed(monkeypatch, {
            "tool_name": "mcp__github__create_issue",
            "tool_input": {"title": "x"}, "session_id": "s1",
        })
        assert hooks.run_pre_tool_use() == 0

    def test_session_start_reports_a_standing_outage(self, audit_db, monkeypatch, capsys):
        audit_db.touch()
        wf.record_failure(audit_db, sqlite3.DatabaseError("database disk image is malformed"))
        _feed(monkeypatch, {"session_id": "s1"})
        assert hooks.run_session_start() == 0

        err = capsys.readouterr().err
        assert "THE AUDIT TRAIL IS NOT RECORDING" in err
        assert "integrity_check" in err

    def test_session_start_surfaces_a_damaged_trail_with_no_marker(
        self, audit_db, monkeypatch, capsys
    ):
        """A trail damaged while nothing was writing still gets reported."""
        audit_db.write_bytes(b"\x00" * 8192)
        _feed(monkeypatch, {"session_id": "s1"})
        assert hooks.run_session_start() == 0

        err = capsys.readouterr().err
        assert "integrity_check" in err
        assert "Do not delete the file" in err

    def test_the_quick_check_branch_reports_a_trail_that_still_opens(
        self, tmp_path: Path, capsys
    ):
        """The belt for damage that no write has hit yet.

        Reached directly because the marker branch runs first, and a trail
        damaged badly enough to fail to open lands there instead.
        """
        db = _populated_trail(tmp_path)
        _corrupt_a_later_page(db)
        hooks._report_trail_health({"notifications": False}, db, existed=True)
        assert "does not read clean" in capsys.readouterr().err

    def test_session_start_on_a_healthy_trail_is_quiet(self, audit_db, monkeypatch, capsys):
        SQLiteAuditBackend(audit_db).close()
        _feed(monkeypatch, {"session_id": "s1"})
        assert hooks.run_session_start() == 0

        err = capsys.readouterr().err
        assert "NOT RECORDING" not in err
        assert "does not read clean" not in err

    def test_a_recovered_trail_stops_nagging(self, audit_db, monkeypatch, capsys):
        SQLiteAuditBackend(audit_db).close()
        wf.record_failure(audit_db, RuntimeError("x"))
        wf.mark_recovered(audit_db)

        _feed(monkeypatch, {"session_id": "s1"})
        hooks.run_session_start()
        assert "NOT RECORDING" not in capsys.readouterr().err

    def test_the_recorded_event_type_is_unchanged_when_the_trail_works(
        self, audit_db, monkeypatch
    ):
        _feed(monkeypatch, {
            "tool_name": "Bash", "tool_input": {"command": "ls"}, "session_id": "s1",
        })
        assert hooks.run_pre_tool_use() == 0
        con = sqlite3.connect(audit_db)
        try:
            kinds = [row[0] for row in con.execute("select event_type from audit_records")]
        finally:
            con.close()
        assert EventType.ACTION_REQUESTED.value in kinds
