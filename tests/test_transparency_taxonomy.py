"""Tests for v0.6 prEN ISO/IEC 12792 four-axis transparency taxonomy."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import (
    TRANSPARENCY_DEFAULTS,
    AuditRecord,
    EventType,
)


def _record(event_type: EventType, **kwargs) -> AuditRecord:
    """Helper for building minimal records."""
    base = dict(
        record_id=kwargs.pop("record_id", "rec"),
        action_id=kwargs.pop("action_id", "act"),
        event_type=event_type,
        timestamp=kwargs.pop("timestamp", time.time()),
        agent_id="ag",
        tool_name="t",
    )
    return AuditRecord(**base, **kwargs)


# ── Default classification ─────────────────────────────────────────────────

def test_defaults_cover_every_event_type() -> None:
    for event_type in list(EventType):
        assert event_type in TRANSPARENCY_DEFAULTS
        d = TRANSPARENCY_DEFAULTS[event_type]
        for axis in ("system_operation", "data_usage", "decision_making"):
            assert isinstance(d[axis], str)


def test_action_requested_defaults() -> None:
    rec = _record(EventType.ACTION_REQUESTED)
    assert rec.system_operation == "logging_intake"
    assert rec.data_usage == "tool_args+context"
    assert rec.decision_making == "n/a"
    assert rec.limitations is None


def test_risk_scored_defaults() -> None:
    rec = _record(EventType.RISK_SCORED)
    assert rec.system_operation == "scoring"
    assert rec.decision_making == "heuristic_score"


def test_human_oversight_defaults() -> None:
    rec = _record(EventType.ESCALATION_RESOLVED)
    assert rec.system_operation == "human_oversight"
    assert rec.decision_making == "human_decision"


# ── Override ───────────────────────────────────────────────────────────────

def test_per_record_override_takes_precedence() -> None:
    rec = _record(
        EventType.RISK_SCORED,
        system_operation="custom_scoring",
        data_usage="custom_data",
        decision_making="custom_decision",
        limitations="custom_limits",
    )
    assert rec.system_operation == "custom_scoring"
    assert rec.data_usage == "custom_data"
    assert rec.decision_making == "custom_decision"
    assert rec.limitations == "custom_limits"


def test_loaded_record_keeps_null_fields() -> None:
    """Records constructed with a pre-set record_hash skip default-filling.

    This is the load-from-DB path. Filling defaults retroactively would
    misrepresent a historical record's actual decision-making mode.
    """
    rec = _record(
        EventType.RISK_SCORED,
        record_hash="sha256:fake-but-non-empty",
    )
    assert rec.system_operation is None
    assert rec.data_usage is None
    assert rec.decision_making is None


# ── Hash chain backward compat ──────────────────────────────────────────────

def test_compute_hash_excludes_transparency_fields() -> None:
    """Transparency fields are NOT tamper-evident — old chains stay valid."""
    rec_with = _record(EventType.RISK_SCORED, timestamp=1234567890.0)
    # Same content, but explicitly null transparency (mimics a loaded old record).
    rec_without = _record(
        EventType.RISK_SCORED,
        timestamp=1234567890.0,
        record_hash="x",  # forces skip-defaults path
    )
    rec_without.record_hash = ""  # reset for compute
    assert rec_with.compute_hash() == rec_without.compute_hash()


# ── SQLite persistence + migration ─────────────────────────────────────────

def test_write_and_load_persists_transparency(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    with SQLiteAuditBackend(db) as backend:
        rec = _record(
            EventType.DECISION_MADE,
            record_id="r-persist",
            action_id="a-persist",
            limitations="custom_limit",
        )
        rec.record_hash = rec.compute_hash()
        backend.write_record(rec)

        loaded = backend.query_by_action("a-persist")
        assert len(loaded) == 1
        lr = loaded[0]
        assert lr.system_operation == "decision_threshold"
        assert lr.decision_making == "threshold_match"
        assert lr.limitations == "custom_limit"


def test_migration_from_v2_to_v3(tmp_path: Path) -> None:
    """Pre-v0.6 DB (schema v2) opens cleanly under v3 backend."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO audit_meta (key, value) VALUES ('schema_version', '2');
        CREATE TABLE audit_records (
            record_id     TEXT PRIMARY KEY,
            action_id     TEXT NOT NULL,
            event_type    TEXT NOT NULL,
            timestamp     REAL NOT NULL,
            agent_id      TEXT NOT NULL,
            tool_name     TEXT NOT NULL,
            data          TEXT NOT NULL DEFAULT '{}',
            regulatory    TEXT NOT NULL DEFAULT '[]',
            previous_hash TEXT NOT NULL DEFAULT '',
            record_hash   TEXT NOT NULL DEFAULT '',
            seq           INTEGER NOT NULL,
            tenant_id     TEXT NOT NULL DEFAULT ''
        );
        INSERT INTO audit_records VALUES (
            'old-rec', 'old-act', 'risk_scored', 1700000000.0,
            'ag', 't', '{}', '[]', '', 'OLD_HASH', 0, ''
        );
        """
    )
    conn.commit()
    conn.close()

    with SQLiteAuditBackend(db) as backend:
        loaded = backend.query_by_action("old-act")
        assert len(loaded) == 1
        old = loaded[0]
        assert old.record_hash == "OLD_HASH"   # stored hash preserved
        assert old.system_operation is None    # transparency columns NULL
        assert old.data_usage is None
        assert old.decision_making is None
        assert old.limitations is None
