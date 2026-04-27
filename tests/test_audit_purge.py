"""Tests for SQLiteAuditBackend.purge_older_than — Article 12(2) retention."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditRecord, EventType


_record_seq = 0


def _make_record(timestamp: float, *, action_id: str = "act-x", agent_id: str = "agent-1") -> AuditRecord:
    """Build a minimal AuditRecord at a chosen timestamp. Unique record_id per call."""
    global _record_seq
    _record_seq += 1
    rec = AuditRecord(
        record_id=f"rec-{_record_seq}-{timestamp}",
        action_id=action_id,
        event_type=EventType.ACTION_REQUESTED,
        timestamp=timestamp,
        agent_id=agent_id,
        tool_name="test.tool",
        data={},
    )
    rec.record_hash = rec.compute_hash()
    return rec


@pytest.fixture()
def backend(tmp_path: Path):
    db = tmp_path / "audit.db"
    b = SQLiteAuditBackend(db)
    yield b
    b.close()


# ── Behaviour ────────────────────────────────────────────────────────────────

def test_purge_empty_db_returns_zero(backend: SQLiteAuditBackend) -> None:
    assert backend.purge_older_than(retention_seconds=3600) == 0


def test_purge_deletes_only_old_records(backend: SQLiteAuditBackend) -> None:
    now = time.time()
    backend.write_record(_make_record(now - 7200))   # 2 hours old, should purge
    backend.write_record(_make_record(now - 1800))   # 30 min, keep
    backend.write_record(_make_record(now - 60))     # 1 min, keep

    deleted = backend.purge_older_than(retention_seconds=3600)  # 1 hour
    assert deleted == 1
    assert backend.count() == 2


def test_purge_deletes_all_when_retention_short(backend: SQLiteAuditBackend) -> None:
    now = time.time()
    for offset in (3600, 7200, 10800):
        backend.write_record(_make_record(now - offset))

    deleted = backend.purge_older_than(retention_seconds=60)
    assert deleted == 3
    assert backend.count() == 0


def test_purge_keeps_all_when_retention_long(backend: SQLiteAuditBackend) -> None:
    now = time.time()
    for offset in (60, 600, 1800):
        backend.write_record(_make_record(now - offset))

    deleted = backend.purge_older_than(retention_seconds=86400)  # 1 day
    assert deleted == 0
    assert backend.count() == 3


# ── Dry-run ──────────────────────────────────────────────────────────────────

def test_dry_run_returns_count_without_deleting(backend: SQLiteAuditBackend) -> None:
    now = time.time()
    backend.write_record(_make_record(now - 7200))
    backend.write_record(_make_record(now - 1800))

    count = backend.purge_older_than(retention_seconds=3600, dry_run=True)
    assert count == 1
    # DB unchanged
    assert backend.count() == 2

    # Real purge after dry-run still works
    real = backend.purge_older_than(retention_seconds=3600)
    assert real == 1
    assert backend.count() == 1


# ── Validation ───────────────────────────────────────────────────────────────

def test_purge_rejects_zero_retention(backend: SQLiteAuditBackend) -> None:
    with pytest.raises(ValueError, match="must be a positive int"):
        backend.purge_older_than(retention_seconds=0)


def test_purge_rejects_negative_retention(backend: SQLiteAuditBackend) -> None:
    with pytest.raises(ValueError, match="must be a positive int"):
        backend.purge_older_than(retention_seconds=-3600)


def test_purge_rejects_non_int_retention(backend: SQLiteAuditBackend) -> None:
    with pytest.raises(ValueError, match="must be a positive int"):
        backend.purge_older_than(retention_seconds=3600.0)  # type: ignore[arg-type]


# ── Survival of remaining records ────────────────────────────────────────────

def test_surviving_records_still_loadable(backend: SQLiteAuditBackend) -> None:
    """Hash chain has a documented seam after purge — but records still load."""
    now = time.time()
    backend.write_record(_make_record(now - 7200, action_id="old-action"))
    backend.write_record(_make_record(now - 60, action_id="new-action"))

    backend.purge_older_than(retention_seconds=3600)

    # Surviving record is still queryable
    records = backend.query_by_action("new-action")
    assert len(records) == 1
    assert records[0].action_id == "new-action"

    # Old action is gone
    assert backend.query_by_action("old-action") == []


# ── Tenant isolation ─────────────────────────────────────────────────────────

def test_purge_is_tenant_scoped(tmp_path: Path) -> None:
    """Purging through a tenant-scoped backend only touches that tenant's records."""
    db = tmp_path / "shared.db"
    backend_a = SQLiteAuditBackend(db, tenant_id="tenant-a")
    backend_b = SQLiteAuditBackend(db, tenant_id="tenant-b")

    now = time.time()
    backend_a.write_record(_make_record(now - 7200, agent_id="a"))
    backend_b.write_record(_make_record(now - 7200, agent_id="b"))

    try:
        deleted = backend_a.purge_older_than(retention_seconds=3600)
        assert deleted == 1
        assert backend_a.count() == 0
        assert backend_b.count() == 1   # other tenant untouched
    finally:
        backend_a.close()
        backend_b.close()
