"""Tests for the SQLite audit backend."""

import tempfile
from pathlib import Path

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail, EventType
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)
    # Clean WAL files
    Path(str(path) + "-wal").unlink(missing_ok=True)
    Path(str(path) + "-shm").unlink(missing_ok=True)


@pytest.fixture
def sample_action_type():
    return ActionType(
        "tx.transfer", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
        BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
        frozenset({RegulatoryDomain.MIFID2, RegulatoryDomain.DORA}),
    )


class TestSQLiteBackend:
    def test_create_and_count(self, db_path):
        with SQLiteAuditBackend(db_path) as backend:
            assert backend.count() == 0

    def test_write_and_read(self, db_path, sample_action_type):
        # Write records via trail
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            req = ActionRequest(
                agent_id="agent-1", tool_name="tx.transfer",
                action_type=sample_action_type,
                parameters={"amount": 1000},
            )
            action_id = trail.record_action_requested(req)
            trail.record_decision(
                action_id=action_id, agent_id="agent-1",
                tool_name="tx.transfer",
                decision="deny", reason="too risky", risk_score=0.8,
            )
            assert backend.count() == 2

        # Read back from fresh connection
        with SQLiteAuditBackend(db_path) as backend:
            assert backend.count() == 2
            records = backend.query_by_action(action_id)
            assert len(records) == 2
            assert records[0].event_type == EventType.ACTION_REQUESTED
            assert records[1].event_type == EventType.ACTION_BLOCKED

    def test_load_trail_with_chain_verification(self, db_path, sample_action_type):
        # Write
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            for i in range(10):
                req = ActionRequest(
                    agent_id=f"agent-{i % 3}", tool_name="tx.transfer",
                    action_type=sample_action_type,
                )
                trail.record_action_requested(req)
            assert backend.count() == 10

        # Reload
        with SQLiteAuditBackend(db_path) as backend:
            loaded = backend.load_trail()
            assert loaded.size == 10
            assert loaded.chain_intact

    def test_query_by_agent(self, db_path, sample_action_type):
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            for i in range(5):
                req = ActionRequest(
                    agent_id="target" if i < 3 else "other",
                    tool_name="tx.transfer",
                    action_type=sample_action_type,
                )
                trail.record_action_requested(req)

            results = backend.query_by_agent("target")
            assert len(results) == 3

    def test_query_by_regulation(self, db_path, sample_action_type):
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            req = ActionRequest(
                agent_id="agent-1", tool_name="tx.transfer",
                action_type=sample_action_type,
            )
            trail.record_action_requested(req)

            dora_records = backend.query_by_regulation("dora")
            assert len(dora_records) >= 1

    def test_query_blocked(self, db_path):
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            trail.record_decision(
                action_id="a1", agent_id="agent", tool_name="danger",
                decision="deny", reason="blocked", risk_score=0.9,
            )
            blocked = backend.query_blocked()
            assert len(blocked) == 1

    def test_stats(self, db_path, sample_action_type):
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            for i in range(5):
                req = ActionRequest(
                    agent_id=f"agent-{i}", tool_name="tx.transfer",
                    action_type=sample_action_type,
                )
                trail.record_action_requested(req)

            stats = backend.stats()
            assert stats["total_records"] == 5
            assert stats["unique_agents"] == 5
            assert "action_requested" in stats["by_event_type"]

    def test_export_jsonl(self, db_path, sample_action_type):
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            for _i in range(3):
                req = ActionRequest(
                    agent_id="agent", tool_name="tx.transfer",
                    action_type=sample_action_type,
                )
                trail.record_action_requested(req)

            export_path = db_path.with_suffix(".jsonl")
            count = backend.export_jsonl(export_path)
            assert count == 3
            lines = export_path.read_text().strip().split("\n")
            assert len(lines) == 3
            export_path.unlink(missing_ok=True)

    def test_persistence_across_connections(self, db_path, sample_action_type):
        # First session: write 5 records
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            for _ in range(5):
                req = ActionRequest(
                    agent_id="agent", tool_name="data.read",
                    action_type=sample_action_type,
                )
                trail.record_action_requested(req)

        # Second session: write 3 more
        with SQLiteAuditBackend(db_path) as backend:
            trail = AuditTrail(on_record=backend.write_record)
            for _ in range(3):
                req = ActionRequest(
                    agent_id="agent", tool_name="data.read",
                    action_type=sample_action_type,
                )
                trail.record_action_requested(req)

            assert backend.count() == 8


class TestSkeletonRecordsCounter:
    """Loop 51: load_trail reports skeleton rows via log only; the count
    is lost to callers. Ops dashboards polling trail.persistence_failures
    (write side) would miss reload-time corruption. Expose as a parallel
    counter trail.skeleton_records, surfaced in pipeline.status().
    """

    def test_corrupt_row_bumps_skeleton_counter(self, db_path, sample_action_type):
        backend = SQLiteAuditBackend(db_path)
        trail = AuditTrail(on_record=backend.write_record)
        for i in range(3):
            req = ActionRequest(
                agent_id=f"agent-{i}", tool_name="tx.transfer",
                action_type=sample_action_type,
            )
            trail.record_action_requested(req)

        # Corrupt row 1's data column directly in the DB so from_dict fails.
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE audit_records SET data = 'not-json' WHERE seq = 1"
        )
        conn.commit()
        conn.close()

        reloaded = backend.load_trail()
        assert reloaded.skeleton_records == 1
        # Fresh trail (no on_record writes) starts at 0
        fresh = AuditTrail()
        assert fresh.skeleton_records == 0
        assert fresh.persistence_failures == 0

    def test_pipeline_status_exposes_skeleton_count(self):
        # Fresh pipeline has zero skeletons
        from vaara.pipeline import InterceptionPipeline
        p = InterceptionPipeline()
        status = p.status()
        assert "trail_skeleton_records" in status
        assert status["trail_skeleton_records"] == 0
