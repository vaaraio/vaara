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


class TestSchemaUpgrade:
    """Opening a DB at any older schema version must migrate cleanly.

    Regression for the v0.19.0 init bug where SCHEMA_SQL ran before
    migrations and crashed with `no such column: tenant_id` on any DB
    that had not yet been brought to the tenant_id-bearing version.
    """

    _V0_AUDIT_RECORDS_SQL = """
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
        seq           INTEGER NOT NULL
    );
    """

    def _seed_v0_db(self, path: Path) -> None:
        """Build a pre-versioning DB by hand: no audit_meta, no tenant_id."""
        import sqlite3
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.executescript(self._V0_AUDIT_RECORDS_SQL)
        conn.close()

    def _seed_v1_db(self, path: Path) -> None:
        """Build a v1 DB: audit_meta exists with schema_version='1',
        audit_records has no tenant_id yet."""
        import sqlite3
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.executescript(self._V0_AUDIT_RECORDS_SQL)
        conn.execute("CREATE TABLE audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO audit_meta (key, value) VALUES ('schema_version', '1')")
        conn.close()

    def _assert_current(self, path: Path) -> None:
        import sqlite3
        from vaara.audit.sqlite_backend import SCHEMA_VERSION
        conn = sqlite3.connect(str(path))
        v = conn.execute(
            "SELECT value FROM audit_meta WHERE key='schema_version'"
        ).fetchone()
        assert v is not None
        assert int(v[0]) == SCHEMA_VERSION
        cols = [r[1] for r in conn.execute("PRAGMA table_info(audit_records)").fetchall()]
        assert "tenant_id" in cols
        assert "system_operation" in cols
        assert "data_usage" in cols
        assert "decision_making" in cols
        assert "limitations" in cols
        assert "chain_version" in cols
        conn.close()

    def test_preversion_db_migrates(self, db_path):
        self._seed_v0_db(db_path)
        SQLiteAuditBackend(db_path).close()
        self._assert_current(db_path)

    def test_v1_db_migrates(self, db_path):
        self._seed_v1_db(db_path)
        SQLiteAuditBackend(db_path).close()
        self._assert_current(db_path)

    def test_reopening_current_db_is_idempotent(self, db_path):
        SQLiteAuditBackend(db_path).close()
        SQLiteAuditBackend(db_path).close()
        self._assert_current(db_path)

    def test_pre_v047_record_verifies_after_v4_migration(self, db_path):
        """A legacy record whose record_hash was computed the v1 way (tenant_id
        NOT in the hash) must still re-verify after migrating to schema v4,
        even though its tenant_id column is populated. This is the backward-
        compat guarantee of the chain_version flag."""
        import sqlite3

        from vaara.audit.trail import AuditRecord, EventType

        # Hash computed under v1 rules (chain_version defaults to 1).
        rec = AuditRecord(
            record_id="r1", action_id="a1",
            event_type=EventType.ACTION_REQUESTED, timestamp=1.0,
            agent_id="agent", tool_name="t",
        )
        legacy_hash = rec.compute_hash()
        assert rec.chain_version == 1

        # Seed a schema-v3 DB (tenant + transparency cols, NO chain_version)
        # with that record, tenant_id populated in the column.
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.executescript(
            self._V0_AUDIT_RECORDS_SQL.replace(
                "seq           INTEGER NOT NULL\n    );",
                "seq           INTEGER NOT NULL,\n"
                "        tenant_id TEXT NOT NULL DEFAULT '',\n"
                "        system_operation TEXT, data_usage TEXT,\n"
                "        decision_making TEXT, limitations TEXT\n    );",
            )
        )
        conn.execute("CREATE TABLE audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO audit_meta (key, value) VALUES ('schema_version', '3')")
        conn.execute(
            "INSERT INTO audit_records (record_id, action_id, event_type, "
            "timestamp, agent_id, tool_name, data, regulatory, previous_hash, "
            "record_hash, seq, tenant_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("r1", "a1", "action_requested", 1.0, "agent", "t", "{}", "[]",
             "", legacy_hash, 0, "tenant-a"),
        )
        conn.close()

        backend = SQLiteAuditBackend(db_path)  # migrates 3 -> 4
        try:
            reloaded = backend.load_trail(strict=True)  # raises if chain broke
        finally:
            backend.close()
        self._assert_current(db_path)
        assert reloaded.verify_chain() is None
        loaded = reloaded._records[0]
        assert loaded.chain_version == 1          # legacy stays v1
        assert loaded.tenant_id == "tenant-a"     # column value preserved
        assert loaded.record_hash == legacy_hash  # tenant NOT folded into hash


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
