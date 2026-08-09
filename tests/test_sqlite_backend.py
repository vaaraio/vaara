"""Tests for the SQLite audit backend."""

import multiprocessing
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


def _append_from_child(db_path: str, label: str, count: int) -> None:
    """Write ``count`` records to ``db_path`` from a separate OS process.

    Module level and picklable so this works under the spawn start method
    (the default on macOS) as well as fork.
    """
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.taxonomy.actions import (
        ActionCategory, ActionRequest, ActionType, BlastRadius, Reversibility,
        UrgencyClass,
    )

    action_type = ActionType(
        "tx.transfer", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
        BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
    )
    backend = SQLiteAuditBackend(db_path)
    try:
        trail = backend.load_trail()
        for i in range(count):
            trail.record_action_requested(ActionRequest(
                agent_id=f"{label}-{i}", tool_name="tx.transfer",
                action_type=action_type,
            ))
    finally:
        backend.close()


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

        with SQLiteAuditBackend(db_path) as backend:  # migrates 3 -> 4
            reloaded = backend.load_trail(strict=True)  # raises if chain broke
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


class TestCrossProcessChainHead:
    """The hash chain must stay single-threaded across writers sharing one DB.

    ``previous_hash`` came from ``AuditTrail._last_hash``, per-process memory
    guarded by a ``threading.Lock``. Two writers on one ``audit.db`` each held
    their own copy, so both appended children off the same parent and the chain
    forked. ``seq`` was already immune (computed by subquery inside the INSERT,
    see ``write_record``); ``previous_hash`` was not, and it is the half that
    carries the tamper-evidence claim.

    Observed live: 34 forks and 4 mid-trail genesis rows in a real trail, the
    fork siblings 57-168 ms apart and split between the Claude Code hook and
    the vaara-memory MCP server writing the same tool call.
    """

    def _write(self, trail, agent, action_type):
        req = ActionRequest(
            agent_id=agent, tool_name="tx.transfer", action_type=action_type,
        )
        return trail.record_action_requested(req)

    def test_two_writers_sharing_a_db_do_not_fork_the_chain(
            self, db_path, sample_action_type):
        """Two backends on one file, both loaded, appending alternately."""
        a = SQLiteAuditBackend(db_path)
        trail_a = a.load_trail()
        self._write(trail_a, "writer-a", sample_action_type)

        # Second writer opens the same DB and loads the same head.
        b = SQLiteAuditBackend(db_path)
        trail_b = b.load_trail()

        # Interleave. Each writer's in-process _last_hash goes stale the moment
        # the other one appends.
        for i in range(5):
            self._write(trail_a, f"writer-a-{i}", sample_action_type)
            self._write(trail_b, f"writer-b-{i}", sample_action_type)

        a.close()
        b.close()

        reloaded = SQLiteAuditBackend(db_path).load_trail()
        assert reloaded.verify_chain() is None, "chain forked across writers"

    def test_second_writer_does_not_start_a_new_genesis(
            self, db_path, sample_action_type):
        """A trail built straight from ``on_record`` against a non-empty DB
        must chain onto the stored tail, not restart at ``previous_hash=''``."""
        first = SQLiteAuditBackend(db_path)
        trail = first.load_trail()
        self._write(trail, "writer-a", sample_action_type)
        first.close()

        second = SQLiteAuditBackend(db_path)
        cold = AuditTrail(on_record=second.write_record)
        self._write(cold, "writer-b", sample_action_type)
        second.close()

        reloaded = SQLiteAuditBackend(db_path).load_trail()
        records = reloaded._records
        assert len(records) == 2
        assert records[1].previous_hash == records[0].record_hash, (
            "second writer restarted the chain mid-trail")
        assert reloaded.verify_chain() is None


class TestWindowedTrailVerification:
    """A live trail holds a window onto the chain, and must say so honestly.

    Once the store owns the head, a process's own ``_records`` are no longer
    the whole chain: another writer's records sit between them on disk. The
    naive walk reads that as a break, which would be worse than the fork it
    replaced — ``ComplianceEngine`` downgrades every article to
    EVIDENCE_INSUFFICIENT on a failed ``verify_chain()``, so a correctly
    chained trail would render as BROKEN on every dashboard.
    """

    def _write(self, trail, agent, action_type):
        req = ActionRequest(
            agent_id=agent, tool_name="tx.transfer", action_type=action_type,
        )
        return trail.record_action_requested(req)

    def test_cold_writer_on_a_non_empty_db_verifies(
            self, db_path, sample_action_type):
        """The MCP server and proxy pattern: fresh trail, existing DB."""
        first = SQLiteAuditBackend(db_path)
        self._write(first.load_trail(), "writer-a", sample_action_type)
        first.close()

        second = SQLiteAuditBackend(db_path)
        cold = AuditTrail(on_record=second.write_record)
        self._write(cold, "writer-b", sample_action_type)

        assert cold.verify_chain() is None
        assert cold.chain_intact
        second.close()

    def test_interleaved_writers_each_verify_their_own_window(
            self, db_path, sample_action_type):
        a = SQLiteAuditBackend(db_path)
        b = SQLiteAuditBackend(db_path)
        trail_a = a.load_trail()
        trail_b = b.load_trail()

        for i in range(4):
            self._write(trail_a, f"writer-a-{i}", sample_action_type)
            self._write(trail_b, f"writer-b-{i}", sample_action_type)

        assert trail_a.verify_chain() is None
        assert trail_b.verify_chain() is None
        a.close()
        b.close()
        assert SQLiteAuditBackend(db_path).load_trail().verify_chain() is None

    def test_tampering_still_breaks_a_windowed_trail(
            self, db_path, sample_action_type):
        """The relaxation must not become a hole to hide edits in."""
        a = SQLiteAuditBackend(db_path)
        b = SQLiteAuditBackend(db_path)
        trail_a = a.load_trail()
        trail_b = b.load_trail()
        for i in range(3):
            self._write(trail_a, f"writer-a-{i}", sample_action_type)
            self._write(trail_b, f"writer-b-{i}", sample_action_type)
        assert trail_a.verify_chain() is None

        trail_a._records[1].data["parameters"] = {"amount": 999_999}
        assert trail_a.verify_chain() is not None
        a.close()
        b.close()

    def test_forged_anchor_is_not_accepted(
            self, db_path, sample_action_type):
        """Only a head the store actually handed out excuses a discontinuity.

        Rewriting ``previous_hash`` to an unobserved value has to break the
        chain even where a legitimate anchor sits nearby, otherwise a record
        could be re-parented onto any hash at all.
        """
        a = SQLiteAuditBackend(db_path)
        b = SQLiteAuditBackend(db_path)
        trail_a = a.load_trail()
        trail_b = b.load_trail()
        for i in range(3):
            self._write(trail_a, f"writer-a-{i}", sample_action_type)
            self._write(trail_b, f"writer-b-{i}", sample_action_type)

        victim = trail_a._records[2]
        victim.previous_hash = "f" * 64
        victim.record_hash = victim.compute_hash()  # re-seal the forgery
        assert trail_a.verify_chain() is not None
        a.close()
        b.close()

    def test_in_memory_trail_still_verifies_strictly(self, sample_action_type):
        """No store, no window: the genesis and every link stay mandatory."""
        trail = AuditTrail()
        for i in range(3):
            self._write(trail, f"agent-{i}", sample_action_type)
        assert trail.verify_chain() is None

        trail._records[1].previous_hash = "0" * 64
        trail._records[1].record_hash = trail._records[1].compute_hash()
        assert trail.verify_chain() is not None


class TestAppendPathUsesTheSeqIndex:
    """Writing a record must not get slower as the trail gets longer.

    Both queries on the append path order by seq: the subquery that computes
    the next seq inside the INSERT, and chain_head's ORDER BY seq DESC LIMIT
    1. With no index on seq, SQLite scanned the whole table and built a temp
    B-tree to sort it, on every single append. Measured cost of one record:
    0.12 ms on an empty trail, 4.7 ms at 10k rows, 16 ms at 31k. With the
    index, 0.13 ms at 31k.

    This asserts the query plan rather than a wall-clock number, because a
    timing assertion on a shared CI runner is a flake generator. The plan is
    the thing that actually regressed.
    """

    def test_chain_head_seeks_instead_of_scanning(self, db_path):
        backend = SQLiteAuditBackend(db_path)
        try:
            plan = " ".join(
                str(row[-1]) for row in backend._conn.execute(
                    "EXPLAIN QUERY PLAN "
                    "SELECT record_hash FROM audit_records WHERE 1=1 "
                    "ORDER BY seq DESC LIMIT 1"
                ).fetchall()
            )
            assert "idx_seq" in plan, f"chain_head is not using the seq index: {plan}"
            assert "TEMP B-TREE" not in plan, (
                f"chain_head is sorting the whole table on every append: {plan}")
        finally:
            backend.close()

    def test_seq_subquery_does_not_scan(self, db_path):
        backend = SQLiteAuditBackend(db_path)
        try:
            plan = " ".join(
                str(row[-1]) for row in backend._conn.execute(
                    "EXPLAIN QUERY PLAN "
                    "SELECT COALESCE(MAX(seq), -1) + 1 FROM audit_records"
                ).fetchall()
            )
            assert "SCAN" not in plan, f"the seq subquery scans the table: {plan}"
        finally:
            backend.close()

    def test_existing_trail_gains_the_index_on_open(self, db_path, sample_action_type):
        """A v5 trail written before this release is migrated, not left slow."""
        import sqlite3

        first = SQLiteAuditBackend(db_path)
        trail = first.load_trail()
        trail.record_action_requested(ActionRequest(
            agent_id="a", tool_name="tx.transfer", action_type=sample_action_type,
        ))
        first._conn.execute("DROP INDEX IF EXISTS idx_seq")
        first._conn.execute(
            "UPDATE audit_meta SET value='5' WHERE key='schema_version'")
        first.close()

        conn = sqlite3.connect(str(db_path))
        before = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_seq'"
        ).fetchone()
        conn.close()
        assert before is None, "test setup failed to remove the index"

        reopened = SQLiteAuditBackend(db_path)
        try:
            row = reopened._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_seq'"
            ).fetchone()
            assert row is not None, "reopening a v5 trail did not add the seq index"
            assert reopened.load_trail().verify_chain() is None
        finally:
            reopened.close()


class TestConcurrentProcessAppend:
    """The claim, tested the way it is deployed: separate OS processes.

    The tests above interleave two backends inside one interpreter, which
    reproduces the stale-head mechanism but shares a GIL and a page cache.
    This one spawns real processes contending for SQLite's write lock, which
    is the only version of the claim a deployment cares about: the Claude
    Code hook, an MCP server and `vaara check` all writing one audit.db.
    """

    def test_four_processes_writing_one_db_produce_one_chain(self, db_path):
        writers, per_writer = 4, 15
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(
                target=_append_from_child,
                args=(str(db_path), f"proc-{n}", per_writer),
            )
            for n in range(writers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)

        assert all(p.exitcode == 0 for p in procs), (
            f"writer processes failed: {[p.exitcode for p in procs]}")

        backend = SQLiteAuditBackend(db_path)
        try:
            trail = backend.load_trail()
            assert trail.size == writers * per_writer, (
                "records were lost to write-lock contention")
            assert trail.verify_chain() is None, "chain forked across processes"
            seqs = [
                row[0] for row in backend._conn.execute(
                    "SELECT seq FROM audit_records ORDER BY seq")
            ]
            assert seqs == list(range(len(seqs))), f"seq gaps or repeats: {seqs}"
        finally:
            backend.close()
