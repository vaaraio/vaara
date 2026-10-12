"""A damaged index must not hide a record from the trail load.

``PRAGMA integrity_check`` reporting "row N missing from index idx_seq" means
the record is in the table and only the index lost it. A load that walks the
index then skips the record, and the chain check reports "seq N missing": a
chain break that is not there, printed on every hook call. Seen on a live
trail on 2026-10-12.
"""

import sqlite3
from pathlib import Path

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline

N_CALLS = 30
LOST = 17


def _build(path: Path) -> None:
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(N_CALLS):
        pipeline.intercept(agent_id="index-test", tool_name="file.read",
                           parameters={"path": f"/srv/{i}.txt"})
    backend.close()


def _drop_from_index(path: Path, seq: int) -> None:
    """Leave idx_seq without one row while its schema says it holds all."""
    conn = sqlite3.connect(path)
    full = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'idx_seq'").fetchone()[0]
    conn.execute("DROP INDEX idx_seq")
    conn.execute(f"CREATE UNIQUE INDEX idx_seq ON audit_records(seq) WHERE seq != {seq:d}")
    conn.execute("PRAGMA writable_schema = ON")
    conn.execute("UPDATE sqlite_master SET sql = ? WHERE name = 'idx_seq'", (full,))
    conn.commit()
    conn.close()


def _integrity(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute("PRAGMA integrity_check")]
    finally:
        conn.close()


def test_a_row_the_index_lost_is_still_loaded(tmp_path):
    db = tmp_path / "audit.db"
    _build(db)
    _drop_from_index(db, LOST)
    assert any("missing from index idx_seq" in line for line in _integrity(db))

    backend = SQLiteAuditBackend(db)
    trail = backend.load_trail()
    backend.close()

    conn = sqlite3.connect(db)
    in_table = conn.execute("SELECT count(*) FROM audit_records NOT INDEXED").fetchone()[0]
    conn.close()
    assert len(trail._records) == in_table
    assert trail._load_verdict.error is None
    assert trail.verify_chain() is None


def test_load_order_is_seq_order(tmp_path):
    db = tmp_path / "audit.db"
    _build(db)
    _drop_from_index(db, LOST)
    backend = SQLiteAuditBackend(db)
    trail = backend.load_trail()
    backend.close()
    seqs = [trail._store_seqs[r.record_id] for r in trail._records]
    assert seqs == sorted(seqs)


def test_streaming_verify_walks_a_row_the_index_lost(tmp_path):
    db = tmp_path / "audit.db"
    _build(db)
    _drop_from_index(db, LOST)
    backend = SQLiteAuditBackend(db)
    verdict = backend.verify_chain_streaming()
    backend.close()
    assert verdict is None


def test_export_holds_a_row_the_index_lost(tmp_path):
    db = tmp_path / "audit.db"
    _build(db)
    _drop_from_index(db, LOST)
    backend = SQLiteAuditBackend(db)
    out = tmp_path / "trail.jsonl"
    exported = backend.export_jsonl(out)
    backend.close()
    conn = sqlite3.connect(db)
    in_table = conn.execute("SELECT count(*) FROM audit_records NOT INDEXED").fetchone()[0]
    conn.close()
    assert exported == in_table == len(out.read_text().splitlines())
