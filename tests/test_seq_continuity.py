"""Missing seqs are a gap even when the hash chain links across them.

Salvage keeps the readable rows in a fresh file and, when the lost seqs run
past the last one kept, records a seq floor. The next record then takes a
seq above the floor and chains to the last record kept, so the chain links
cleanly across the hole. The repair's own REPAIR_GAP record is normally that
next record. If anything else is written there first, or the declaration
never comes, both walks used to call the trail intact: they looked for
missing seqs only where a previous_hash failed to match.

These tests pin the rule: a seq hole in an untenanted walk is held like a
break, a REPAIR_GAP record may declare the hole it bridges, and a
tenant-scoped walk, whose seqs interleave with other tenants', is not
checked for holes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from vaara.audit.sqlite_backend import SQLiteAuditBackend

from tests.test_declared_gaps import _build, _calls, _declare, _verdicts


def _lose_tail_as_salvage_leaves_it(path: Path, n: int) -> list[int]:
    """Drop the last ``n`` records the way a salvaged file comes back."""
    conn = sqlite3.connect(path)
    top = conn.execute("SELECT MAX(seq) FROM audit_records").fetchone()[0]
    lost = list(range(top - n + 1, top + 1))
    conn.executemany("DELETE FROM audit_records WHERE seq = ?", [(s,) for s in lost])
    # A salvaged file is a fresh database: no recorded head, and a floor
    # above the lost seqs so they are never reused.
    conn.execute("DELETE FROM audit_meta WHERE key LIKE 'chain_head%'")
    conn.execute(
        "INSERT OR REPLACE INTO audit_meta (key, value) VALUES ('seq_floor', ?)",
        (str(top),),
    )
    conn.commit()
    conn.close()
    return lost


def test_a_hole_the_chain_links_across_is_broken_when_nobody_declares_it(tmp_path: Path):
    path = _build(tmp_path)
    lost = _lose_tail_as_salvage_leaves_it(path, 3)
    backend = SQLiteAuditBackend(path)
    _calls(backend, 1, "after")
    backend.close()

    conn = sqlite3.connect(path)
    first_after = conn.execute(
        "SELECT seq FROM audit_records WHERE seq > ? ORDER BY seq LIMIT 1", (lost[-1],),
    ).fetchone()
    conn.close()
    assert first_after is not None, "the floor should push the next seq past the hole"

    _trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"
    assert "not declared by any repair record" in mem.error
    assert "not declared by any repair record" in streamed.error


def test_the_repair_record_that_bridges_the_hole_declares_it(tmp_path: Path):
    path = _build(tmp_path)
    lost = _lose_tail_as_salvage_leaves_it(path, 3)
    backend = SQLiteAuditBackend(path)
    _declare(backend, lost)
    _calls(backend, 1, "after")
    backend.close()

    _trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "declared_gaps"
    (gap,) = mem.declared_gaps
    assert gap.lost_seqs == tuple(lost)
    assert gap.resumes_at_seq == lost[-1] + 1
    assert streamed.declared_gaps[0].lost_seqs == tuple(lost)
    assert streamed.declared_gaps[0].declared_by_seqs == gap.declared_by_seqs == (lost[-1] + 1,)


def test_a_tenant_walk_is_not_checked_for_holes(tmp_path: Path):
    """Seqs are allocated across the whole table, so one tenant's records
    carry the seqs between them that the other tenant used."""
    path = tmp_path / "audit.db"
    a = SQLiteAuditBackend(path, tenant_id="tenant-a")
    b = SQLiteAuditBackend(path, tenant_id="tenant-b")
    for _ in range(3):
        _calls(a, 1, "a")
        _calls(b, 1, "b")
    a.close()
    b.close()

    for tenant in ("tenant-a", "tenant-b"):
        backend = SQLiteAuditBackend(path, tenant_id=tenant)
        try:
            trail = backend.load_trail()
            assert trail.verify_chain_verdict().state == "intact"
            assert backend.verify_chain_streaming_verdict().state == "intact"
        finally:
            backend.close()
