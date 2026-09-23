"""A gap a repair declared is its own state: not intact, and not a break.

The hook trail on the build box lost seq 1505 to page damage. The repair
kept everything else and appended REPAIR_GAP records naming 1505, yet every
tool call printed AUDIT CHAIN INTEGRITY FAILURE because the verifier never
read the declaration. These tests pin the rule that replaced that: a break
across missing seqs that a later REPAIR_GAP record names in full is reported
as a declared gap, in both walks, and never as intact. Anything less than
that is still broken.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.integrations import claude_code_hooks
from vaara.pipeline import InterceptionPipeline

N_CALLS = 10


def _calls(backend: SQLiteAuditBackend, n: int, tag: str) -> None:
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(n):
        pipeline.intercept(agent_id="gap-test", tool_name="file.read",
                           parameters={"path": f"/srv/{tag}/{i}"})


def _declare(backend: SQLiteAuditBackend, lost: list[int]) -> None:
    backend.load_trail().record_repair_gap(SimpleNamespace(
        method="salvage", problem="test", lost_seqs=lost,
        unreadable_rowids=0, tables_damaged=["audit_records"],
        records_kept=0, damaged_copy="",
    ))


def _delete(path: Path, *seqs: int) -> None:
    conn = sqlite3.connect(path)
    conn.executemany("DELETE FROM audit_records WHERE seq = ?", [(s,) for s in seqs])
    conn.commit()
    conn.close()


def _build(tmp_path: Path) -> Path:
    path = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(path)
    _calls(backend, N_CALLS, "before")
    backend.close()
    return path


def _verdicts(path: Path):
    backend = SQLiteAuditBackend(path)
    try:
        trail = backend.load_trail()
        return trail, trail.verify_chain_verdict(), backend.verify_chain_streaming_verdict()
    finally:
        backend.close()


def test_a_lost_record_nobody_declared_is_broken(tmp_path: Path):
    path = _build(tmp_path)
    _delete(path, 7)
    trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"
    assert "not declared by any repair record" in mem.error
    assert trail.verify_chain().startswith("Chain broken at record")


def test_a_declared_gap_is_reported_as_declared_and_never_intact(tmp_path: Path):
    path = _build(tmp_path)
    _delete(path, 7, 8)
    backend = SQLiteAuditBackend(path)
    _declare(backend, [7, 8])
    backend.close()

    trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "declared_gaps"
    (gap,) = mem.declared_gaps
    assert gap.lost_seqs == (7, 8)
    assert gap.resumes_at_seq == 9
    assert streamed.declared_gaps[0].lost_seqs == (7, 8)
    assert streamed.declared_gaps[0].declared_by_seqs == gap.declared_by_seqs
    # Callers that read the flat answer still see a trail that is not intact.
    assert trail.chain_intact is False
    assert trail.verify_chain().startswith("Chain not intact: 1 declared gap")
    assert SQLiteAuditBackend(path).verify_chain_streaming().startswith("Chain not intact")


def test_load_does_not_cry_integrity_failure_over_a_declared_gap(tmp_path: Path, caplog):
    path = _build(tmp_path)
    _delete(path, 5)
    backend = SQLiteAuditBackend(path)
    _declare(backend, [5])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="vaara.audit.sqlite_backend"):
        trail = backend.load_trail(strict=True)
    backend.close()
    assert trail._load_verdict.state == "declared_gaps"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("declared gap" in r.getMessage() for r in caplog.records)


def test_a_declaration_that_misses_one_seq_does_not_cover_the_gap(tmp_path: Path):
    path = _build(tmp_path)
    _delete(path, 7, 8)
    backend = SQLiteAuditBackend(path)
    _declare(backend, [7])
    backend.close()
    _trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"


def test_a_declaration_written_before_the_break_does_not_count(tmp_path: Path):
    path = _build(tmp_path)
    backend = SQLiteAuditBackend(path)
    future = backend.count() + 3
    _declare(backend, [future])
    _calls(backend, 3, "after")
    backend.close()
    _delete(path, future)
    _trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"


def test_tampering_after_a_declared_gap_still_fails(tmp_path: Path):
    path = _build(tmp_path)
    _delete(path, 7)
    backend = SQLiteAuditBackend(path)
    _declare(backend, [7])
    backend.close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE audit_records SET tool_name = 'file.write' WHERE seq = 12")
    conn.commit()
    conn.close()
    _trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"
    assert mem.error.startswith("Hash mismatch")


def test_the_hook_says_it_once_at_session_start(tmp_path: Path, monkeypatch):
    path = _build(tmp_path)
    _delete(path, 7)
    backend = SQLiteAuditBackend(path)
    _declare(backend, [7])
    verdict = backend.load_trail()._load_verdict
    backend.close()

    said: list[str] = []
    monkeypatch.setattr(claude_code_hooks, "_emit", said.append)
    claude_code_hooks._report_trail_health({}, path, True, verdict)
    lines = [s for s in said if "declared gap" in s]
    assert len(lines) == 1
    assert "is not intact" in lines[0] and "seq 7 lost" in lines[0]
