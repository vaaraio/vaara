"""Records deleted from the end of the trail are reported, not verified clean.

C1 (25.9) deleted the newest row of a trail with plain SQL: what was left
still chained from genesis to a shorter tail, so the compliance report said
intact and ``vaara trail export`` signed "chain intact: True". The store now
records its head (seq and hash) in the transaction that writes each record,
and every verify checks that the recorded head is still stored. A write after
the loss chains to the recorded head, so the loss stays a break in the chain
even once the head moves on.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from vaara.audit.export import _snapshot_trail
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import _CURRENT_CHAIN_VERSION, AuditRecord, EventType
from vaara.pipeline import InterceptionPipeline

N_CALLS = 5


def _calls(path: Path, n: int, tag: str, tenant_id: str = "") -> None:
    with SQLiteAuditBackend(path, tenant_id=tenant_id) as backend:
        pipeline = InterceptionPipeline(trail=backend.load_trail())
        for i in range(n):
            pipeline.intercept(agent_id="tail-test", tool_name="file.read",
                               parameters={"path": f"/srv/{tag}/{i}"})


def _sql(path: Path, *statements: str) -> list:
    conn = sqlite3.connect(path)
    try:
        out = [conn.execute(s).fetchall() for s in statements]
        conn.commit()
        return out
    finally:
        conn.close()


def _max_seq(path: Path) -> int:
    return _sql(path, "SELECT MAX(seq) FROM audit_records")[0][0][0]


def _truncate(path: Path, n: int) -> int:
    top = _max_seq(path)
    _sql(path, f"DELETE FROM audit_records WHERE seq > {top - n}")
    return top


def _verdicts(path: Path):
    with SQLiteAuditBackend(path) as backend:
        trail = backend.load_trail()
        return trail, trail.verify_chain_verdict(), backend.verify_chain_streaming_verdict()


def _head(path: Path, key: str = "chain_head") -> dict:
    rows = _sql(path, f"SELECT value FROM audit_meta WHERE key = '{key}'")[0]
    return json.loads(rows[0][0])


def test_the_head_is_recorded_with_every_write(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    top = _max_seq(path)
    stored = _sql(path, f"SELECT record_hash FROM audit_records WHERE seq = {top}")[0][0][0]
    assert _head(path) == {"seq": top, "hash": stored}
    trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "intact"
    assert trail.chain_intact


def test_deleting_the_newest_records_is_reported_by_every_verify(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    top = _truncate(path, 3)
    trail, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"
    assert mem.error.startswith(f"Tail truncated: seq {top}")
    assert f"newest stored record is seq {top - 3}" in mem.error
    assert streamed.error == mem.error
    assert trail.chain_intact is False
    # The export signs what verify_chain says, so it no longer signs intact.
    assert _snapshot_trail(trail, tmp_path / "out.zip")[4] is False


def test_the_compliance_report_does_not_call_a_truncated_trail_intact(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    _truncate(path, 1)
    out = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "compliance", "report",
         "--db", str(path), "--format", "json"],
        capture_output=True, text=True, check=True,
    ).stdout
    report = json.loads(out)
    assert report["trail_integrity"]["chain_intact"] is False
    assert report["overall_status"] == "evidence_insufficient"


def test_a_write_after_the_loss_keeps_it_on_the_chain(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    top = _truncate(path, 3)
    # The next write moves the recorded head on. Chained to the shortened
    # tail it would verify clean; chained to the recorded head it cannot.
    _calls(path, 1, "b")
    seqs = [r[0] for r in _sql(path, "SELECT seq FROM audit_records ORDER BY seq")[0]]
    assert min(s for s in seqs if s > top - 3) == top + 1, "seq reused a lost record's number"
    _, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"
    assert "not declared by any repair record" in mem.error


def _old_writer_append(path: Path) -> None:
    """Append as a vaara older than the head record does: chained to the
    stored tail, seq from the table, head left where it was."""
    backend = SQLiteAuditBackend(path)
    record = AuditRecord(
        record_id=f"old-writer-{time.time_ns()}", action_id="old",
        event_type=EventType.ACTION_REQUESTED, timestamp=time.time(),
        agent_id="old-writer", tool_name="file.read", data={},
        previous_hash=backend.chain_head(),
    )
    record.chain_version = _CURRENT_CHAIN_VERSION
    record.record_hash = record.compute_hash()
    backend.write_record(record)
    backend.close()


def test_a_head_replaced_after_truncation_is_reported(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    top = _truncate(path, 1)
    # A record at the lost seq, chained cleanly to what is left: the walk
    # alone finds nothing wrong.
    _old_writer_append(path)
    assert _max_seq(path) == top
    _, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"
    assert mem.error.startswith(f"Chain head replaced: the record stored at seq {top}")


def test_a_writer_that_does_not_move_the_head_is_not_truncation(tmp_path: Path):
    """An older vaara on the same file appends without recording the head."""
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    head = _head(path)
    _old_writer_append(path)
    assert _head(path) == head
    _, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "intact"


def test_a_trail_written_before_the_head_existed_still_verifies(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    _sql(path, "DELETE FROM audit_meta WHERE key = 'chain_head'")
    _, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "intact"
    _calls(path, 1, "b")
    assert _head(path)["seq"] == _max_seq(path)


def test_a_purge_that_removes_the_head_is_retention_not_truncation(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    top = _max_seq(path)
    _sql(path, "UPDATE audit_records SET timestamp = timestamp - 10000000")
    backend = SQLiteAuditBackend(path)
    assert backend.purge_older_than(86400) > 0
    assert backend.head_problem() is None
    backend.close()
    assert _head(path)["purged"] is True
    # The next record continues the numbering and chains to the purged head.
    _calls(path, 1, "b")
    assert _sql(path, "SELECT MIN(seq) FROM audit_records")[0][0][0] == top + 1
    assert "purged" not in _head(path)


def test_a_purge_does_not_launder_an_earlier_truncation(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    _truncate(path, 2)
    _sql(path, "UPDATE audit_records SET timestamp = timestamp - 10000000")
    backend = SQLiteAuditBackend(path)
    backend.purge_older_than(86400)
    assert backend.head_problem() is not None
    backend.close()
    assert "purged" not in _head(path)


def test_each_tenant_scoped_backend_keeps_its_own_head(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a", tenant_id="t1")
    _calls(path, N_CALLS, "b", tenant_id="t2")
    t1 = _head(path, "chain_head:t1")
    _sql(path, f"DELETE FROM audit_records WHERE seq = {t1['seq']}")
    one = SQLiteAuditBackend(path, tenant_id="t1")
    two = SQLiteAuditBackend(path, tenant_id="t2")
    try:
        assert one.head_problem().startswith(f"Tail truncated: seq {t1['seq']}")
        assert two.head_problem() is None
    finally:
        one.close()
        two.close()


def test_an_unreadable_head_is_reported(tmp_path: Path):
    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    _sql(path, "UPDATE audit_meta SET value = 'not json' WHERE key = 'chain_head'")
    _, mem, _ = _verdicts(path)
    assert mem.error.startswith("Chain head record in audit_meta is unreadable")


def test_the_head_check_and_the_rows_read_one_snapshot(tmp_path: Path, monkeypatch):
    """A deletion committed between the head check and the row read must not
    split them: the load sees the trail as it was when the check ran."""
    from vaara.audit import sqlite_backend

    path = tmp_path / "audit.db"
    _calls(path, N_CALLS, "a")
    top = _max_seq(path)
    real = sqlite_backend._head_problem

    def check_then_truncate(*args):
        found = real(*args)
        _sql(path, f"DELETE FROM audit_records WHERE seq = {top}")
        return found

    monkeypatch.setattr(sqlite_backend, "_head_problem", check_then_truncate)
    with SQLiteAuditBackend(path) as backend:
        trail = backend.load_trail()
    assert trail.size == top + 1, "the rows came from after the deletion"
    assert trail.chain_intact
    monkeypatch.setattr(sqlite_backend, "_head_problem", real)
    assert _verdicts(path)[1].error.startswith(f"Tail truncated: seq {top}")
