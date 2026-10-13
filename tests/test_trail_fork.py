"""A branch of the trail is declared, not merged.

When a store is swapped out and a fresh file starts from an older copy, two
trails share every record up to some seq and then go on separately. Both are
the record as written. `vaara trail declare-fork` finds the last shared seq,
checks the other branch chains on from it to its head, and appends one
TRAIL_FORKED record naming that head by hash. A copy of the prefix, an
extension, an unrelated store, a branch with a record that does not recompute
and a branch already declared are refused.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from vaara.audit.fork import check_fork
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline


def _add(path: Path, calls: int, agent: str) -> None:
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(calls):
        pipeline.intercept(
            agent_id=agent, tool_name="file.read", parameters={"path": f"/srv/{agent}/{i}.txt"},
        )
    backend.close()


def _max_seq(path: Path) -> int:
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT max(seq) FROM audit_records").fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def forked(tmp_path: Path):
    """A live trail and a branch that share a prefix and then diverge."""
    live = tmp_path / "trail" / "audit.db"
    live.parent.mkdir()
    _add(live, 10, "shared")
    shared = _max_seq(live)
    branch = tmp_path / "volume" / "audit.db"
    branch.parent.mkdir()
    shutil.copy(live, branch)
    _add(live, 5, "live")
    _add(branch, 8, "branch")
    return live, branch, shared


def test_the_fork_point_is_the_last_shared_seq(forked):
    live, branch, shared = forked
    check = check_fork(live, branch)
    assert check.ok, check.problems
    assert check.fork_seq == shared
    assert check.branch_head_seq == _max_seq(branch)
    assert check.branch_record_count == check.branch_head_seq - shared
    assert check.links_verified == check.branch_record_count
    assert check.branch_missing_seqs == []


def test_declaring_appends_one_record_and_the_trail_still_verifies(forked):
    from vaara.cli import main
    live, branch, shared = forked
    head = check_fork(live, branch).branch_head_hash
    assert main(["trail", "declare-fork", "--db", str(live), "--branch", str(branch),
                 "--source", "volume dropped"]) in (None, 0)
    con = sqlite3.connect(live)
    rows = con.execute(
        "SELECT data FROM audit_records WHERE event_type = ?", (EventType.TRAIL_FORKED.value,)
    ).fetchall()
    con.close()
    assert len(rows) == 1
    data = json.loads(rows[0][0])
    assert data["fork_seq"] == shared
    assert data["branch_head_hash"] == head
    assert data["branch_path"] == "volume/audit.db"
    assert data["source"] == "volume dropped"
    backend = SQLiteAuditBackend(live)
    try:
        assert backend.load_trail().verify_chain() is None
    finally:
        backend.close()


def test_the_same_branch_is_not_declared_twice(forked):
    from vaara.cli import main
    live, branch, _ = forked
    assert main(["trail", "declare-fork", "--db", str(live), "--branch", str(branch)]) in (None, 0)
    again = check_fork(live, branch)
    assert not again.ok
    assert any("already declared" in p for p in again.problems)


def test_a_dry_run_appends_nothing(forked, capsys):
    from vaara.cli import main
    live, branch, _ = forked
    before = _max_seq(live)
    assert main(["trail", "declare-fork", "--db", str(live), "--branch", str(branch),
                 "--dry-run", "--format", "json"]) in (None, 0)
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["dry_run"]
    assert _max_seq(live) == before


def test_a_tampered_branch_record_is_refused(forked):
    live, branch, shared = forked
    con = sqlite3.connect(branch)
    con.execute("UPDATE audit_records SET agent_id = 'someone-else' WHERE seq = ?", (shared + 3,))
    con.commit()
    con.close()
    check = check_fork(live, branch)
    assert not check.ok
    assert any(f"seq {shared + 3} does not recompute" in p for p in check.problems)


def test_an_undeclared_hole_in_the_branch_is_refused(forked):
    live, branch, shared = forked
    con = sqlite3.connect(branch)
    con.execute("DELETE FROM audit_records WHERE seq = ?", (shared + 2,))
    con.commit()
    con.close()
    check = check_fork(live, branch)
    assert not check.ok
    assert any("never declared it lost" in p for p in check.problems)


def test_a_copy_of_the_prefix_is_not_a_fork(tmp_path):
    live = tmp_path / "audit.db"
    _add(live, 6, "a")
    copy = tmp_path / "copy.db"
    shutil.copy(live, copy)
    _add(live, 3, "a")
    check = check_fork(live, copy)
    assert not check.ok
    assert any("not a fork" in p for p in check.problems)


def test_an_extension_is_not_a_fork(tmp_path):
    live = tmp_path / "audit.db"
    _add(live, 6, "a")
    ahead = tmp_path / "ahead.db"
    shutil.copy(live, ahead)
    _add(ahead, 3, "a")
    check = check_fork(live, ahead)
    assert not check.ok
    assert any("extension" in p for p in check.problems)


def test_unrelated_stores_are_refused(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    _add(a, 4, "a")
    _add(b, 4, "b")
    check = check_fork(a, b)
    assert not check.ok
    assert any("first record" in p for p in check.problems)


def test_the_cli_refuses_the_trail_as_its_own_branch(forked):
    from vaara.cli import main
    live, _, _ = forked
    assert main(["trail", "declare-fork", "--db", str(live), "--branch", str(live)]) == 2
