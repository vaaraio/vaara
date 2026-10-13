# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Declare that another store is a branch of this trail.

Two trail files can share every record up to some seq and then go on
separately: a store that was swapped out and back, a volume that was dropped
and a fresh file started from an older copy. Both branches are the record as
written. Neither is merged into the other and nothing is rewritten.

``check_fork`` finds the last seq both files hold with the same record hash,
then checks the other branch from there to its head: every record recomputes
its own hash, each names the one before it, and any seq missing from it is
one that branch itself declared lost. ``declare_fork`` appends a TRAIL_FORKED
record to the live trail naming that branch by its head hash, which commits
to every record on it.

Both files are opened read-only for the check.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vaara.audit.carve import _declared_lost, _record_hash_holds

_COLUMNS = (
    "record_id, action_id, event_type, timestamp, agent_id, tool_name, data, "
    "regulatory, previous_hash, record_hash, seq, tenant_id, chain_version"
)


@dataclass
class ForkCheck:
    """Whether a store may be declared a branch of a live trail."""

    branch: str
    fork_seq: Optional[int] = None
    fork_hash: str = ""
    branch_head_seq: Optional[int] = None
    branch_head_hash: str = ""
    branch_record_count: int = 0
    branch_missing_seqs: list[int] = field(default_factory=list)
    branch_first_ts: Optional[float] = None
    branch_last_ts: Optional[float] = None
    links_verified: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and self.fork_seq is not None

    def to_dict(self) -> dict:
        return {
            "branch": self.branch, "fork_seq": self.fork_seq, "fork_hash": self.fork_hash,
            "branch_head_seq": self.branch_head_seq, "branch_head_hash": self.branch_head_hash,
            "branch_record_count": self.branch_record_count,
            "branch_missing_seqs": list(self.branch_missing_seqs),
            "branch_first_ts": self.branch_first_ts, "branch_last_ts": self.branch_last_ts,
            "links_verified": self.links_verified,
            "ok": self.ok, "problems": list(self.problems),
        }


def _open_ro(db: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"{Path(db).expanduser().resolve().as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _hash_at(con: sqlite3.Connection, seq: int) -> Optional[str]:
    row = con.execute(
        "SELECT record_hash FROM audit_records NOT INDEXED WHERE seq = ?", (seq,)
    ).fetchone()
    return row[0] if row is not None else None


def _last_shared_seq(live: sqlite3.Connection, branch: sqlite3.Connection, top: int) -> Optional[int]:
    """The highest seq both stores hold under one hash.

    A hash chain cannot agree again once it has diverged, so agreement is a
    prefix and a binary search finds its end. A seq missing from either side
    counts as disagreement at that point.
    """
    if _hash_at(live, 0) is None or _hash_at(live, 0) != _hash_at(branch, 0):
        return None
    lo, hi = 0, top
    while lo < hi:
        mid = (lo + hi + 1) // 2
        h = _hash_at(live, mid)
        if h is not None and h == _hash_at(branch, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _declared_forks(con: sqlite3.Connection) -> set[str]:
    heads: set[str] = set()
    for (data,) in con.execute(
        "SELECT data FROM audit_records NOT INDEXED WHERE event_type = 'trail_forked'"
    ):
        try:
            heads.add(str(json.loads(data).get("branch_head_hash", "")))
        except (ValueError, AttributeError):
            continue
    return heads


def check_fork(db: str | Path, branch: str | Path) -> ForkCheck:
    """Check that ``branch`` is a branch of the live trail at ``db``."""
    check = ForkCheck(branch=str(Path(branch).expanduser()))
    live = _open_ro(db)
    other = _open_ro(branch)
    try:
        live_top = live.execute("SELECT max(seq) FROM audit_records NOT INDEXED").fetchone()[0]
        head = other.execute(
            "SELECT seq, record_hash FROM audit_records NOT INDEXED ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if live_top is None or head is None:
            check.problems.append("one of the two stores holds no records")
            return check
        check.branch_head_seq, check.branch_head_hash = head[0], head[1]
        fork = _last_shared_seq(live, other, min(live_top, head[0]))
        if fork is None:
            check.problems.append("the two stores do not share their first record")
            return check
        if fork == head[0]:
            check.problems.append(
                f"the branch ends at seq {fork}, which the trail already holds: "
                "it is a copy of this trail's prefix, not a fork")
            return check
        if fork == live_top:
            check.problems.append(
                f"the trail ends at seq {fork} and the branch continues it: "
                "it is an extension of this trail, not a fork")
            return check
        check.fork_seq = fork
        check.fork_hash = _hash_at(live, fork) or ""

        lost = _declared_lost(other)
        prev: Optional[dict] = None
        expected = fork + 1
        seen = 0
        for row in other.execute(
            f"SELECT {_COLUMNS} FROM audit_records NOT INDEXED WHERE seq > ? ORDER BY seq",
            (fork,),
        ):
            rec = dict(row)
            seen += 1
            for missing in range(expected, rec["seq"]):
                check.branch_missing_seqs.append(missing)
                if missing not in lost:
                    check.problems.append(
                        f"branch seq {missing} is missing and the branch never declared it lost")
            if not _record_hash_holds(rec):
                check.problems.append(f"branch seq {rec['seq']} does not recompute its record_hash")
            before_hash = prev["record_hash"] if prev is not None else check.fork_hash
            if rec["seq"] == expected:
                if rec["previous_hash"] == before_hash:
                    check.links_verified += 1
                else:
                    check.problems.append(
                        f"branch seq {rec['seq']} does not name seq {rec['seq'] - 1}")
            if check.branch_first_ts is None:
                check.branch_first_ts = rec["timestamp"]
            check.branch_last_ts = rec["timestamp"]
            prev = rec
            expected = rec["seq"] + 1
        check.branch_record_count = seen
        if len(check.problems) > 20:
            check.problems[20:] = [f"... and {len(check.problems) - 20} more"]
        if check.branch_head_hash in _declared_forks(live):
            check.problems.append("this branch is already declared on the trail")
    finally:
        live.close()
        other.close()
    return check


def declare_fork(check: ForkCheck, db: str | Path, source: str = "") -> None:
    """Append the TRAIL_FORKED record for a passing check to the live trail."""
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.audit.trail import AuditTrail

    if not check.ok:
        raise ValueError("fork check did not pass: " + "; ".join(check.problems))
    db_path = Path(db).expanduser().resolve()
    branch_path = Path(check.branch).expanduser().resolve()
    try:
        label = branch_path.relative_to(db_path.parent.parent).as_posix()
    except ValueError:
        label = branch_path.as_posix()
    backend = SQLiteAuditBackend(db_path)
    try:
        AuditTrail(on_record=backend.write_record).record_trail_forked({
            "fork_seq": check.fork_seq,
            "fork_hash": check.fork_hash,
            "branch_head_seq": check.branch_head_seq,
            "branch_head_hash": check.branch_head_hash,
            "branch_record_count": check.branch_record_count,
            "branch_missing_seqs": check.branch_missing_seqs,
            "branch_first_ts": check.branch_first_ts,
            "branch_last_ts": check.branch_last_ts,
            "links_verified": check.links_verified,
            "branch_path": label,
            "source": source,
        })
    finally:
        backend.close()
