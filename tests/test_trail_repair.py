"""Trail repair: keep every readable record, and declare the ones that are gone.

The trail on the maintainer's machine was damaged seven times on a virtiofs
mount. One morning repair used `.recover` and silently dropped seq 1505; the
evening's REINDEX dropped nothing. Seq 30415 of the llm-proxy trail is missing
from every copy on disk and nothing in the trail says so. These tests damage a
real trail at the page level the way those files were damaged, then check the
two properties the repair has to have: nothing readable is lost, and anything
that is lost is named inside the chain.

The page walker below reads the b-tree directly rather than through `dbstat`,
which not every SQLite build carries.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend, repair_trail_file
from vaara.audit.trail import AuditTrail, EventType
from vaara.pipeline import InterceptionPipeline

N_CALLS = 400  # three to four records each, enough for several leaves per b-tree


def _build(path: Path) -> None:
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(N_CALLS):
        pipeline.intercept(
            agent_id="repair-test", tool_name="file.read",
            parameters={"path": f"/srv/data/{i:05d}.txt", "pad": "x" * 120},
        )
    backend.close()


def _page_size(data: bytes) -> int:
    size = struct.unpack(">H", data[16:18])[0]
    return 65536 if size == 1 else size


def _root(path: Path, name: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT rootpage FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()[0]
    finally:
        conn.close()


def _leaves(data: bytes, root: int) -> list[int]:
    ps = _page_size(data)
    out: list[int] = []

    def walk(n: int) -> None:
        base = (n - 1) * ps
        hdr = base + (100 if n == 1 else 0)
        kind = data[hdr]
        if kind in (0x0A, 0x0D):
            out.append(n)
            return
        assert kind in (0x02, 0x05), f"page {n} is not a b-tree page"
        cells = struct.unpack(">H", data[hdr + 3:hdr + 5])[0]
        right = struct.unpack(">I", data[hdr + 8:hdr + 12])[0]
        for i in range(cells):
            ptr = struct.unpack(">H", data[hdr + 12 + 2 * i:hdr + 14 + 2 * i])[0]
            walk(struct.unpack(">I", data[base + ptr:base + ptr + 4])[0])
        walk(right)

    walk(root)
    return out


def _seqs(path: Path) -> set[int]:
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT seq FROM audit_records NOT INDEXED")}
    finally:
        conn.close()


def _integrity(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute("PRAGMA integrity_check")]
    finally:
        conn.close()


@pytest.fixture
def trail_db(tmp_path: Path) -> Path:
    path = tmp_path / "audit.db"
    _build(path)
    assert _integrity(path) == ["ok"]
    return path


def test_a_clean_trail_is_left_alone(trail_db: Path):
    before = hashlib.sha256(trail_db.read_bytes()).hexdigest()
    report = repair_trail_file(trail_db)
    assert report.method == "clean"
    assert not report.lost
    assert hashlib.sha256(trail_db.read_bytes()).hexdigest() == before


def test_index_damage_is_rebuilt_and_nothing_is_lost(trail_db: Path):
    seqs = _seqs(trail_db)
    data = bytearray(trail_db.read_bytes())
    ps = _page_size(data)
    leaves = _leaves(bytes(data), _root(trail_db, "idx_seq"))
    assert len(leaves) >= 2
    a, b = leaves[0], leaves[-1]
    pa, pb = data[(a - 1) * ps:a * ps], data[(b - 1) * ps:b * ps]
    data[(a - 1) * ps:a * ps], data[(b - 1) * ps:b * ps] = pb, pa
    trail_db.write_bytes(bytes(data))
    assert _integrity(trail_db) != ["ok"]

    backend = SQLiteAuditBackend(trail_db)
    trail = AuditTrail(on_record=backend.write_record)
    report = trail.repair_store()
    backend.close()

    assert report.method == "reindex"
    assert not report.lost
    assert _integrity(trail_db) == ["ok"]
    assert _seqs(trail_db) == seqs
    assert not list(trail_db.parent.glob("audit.db.corrupt-*"))
    # Nothing was lost, so nothing is declared.
    conn = sqlite3.connect(trail_db)
    gaps = conn.execute(
        "SELECT count(*) FROM audit_records WHERE event_type = 'repair_gap'"
    ).fetchone()[0]
    conn.close()
    assert gaps == 0


def test_table_damage_keeps_every_readable_record_and_declares_the_rest(trail_db: Path):
    seqs = _seqs(trail_db)
    conn = sqlite3.connect(trail_db)
    head_before = dict(conn.execute("SELECT seq, record_hash FROM audit_records").fetchall())
    conn.close()
    data = bytearray(trail_db.read_bytes())
    ps = _page_size(data)
    leaves = _leaves(bytes(data), _root(trail_db, "audit_records"))
    assert len(leaves) >= 3
    victim = leaves[len(leaves) // 2]
    data[(victim - 1) * ps:victim * ps] = bytes(ps)
    trail_db.write_bytes(bytes(data))

    backend = SQLiteAuditBackend(trail_db)
    trail = AuditTrail(on_record=backend.write_record)
    report = trail.repair_store()

    assert report.method == "salvage"
    assert report.lost_seqs, "zeroing a leaf page must lose the rows on it"
    survivors = seqs - set(report.lost_seqs)
    assert report.records_kept == len(survivors)
    assert report.damaged_copy and Path(report.damaged_copy).is_file()
    assert _integrity(trail_db) == ["ok"]

    conn = sqlite3.connect(trail_db)
    rows = conn.execute(
        "SELECT seq, record_hash, event_type, data FROM audit_records ORDER BY seq"
    ).fetchall()
    conn.close()
    kept = {s: h for s, h, _e, _d in rows[:-1]}
    # Every surviving record keeps its seq and its original hash.
    assert set(kept) == survivors
    assert all(kept[s] == head_before[s] for s in kept)
    # The last record is the trail's own statement of the hole.
    import json
    seq, _h, event, payload = rows[-1]
    assert event == EventType.REPAIR_GAP.value
    body = json.loads(payload)
    assert body["lost_seqs"] == report.lost_seqs
    assert body["lost_seq_count"] == len(report.lost_seqs)
    assert body["method"] == "salvage"
    assert seq == max(survivors) + 1

    # And the repaired store takes new records.
    pipeline = InterceptionPipeline(trail=trail)
    pipeline.intercept(agent_id="repair-test", tool_name="file.read",
                       parameters={"path": "/srv/after-repair"})
    backend.close()
    assert max(_seqs(trail_db)) > seq


def test_a_trail_nothing_can_read_is_left_exactly_where_it_was(trail_db: Path):
    data = bytearray(trail_db.read_bytes())
    data[:_page_size(data)] = bytes(_page_size(data))
    trail_db.write_bytes(bytes(data))
    before = hashlib.sha256(trail_db.read_bytes()).hexdigest()

    report = repair_trail_file(trail_db)

    assert report.method == "failed"
    assert report.error
    assert hashlib.sha256(trail_db.read_bytes()).hexdigest() == before
    assert not list(trail_db.parent.glob("audit.db.corrupt-*"))
    assert not list(trail_db.parent.glob("audit.db.repair-*"))


def test_a_missing_file_is_a_failed_repair_not_a_new_trail(tmp_path: Path):
    report = repair_trail_file(tmp_path / "nope.db")
    assert report.method == "failed"
    assert not (tmp_path / "nope.db").exists()


def _zero_a_middle_leaf(path: Path) -> None:
    data = bytearray(path.read_bytes())
    ps = _page_size(data)
    leaves = _leaves(bytes(data), _root(path, "audit_records"))
    victim = leaves[len(leaves) // 2]
    data[(victim - 1) * ps:victim * ps] = bytes(ps)
    path.write_bytes(bytes(data))


def test_the_cli_repairs_and_declares(trail_db: Path, capsys):
    import json

    from vaara.cli import main as cli_main

    _zero_a_middle_leaf(trail_db)
    code = cli_main(["trail", "repair", "--db", str(trail_db), "--format", "json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["method"] == "salvage"
    assert report["lost_seqs"]
    conn = sqlite3.connect(trail_db)
    gap = conn.execute(
        "SELECT data FROM audit_records WHERE event_type = 'repair_gap'"
    ).fetchone()
    conn.close()
    assert json.loads(gap[0])["lost_seqs"] == report["lost_seqs"]


def test_the_cli_says_clean_and_changes_nothing(trail_db: Path, capsys):
    from vaara.cli import main as cli_main

    before = hashlib.sha256(trail_db.read_bytes()).hexdigest()
    assert cli_main(["trail", "repair", "--db", str(trail_db)]) == 0
    assert "reads clean" in capsys.readouterr().out
    assert hashlib.sha256(trail_db.read_bytes()).hexdigest() == before


def test_an_open_writer_follows_a_trail_replaced_under_it(trail_db: Path, tmp_path: Path):
    backend = SQLiteAuditBackend(trail_db)
    trail = backend.load_trail()
    # Another process repairs the trail by hand while this writer holds it.
    _zero_a_middle_leaf(trail_db)
    report = repair_trail_file(trail_db)
    assert report.method == "salvage"
    old_copy = Path(report.damaged_copy)
    old_bytes = old_copy.read_bytes()

    pipeline = InterceptionPipeline(trail=trail)
    pipeline.intercept(agent_id="repair-test", tool_name="file.read",
                       parameters={"path": "/srv/after-hand-repair"})
    backend.close()

    conn = sqlite3.connect(trail_db)
    landed = conn.execute(
        "SELECT count(*) FROM audit_records WHERE data LIKE '%after-hand-repair%'"
    ).fetchone()[0]
    conn.close()
    assert landed >= 1
    assert old_copy.read_bytes() == old_bytes
