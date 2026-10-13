"""Page carving: records a damaged b-tree no longer reaches are still on disk.

On 12.10 the maintainer's trail lost 699 records to a salvage: the table's
interior pages no longer pointed at the leaves that held them, so SQLite
could not reach them, though the bytes were intact. `carve_trail` reads the
file page by page without SQLite, and keeps a row only when its record_hash
recomputes from its own content. These tests cut the b-tree off from its
leaves, check that carving returns every record with its original hash, and
check that a record altered on disk is refused rather than recovered.
"""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import pytest

from vaara.audit.carve import carve_trail, write_jsonl
from vaara.audit.sqlite_backend import SQLiteAuditBackend, repair_trail_file
from vaara.pipeline import InterceptionPipeline

N_CALLS = 200


def _build(path: Path) -> None:
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(N_CALLS):
        pipeline.intercept(
            agent_id="carve-test", tool_name="file.read",
            parameters={"path": f"/srv/data/{i:05d}.txt", "pad": "x" * 120},
        )
    backend.close()


def _page_size(data: bytes) -> int:
    size = struct.unpack(">H", data[16:18])[0]
    return 65536 if size == 1 else size


def _root(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT rootpage FROM sqlite_master WHERE name = 'audit_records'"
        ).fetchone()[0]
    finally:
        conn.close()


def _rows(path: Path) -> dict[int, str]:
    conn = sqlite3.connect(path)
    try:
        return dict(conn.execute("SELECT seq, record_hash FROM audit_records"))
    finally:
        conn.close()


@pytest.fixture
def trail_db(tmp_path: Path) -> Path:
    path = tmp_path / "audit.db"
    _build(path)
    return path


def test_a_clean_trail_carves_to_exactly_its_records(trail_db: Path):
    expected = _rows(trail_db)
    result = carve_trail(trail_db)
    assert {r["seq"]: r["record_hash"] for r in result.records} == expected
    assert result.conflicts == []
    # Every record but the first names the one before it.
    assert [s for s, ok in result.linked.items() if not ok] == [min(expected)]


def test_leaves_cut_off_from_the_tree_are_recovered(trail_db: Path):
    expected = _rows(trail_db)
    data = bytearray(trail_db.read_bytes())
    ps = _page_size(data)
    root = _root(trail_db)
    assert data[(root - 1) * ps] == 0x05, "the table needs an interior root"
    # The root no longer points anywhere: every leaf is orphaned.
    data[(root - 1) * ps:root * ps] = bytes([0x0D]) + bytes(ps - 1)
    trail_db.write_bytes(bytes(data))
    before = trail_db.read_bytes()

    result = carve_trail(trail_db)
    assert {r["seq"]: r["record_hash"] for r in result.records} == expected
    # Read-only: the damaged file is exactly as it was.
    assert trail_db.read_bytes() == before

    # A salvage through SQLite cannot reach the same rows.
    report = repair_trail_file(trail_db)
    assert report.records_kept < len(expected)


def test_a_record_altered_on_disk_is_refused(trail_db: Path):
    expected = _rows(trail_db)
    data = bytearray(trail_db.read_bytes())
    at = data.find(b"/srv/data/00042.txt")
    assert at > 0
    data[at:at + 19] = b"/srv/data/99999.txt"
    trail_db.write_bytes(bytes(data))

    result = carve_trail(trail_db)
    got = {r["seq"]: r["record_hash"] for r in result.records}
    missing = set(expected) - set(got)
    assert len(missing) == 1
    (seq,) = missing
    assert result.linked[seq + 1] is False
    assert all(got[s] == expected[s] for s in got)


def test_jsonl_holds_the_wanted_range(trail_db: Path, tmp_path: Path):
    result = carve_trail(trail_db)
    out = tmp_path / "recovered.jsonl"
    n = write_jsonl(result, out, range(10, 20))
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert n == 10 and [r["seq"] for r in lines] == list(range(10, 20))
    summary = result.summary(range(10, 20))
    assert summary["wanted_recovered"] == 10 and summary["wanted_missing"] == []


def test_the_cli_writes_the_records_and_reports(trail_db: Path, tmp_path: Path, capsys):
    from vaara.cli import main
    out = tmp_path / "recovered.jsonl"
    rc = main(["trail", "carve", "--db", str(trail_db), "--out", str(out),
               "--seq-from", "5", "--seq-to", "9", "--format", "json"])
    assert rc in (None, 0)
    report = json.loads(capsys.readouterr().out)
    assert report["wanted_recovered"] == 5
    assert report["written"] == 5
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5
