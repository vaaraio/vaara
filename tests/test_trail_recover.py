"""Recovered records are declared, not spliced back.

After a salvage, the trail's REPAIR_GAP names the seqs it could not keep.
`vaara trail carve` can often read those records out of the damaged copy.
`vaara trail recover` checks a carved bundle against the live trail and
appends one REPAIR_RECOVERED record naming the bundle by digest. History is
not rewritten: the REPAIR_GAP stays true, and a verifier holding the bundle
can recompute every record and see it fills the declared gap. A bundle with a
record that does not recompute, a seq the trail never declared lost, or a
digest already declared is refused.
"""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import pytest

from vaara.audit.carve import carve_trail, check_recovery, write_jsonl
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail, EventType
from vaara.pipeline import InterceptionPipeline


def _build(path: Path, calls: int = 200) -> None:
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(calls):
        pipeline.intercept(
            agent_id="recover-test", tool_name="file.read",
            parameters={"path": f"/srv/data/{i:05d}.txt", "pad": "x" * 120},
        )
    backend.close()


def _leaves(data: bytes, root: int) -> list[int]:
    ps = struct.unpack(">H", data[16:18])[0]
    out: list[int] = []

    def walk(n: int) -> None:
        base = (n - 1) * ps
        hdr = base + (100 if n == 1 else 0)
        if data[hdr] == 0x0D:
            out.append(n)
            return
        cells = struct.unpack(">H", data[hdr + 3:hdr + 5])[0]
        right = struct.unpack(">I", data[hdr + 8:hdr + 12])[0]
        for i in range(cells):
            ptr = struct.unpack(">H", data[hdr + 12 + 2 * i:hdr + 14 + 2 * i])[0]
            walk(struct.unpack(">I", data[base + ptr:base + ptr + 4])[0])
        walk(right)

    walk(root)
    return out


@pytest.fixture
def repaired(tmp_path: Path):
    """A trail that lost one leaf page to a salvage, and its damaged copy."""
    db = tmp_path / "audit.db"
    _build(db)
    con = sqlite3.connect(db)
    root = con.execute("SELECT rootpage FROM sqlite_master WHERE name='audit_records'").fetchone()[0]
    con.close()
    data = bytearray(db.read_bytes())
    ps = struct.unpack(">H", data[16:18])[0]
    leaves = _leaves(bytes(data), root)
    victim = leaves[len(leaves) // 2]
    # Cut the interior page's pointer to the victim leaf by zeroing the leaf's
    # header byte: SQLite reads it as damaged, the carver still finds the cells
    # once the header byte is restored in the copy it reads.
    saved = bytes(data[(victim - 1) * ps:victim * ps])
    data[(victim - 1) * ps:victim * ps] = bytes(ps)
    db.write_bytes(bytes(data))
    backend = SQLiteAuditBackend(db)
    report = AuditTrail(on_record=backend.write_record).repair_store()
    backend.close()
    assert report.method == "salvage" and report.lost_seqs
    damaged = Path(report.damaged_copy)
    # The damaged copy as it would be if the leaf's bytes had survived and only
    # the tree had lost it: put the leaf back, keep the parent pointing nowhere.
    raw = bytearray(damaged.read_bytes())
    raw[(victim - 1) * ps:victim * ps] = saved
    damaged.write_bytes(bytes(raw))
    return db, damaged, list(report.lost_seqs)


def _bundle(damaged: Path, lost: list[int], out: Path) -> Path:
    result = carve_trail(damaged)
    write_jsonl(result, out, range(min(lost), max(lost) + 1))
    return out


def _events(db: Path, kind: str) -> list[dict]:
    con = sqlite3.connect(db)
    try:
        return [json.loads(d) for (d,) in con.execute(
            "SELECT data FROM audit_records WHERE event_type = ?", (kind,))]
    finally:
        con.close()


def test_a_carved_bundle_is_declared_and_the_chain_still_verifies(repaired, tmp_path):
    from vaara.cli import main
    db, damaged, lost = repaired
    bundle = _bundle(damaged, lost, tmp_path / "bundle.jsonl")

    check = check_recovery(bundle, db)
    assert check.ok, check.problems
    assert check.seqs == lost
    assert check.anchored_to_seq == min(lost) - 1
    assert check.links_verified == len(lost) - 1

    assert main(["trail", "recover", "--db", str(db), "--bundle", str(bundle),
                 "--source", "test damaged copy"]) in (None, 0)
    (rec,) = _events(db, EventType.REPAIR_RECOVERED.value)
    assert rec["recovered_seqs"] == lost
    assert rec["bundle_sha256"] == check.bundle_sha256
    kept = db.parent / rec["bundle_path"]
    assert kept.read_bytes() == bundle.read_bytes()
    # The gap stays declared, and nothing was written under the lost seqs.
    assert _events(db, EventType.REPAIR_GAP.value)
    con = sqlite3.connect(db)
    seqs = {s for (s,) in con.execute("SELECT seq FROM audit_records")}
    con.close()
    assert not seqs & set(lost)
    backend = SQLiteAuditBackend(db)
    try:
        # Recovery does not rewrite history: the gap stays declared, and it
        # is the only break the verifier reports.
        verdict = backend.load_trail().verify_chain()
        assert verdict is not None and "1 declared gap" in verdict
        assert "Every other record verifies" in verdict
    finally:
        backend.close()


def test_the_same_bundle_is_not_declared_twice(repaired, tmp_path):
    from vaara.cli import main
    db, damaged, lost = repaired
    bundle = _bundle(damaged, lost, tmp_path / "bundle.jsonl")
    assert main(["trail", "recover", "--db", str(db), "--bundle", str(bundle)]) in (None, 0)
    again = check_recovery(bundle, db)
    assert not again.ok
    assert any("already declared" in p for p in again.problems)


def test_a_record_that_does_not_recompute_is_refused(repaired, tmp_path):
    db, damaged, lost = repaired
    bundle = _bundle(damaged, lost, tmp_path / "bundle.jsonl")
    lines = bundle.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["agent_id"] = "someone-else"
    lines[0] = json.dumps(row, sort_keys=True)
    bundle.write_text("\n".join(lines) + "\n", encoding="utf-8")
    check = check_recovery(bundle, db)
    assert not check.ok
    assert any("does not recompute" in p for p in check.problems)


def test_a_seq_never_declared_lost_is_refused(tmp_path):
    db = tmp_path / "audit.db"
    _build(db, calls=20)
    bundle = tmp_path / "bundle.jsonl"
    write_jsonl(carve_trail(db), bundle, range(5, 8))
    check = check_recovery(bundle, db)
    assert not check.ok
    assert any("never declared lost" in p for p in check.problems)
    assert any("already on the trail" in p for p in check.problems)


def test_dry_run_appends_nothing(repaired, tmp_path):
    from vaara.cli import main
    db, damaged, lost = repaired
    bundle = _bundle(damaged, lost, tmp_path / "bundle.jsonl")
    assert main(["trail", "recover", "--db", str(db), "--bundle", str(bundle),
                 "--dry-run"]) in (None, 0)
    assert _events(db, EventType.REPAIR_RECOVERED.value) == []
    assert not (db.parent / "recovered").exists()
