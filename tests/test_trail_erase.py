"""Erasing a record keeps the chain verifiable and says so.

GDPR Art. 17 meets an append-only trail. `vaara trail erase` replaces one
record's data with a tombstone, keeps its stored record_hash (the next record's
previous_hash already pins it), and appends a RECORD_ERASED record naming the
record by id and hash. Both walks then report the slot as erased: the link
verifies, the content cannot be checked. That is never intact and never a
break. A tombstone no erasure record accounts for is a hash mismatch as before.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.audit.erase import ErasureRefused, erase_record
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import ERASED_DATA_KEY, EventType
from vaara.pipeline import InterceptionPipeline

SECRET = "/home/matti-meikalainen/terveystiedot.txt"


def _build(tmp_path: Path) -> Path:
    path = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(4):
        pipeline.intercept(agent_id="erase-test", tool_name="file.read",
                           parameters={"path": f"/srv/plain/{i}.txt"})
    pipeline.intercept(agent_id="erase-test", tool_name="file.read",
                       parameters={"path": SECRET})
    for i in range(4):
        pipeline.intercept(agent_id="erase-test", tool_name="file.read",
                           parameters={"path": f"/srv/after/{i}.txt"})
    backend.close()
    return path


def _rows(path: Path, needle: str) -> list[tuple[str, int, str, str]]:
    con = sqlite3.connect(path)
    try:
        return [r for r in con.execute(
            "SELECT record_id, seq, event_type, data FROM audit_records ORDER BY seq"
        ) if needle in r[3]]
    finally:
        con.close()


def _target(path: Path) -> tuple[str, int]:
    record_id, seq, _, _ = _rows(path, SECRET)[0]
    return record_id, seq


def _verdicts(path: Path):
    backend = SQLiteAuditBackend(path)
    try:
        trail = backend.load_trail()
        return trail, trail.verify_chain_verdict(), backend.verify_chain_streaming_verdict()
    finally:
        backend.close()


def test_an_erased_record_is_its_own_state_in_both_walks(tmp_path: Path):
    path = _build(tmp_path)
    record_id, seq = _target(path)

    result = erase_record(path, record_id, reason="data subject request 2026-10-17")

    assert result["erased_record_id"] == record_id
    assert result["erased_seq"] == seq
    trail, mem, streamed = _verdicts(path)
    assert mem.error is None and streamed.error is None
    assert mem.state == streamed.state == "erased"
    (slot,) = mem.erased
    assert slot.record_id == record_id and slot.seq == seq
    assert slot.record_hash == result["erased_record_hash"]
    assert streamed.erased[0].record_id == record_id
    summary = mem.summary()
    assert summary is not None and "erased: link verified, content not checkable" in summary
    assert trail.verify_chain() == summary


def test_the_content_is_gone_from_the_file(tmp_path: Path):
    path = _build(tmp_path)
    record_id, _ = _target(path)
    erase_record(path, record_id, reason="dsr")

    assert _rows(path, SECRET) == []
    for name in (path.name, path.name + "-wal"):
        f = path.parent / name
        if f.exists():
            assert SECRET.encode() not in f.read_bytes()


def test_the_erasure_record_names_the_slot_and_the_tombstone_points_back(tmp_path: Path):
    path = _build(tmp_path)
    record_id, seq = _target(path)
    result = erase_record(path, record_id, reason="dsr")

    con = sqlite3.connect(path)
    try:
        (data,) = con.execute(
            "SELECT data FROM audit_records WHERE event_type = ?",
            (EventType.RECORD_ERASED.value,),
        ).fetchone()
        (tomb,) = con.execute(
            "SELECT data FROM audit_records WHERE record_id = ?", (record_id,)
        ).fetchone()
    finally:
        con.close()
    data = json.loads(data)
    assert data["erased_record_id"] == record_id
    assert data["erased_seq"] == seq
    assert data["erased_record_hash"] == result["erased_record_hash"]
    assert data["reason"] == "dsr"
    assert json.loads(tomb) == {ERASED_DATA_KEY: {"erasure_record_id": result["erasure_record_id"]}}


def test_a_tombstone_no_erasure_record_accounts_for_is_broken(tmp_path: Path):
    path = _build(tmp_path)
    record_id, _ = _target(path)
    con = sqlite3.connect(path)
    con.execute("UPDATE audit_records SET data = ? WHERE record_id = ?",
                (json.dumps({ERASED_DATA_KEY: {"erasure_record_id": "forged"}}), record_id))
    con.commit()
    con.close()

    _, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"
    assert "Hash mismatch" in mem.error and "no erasure record" in mem.error
    assert "Hash mismatch" in streamed.error


def test_an_erasure_record_naming_another_hash_does_not_cover_the_slot(tmp_path: Path):
    path = _build(tmp_path)
    record_id, _ = _target(path)
    erase_record(path, record_id, reason="dsr")
    con = sqlite3.connect(path)
    con.execute("UPDATE audit_records SET record_hash = ? WHERE record_id = ?",
                ("0" * 64, record_id))
    con.commit()
    con.close()

    _, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"


def test_a_rewrite_elsewhere_is_still_a_break(tmp_path: Path):
    path = _build(tmp_path)
    record_id, _ = _target(path)
    erase_record(path, record_id, reason="dsr")
    victim = _rows(path, "/srv/after/2.txt")[0][0]
    con = sqlite3.connect(path)
    con.execute("UPDATE audit_records SET data = ? WHERE record_id = ?",
                (json.dumps({"path": "/srv/after/rewritten"}), victim))
    con.commit()
    con.close()

    _, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "broken"
    assert victim in mem.error


def test_unknown_system_and_already_erased_records_are_refused(tmp_path: Path):
    path = _build(tmp_path)
    record_id, _ = _target(path)
    with pytest.raises(ErasureRefused, match="no record"):
        erase_record(path, "no-such-id", reason="dsr")
    result = erase_record(path, record_id, reason="dsr")
    with pytest.raises(ErasureRefused, match="already erased"):
        erase_record(path, record_id, reason="dsr")
    with pytest.raises(ErasureRefused, match="record_erased"):
        erase_record(path, result["erasure_record_id"], reason="dsr")
    with pytest.raises(ErasureRefused, match="reason"):
        erase_record(path, _rows(path, "/srv/plain/1.txt")[0][0], reason="  ")


def test_a_trail_keeps_recording_after_an_erasure(tmp_path: Path):
    path = _build(tmp_path)
    record_id, _ = _target(path)
    erase_record(path, record_id, reason="dsr")
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    pipeline.intercept(agent_id="erase-test", tool_name="file.read",
                       parameters={"path": "/srv/later.txt"})
    backend.close()

    _, mem, streamed = _verdicts(path)
    assert mem.state == streamed.state == "erased"


def test_cli_erase(tmp_path: Path):
    path = _build(tmp_path)
    record_id, seq = _target(path)
    out = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "trail", "erase", "--db", str(path),
         "--record-id", record_id, "--reason", "dsr 17.10", "--format", "json"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["erased_record_id"] == record_id and report["erased_seq"] == seq

    again = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "trail", "erase", "--db", str(path),
         "--record-id", record_id, "--reason", "dsr"],
        capture_output=True, text=True,
    )
    assert again.returncode == 1
    assert "already erased" in again.stderr


def test_compliance_names_an_erasure_and_not_a_compromise(tmp_path: Path):
    from vaara.compliance.engine import create_default_engine

    path = _build(tmp_path)
    record_id, _ = _target(path)
    erase_record(path, record_id, reason="dsr")
    backend = SQLiteAuditBackend(path)
    try:
        report = create_default_engine().assess(backend.load_trail())
    finally:
        backend.close()
    gaps = [g for a in report.articles for g in a.gaps]
    assert any("erased records are named by erasure records" in g for g in gaps)
    assert not any("compromised" in g for g in gaps)
