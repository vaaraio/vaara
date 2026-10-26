"""A purge scrubs the bytes it deletes, and says when it could not finish.

``purge_older_than`` (and so ``vaara trail purge`` and ``vaara trail rotate``)
used to run a plain DELETE. SQLite keeps a deleted row's bytes in the page
until something reuses it, and in the -wal file until a checkpoint, so a
retention purge left the personal data it was meant to drop readable in the
file. The purge now sets ``secure_delete`` and truncates the WAL, and reports
when another reader kept the WAL from being truncated, the way
``vaara trail erase`` does.
"""
from __future__ import annotations

import sqlite3
import time
from argparse import Namespace
from pathlib import Path

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail
from vaara.cli import _cmd_trail_purge

SECRET = "/home/matti-meikalainen/terveystiedot.txt"
DAY = 86400


def _build(tmp_path: Path, monkeypatch) -> Path:
    """Three decisions three days old (one names SECRET) and one from now, chained."""
    path = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(path)
    trail = AuditTrail(on_record=backend.write_record)
    real_time = time.time
    monkeypatch.setattr("vaara.audit.trail.time.time", lambda: real_time() - 3 * DAY)
    trail.record_decision("old0", "purge-test", "file.read", "allow", "plain", 0.1)
    trail.record_decision("old1", "purge-test", "file.read", "allow", SECRET, 0.1)
    trail.record_decision("old2", "purge-test", "file.read", "allow", "plain", 0.1)
    monkeypatch.setattr("vaara.audit.trail.time.time", real_time)
    trail.record_decision("new0", "purge-test", "file.read", "allow", "plain", 0.1)
    backend.close()
    return path


def _file_bytes(path: Path) -> bytes:
    out = b""
    for name in (path.name, path.name + "-wal"):
        f = path.parent / name
        if f.exists():
            out += f.read_bytes()
    return out


def _hold_a_reader(path: Path) -> sqlite3.Connection:
    """Another process's open read snapshot: it keeps the WAL from truncating."""
    reader = sqlite3.connect(path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM audit_records").fetchone()
    return reader


def test_a_plain_delete_would_leave_the_bytes(tmp_path: Path, monkeypatch):
    """The precondition: without the scrub, the purged content stays readable."""
    path = _build(tmp_path, monkeypatch)
    assert SECRET.encode() in _file_bytes(path)
    con = sqlite3.connect(path)
    con.execute("DELETE FROM audit_records WHERE timestamp < ?", (time.time() - DAY,))
    con.commit()
    con.close()
    assert SECRET.encode() in _file_bytes(path)


def test_purged_content_is_gone_from_the_file_and_the_wal(tmp_path: Path, monkeypatch):
    path = _build(tmp_path, monkeypatch)
    backend = SQLiteAuditBackend(path)
    try:
        assert backend.purge_older_than(DAY) == 3
    finally:
        backend.close()

    assert SECRET.encode() not in _file_bytes(path)


def test_a_finished_purge_reports_the_wal_scrubbed(tmp_path: Path, monkeypatch):
    path = _build(tmp_path, monkeypatch)
    backend = SQLiteAuditBackend(path)
    try:
        backend.purge_older_than(DAY)
        assert backend.last_purge_scrubbed is True
    finally:
        backend.close()


def test_a_dry_run_scrubs_nothing_and_reports_nothing(tmp_path: Path, monkeypatch):
    path = _build(tmp_path, monkeypatch)
    backend = SQLiteAuditBackend(path)
    try:
        assert backend.purge_older_than(DAY, dry_run=True) == 3
        assert backend.last_purge_scrubbed is None
    finally:
        backend.close()
    assert SECRET.encode() in _file_bytes(path)


def test_a_purge_that_deletes_nothing_does_not_claim_a_scrub(tmp_path: Path, monkeypatch):
    path = _build(tmp_path, monkeypatch)
    backend = SQLiteAuditBackend(path)
    try:
        assert backend.purge_older_than(30 * DAY) == 0
        assert backend.last_purge_scrubbed is None
    finally:
        backend.close()


def test_a_reader_holding_the_wal_is_reported_not_hidden(tmp_path: Path, monkeypatch):
    path = _build(tmp_path, monkeypatch)
    backend = SQLiteAuditBackend(path)
    reader = _hold_a_reader(path)
    try:
        backend._conn.execute("PRAGMA busy_timeout = 0")
        assert backend.purge_older_than(DAY) == 3
        assert backend.last_purge_scrubbed is False
    finally:
        reader.close()
        backend.close()


def _purge_args(path: Path) -> Namespace:
    return Namespace(db=str(path), retention_days=1, dry_run=False,
                     all_tenants=False, tenant="")


def test_the_purge_command_stays_quiet_when_the_wal_was_truncated(tmp_path: Path, monkeypatch, capsys):
    path = _build(tmp_path, monkeypatch)

    assert _cmd_trail_purge(_purge_args(path)) == 0

    assert "-wal" not in capsys.readouterr().err
    assert SECRET.encode() not in _file_bytes(path)


def test_the_purge_command_names_a_wal_it_could_not_truncate(tmp_path: Path, monkeypatch, capsys):
    path = _build(tmp_path, monkeypatch)
    reader = _hold_a_reader(path)
    try:
        assert _cmd_trail_purge(_purge_args(path)) == 0
    finally:
        reader.close()

    assert "the purged bytes stay in the -wal file until the next checkpoint" in capsys.readouterr().err


def test_rotate_scrubs_what_it_purges(tmp_path: Path, monkeypatch):
    pytest.importorskip("cryptography")
    from vaara.audit.rotate import rotate

    path = _build(tmp_path, monkeypatch)
    result = rotate(db_path=path, out_path=tmp_path / "archive.zip",
                    signer_key=_signer_key(tmp_path), retention_days=1)

    assert result.ok and result.purged_records == 3
    assert result.wal_scrubbed is True
    assert SECRET.encode() not in _file_bytes(path)


def test_rotate_reports_a_wal_it_could_not_truncate(tmp_path: Path, monkeypatch):
    pytest.importorskip("cryptography")
    from vaara.audit.rotate import rotate

    path = _build(tmp_path, monkeypatch)
    reader = _hold_a_reader(path)
    try:
        result = rotate(db_path=path, out_path=tmp_path / "archive.zip",
                        signer_key=_signer_key(tmp_path), retention_days=1)
    finally:
        reader.close()

    assert result.ok and result.purged_records == 3
    assert result.wal_scrubbed is False


def _signer_key(tmp_path: Path) -> Path:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    path = tmp_path / "signer.pem"
    path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return path
