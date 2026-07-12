"""vaara trail rotate: export a signed archive, verify it, then purge.

The documented retention workflow (export signed zip before purge, archive
externally) was manual; rotate makes it one fail-closed command. The purge
must never run unless the exported archive verified.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.audit.rotate import RotateResult, rotate
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail
from vaara.audit.verify import verify_signed

_TEN_DAYS = 10 * 86400.0


def _key_pem(tmp_path):
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "signer.pem"
    path.write_bytes(pem)
    return path


def _seed(db_path, monkeypatch):
    """Two records 10 days old, two written now, all properly chained."""
    backend = SQLiteAuditBackend(db_path)
    trail = AuditTrail(on_record=backend.write_record)

    real_time = time.time
    monkeypatch.setattr("vaara.audit.trail.time.time", lambda: real_time() - _TEN_DAYS)
    trail.record_decision("old1", "agent-x", "shell_exec", "deny", "high risk", 0.9)
    trail.record_decision("old2", "agent-x", "read_file", "allow", "low risk", 0.1)
    monkeypatch.setattr("vaara.audit.trail.time.time", real_time)
    trail.record_decision("new1", "agent-x", "read_file", "allow", "low risk", 0.1)
    trail.record_decision("new2", "agent-x", "read_file", "allow", "low risk", 0.1)
    backend.close()


def test_rotate_exports_verifies_and_purges(tmp_path, monkeypatch):
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    out = tmp_path / "archive.zip"

    result = rotate(
        db_path=db,
        out_path=out,
        signer_key=_key_pem(tmp_path),
        retention_days=7,
    )

    assert isinstance(result, RotateResult)
    assert result.ok
    assert result.exported_records == 4
    assert result.purged_records == 2
    assert out.exists()
    assert verify_signed(out).ok

    backend = SQLiteAuditBackend(db)
    try:
        assert backend.count() == 2
    finally:
        backend.close()


def test_rotate_dry_run_purges_nothing(tmp_path, monkeypatch):
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    out = tmp_path / "archive.zip"

    result = rotate(
        db_path=db,
        out_path=out,
        signer_key=_key_pem(tmp_path),
        retention_days=7,
        dry_run=True,
    )

    assert result.ok
    assert result.purged_records == 2  # would purge
    backend = SQLiteAuditBackend(db)
    try:
        assert backend.count() == 4
    finally:
        backend.close()


def test_rotate_bad_key_fails_closed(tmp_path, monkeypatch):
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    bad_key = tmp_path / "bad.pem"
    bad_key.write_text("not a key")

    result = rotate(
        db_path=db,
        out_path=tmp_path / "archive.zip",
        signer_key=bad_key,
        retention_days=7,
    )

    assert not result.ok
    assert result.purged_records == 0
    backend = SQLiteAuditBackend(db)
    try:
        assert backend.count() == 4  # nothing purged
    finally:
        backend.close()


def test_cli_trail_rotate(tmp_path, monkeypatch, capsys):
    from vaara.cli import main

    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    out = tmp_path / "archive.zip"

    rc = main([
        "trail", "rotate",
        "--db", str(db),
        "--out", str(out),
        "--key", str(_key_pem(tmp_path)),
        "--retention-days", "7",
        "--all-tenants",
    ])
    assert rc == 0
    stdout = capsys.readouterr().out.lower()
    assert "purged 2" in stdout
    assert out.exists()
