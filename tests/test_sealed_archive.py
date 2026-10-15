# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encrypted retention archives with a public, signed seal.

A retention archive has two readers. The key holder reads the records; anyone
else checks, without the key, that the archive exists, is intact, is the one
the trail's key signed, and ends where the live trail's seam begins. These
tests hold both sides, the purge ordering in rotate, and the access record
unseal writes before it decrypts.
"""

from __future__ import annotations

import json
import stat
import sys
import time

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from vaara.audit import sealed_archive as sa  # noqa: E402
from vaara.audit.export import export_signed  # noqa: E402
from vaara.audit.rotate import rotate  # noqa: E402
from vaara.audit.sqlite_backend import SQLiteAuditBackend  # noqa: E402
from vaara.audit.trail import AuditTrail  # noqa: E402
from vaara.audit.verify import verify_signed  # noqa: E402

_TEN_DAYS = 10 * 86400.0


def _signer(tmp_path, name="signer"):
    key = Ed25519PrivateKey.generate()
    priv = tmp_path / f"{name}.pem"
    priv.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    pub = tmp_path / f"{name}.pub.pem"
    pub.write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv, pub


def _recipient(tmp_path, name="dpo"):
    priv, pub = tmp_path / f"{name}.key", tmp_path / f"{name}.pub"
    sa.generate_recipient_key(priv, pub)
    return priv, pub


def _seed(db_path, monkeypatch):
    backend = SQLiteAuditBackend(db_path)
    trail = AuditTrail(on_record=backend.write_record)
    real_time = time.time
    monkeypatch.setattr("vaara.audit.trail.time.time", lambda: real_time() - _TEN_DAYS)
    trail.record_decision("old1", "agent-x", "shell_exec", "deny", "high risk", 0.9)
    trail.record_decision("old2", "agent-x", "read_file", "allow", "low risk", 0.1)
    monkeypatch.setattr("vaara.audit.trail.time.time", real_time)
    trail.record_decision("new1", "agent-x", "read_file", "allow", "low risk", 0.1)
    backend.close()


def _export(tmp_path, signer_priv, records=3):
    trail = AuditTrail()
    for i in range(records):
        trail.record_decision(f"a{i}", "agent-x", "read_file", "allow", "ok", 0.1)
    out = tmp_path / "archive.zip"
    export_signed(trail, out, signer_key=signer_priv)
    return out


class TestCipher:
    @pytest.mark.parametrize("size", [0, 1, sa.CHUNK - 1, sa.CHUNK, sa.CHUNK + 1, 3 * sa.CHUNK + 7])
    def test_round_trip_at_chunk_edges(self, tmp_path, size):
        rpriv, rpub = _recipient(tmp_path)
        plain = tmp_path / "p.bin"
        plain.write_bytes(bytes((i * 31) % 251 for i in range(size)))
        sealed = tmp_path / "p.sealed"
        sa.encrypt_file(plain, sealed, sa._load_recipient_public(rpub))
        back = tmp_path / "back.bin"
        sa.decrypt_file(sealed, back, sa._load_recipient_private(rpriv))
        assert back.read_bytes() == plain.read_bytes()

    def _sealed(self, tmp_path, size=2 * sa.CHUNK + 5):
        rpriv, rpub = _recipient(tmp_path)
        plain = tmp_path / "p.bin"
        plain.write_bytes(b"x" * size)
        sealed = tmp_path / "p.sealed"
        sa.encrypt_file(plain, sealed, sa._load_recipient_public(rpub))
        return sealed, sa._load_recipient_private(rpriv)

    def test_a_flipped_byte_fails_and_leaves_nothing(self, tmp_path):
        sealed, key = self._sealed(tmp_path)
        data = bytearray(sealed.read_bytes())
        data[-100] ^= 1
        sealed.write_bytes(bytes(data))
        out = tmp_path / "out.bin"
        with pytest.raises(sa.SealError, match="authentication"):
            sa.decrypt_file(sealed, out, key)
        assert not out.exists()

    def test_a_dropped_last_chunk_fails(self, tmp_path):
        sealed, key = self._sealed(tmp_path)
        data = sealed.read_bytes()
        sealed.write_bytes(data[: len(data) - (5 + 16)])
        with pytest.raises(sa.SealError):
            sa.decrypt_file(sealed, tmp_path / "out.bin", key)

    def test_swapped_chunks_fail(self, tmp_path):
        sealed, key = self._sealed(tmp_path, size=3 * sa.CHUNK)
        data = sealed.read_bytes()
        hdr_len = len(sa.MAGIC) + 2 + int.from_bytes(data[len(sa.MAGIC):len(sa.MAGIC) + 2], "big")
        c = sa.CHUNK + 16
        body = data[hdr_len:]
        swapped = body[c:2 * c] + body[:c] + body[2 * c:]
        sealed.write_bytes(data[:hdr_len] + swapped)
        with pytest.raises(sa.SealError):
            sa.decrypt_file(sealed, tmp_path / "out.bin", key)

    def test_the_wrong_recipient_is_named(self, tmp_path):
        sealed, _ = self._sealed(tmp_path)
        other, _ = _recipient(tmp_path, "other")
        with pytest.raises(sa.SealError, match="different recipient"):
            sa.decrypt_file(sealed, tmp_path / "out.bin", sa._load_recipient_private(other))

    @pytest.mark.skipif(sys.platform == "win32", reason="no POSIX mode bits")
    def test_the_private_key_is_owner_only_and_never_overwritten(self, tmp_path):
        priv, pub = _recipient(tmp_path)
        assert stat.S_IMODE(priv.stat().st_mode) == 0o600
        with pytest.raises(FileExistsError):
            sa.generate_recipient_key(priv, pub)


class TestSeal:
    def test_seal_names_the_chain_and_verifies_without_the_key(self, tmp_path):
        spriv, spub = _signer(tmp_path)
        _, rpub = _recipient(tmp_path)
        zip_path = _export(tmp_path, spriv)
        export_manifest = json.loads(__import__("zipfile").ZipFile(zip_path).read("manifest.json"))
        sealed, seal, manifest = sa.seal_archive(zip_path, rpub, spriv)
        assert manifest["trail"]["record_count"] == 3
        assert manifest["trail"]["first_hash"] == export_manifest["first_hash"]
        assert manifest["trail"]["last_hash"] == export_manifest["last_hash"]
        assert manifest["trail"]["chain_intact_at_export"] is True
        verdict = sa.verify_seal(seal, sealed, expected_signer=spub)
        assert verdict.ok, verdict.errors

    def test_a_seal_signed_by_another_key_than_the_export_is_refused(self, tmp_path):
        spriv, _ = _signer(tmp_path)
        other, _ = _signer(tmp_path, "other")
        _, rpub = _recipient(tmp_path)
        zip_path = _export(tmp_path, spriv)
        with pytest.raises(sa.SealError, match="not the key that signed the export"):
            sa.seal_archive(zip_path, rpub, other)
        assert not zip_path.with_name(zip_path.name + ".sealed").exists()

    def test_an_edited_seal_fails(self, tmp_path):
        spriv, _ = _signer(tmp_path)
        _, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        outer = json.loads(seal.read_text())
        outer["manifest"] = outer["manifest"].replace('"record_count": 3', '"record_count": 2')
        seal.write_text(json.dumps(outer))
        assert "seal signature does not verify" in sa.verify_seal(seal, sealed).errors

    def test_a_swapped_ciphertext_fails(self, tmp_path):
        spriv, _ = _signer(tmp_path)
        _, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        sealed.write_bytes(sealed.read_bytes()[:-1] + b"\x00")
        verdict = sa.verify_seal(seal, sealed)
        assert not verdict.ok and "does not match the seal" in verdict.errors[0]

    def test_a_pinned_signer_rejects_a_self_consistent_forgery(self, tmp_path):
        spriv, spub = _signer(tmp_path)
        forger, _ = _signer(tmp_path, "forger")
        _, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, forger), rpub, forger)
        assert sa.verify_seal(seal, sealed).ok
        assert not sa.verify_seal(seal, sealed, expected_signer=spub).ok


class TestUnseal:
    def test_unseal_records_the_access_first_and_returns_a_verified_export(self, tmp_path):
        spriv, _ = _signer(tmp_path)
        rpriv, rpub = _recipient(tmp_path)
        sealed, seal, manifest = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        db = tmp_path / "live.db"
        backend = SQLiteAuditBackend(db)
        trail = backend.load_trail()
        out = sa.unseal(sealed, seal, rpriv, tmp_path / "opened.zip", trail=trail,
                        on_behalf_of="dpo@example.org", basis="legal hold 2026-114")
        backend.close()
        assert verify_signed(out).ok
        reread = SQLiteAuditBackend(db)
        try:
            records = reread.load_trail()._records
        finally:
            reread.close()
        access = [r for r in records if r.event_type.value == "access_recorded"]
        assert len(access) == 1
        params = access[0].data["parameters"]
        assert params["subject"] == f"sealed-archive:sha256:{manifest['ciphertext']['sha256']}"
        assert params["returned_sha256"] == f"sha256:{manifest['archive']['sha256']}"
        assert params["basis"] == "legal hold 2026-114"

    def test_a_tampered_archive_is_not_opened(self, tmp_path):
        spriv, _ = _signer(tmp_path)
        rpriv, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        sealed.write_bytes(sealed.read_bytes()[:-1] + b"\x00")
        with pytest.raises(sa.SealError):
            sa.unseal(sealed, seal, rpriv, tmp_path / "opened.zip")
        assert not (tmp_path / "opened.zip").exists()


class TestRotate:
    def test_rotate_encrypts_removes_the_plaintext_then_purges(self, tmp_path, monkeypatch):
        db = tmp_path / "audit.db"
        _seed(db, monkeypatch)
        spriv, spub = _signer(tmp_path)
        rpriv, rpub = _recipient(tmp_path)
        out = tmp_path / "archive.zip"
        result = rotate(db_path=db, out_path=out, signer_key=spriv,
                        retention_days=7, encrypt_to=rpub)
        assert result.ok, result.errors
        assert not out.exists()
        assert result.archive_path == result.sealed_path
        assert sa.verify_seal(result.seal_path, result.sealed_path, expected_signer=spub).ok
        assert result.purged_records == 2
        last_hash = sa.verify_seal(result.seal_path).manifest["trail"]["last_hash"]
        backend = SQLiteAuditBackend(db)
        try:
            live = [r.record_hash for r in backend.load_trail()._records]
        finally:
            backend.close()
        assert last_hash in live, "the seal's last hash is a record the purge kept"
        opened = sa.unseal(result.sealed_path, result.seal_path, rpriv, tmp_path / "o.zip")
        assert verify_signed(opened).ok

    def test_a_bad_recipient_key_skips_the_purge(self, tmp_path, monkeypatch):
        db = tmp_path / "audit.db"
        _seed(db, monkeypatch)
        spriv, spub = _signer(tmp_path)
        result = rotate(db_path=db, out_path=tmp_path / "a.zip", signer_key=spriv,
                        retention_days=7, encrypt_to=spub)
        assert not result.ok and "purge skipped" in result.errors[0]
        backend = SQLiteAuditBackend(db)
        try:
            assert backend.count() == 3
        finally:
            backend.close()


class TestCli:
    def test_keygen_rotate_verify_unseal(self, tmp_path, monkeypatch, capsys):
        from vaara.cli import main

        db = tmp_path / "audit.db"
        _seed(db, monkeypatch)
        spriv, spub = _signer(tmp_path)
        rpriv, rpub = tmp_path / "r.key", tmp_path / "r.pub"
        assert main(["trail", "seal-keygen", "--private", str(rpriv), "--public", str(rpub)]) == 0
        out = tmp_path / "a.zip"
        assert main(["trail", "rotate", "--db", str(db), "--out", str(out), "--key", str(spriv),
                     "--retention-days", "7", "--all-tenants", "--encrypt-to", str(rpub)]) == 0
        sealed, seal = tmp_path / "a.zip.sealed", tmp_path / "a.zip.seal.json"
        assert main(["trail", "verify-seal", str(seal), "--ciphertext", str(sealed),
                     "--signer-pub", str(spub)]) == 0
        assert main(["trail", "unseal", str(sealed), "--seal", str(seal), "--key", str(rpriv),
                     "--out", str(tmp_path / "o.zip")]) == 2
        assert main(["trail", "unseal", str(sealed), "--seal", str(seal), "--key", str(rpriv),
                     "--out", str(tmp_path / "o.zip"), "--db", str(db),
                     "--on-behalf-of", "dpo", "--basis", "audit request"]) == 0
        assert verify_signed(tmp_path / "o.zip").ok
