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

import base64
import json
import os
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
        sa.encrypt_file(plain, sealed, sa.load_recipient_public(rpub))
        back = tmp_path / "back.bin"
        sa.decrypt_file(sealed, back, sa.load_recipient_private(rpriv))
        assert back.read_bytes() == plain.read_bytes()

    def _sealed(self, tmp_path, size=2 * sa.CHUNK + 5):
        rpriv, rpub = _recipient(tmp_path)
        plain = tmp_path / "p.bin"
        plain.write_bytes(b"x" * size)
        sealed = tmp_path / "p.sealed"
        sa.encrypt_file(plain, sealed, sa.load_recipient_public(rpub))
        return sealed, sa.load_recipient_private(rpriv)

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
            sa.decrypt_file(sealed, tmp_path / "out.bin", sa.load_recipient_private(other))

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
        data = sealed.read_bytes()
        sealed.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
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
        spub = tmp_path / "signer.pub.pem"
        sealed, seal, manifest = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        db = tmp_path / "live.db"
        backend = SQLiteAuditBackend(db)
        trail = backend.load_trail()
        out = sa.unseal(sealed, seal, rpriv, tmp_path / "opened.zip", expected_signer=spub,
                        trail=trail,
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
        spriv, spub = _signer(tmp_path)
        rpriv, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        data = sealed.read_bytes()
        sealed.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
        with pytest.raises(sa.SealError):
            sa.unseal(sealed, seal, rpriv, tmp_path / "opened.zip", expected_signer=spub)
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
        opened = sa.unseal(result.sealed_path, result.seal_path, rpriv, tmp_path / "o.zip",
                           expected_signer=spub)
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
                     "--signer-pub", str(spub), "--out", str(tmp_path / "o.zip")]) == 2
        assert main(["trail", "unseal", str(sealed), "--seal", str(seal), "--key", str(rpriv),
                     "--signer-pub", str(spub), "--out", str(tmp_path / "o.zip"), "--db", str(db),
                     "--on-behalf-of", "dpo", "--basis", "audit request"]) == 0
        assert verify_signed(tmp_path / "o.zip").ok


def _access_records(db):
    backend = SQLiteAuditBackend(db)
    try:
        return [r for r in backend.load_trail()._records
                if r.event_type.value == "access_recorded"]
    finally:
        backend.close()


class TestReviewFindings:
    """One test per finding of the 15.10 review of this module."""

    def _sealed(self, tmp_path):
        spriv, spub = _signer(tmp_path)
        rpriv, rpub = _recipient(tmp_path)
        sealed, seal, manifest = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        return spriv, spub, rpriv, rpub, sealed, seal, manifest

    @pytest.mark.skipif(not hasattr(os, "pwrite"), reason="os.pwrite is POSIX only")
    def test_a_write_fault_is_caught_before_the_file_is_published(self, tmp_path, monkeypatch):
        _, rpub = _recipient(tmp_path)
        plain = tmp_path / "p.bin"
        plain.write_bytes(b"y" * (2 * sa.CHUNK))
        real_fsync = sa.os.fsync

        def corrupting_fsync(fd):
            # Sixteen bytes, not one: ciphertext is random, so a single fixed byte is
            # already 0xff one time in 256 and the write would change nothing.
            os.pwrite(fd, b"\xff" * 16, 300)
            real_fsync(fd)

        monkeypatch.setattr(sa.os, "fsync", corrupting_fsync)
        sealed = tmp_path / "p.sealed"
        with pytest.raises(sa.SealError):
            sa.encrypt_file(plain, sealed, sa.load_recipient_public(rpub))
        assert not sealed.exists()
        assert not list(tmp_path.glob("p.sealed.*.tmp"))

    def test_unseal_rejects_a_forgery_signed_with_its_own_key(self, tmp_path):
        spriv, spub = _signer(tmp_path)
        forger, _ = _signer(tmp_path, "forger")
        rpriv, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, forger), rpub, forger)
        with pytest.raises(sa.SealError, match="fingerprint"):
            sa.unseal(sealed, seal, rpriv, tmp_path / "o.zip", expected_signer=spub)
        assert not (tmp_path / "o.zip").exists()

    def test_a_wrong_key_writes_no_access_record(self, tmp_path):
        _, spub, _, _, sealed, seal, _ = self._sealed(tmp_path)
        other, _ = _recipient(tmp_path, "other")
        db = tmp_path / "live.db"
        backend = SQLiteAuditBackend(db)
        trail = backend.load_trail()
        with pytest.raises(sa.SealError, match="different recipient"):
            sa.unseal(sealed, seal, other, tmp_path / "o.zip", expected_signer=spub,
                      trail=trail, on_behalf_of="dpo", basis="test")
        backend.close()
        assert _access_records(db) == []

    @pytest.mark.parametrize("field,value", [
        ("ephemeral_public", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
        ("nonce_prefix", 7),
        ("ephemeral_public", "not base64!"),
    ])
    def test_a_malformed_header_is_a_seal_error_and_records_nothing(self, tmp_path, field, value):
        _, spub, rpriv, rpub, sealed, seal, _ = self._sealed(tmp_path)
        data = sealed.read_bytes()
        n = int.from_bytes(data[len(sa.MAGIC):len(sa.MAGIC) + 2], "big")
        header = json.loads(data[len(sa.MAGIC) + 2:len(sa.MAGIC) + 2 + n])
        header[field] = value
        raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        sealed.write_bytes(sa.MAGIC + len(raw).to_bytes(2, "big") + raw
                           + data[len(sa.MAGIC) + 2 + n:])
        with pytest.raises(sa.SealError):
            sa.decrypt_file(sealed, tmp_path / "o.zip", sa.load_recipient_private(rpriv))

    def test_an_all_zero_ephemeral_key_is_a_seal_error(self, tmp_path):
        _, _, rpriv, _, sealed, _, _ = self._sealed(tmp_path)
        data = sealed.read_bytes()
        n = int.from_bytes(data[len(sa.MAGIC):len(sa.MAGIC) + 2], "big")
        header = json.loads(data[len(sa.MAGIC) + 2:len(sa.MAGIC) + 2 + n])
        header["ephemeral_public"] = base64.b64encode(bytes(32)).decode()
        raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        sealed.write_bytes(sa.MAGIC + len(raw).to_bytes(2, "big") + raw
                           + data[len(sa.MAGIC) + 2 + n:])
        with pytest.raises(sa.SealError):
            sa.decrypt_file(sealed, tmp_path / "o.zip", sa.load_recipient_private(rpriv))

    def test_a_self_signed_seal_with_odd_types_is_reported_not_raised(self, tmp_path):
        spriv, _, _, _, sealed, seal, _ = self._sealed(tmp_path)
        key = Ed25519PrivateKey.from_private_bytes(
            serialization.load_pem_private_key(spriv.read_bytes(), None).private_bytes_raw())
        outer = json.loads(seal.read_text())
        manifest = json.loads(outer["manifest"])
        manifest["ciphertext"] = "x"
        text = json.dumps(manifest, sort_keys=True, indent=2)
        outer["manifest"] = text
        outer["signature"] = base64.b64encode(key.sign(text.encode())).decode()
        seal.write_text(json.dumps(outer))
        verdict = sa.verify_seal(seal, sealed)
        assert not verdict.ok and "ciphertext" in verdict.errors[0]

    def test_dry_run_keeps_the_plaintext(self, tmp_path, monkeypatch):
        db = tmp_path / "audit.db"
        _seed(db, monkeypatch)
        spriv, _ = _signer(tmp_path)
        _, rpub = _recipient(tmp_path)
        out = tmp_path / "a.zip"
        result = rotate(db_path=db, out_path=out, signer_key=spriv, retention_days=7,
                        encrypt_to=rpub, dry_run=True)
        assert result.ok and out.exists() and result.archive_path == out
        assert result.sealed_path.exists()

    def test_a_failed_seal_leaves_no_orphan(self, tmp_path, monkeypatch):
        db = tmp_path / "audit.db"
        _seed(db, monkeypatch)
        spriv, _ = _signer(tmp_path)
        _, rpub = _recipient(tmp_path)
        out = tmp_path / "a.zip"
        monkeypatch.setattr(sa, "verify_seal", lambda *a, **k: sa.SealVerdict(False, {}, ["forced"]))
        result = rotate(db_path=db, out_path=out, signer_key=spriv, retention_days=7,
                        encrypt_to=rpub)
        assert not result.ok and out.exists()
        assert not (tmp_path / "a.zip.sealed").exists()
        assert not (tmp_path / "a.zip.seal.json").exists()

    def test_an_unsupported_key_type_skips_the_purge_cleanly(self, tmp_path, monkeypatch):
        db = tmp_path / "audit.db"
        _seed(db, monkeypatch)
        spriv, _ = _signer(tmp_path)
        bad = tmp_path / "bad.pem"
        bad.write_bytes(b"-----BEGIN PUBLIC KEY-----\nAAAA\n-----END PUBLIC KEY-----\n")
        result = rotate(db_path=db, out_path=tmp_path / "a.zip", signer_key=spriv,
                        retention_days=7, encrypt_to=bad)
        assert not result.ok and "purge skipped" in result.errors[0]


class TestSecondReview:
    """Findings of the second review pass, 15.10."""

    def test_a_rerun_with_the_same_out_never_replaces_last_periods_archive(self, tmp_path, monkeypatch):
        db = tmp_path / "audit.db"
        _seed(db, monkeypatch)
        spriv, _ = _signer(tmp_path)
        _, rpub = _recipient(tmp_path)
        out = tmp_path / "a.zip"
        first = rotate(db_path=db, out_path=out, signer_key=spriv, retention_days=7, encrypt_to=rpub)
        assert first.ok
        before = first.sealed_path.read_bytes()
        again = rotate(db_path=db, out_path=out, signer_key=spriv, retention_days=7, encrypt_to=rpub)
        assert not again.ok and "never replaced" in again.errors[0]
        assert first.sealed_path.read_bytes() == before

    def test_seal_archive_refuses_an_existing_target_and_leaves_it(self, tmp_path):
        spriv, _ = _signer(tmp_path)
        _, rpub = _recipient(tmp_path)
        zip_path = _export(tmp_path, spriv)
        sealed = zip_path.with_name(zip_path.name + ".sealed")
        sealed.write_bytes(b"earlier period")
        with pytest.raises(sa.SealError, match="exists"):
            sa.seal_archive(zip_path, rpub, spriv)
        assert sealed.read_bytes() == b"earlier period"

    def test_unseal_refuses_an_existing_out_before_recording(self, tmp_path):
        spriv, spub = _signer(tmp_path)
        rpriv, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        victim = tmp_path / "important.db"
        victim.write_bytes(b"keep me")
        db = tmp_path / "live.db"
        backend = SQLiteAuditBackend(db)
        trail = backend.load_trail()
        with pytest.raises(sa.SealError, match="exists"):
            sa.unseal(sealed, seal, rpriv, victim, expected_signer=spub, trail=trail,
                      on_behalf_of="dpo", basis="test")
        backend.close()
        assert victim.read_bytes() == b"keep me"
        assert _access_records(db) == []

    def test_unseal_decrypts_nothing_when_the_access_record_is_not_persisted(self, tmp_path):
        spriv, spub = _signer(tmp_path)
        rpriv, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)

        def refuse(record):
            raise OSError("disk full")

        trail = AuditTrail(on_record=refuse)
        with pytest.raises(sa.SealError, match="could not be written"):
            sa.unseal(sealed, seal, rpriv, tmp_path / "o.zip", expected_signer=spub,
                      trail=trail, on_behalf_of="dpo", basis="test")
        assert not (tmp_path / "o.zip").exists()

    @pytest.mark.parametrize("patch", [
        {"ciphertext": {"sha256": "x", "size": True}},
        {"trail": {"record_count": "all"}},
    ])
    def test_odd_types_in_a_self_signed_seal_are_reported(self, tmp_path, patch):
        spriv, _ = _signer(tmp_path)
        _, rpub = _recipient(tmp_path)
        sealed, seal, _ = sa.seal_archive(_export(tmp_path, spriv), rpub, spriv)
        key = serialization.load_pem_private_key(spriv.read_bytes(), None)
        outer = json.loads(seal.read_text())
        manifest = json.loads(outer["manifest"])
        manifest.update(patch)
        text = json.dumps(manifest, sort_keys=True, indent=2)
        outer["manifest"] = text
        outer["signature"] = base64.b64encode(key.sign(text.encode())).decode()
        seal.write_text(json.dumps(outer))
        assert not sa.verify_seal(seal, sealed).ok

    def test_keygen_leaves_no_orphan_private_key(self, tmp_path):
        priv = tmp_path / "k.key"
        with pytest.raises(OSError):
            sa.generate_recipient_key(priv, tmp_path / "no-such-dir" / "k.pub")
        assert not priv.exists()
