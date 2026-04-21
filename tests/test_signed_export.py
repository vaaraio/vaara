"""Tests for signed-trail export / verify round-trip."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")  # skip entire module when extra missing

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.audit.export import export_signed
from vaara.audit.trail import AuditTrail
from vaara.audit.verify import verify_signed
from vaara.taxonomy.actions import ActionRequest, create_default_registry

_REGISTRY = create_default_registry()
_TX_TRANSFER = _REGISTRY.get("tx.transfer")


def _make_trail() -> AuditTrail:
    trail = AuditTrail()
    for i in range(3):
        req = ActionRequest(
            agent_id=f"agent-{i}",
            tool_name="send_funds",
            action_type=_TX_TRANSFER,
            parameters={"to": f"0xabc{i}", "amount": 10 * i},
        )
        trail.record_action_requested(req)
    return trail


def _pem_keys(tmp_path: Path, name: str = "signer") -> tuple[Path, Path]:
    key = Ed25519PrivateKey.generate()
    priv_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv = tmp_path / f"{name}.pem"
    pub = tmp_path / f"{name}.pub.pem"
    priv.write_bytes(priv_pem)
    pub.write_bytes(pub_pem)
    return priv, pub


# ── Happy path ────────────────────────────────────────────────────────────

def test_round_trip_happy_path(tmp_path):
    trail = _make_trail()
    priv, pub = _pem_keys(tmp_path)
    out = tmp_path / "trail.zip"

    result = export_signed(trail, out_path=out, signer_key=priv, agent_id="test-agent")

    assert result.path == out
    assert result.chain_intact is True
    assert result.manifest["record_count"] == 3
    assert result.manifest["agent_id"] == "test-agent"
    assert len(result.manifest["signer_pubkey_fingerprint"]) == 32

    # Embedded public key path
    vr = verify_signed(out)
    assert vr.ok, vr.errors

    # Explicit public key path
    vr2 = verify_signed(out, public_key=pub)
    assert vr2.ok, vr2.errors


def test_empty_trail_round_trip(tmp_path):
    trail = AuditTrail()
    priv, _pub = _pem_keys(tmp_path)
    out = tmp_path / "empty.zip"

    result = export_signed(trail, out_path=out, signer_key=priv)
    assert result.manifest["record_count"] == 0
    assert result.manifest["first_hash"] == ""
    assert result.manifest["last_hash"] == ""

    vr = verify_signed(out)
    assert vr.ok, vr.errors


# ── Tamper detection ──────────────────────────────────────────────────────

def _rewrite_zip(src: Path, dst: Path, mutations: dict[str, bytes]) -> None:
    """Copy `src` zip to `dst`, replacing named entries from `mutations`."""
    with zipfile.ZipFile(src, "r") as zin:
        payload = {n: zin.read(n) for n in zin.namelist()}
    payload.update(mutations)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for n, b in payload.items():
            zout.writestr(n, b)


def test_tampered_record_detected(tmp_path):
    trail = _make_trail()
    priv, _ = _pem_keys(tmp_path)
    out = tmp_path / "trail.zip"
    export_signed(trail, out_path=out, signer_key=priv)

    with zipfile.ZipFile(out) as zf:
        trail_bytes = zf.read("trail.jsonl")
    lines = trail_bytes.splitlines()
    rec = json.loads(lines[1])
    rec["agent_id"] = "forged-agent"  # tamper but keep hash untouched
    lines[1] = json.dumps(rec).encode("utf-8")
    tampered_bytes = b"\n".join(lines) + b"\n"

    bad = tmp_path / "bad.zip"
    _rewrite_zip(out, bad, {"trail.jsonl": tampered_bytes})

    vr = verify_signed(bad)
    assert not vr.ok
    joined = " | ".join(vr.errors)
    assert "SHA-256" in joined or "hash" in joined or "signature" in joined


def test_tampered_manifest_detected(tmp_path):
    trail = _make_trail()
    priv, _ = _pem_keys(tmp_path)
    out = tmp_path / "trail.zip"
    export_signed(trail, out_path=out, signer_key=priv)

    with zipfile.ZipFile(out) as zf:
        m = json.loads(zf.read("manifest.json"))
    m["record_count"] = 999
    tampered = json.dumps(m, sort_keys=True, indent=2).encode("utf-8")

    bad = tmp_path / "bad.zip"
    _rewrite_zip(out, bad, {"manifest.json": tampered})

    vr = verify_signed(bad)
    assert not vr.ok


def test_wrong_public_key_rejected(tmp_path):
    trail = _make_trail()
    priv, _ = _pem_keys(tmp_path)
    _other_priv, other_pub = _pem_keys(tmp_path, name="other")  # different keypair
    out = tmp_path / "trail.zip"
    export_signed(trail, out_path=out, signer_key=priv)

    vr = verify_signed(out, public_key=other_pub)
    assert not vr.ok
    assert any("signature" in e.lower() for e in vr.errors)


def test_missing_signature_file_rejected(tmp_path):
    trail = _make_trail()
    priv, _ = _pem_keys(tmp_path)
    out = tmp_path / "trail.zip"
    export_signed(trail, out_path=out, signer_key=priv)

    with zipfile.ZipFile(out) as zf:
        payload = {n: zf.read(n) for n in zf.namelist() if n != "trail.sig"}
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zout:
        for n, b in payload.items():
            zout.writestr(n, b)

    vr = verify_signed(bad)
    assert not vr.ok
    assert any("missing" in e.lower() for e in vr.errors)


def test_corrupt_zip_rejected(tmp_path):
    bad = tmp_path / "not-a-zip.zip"
    bad.write_bytes(b"this is not a zip file")
    vr = verify_signed(bad)
    assert not vr.ok
    assert any("corrupt zip" in e.lower() or "bad zip" in e.lower() for e in vr.errors)


def test_missing_file_rejected(tmp_path):
    vr = verify_signed(tmp_path / "does-not-exist.zip")
    assert not vr.ok
    assert any("not found" in e.lower() for e in vr.errors)


def test_export_rejects_non_trail(tmp_path):
    priv, _ = _pem_keys(tmp_path)
    with pytest.raises(TypeError):
        export_signed("not a trail", out_path=tmp_path / "x.zip", signer_key=priv)  # type: ignore[arg-type]


def test_export_manifest_fingerprint_matches_verify(tmp_path):
    trail = _make_trail()
    priv, pub = _pem_keys(tmp_path)
    out = tmp_path / "trail.zip"
    result = export_signed(trail, out_path=out, signer_key=priv)

    vr = verify_signed(out, public_key=pub)
    assert vr.ok
    assert vr.manifest["signer_pubkey_fingerprint"] == result.manifest["signer_pubkey_fingerprint"]


# ── Missing-cryptography install hint ─────────────────────────────────────
# These tests simulate an install *without* the `[export]` extra by flipping
# the module-level `_HAS_CRYPTO` flag, then assert the ImportError message
# guides the user to `pip install 'vaara[export]'`.

def test_export_missing_crypto_raises_install_hint(tmp_path, monkeypatch):
    from vaara.audit import export as export_mod

    monkeypatch.setattr(export_mod, "_HAS_CRYPTO", False)
    trail = _make_trail()
    with pytest.raises(ImportError, match=r"vaara\[export\]"):
        export_mod.export_signed(trail, out_path=tmp_path / "x.zip", signer_key=tmp_path / "nokey")


def test_verify_missing_crypto_raises_install_hint(tmp_path, monkeypatch):
    from vaara.audit import verify as verify_mod

    monkeypatch.setattr(verify_mod, "_HAS_CRYPTO", False)
    bogus = tmp_path / "any.zip"
    bogus.write_bytes(b"doesnt matter, crypto check fires first")
    with pytest.raises(ImportError, match=r"vaara\[export\]"):
        verify_mod.verify_signed(bogus)
