"""Tests for the pluggable signer abstraction (Ed25519 + ML-DSA-65)."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
except ImportError:
    pytest.skip(
        "cryptography not installed (pip install 'vaara[export]')",
        allow_module_level=True,
    )

from vaara.audit.export import export_signed
from vaara.audit.signer import (
    Ed25519Signer,
    Ed25519Verifier,
    Signer,
    Verifier,
    verifier_for,
)
from vaara.audit.trail import AuditTrail
from vaara.audit.verify import verify_signed
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)


def _populated_trail() -> AuditTrail:
    trail = AuditTrail()
    action_type = ActionType(
        name="data.read",
        category=ActionCategory.DATA,
        reversibility=Reversibility.FULLY,
        blast_radius=BlastRadius.LOCAL,
        urgency=UrgencyClass.DEFERRABLE,
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    req = ActionRequest(
        agent_id="a-1", tool_name="data.read",
        action_type=action_type, parameters={}, confidence=0.9,
    )
    action_id = trail.record_action_requested(req)
    trail.record_decision(
        action_id=action_id, agent_id="a-1", tool_name="data.read",
        decision="allow", reason="ok", risk_score=0.1,
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    return trail


# ── Ed25519 surface ──────────────────────────────────────────────────


def test_ed25519_signer_implements_protocol():
    key = Ed25519PrivateKey.generate()
    signer = Ed25519Signer(key)
    assert isinstance(signer, Signer)
    assert signer.algorithm == "Ed25519"
    sig = signer.sign(b"hello")
    assert len(sig) == 64


def test_ed25519_signer_rejects_wrong_key_type():
    with pytest.raises(TypeError):
        Ed25519Signer("not-a-key")  # type: ignore[arg-type]


def test_ed25519_verifier_round_trip():
    key = Ed25519PrivateKey.generate()
    signer = Ed25519Signer(key)
    verifier = Ed25519Verifier(signer.public_key_bytes())
    assert isinstance(verifier, Verifier)
    msg = b"audit-message"
    sig = signer.sign(msg)
    assert verifier.verify(msg, sig) is True
    assert verifier.verify(b"tampered", sig) is False


def test_export_signed_via_signer_argument(tmp_path: Path):
    key = Ed25519PrivateKey.generate()
    signer = Ed25519Signer(key)
    result = export_signed(
        _populated_trail(), tmp_path / "trail.zip", signer=signer,
    )
    assert result.path.is_file()
    assert result.manifest["signature_algorithm"] == "Ed25519"
    verify = verify_signed(result.path)
    assert verify.ok, verify.errors


def test_export_signed_rejects_both_arguments(tmp_path: Path):
    key = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="not both"):
        export_signed(
            _populated_trail(), tmp_path / "x.zip",
            signer_key=key, signer=Ed25519Signer(key),
        )


def test_export_signed_rejects_neither_argument(tmp_path: Path):
    with pytest.raises(ValueError, match="required"):
        export_signed(_populated_trail(), tmp_path / "x.zip")


def test_verifier_for_dispatches_by_algorithm():
    key = Ed25519PrivateKey.generate()
    signer = Ed25519Signer(key)
    v = verifier_for("Ed25519", signer.public_key_bytes())
    assert v.algorithm == "Ed25519"
    with pytest.raises(ValueError):
        verifier_for("RSA-PSS", b"not-a-key")


# ── ML-DSA-65 surface (requires vaara[pq]) ──────────────────────────


def _require_pq():
    return pytest.importorskip("dilithium_py")


def test_mldsa_signer_round_trip():
    _require_pq()
    from vaara.audit.signer import MLDSA65Signer, MLDSA65Verifier

    signer, pub = MLDSA65Signer.generate()
    assert signer.algorithm == "ML-DSA-65"
    sig = signer.sign(b"hello pq")
    assert len(sig) > 1000  # ML-DSA-65 sigs are ~3.3 KB
    verifier = MLDSA65Verifier(pub)
    assert verifier.verify(b"hello pq", sig) is True
    assert verifier.verify(b"tampered", sig) is False


def test_mldsa_signer_via_export_zip(tmp_path: Path):
    _require_pq()
    from vaara.audit.signer import MLDSA65Signer

    signer, _ = MLDSA65Signer.generate()
    result = export_signed(
        _populated_trail(), tmp_path / "trail-pq.zip", signer=signer,
    )
    assert result.manifest["signature_algorithm"] == "ML-DSA-65"
    verify = verify_signed(result.path)
    assert verify.ok, verify.errors


def test_mldsa_zip_rejects_tampered_trail(tmp_path: Path):
    import zipfile
    _require_pq()
    from vaara.audit.signer import MLDSA65Signer

    signer, _ = MLDSA65Signer.generate()
    zpath = tmp_path / "trail-pq.zip"
    export_signed(_populated_trail(), zpath, signer=signer)

    # Tamper the trail inside the zip.
    with zipfile.ZipFile(zpath, "r") as zin:
        trail_bytes = zin.read("trail.jsonl")
        manifest_bytes = zin.read("manifest.json")
        sig_bytes = zin.read("trail.sig")
        pub_bytes = zin.read("signer_pubkey.bin")
    tampered = trail_bytes.replace(b'"allow"', b'"deny"')
    bad_zip = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bad_zip, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("trail.jsonl", tampered)
        zout.writestr("manifest.json", manifest_bytes)
        zout.writestr("trail.sig", sig_bytes)
        zout.writestr("signer_pubkey.bin", pub_bytes)

    result = verify_signed(bad_zip)
    assert not result.ok
