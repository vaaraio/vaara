"""Tests for k-of-n threshold-signed trail export / verify.

See ``docs/design/threshold-signing-spec.md``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")  # skip entire module when extra missing

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.audit.export import _pubkey_member_bytes, export_signed_threshold
from vaara.audit.signer import Ed25519Signer
from vaara.audit.trail import AuditTrail
from vaara.audit.verify import verify_signed
from vaara.taxonomy.actions import ActionRequest, create_default_registry

_REGISTRY = create_default_registry()
_TX_TRANSFER = _REGISTRY.get("tx.transfer")


def _make_trail(n: int = 3) -> AuditTrail:
    trail = AuditTrail()
    for i in range(n):
        req = ActionRequest(
            agent_id=f"agent-{i}",
            tool_name="send_funds",
            action_type=_TX_TRANSFER,
            parameters={"to": f"0xabc{i}", "amount": 10 * i},
        )
        trail.record_action_requested(req)
    return trail


def _signers(n: int) -> list[Ed25519Signer]:
    return [Ed25519Signer(Ed25519PrivateKey.generate()) for _ in range(n)]


def _drop_sigs(src: Path, dst: Path, keep: int) -> None:
    """Copy the zip keeping only the first ``keep`` sigs/ entries."""
    with zipfile.ZipFile(src) as zin:
        sigs = sorted(n for n in zin.namelist() if n.startswith("sigs/"))
        keepset = set(sigs[:keep])
        with zipfile.ZipFile(dst, "w") as zout:
            for name in zin.namelist():
                if name.startswith("sigs/") and name not in keepset:
                    continue
                zout.writestr(name, zin.read(name))


# ── Happy path ─────────────────────────────────────────────────────────────

def test_3_of_5_all_sign_verifies(tmp_path):
    out = tmp_path / "t.zip"
    res = export_signed_threshold(
        _make_trail(), out, signers=_signers(5), threshold_k=3
    )
    assert res.manifest["signature_algorithm"] == "threshold-Ed25519"
    assert res.manifest["threshold_k"] == 3
    assert res.manifest["signers_n"] == 5
    assert res.manifest["member_algorithm"] == "Ed25519"
    assert len(res.manifest["signer_fingerprints"]) == 5
    with zipfile.ZipFile(out) as zf:
        assert sum(n.startswith("sigs/") for n in zf.namelist()) == 5
        assert sum(n.startswith("pubkeys/") for n in zf.namelist()) == 5
    assert verify_signed(out).ok


def test_exactly_k_sigs_present_verifies(tmp_path):
    out = tmp_path / "t.zip"
    export_signed_threshold(_make_trail(), out, signers=_signers(5), threshold_k=3)
    trimmed = tmp_path / "k.zip"
    _drop_sigs(out, trimmed, keep=3)
    assert verify_signed(trimmed).ok


def test_below_k_sigs_fails(tmp_path):
    out = tmp_path / "t.zip"
    export_signed_threshold(_make_trail(), out, signers=_signers(5), threshold_k=3)
    trimmed = tmp_path / "k.zip"
    _drop_sigs(out, trimmed, keep=2)
    res = verify_signed(trimmed)
    assert not res.ok
    assert any("threshold not met" in e for e in res.errors)


# ── Attacks ────────────────────────────────────────────────────────────────

def test_downgrade_k_to_1_fails(tmp_path):
    """Lowering threshold_k in the manifest breaks every member signature."""
    out = tmp_path / "t.zip"
    export_signed_threshold(_make_trail(), out, signers=_signers(5), threshold_k=3)
    tampered = tmp_path / "d.zip"
    with zipfile.ZipFile(out) as zin:
        man = json.loads(zin.read("manifest.json"))
        man["threshold_k"] = 1
        with zipfile.ZipFile(tampered, "w") as zout:
            for name in zin.namelist():
                if name == "manifest.json":
                    zout.writestr(
                        name, json.dumps(man, sort_keys=True, indent=2).encode()
                    )
                else:
                    zout.writestr(name, zin.read(name))
    assert not verify_signed(tampered).ok


def test_unauthorized_extra_signature_ignored(tmp_path):
    """An extra valid sig from a non-member key does not count toward k."""
    out = tmp_path / "t.zip"
    export_signed_threshold(_make_trail(), out, signers=_signers(5), threshold_k=3)
    trimmed = tmp_path / "k.zip"
    _drop_sigs(out, trimmed, keep=2)  # below quorum

    with zipfile.ZipFile(trimmed) as zf:
        digest = hashlib.sha256(
            zf.read("trail.jsonl") + zf.read("manifest.json")
        ).digest()
    rogue = Ed25519Signer(Ed25519PrivateKey.generate())
    rogue_fp = rogue.public_key_fingerprint()[:32]
    injected = tmp_path / "inj.zip"
    with zipfile.ZipFile(trimmed) as zin, zipfile.ZipFile(injected, "w") as zout:
        for name in zin.namelist():
            zout.writestr(name, zin.read(name))
        zout.writestr(f"sigs/{rogue_fp}.sig", rogue.sign(digest))
    res = verify_signed(injected)
    assert not res.ok  # 2 authorized + 1 rogue = still below k=3
    assert any("threshold not met" in e for e in res.errors)


def test_substituted_pubkey_fails(tmp_path):
    """Swapping a member pubkey file (different fingerprint) is rejected."""
    out = tmp_path / "t.zip"
    export_signed_threshold(_make_trail(), out, signers=_signers(5), threshold_k=3)
    swapped = tmp_path / "s.zip"
    rogue_raw = Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    _, rogue_pem = _pubkey_member_bytes(rogue_raw, "Ed25519")
    with zipfile.ZipFile(out) as zin:
        target = next(n for n in zin.namelist() if n.startswith("pubkeys/"))
        with zipfile.ZipFile(swapped, "w") as zout:
            for name in zin.namelist():
                body = rogue_pem if name == target else zin.read(name)
                zout.writestr(name, body)
    res = verify_signed(swapped)
    assert not res.ok
    assert any("fingerprint" in e or "threshold not met" in e for e in res.errors)


def test_tampered_record_fails_chain(tmp_path):
    out = tmp_path / "t.zip"
    export_signed_threshold(_make_trail(), out, signers=_signers(5), threshold_k=3)
    tampered = tmp_path / "c.zip"
    with zipfile.ZipFile(out) as zin:
        lines = zin.read("trail.jsonl").splitlines()
        rec = json.loads(lines[0])
        rec["data"] = {"tampered": True}
        lines[0] = json.dumps(rec).encode()
        with zipfile.ZipFile(tampered, "w") as zout:
            for name in zin.namelist():
                body = b"\n".join(lines) if name == "trail.jsonl" else zin.read(name)
                zout.writestr(name, body)
    assert not verify_signed(tampered).ok


# ── Authorized-set semantics ────────────────────────────────────────────────

def test_k_of_n_with_subset_present(tmp_path):
    """Pass the full authorized set but only k signers available now."""
    all_signers = _signers(5)
    authorized = [s.public_key_bytes() for s in all_signers]
    out = tmp_path / "t.zip"
    res = export_signed_threshold(
        _make_trail(),
        out,
        signers=all_signers[:3],
        threshold_k=3,
        authorized_pubkeys=authorized,
    )
    assert res.manifest["signers_n"] == 5
    with zipfile.ZipFile(out) as zf:
        assert sum(n.startswith("sigs/") for n in zf.namelist()) == 3
        assert sum(n.startswith("pubkeys/") for n in zf.namelist()) == 5
    assert verify_signed(out).ok


# ── Validation ──────────────────────────────────────────────────────────────

def test_k_exceeds_n_raises(tmp_path):
    with pytest.raises(ValueError, match="exceeds n"):
        export_signed_threshold(
            _make_trail(), tmp_path / "t.zip", signers=_signers(3), threshold_k=5
        )


def test_too_few_signers_for_k_raises(tmp_path):
    all_signers = _signers(5)
    authorized = [s.public_key_bytes() for s in all_signers]
    with pytest.raises(ValueError, match="need at least"):
        export_signed_threshold(
            _make_trail(),
            tmp_path / "t.zip",
            signers=all_signers[:2],
            threshold_k=3,
            authorized_pubkeys=authorized,
        )


def test_signer_not_in_authorized_set_raises(tmp_path):
    authorized = [s.public_key_bytes() for s in _signers(3)]  # unrelated set
    with pytest.raises(ValueError, match="not in the authorized set"):
        export_signed_threshold(
            _make_trail(),
            tmp_path / "t.zip",
            signers=_signers(3),
            threshold_k=2,
            authorized_pubkeys=authorized,
        )


# ── Standalone verifier parity ──────────────────────────────────────────────

def _standalone(zip_path: Path) -> int:
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_vaara_trail.py"
    return subprocess.run(
        [sys.executable, str(script), str(zip_path)],
        capture_output=True,
    ).returncode


def test_standalone_verifier_parity(tmp_path):
    out = tmp_path / "t.zip"
    export_signed_threshold(_make_trail(), out, signers=_signers(5), threshold_k=3)
    assert _standalone(out) == 0

    below = tmp_path / "b.zip"
    _drop_sigs(out, below, keep=2)
    assert _standalone(below) == 1
