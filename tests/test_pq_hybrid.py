"""Post-quantum hybrid signing for execution receipts (Track E1).

Covers the parallel ML-DSA-65 signature over the classical receipt preimage,
the ``sigSuite`` downgrade commitment, the ``pq_verdict`` quantum-resistance
tiers, and the ``pq_hybrid_v0`` conformance vectors (Vaara and a standalone
Vaara-free checker both reproduce the verdicts).

See ``docs/design/pq-hybrid-signing-spec.md``.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography", "dilithium_py"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "pq extra not installed (pip install 'vaara[attestation,pq]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from dilithium_py.ml_dsa import ML_DSA_65  # noqa: E402

from vaara.attestation._receipt_pq import _b64u_encode  # noqa: E402
from vaara.attestation.receipt import (  # noqa: E402
    BackLink,
    OutcomeDerived,
    attach_pq_signature,
    emit_receipt,
    parse_receipt,
    pq_verdict,
)

VECTORS = Path(__file__).resolve().parent / "vectors" / "pq_hybrid_v0"
ISS = "did:web:issuer.example"
ES_KEYID = ISS + "#es256-2026"
PQ_KEYID = ISS + "#mldsa-2026"
IAT = "2026-06-11T12:00:00Z"


def _b64u_int(n: int, size: int) -> str:
    return base64.urlsafe_b64encode(n.to_bytes(size, "big")).rstrip(b"=").decode()


@pytest.fixture(scope="module")
def world():
    es_key = ec.generate_private_key(ec.SECP256R1())
    pq_pub, pq_sec = ML_DSA_65.keygen()
    nums = es_key.public_key().public_numbers()
    doc = {"id": ISS, "verificationMethod": [
        {"id": ES_KEYID, "type": "JsonWebKey2020", "publicKeyJwk": {
            "kty": "EC", "crv": "P-256",
            "x": _b64u_int(nums.x, 32), "y": _b64u_int(nums.y, 32)}},
        {"id": PQ_KEYID, "type": "JsonWebKey2020", "publicKeyJwk": {
            "kty": "AKP", "alg": "ML-DSA-65", "pub": _b64u_encode(pq_pub)}},
    ]}
    return es_key, pq_sec, doc


def _emit(es_key, suite=None):
    return emit_receipt(
        back_link=BackLink(attestation_digest="sha256:" + "a" * 64,
                           attestation_nonce="att-1"),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=ISS, sub="tool:transfer", secret_version="v1",
        alg="ES256", signing_material=es_key, iat=IAT, nonce="n1",
        sig_suite=suite,
    )


def test_hybrid_verified(world):
    es_key, pq_sec, doc = world
    r = attach_pq_signature(_emit(es_key, "ES256+ML-DSA-65"),
                            pq_secret_key=pq_sec, pq_keyid=PQ_KEYID)
    v = pq_verdict(r, doc)
    assert v.tier == "hybrid-verified"
    assert v.quantum_resistant and v.downgrade_resistant
    assert v.classical_bound and v.pq_bound


def test_classical_only_preimage_is_unchanged(world):
    """A receipt with no committed suite signs the exact pre-feature preimage."""
    es_key, _pq_sec, _doc = world
    r = _emit(es_key)
    assert "sigSuite" not in r.to_dict()["receiptAsserted"]
    assert "pqSignature" not in r.to_dict()
    assert pq_verdict(r, _doc).tier == "classical-only"


def test_pqc_present_is_strippable(world):
    es_key, pq_sec, doc = world
    r = attach_pq_signature(_emit(es_key), pq_secret_key=pq_sec, pq_keyid=PQ_KEYID)
    v = pq_verdict(r, doc)
    assert v.tier == "pqc-present"
    assert v.quantum_resistant and not v.downgrade_resistant


def test_stripping_committed_pq_fails_closed(world):
    es_key, pq_sec, doc = world
    r = attach_pq_signature(_emit(es_key, "ES256+ML-DSA-65"),
                            pq_secret_key=pq_sec, pq_keyid=PQ_KEYID)
    d = r.to_dict()
    d.pop("pqSignature")
    assert pq_verdict(parse_receipt(d), doc).tier == "hybrid-downgraded"


def test_tampered_pq_signature_fails_closed(world):
    es_key, pq_sec, doc = world
    r = attach_pq_signature(_emit(es_key, "ES256+ML-DSA-65"),
                            pq_secret_key=pq_sec, pq_keyid=PQ_KEYID)
    d = r.to_dict()
    raw = bytearray.fromhex(d["pqSignature"]["sig"])
    raw[0] ^= 0x01
    d["pqSignature"]["sig"] = bytes(raw).hex()
    assert pq_verdict(parse_receipt(d), doc).tier == "hybrid-downgraded"


def test_unknown_suite_fails_closed(world):
    es_key, _pq_sec, doc = world
    assert pq_verdict(_emit(es_key, "ES256+BOGUS"), doc).tier == "hybrid-downgraded"


def test_wire_roundtrip_stable(world):
    es_key, pq_sec, doc = world
    r = attach_pq_signature(_emit(es_key, "ES256+ML-DSA-65"),
                            pq_secret_key=pq_sec, pq_keyid=PQ_KEYID)
    again = parse_receipt(r.to_dict())
    assert again.to_dict() == r.to_dict()
    assert pq_verdict(again, doc).tier == "hybrid-verified"


def test_injected_unmodeled_field_is_rejected(world):
    """An extra key under a signed block must fail closed, not be dropped.

    Dropping it would make the model-derived preimage exclude bytes a byte-exact
    verifier (and the independent checker) include, so Vaara could call a record
    hybrid-verified while carrying content neither signature covered.
    """
    from vaara.attestation._sep2787_types import AttestationError

    es_key, pq_sec, _doc = world
    r = attach_pq_signature(_emit(es_key, "ES256+ML-DSA-65"),
                            pq_secret_key=pq_sec, pq_keyid=PQ_KEYID)
    for mutate in (
        lambda d: d["receiptAsserted"].__setitem__("evil", "x"),
        lambda d: d["backLink"].__setitem__("evil", "x"),
        lambda d: d["outcomeDerived"].__setitem__("evil", "x"),
        lambda d: d.__setitem__("evil", "x"),
    ):
        d = r.to_dict()
        mutate(d)
        with pytest.raises(AttestationError):
            parse_receipt(d)


def test_vectors_match_vaara():
    cases = json.loads((VECTORS / "cases.json").read_text())["cases"]
    expected = json.loads((VECTORS / "expected.json").read_text())
    compare = ("tier", "classical_bound", "pq_bound", "suite", "pq_keyid",
               "quantum_resistant", "downgrade_resistant")
    for case in cases:
        v = pq_verdict(parse_receipt(case["receipt"]), case["didDocument"])
        got = {k: v.to_dict()[k] for k in compare}
        want = {k: expected[case["name"]][k] for k in compare}
        assert got == want, case["name"]


def test_independent_checker_reproduces_vectors():
    result = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
