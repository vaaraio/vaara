"""CycloneDX-CBOM crypto-posture derivation and verification (v0).

Covers ``crypto_posture_for`` derivation (classical / HS256 keyed-MAC / hybrid),
the effective NIST post-quantum level, the closed-schema serialization
round-trips and rejections, the ``verify_crypto_posture`` verdicts (ok / absent
/ mismatch / downgrade), the preimage binding (tampering the block breaks the
signature), and an independent recompute of the CBOM block from the receipt
wire bytes using only the published algorithm-to-level table.

See ``docs/design/cbom-crypto-posture-spec.md``.
"""

from __future__ import annotations

import dataclasses

import pytest

from vaara.attestation._receipt_cbom import (
    CBOM_ABSENT,
    CBOM_DOWNGRADE,
    CBOM_MISMATCH,
    CBOM_OK,
    crypto_posture_for,
    verify_crypto_posture,
)
from vaara.attestation._receipt_types import (
    BackLink,
    CryptoAlgorithm,
    ExecutionReceipt,
    OutcomeDerived,
    PqSignature,
    ReceiptAsserted,
    crypto_posture_from_dict,
    crypto_posture_to_dict,
    receipt_asserted_from_dict,
    receipt_asserted_to_dict,
)
from vaara.attestation._sep2787_types import AttestationError


def _receipt(*, alg="ES256", sig_suite=None, crypto_posture=None, pq=False):
    """A structurally valid receipt for verdict tests. Signature is a stub;
    ``verify_crypto_posture`` is pure recomputation and never touches it."""
    ra = ReceiptAsserted(
        iss="did:web:issuer.example",
        sub="tenant/upstream",
        iat="2026-07-02T00:00:00Z",
        nonce="n" * 16,
        secret_version="k1",
        alg=alg,
        sig_suite=sig_suite,
        crypto_posture=crypto_posture,
    )
    back_link = BackLink(
        attestation_digest="sha256:" + "0" * 64, attestation_nonce="a" * 16
    )
    outcome = OutcomeDerived(status="executed", completed_at="2026-07-02T00:00:01Z")
    pq_sig = (
        PqSignature(alg="ML-DSA-65", keyid="did:web:issuer.example#pq", sig="ab")
        if pq
        else None
    )
    return ExecutionReceipt(
        version=1,
        alg=alg,
        back_link=back_link,
        receipt_asserted=ra,
        outcome_derived=outcome,
        signature="00",
        pq_signature=pq_sig,
    )


# --- derivation -----------------------------------------------------------


def test_derive_classical_es256():
    p = crypto_posture_for(alg="ES256")
    assert p.asset_type == "algorithm"
    assert p.nist_quantum_security_level == 0
    assert p.algorithms == (
        CryptoAlgorithm(
            algorithm="ES256", primitive="signature", nist_quantum_security_level=0
        ),
    )


def test_derive_hs256_is_mac_primitive():
    p = crypto_posture_for(alg="HS256")
    assert p.algorithms[0].primitive == "mac"
    assert p.nist_quantum_security_level == 0


def test_derive_hybrid_reaches_mldsa_level():
    p = crypto_posture_for(alg="ES256", sig_suite="ES256+ML-DSA-65")
    assert [a.algorithm for a in p.algorithms] == ["ES256", "ML-DSA-65"]
    assert p.algorithms[1].nist_quantum_security_level == 3
    # Effective level is the max: the ML-DSA leg carries quantum resistance.
    assert p.nist_quantum_security_level == 3


def test_derive_unknown_alg_fails_closed():
    with pytest.raises(AttestationError):
        crypto_posture_for(alg="XYZ999")


def test_derive_unknown_suite_fails_closed():
    with pytest.raises(AttestationError):
        crypto_posture_for(alg="ES256", sig_suite="ES256+NOPE")


def test_derive_suite_alg_disagreement_fails_closed():
    # sigSuite's classical member (ES256) must equal the receipt alg (RS256).
    with pytest.raises(AttestationError):
        crypto_posture_for(alg="RS256", sig_suite="ES256+ML-DSA-65")


# --- serialization --------------------------------------------------------


@pytest.mark.parametrize("sig_suite", [None, "ES256+ML-DSA-65"])
def test_posture_roundtrips(sig_suite):
    p = crypto_posture_for(alg="ES256", sig_suite=sig_suite)
    assert crypto_posture_from_dict(crypto_posture_to_dict(p)) == p


def test_from_dict_rejects_unknown_key():
    d = crypto_posture_to_dict(crypto_posture_for(alg="ES256"))
    d["extra"] = "nope"
    with pytest.raises(AttestationError):
        crypto_posture_from_dict(d)


def test_from_dict_rejects_bad_asset_type():
    d = crypto_posture_to_dict(crypto_posture_for(alg="ES256"))
    d["assetType"] = "certificate"
    with pytest.raises(AttestationError):
        crypto_posture_from_dict(d)


def test_from_dict_rejects_bool_level():
    # bool is an int subclass; True must not slip through as level 1.
    d = crypto_posture_to_dict(crypto_posture_for(alg="ES256"))
    d["nistQuantumSecurityLevel"] = True
    with pytest.raises(AttestationError):
        crypto_posture_from_dict(d)


@pytest.mark.parametrize("level", [-1, 6, 42])
def test_from_dict_rejects_out_of_range_level(level):
    d = crypto_posture_to_dict(crypto_posture_for(alg="ES256"))
    d["nistQuantumSecurityLevel"] = level
    with pytest.raises(AttestationError):
        crypto_posture_from_dict(d)


def test_from_dict_rejects_empty_algorithms():
    d = crypto_posture_to_dict(crypto_posture_for(alg="ES256"))
    d["algorithms"] = []
    with pytest.raises(AttestationError):
        crypto_posture_from_dict(d)


def test_receipt_asserted_carries_posture_through_serialization():
    posture = crypto_posture_for(alg="ES256", sig_suite="ES256+ML-DSA-65")
    ra = _receipt(
        alg="ES256", sig_suite="ES256+ML-DSA-65", crypto_posture=posture
    ).receipt_asserted
    wire = receipt_asserted_to_dict(ra)
    assert wire["cryptoPosture"]["nistQuantumSecurityLevel"] == 3
    assert receipt_asserted_from_dict(wire) == ra


def test_receipt_asserted_without_posture_omits_key():
    # Absent posture keeps the envelope byte-for-byte what it was before.
    ra = _receipt(alg="ES256").receipt_asserted
    assert "cryptoPosture" not in receipt_asserted_to_dict(ra)


# --- verdicts -------------------------------------------------------------


def test_verify_absent_when_no_posture():
    assert verify_crypto_posture(_receipt(alg="ES256")) == CBOM_ABSENT


def test_verify_ok_classical():
    posture = crypto_posture_for(alg="ES256")
    assert (
        verify_crypto_posture(_receipt(alg="ES256", crypto_posture=posture))
        == CBOM_OK
    )


def test_verify_ok_hybrid_with_pq_signature():
    posture = crypto_posture_for(alg="ES256", sig_suite="ES256+ML-DSA-65")
    r = _receipt(
        alg="ES256", sig_suite="ES256+ML-DSA-65", crypto_posture=posture, pq=True
    )
    assert verify_crypto_posture(r) == CBOM_OK


def test_verify_downgrade_when_pq_signature_stripped():
    posture = crypto_posture_for(alg="ES256", sig_suite="ES256+ML-DSA-65")
    r = _receipt(
        alg="ES256", sig_suite="ES256+ML-DSA-65", crypto_posture=posture, pq=False
    )
    assert verify_crypto_posture(r) == CBOM_DOWNGRADE


def test_verify_mismatch_when_quantum_claim_inflated():
    # Posture asserts an ML-DSA leg, but no sigSuite commits it: the recomputed
    # (classical-only) posture differs, so the inflated claim is caught.
    inflated = crypto_posture_for(alg="ES256", sig_suite="ES256+ML-DSA-65")
    r = _receipt(alg="ES256", sig_suite=None, crypto_posture=inflated, pq=True)
    assert verify_crypto_posture(r) == CBOM_MISMATCH


def test_verify_mismatch_when_level_tampered():
    posture = crypto_posture_for(alg="ES256")
    tampered = dataclasses.replace(posture, nist_quantum_security_level=3)
    r = _receipt(alg="ES256", crypto_posture=tampered)
    assert verify_crypto_posture(r) == CBOM_MISMATCH


# --- independent recompute (Vaara-free reconstruction) --------------------

# The published algorithm-to-level table, restated here with NO Vaara import so
# the recompute is genuinely independent: a third party reconstructs the CBOM
# block from the receipt's alg + sigSuite and this public table alone.
_PUBLISHED_TABLE = {
    "HS256": ("mac", 0),
    "ES256": ("signature", 0),
    "RS256": ("signature", 0),
    "Ed25519": ("signature", 0),
    "ML-DSA-65": ("signature", 3),
}


def _independent_posture(receipt_asserted_wire):
    """Rebuild the expected cryptoPosture dict from the wire, Vaara-free."""
    alg = receipt_asserted_wire["alg"]
    algs = [alg]
    suite = receipt_asserted_wire.get("sigSuite")
    if suite is not None:
        members = suite.split("+")
        assert members[0] == alg, "suite classical member must equal alg"
        algs = members
    entries = []
    for a in algs:
        primitive, level = _PUBLISHED_TABLE[a]
        entries.append(
            {
                "algorithm": a,
                "primitive": primitive,
                "nistQuantumSecurityLevel": level,
            }
        )
    return {
        "assetType": "algorithm",
        "nistQuantumSecurityLevel": max(
            e["nistQuantumSecurityLevel"] for e in entries
        ),
        "algorithms": entries,
    }


@pytest.mark.parametrize("sig_suite", [None, "ES256+ML-DSA-65"])
def test_independent_recompute_matches_committed_posture(sig_suite):
    posture = crypto_posture_for(alg="ES256", sig_suite=sig_suite)
    ra_wire = receipt_asserted_to_dict(
        _receipt(
            alg="ES256", sig_suite=sig_suite, crypto_posture=posture
        ).receipt_asserted
    )
    assert ra_wire["cryptoPosture"] == _independent_posture(ra_wire)


# --- preimage binding (needs the JCS/rfc8785 signing path) ----------------


def test_posture_is_inside_preimage_and_tamper_breaks_signature():
    pytest.importorskip("rfc8785")
    from vaara.attestation.receipt import emit_receipt, verify_receipt_signature

    secret = b"a-shared-secret-32-bytes-long!!!"
    posture = crypto_posture_for(alg="HS256")
    template = _receipt()

    receipt = emit_receipt(
        back_link=template.back_link,
        outcome_derived=template.outcome_derived,
        iss="did:web:issuer.example",
        sub="tenant/upstream",
        secret_version="k1",
        alg="HS256",
        signing_material=secret,
        crypto_posture=posture,
    )
    assert receipt.receipt_asserted.crypto_posture == posture
    assert "cryptoPosture" in receipt.to_dict()["receiptAsserted"]
    assert verify_receipt_signature(receipt, verifying_material=secret) is True

    # Editing the committed posture must invalidate the signature: proof the
    # block rides inside the signed preimage, not outside it.
    tampered_posture = dataclasses.replace(
        posture, nist_quantum_security_level=3
    )
    tampered_ra = dataclasses.replace(
        receipt.receipt_asserted, crypto_posture=tampered_posture
    )
    tampered = dataclasses.replace(receipt, receipt_asserted=tampered_ra)
    assert verify_receipt_signature(tampered, verifying_material=secret) is False
