"""CycloneDX-CBOM crypto-posture derivation and verification (v0).

Internal module. Public surface re-exports ``crypto_posture_for`` and
``verify_crypto_posture`` from ``vaara.attestation.receipt``.

An execution receipt is a durable Article 12 record kept for years, so the
crypto that protects it is itself audit-relevant: a reader years from now needs
to know, from the signed bytes alone, whether the record is quantum-resistant
and by what algorithm. This module records that as a small CycloneDX-CBOM
(ECMA-424) crypto-posture block committed *inside* the receipt preimage, next
to ``receiptAsserted.sigSuite``.

Two properties fall out of putting the block inside the preimage:

- Tamper-evidence: the classical signature already covers the block, so editing
  the posture breaks the signature before this module is consulted.
- Downgrade-evidence: the posture is recomputed from the receipt's own ``alg``
  and ``sigSuite``. A posture that commits to an ML-DSA leg while ``pqSignature``
  has been stripped is a signed claim of quantum resistance the record cannot
  back, and ``verify_crypto_posture`` reports it. This mirrors the ``sigSuite``
  downgrade commitment, in the CycloneDX vocabulary an auditor's CBOM tooling
  already speaks.

This is not a scanner. It records the posture of the one algorithm set that
signed this one record; it does not inventory a host's crypto. The verdict is
pure recomputation over the receipt bytes with no keying material, so a
Vaara-free checker reproduces it offline. See
``docs/design/cbom-crypto-posture-spec.md``.
"""

from __future__ import annotations

from typing import Optional

from vaara.attestation._receipt_pq import _HYBRID_SUITES, _PQ_ALG_MLDSA65
from vaara.attestation._receipt_types import (
    CryptoAlgorithm,
    CryptoPosture,
    ExecutionReceipt,
)
from vaara.attestation._attest_types import AttestationError

# alg -> (CycloneDX primitive, NIST post-quantum security category).
# Classical suites carry no quantum resistance (level 0); ML-DSA-65 is FIPS 204
# Category 3. HS256 is a keyed MAC so its primitive is "mac"; the rest are
# asymmetric signatures. A closed table: an unknown alg fails closed rather than
# defaulting to a reassuring 0, which would understate risk by omission.
_NIST_QUANTUM_LEVEL: dict[str, tuple[str, int]] = {
    "HS256": ("mac", 0),
    "ES256": ("signature", 0),
    "RS256": ("signature", 0),
    "Ed25519": ("signature", 0),
    _PQ_ALG_MLDSA65: ("signature", 3),
}

CBOM_OK = "crypto_posture_ok"
CBOM_ABSENT = "crypto_posture_absent"
CBOM_MISMATCH = "crypto_posture_mismatch"
CBOM_DOWNGRADE = "crypto_posture_downgrade"


def _algorithm_entry(alg: str) -> CryptoAlgorithm:
    entry = _NIST_QUANTUM_LEVEL.get(alg)
    if entry is None:
        raise AttestationError(
            f"no CBOM crypto-posture mapping for algorithm {alg!r}"
        )
    primitive, level = entry
    return CryptoAlgorithm(
        algorithm=alg, primitive=primitive, nist_quantum_security_level=level
    )


def crypto_posture_for(
    *, alg: str, sig_suite: Optional[str] = None
) -> CryptoPosture:
    """Derive the CBOM crypto posture for a receipt's signing algorithms.

    ``alg`` is the receipt's classical signing algorithm. ``sig_suite``, when
    set, names an allowlisted hybrid suite (the same closed set the PQ signing
    path uses) whose ML-DSA leg is added as a second algorithm. The effective
    ``nistQuantumSecurityLevel`` is the max over the algorithms, so a hybrid
    reaches its ML-DSA leg's category and a classical-only receipt reports 0.

    Fails closed on an unknown algorithm, an unknown suite, or a suite whose
    classical member disagrees with ``alg`` (which would let the posture
    describe a different key than the one that signed).
    """
    entries = [_algorithm_entry(alg)]
    if sig_suite is not None:
        members = _HYBRID_SUITES.get(sig_suite)
        if members is None:
            raise AttestationError(f"unknown hybrid suite {sig_suite!r}")
        classical_alg, pq_alg = members
        if classical_alg != alg:
            raise AttestationError(
                f"sigSuite {sig_suite!r} names classical member "
                f"{classical_alg!r} but alg is {alg!r}"
            )
        entries.append(_algorithm_entry(pq_alg))
    effective = max(e.nist_quantum_security_level for e in entries)
    return CryptoPosture(
        asset_type="algorithm",
        nist_quantum_security_level=effective,
        algorithms=tuple(entries),
    )


def verify_crypto_posture(receipt: ExecutionReceipt) -> str:
    """Check a receipt's committed crypto posture against what it actually uses.

    Returns one of:

    - ``crypto_posture_absent`` — no posture committed (a classical, pre-CBOM
      record). Not an error: the block is optional and its absence claims
      nothing.
    - ``crypto_posture_ok`` — the committed posture equals the posture
      recomputed from the receipt's own ``alg`` + ``sigSuite``, and any claimed
      ML-DSA leg is backed by a present ``pqSignature``.
    - ``crypto_posture_mismatch`` — the committed posture does not match the
      recomputed one (a forged or drifted quantum-resistance claim, e.g. an
      ML-DSA leg asserted with no ``sigSuite`` to commit it). The posture is
      inside the signed preimage, so on a signature-verified record this only
      fires on an internally inconsistent claim, never on plain tampering
      (which breaks the signature first).
    - ``crypto_posture_downgrade`` — the posture commits to an ML-DSA leg
      (level > 0) but ``pqSignature`` is absent: a signed claim of quantum
      resistance the record cannot back. Mirrors the ``sigSuite`` downgrade
      check in CBOM terms.

    Pure recomputation with no keying material, so an independent checker
    reproduces this verdict from the receipt bytes alone.
    """
    posture = receipt.receipt_asserted.crypto_posture
    if posture is None:
        return CBOM_ABSENT
    try:
        expected = crypto_posture_for(
            alg=receipt.receipt_asserted.alg,
            sig_suite=receipt.receipt_asserted.sig_suite,
        )
    except AttestationError:
        # The record commits a posture the alg/suite pair cannot legitimately
        # produce (unknown alg, or a suite/alg disagreement). Treat as a bad
        # claim rather than propagate.
        return CBOM_MISMATCH
    if posture != expected:
        return CBOM_MISMATCH
    has_pq_leg = any(
        e.nist_quantum_security_level > 0 for e in posture.algorithms
    )
    if has_pq_leg and receipt.pq_signature is None:
        return CBOM_DOWNGRADE
    return CBOM_OK
