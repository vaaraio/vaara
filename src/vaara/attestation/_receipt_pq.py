# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-quantum hybrid signing for execution receipts (v0).

Internal module. Public surface is re-exported from
``vaara.attestation.receipt``.

A receipt's classical signature (ES256 / RS256) is forgeable by a future
quantum adversary, and a receipt is a durable Article 12 record kept for
years, so the live threat is "trust now, forge later". This module adds a
parallel ML-DSA-65 (FIPS 204) signature over the *same* JCS preimage the
classical signature covers. Both must verify; the classical one keeps the
record interoperable today, the ML-DSA one carries the quantum-resistant
guarantee.

``dilithium-py`` (pure Python, ``pip install 'vaara[pq]'``) provides
ML-DSA. It is imported lazily so the base install and every classical path
stay standard-library only. The independent checker uses the same one named
dependency and nothing else of Vaara's.

The downgrade attack (strip ``pqSignature`` and the protection vanishes) is
closed by ``receiptAsserted.sigSuite``, which sits inside the signed preimage
and so is covered by both signatures. See
``docs/design/pq-hybrid-signing-spec.md``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from typing import Any, Optional

from vaara.attestation._receipt_emit import _signing_payload
from vaara.attestation._receipt_identity import (
    _verification_methods,
    verify_receipt_identity,
)
from vaara.attestation._receipt_types import ExecutionReceipt, PqSignature

_PQ_ALG_MLDSA65 = "ML-DSA-65"
_MLDSA65_PUBKEY_BYTES = 1952  # FIPS 204 ML-DSA-65 public-key length

# Allowlisted hybrid suites: name -> (required classical alg, required PQC alg).
# A closed set, mirroring the #2867 fallback-projection allowlist framing.
_HYBRID_SUITES: dict[str, tuple[str, str]] = {
    "ES256+ML-DSA-65": ("ES256", _PQ_ALG_MLDSA65),
    "RS256+ML-DSA-65": ("RS256", _PQ_ALG_MLDSA65),
}


def _b64u_decode(value: str) -> bytes:
    pad = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + pad)


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def receipt_preimage(receipt: ExecutionReceipt) -> bytes:
    """The JCS preimage both signatures cover.

    Identical bytes for the classical and the ML-DSA signature, so the two
    bind exactly the same record. Includes ``receiptAsserted.sigSuite`` when
    set, which is what makes a committed hybrid suite tamper-evident.
    """
    return _signing_payload(
        version=receipt.version,
        alg=receipt.alg,
        back_link=receipt.back_link,
        outcome_derived=receipt.outcome_derived,
        receipt_asserted=receipt.receipt_asserted,
    )


def mldsa65_sign(payload: bytes, secret_key: bytes) -> str:
    """ML-DSA-65 signature over ``payload``, hex-encoded. Needs the pq extra."""
    from dilithium_py.ml_dsa import ML_DSA_65

    return ML_DSA_65.sign(bytes(secret_key), payload).hex()


def mldsa65_verify(payload: bytes, signature_hex: str, public_key: bytes) -> bool:
    """Verify an ML-DSA-65 signature. Returns False on any malformed input.

    Fails closed (returns False) rather than raising on a bad hex string,
    wrong-length key, or a missing pq extra, so a malformed PQC member can
    never crash a verifier into skipping the check.
    """
    if len(public_key) != _MLDSA65_PUBKEY_BYTES:
        return False
    try:
        sig = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    try:
        from dilithium_py.ml_dsa import ML_DSA_65
    except ImportError:
        return False
    try:
        return bool(ML_DSA_65.verify(bytes(public_key), payload, sig))
    except (ValueError, TypeError):
        return False


def mldsa65_public_key_from_method(method: dict[str, Any]) -> Optional[bytes]:
    """Raw ML-DSA-65 public key from a DID verification method, or None.

    Reads an ``AKP`` (Algorithm Key Pair) JWK with ``alg`` ``ML-DSA-65`` and a
    base64url ``pub`` member, the encoding tracked by the JOSE/COSE
    post-quantum drafts. Returns None when the method is not an ML-DSA AKP key
    or the bytes do not decode to the expected length, so the caller treats a
    malformed method as simply not a usable PQC key.
    """
    jwk = method.get("publicKeyJwk")
    if not isinstance(jwk, dict):
        return None
    if jwk.get("kty") != "AKP" or jwk.get("alg") != _PQ_ALG_MLDSA65:
        return None
    pub = jwk.get("pub")
    if not isinstance(pub, str) or not pub:
        return None
    try:
        raw = _b64u_decode(pub)
    except (ValueError, TypeError):
        return None
    if len(raw) != _MLDSA65_PUBKEY_BYTES:
        return None
    return raw


def attach_pq_signature(
    receipt: ExecutionReceipt, *, pq_secret_key: bytes, pq_keyid: str
) -> ExecutionReceipt:
    """Return a copy of ``receipt`` carrying an ML-DSA-65 ``pqSignature``.

    The ML-DSA signature is computed over the receipt's existing preimage, so
    it binds the same bytes the classical signature already did. For a
    downgrade-resistant record the receipt must have been emitted with a
    hybrid ``sig_suite`` so the commitment is inside that preimage; attaching
    to a receipt with no committed suite produces the strippable
    ``pqc-present`` tier instead.
    """
    sig_hex = mldsa65_sign(receipt_preimage(receipt), pq_secret_key)
    return replace(
        receipt,
        pq_signature=PqSignature(alg=_PQ_ALG_MLDSA65, keyid=pq_keyid, sig=sig_hex),
    )


@dataclass(frozen=True)
class PqVerdict:
    """Quantum-resistance tier of a receipt, orthogonal to verifiable/corroborated.

    ``tier`` is one of:

    - ``classical-only`` — no committed hybrid suite and no valid PQC signature.
      Still verifiable today under the classical key, not quantum-resistant.
    - ``pqc-present`` — a valid PQC signature is attached but the issuer did not
      commit ``sigSuite``, so it is strippable (not downgrade-resistant).
    - ``hybrid-verified`` — a committed hybrid suite, the classical signature
      binds to a document key, and the ML-DSA signature binds to an ML-DSA
      document key over the same preimage. Quantum- and downgrade-resistant.
    - ``hybrid-downgraded`` — the fail-closed case: ``sigSuite`` commits to a
      hybrid suite but the ``pqSignature`` is absent, malformed, inconsistent
      with the suite, or does not verify. Not bound for hybrid purposes.

    ``classical_bound`` / ``pq_bound`` are the two member results.
    ``quantum_resistant`` is True when a verifying PQC signature is present
    (``pqc-present`` or ``hybrid-verified``). ``downgrade_resistant`` is True
    only for ``hybrid-verified``.
    """

    tier: str
    classical_bound: bool
    pq_bound: bool
    suite: Optional[str]
    pq_keyid: Optional[str]
    quantum_resistant: bool
    downgrade_resistant: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "classical_bound": self.classical_bound,
            "pq_bound": self.pq_bound,
            "suite": self.suite,
            "pq_keyid": self.pq_keyid,
            "quantum_resistant": self.quantum_resistant,
            "downgrade_resistant": self.downgrade_resistant,
            "reason": self.reason,
        }


def _pq_bound_against_document(
    receipt: ExecutionReceipt, did_document: dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """Whether the receipt's ML-DSA signature binds to a document key.

    Returns (bound, keyid). The verification method is selected by the
    ``pqSignature.keyid`` so a record cannot point its PQC signature at one key
    and have it counted against another. Only ML-DSA-65 is supported in v0.
    """
    pq = receipt.pq_signature
    if pq is None:
        return False, None
    if pq.alg != _PQ_ALG_MLDSA65:
        return False, pq.keyid
    payload = receipt_preimage(receipt)
    for method in _verification_methods(did_document):
        if method.get("id") != pq.keyid:
            continue
        raw = mldsa65_public_key_from_method(method)
        if raw is None:
            continue
        if mldsa65_verify(payload, pq.sig, raw):
            return True, pq.keyid
    return False, pq.keyid


def pq_verdict(
    receipt: ExecutionReceipt,
    did_document: dict[str, Any],
    *,
    expected_keyid: Optional[str] = None,
) -> PqVerdict:
    """Classify a receipt's quantum-resistance tier against a DID document.

    ``expected_keyid`` pins the *classical* verification method (passed through
    to :func:`verify_receipt_identity`); the ML-DSA method is always selected by
    the ``pqSignature.keyid``. Reported alongside ``verify_receipt_retained``:
    one answers "verifiable under a key valid at issuance", this one answers
    "what does the record resist".
    """
    classical = verify_receipt_identity(
        receipt, did_document, expected_keyid=expected_keyid
    )
    classical_bound = classical.bound
    suite = receipt.receipt_asserted.sig_suite
    pq_bound, pq_keyid = _pq_bound_against_document(receipt, did_document)

    if suite is not None:
        members = _HYBRID_SUITES.get(suite)
        if members is None:
            return PqVerdict(
                "hybrid-downgraded", classical_bound, pq_bound, suite, pq_keyid,
                False, False,
                f"receiptAsserted.sigSuite {suite!r} is not an allowlisted suite",
            )
        classical_alg, _pq_alg = members
        if receipt.alg != classical_alg:
            return PqVerdict(
                "hybrid-downgraded", classical_bound, pq_bound, suite, pq_keyid,
                False, False,
                f"sigSuite {suite!r} names classical member {classical_alg!r} "
                f"but the receipt alg is {receipt.alg!r}",
            )
        if receipt.pq_signature is None:
            return PqVerdict(
                "hybrid-downgraded", classical_bound, False, suite, None,
                False, False,
                f"sigSuite commits to {suite!r} but no pqSignature is present "
                "(downgrade)",
            )
        if not pq_bound:
            return PqVerdict(
                "hybrid-downgraded", classical_bound, False, suite, pq_keyid,
                False, False,
                "pqSignature does not bind to an ML-DSA key the document lists",
            )
        if not classical_bound:
            return PqVerdict(
                "hybrid-downgraded", False, pq_bound, suite, pq_keyid,
                False, False,
                f"classical signature not bound: {classical.reason}",
            )
        return PqVerdict(
            "hybrid-verified", True, True, suite, pq_keyid, True, True,
            "classical and ML-DSA-65 signatures both bind the same preimage",
        )

    # No committed suite.
    if pq_bound:
        return PqVerdict(
            "pqc-present", classical_bound, True, None, pq_keyid, True, False,
            "a valid ML-DSA-65 signature is attached but sigSuite was not "
            "committed, so it is strippable",
        )
    return PqVerdict(
        "classical-only", classical_bound, False, None, pq_keyid, False, False,
        "no committed hybrid suite and no verifying PQC signature; "
        "not quantum-resistant",
    )


__all__ = [
    "PqVerdict",
    "attach_pq_signature",
    "mldsa65_public_key_from_method",
    "mldsa65_sign",
    "mldsa65_verify",
    "pq_verdict",
    "receipt_preimage",
]
