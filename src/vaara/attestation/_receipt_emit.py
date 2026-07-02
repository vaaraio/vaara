"""Emit and verify-signature for execution-receipt envelopes.

Internal module. Public surface is in ``vaara.attestation.receipt``.

Reuses the SEP-2787 canonicalization (RFC 8785 JCS) and signing stack
(HS256 / ES256 / RS256) unchanged. The only new wire shape is the
envelope layout; the cryptographic primitives are shared so a verifier
that already handles SEP-2787 signatures handles receipts with no new
crypto code.
"""

from __future__ import annotations

from typing import Any, Optional

from vaara.attestation._receipt_types import (
    BackLink,
    CryptoPosture,
    ExecutionReceipt,
    OutcomeDerived,
    ReceiptAsserted,
    back_link_to_dict,
    outcome_to_dict,
    receipt_asserted_to_dict,
)
from vaara.attestation._sep2787_canonical import (
    canonical_json,
    new_nonce,
    now_iso8601,
)
from vaara.attestation._sep2787_signing import (
    sign_es256,
    sign_hs256,
    sign_rs256,
    verify_es256,
    verify_hs256,
    verify_rs256,
)
from vaara.attestation._sep2787_types import (
    VALID_ALGS,
    Algorithm,
    AttestationError,
)


def _signing_payload(
    *,
    version: int,
    alg: Algorithm,
    back_link: BackLink,
    outcome_derived: OutcomeDerived,
    receipt_asserted: ReceiptAsserted,
) -> bytes:
    """JCS-canonical encoding of the receipt blocks, signature excluded."""
    body = {
        "version": version,
        "alg": alg,
        "backLink": back_link_to_dict(back_link),
        "outcomeDerived": outcome_to_dict(outcome_derived),
        "receiptAsserted": receipt_asserted_to_dict(receipt_asserted),
    }
    return canonical_json(body)


def emit_receipt(
    *,
    back_link: BackLink,
    outcome_derived: OutcomeDerived,
    iss: str,
    sub: str,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    version: int = 1,
    sig_suite: Optional[str] = None,
    crypto_posture: Optional[CryptoPosture] = None,
) -> ExecutionReceipt:
    """Build, JCS-canonicalize, and sign an ExecutionReceipt envelope.

    ``back_link`` joins the receipt to the SEP-2787 attestation it
    answers (build it with ``make_back_link``). ``outcome_derived``
    carries the execution status, completion time, and an optional
    result commitment.

    ``signing_material`` is either a bytes shared secret (HS256) or a
    private-key object from ``cryptography.hazmat`` (ES256 / RS256).

    ``crypto_posture`` is the optional CycloneDX-CBOM crypto-posture block.
    Derive it with ``crypto_posture_for(alg=..., sig_suite=...)`` so it stays
    consistent with the signing algorithm and any hybrid suite; it is written
    into ``receiptAsserted`` and so covered by the signature.
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if not back_link.attestation_digest.startswith("sha256:"):
        raise AttestationError(
            "backLink.attestationDigest MUST be a 'sha256:' digest"
        )
    if not back_link.attestation_nonce:
        raise AttestationError("backLink.attestationNonce MUST be non-empty")

    receipt_asserted = ReceiptAsserted(
        iss=iss,
        sub=sub,
        iat=iat or now_iso8601(),
        nonce=nonce or new_nonce(),
        secret_version=secret_version,
        alg=alg,
        sig_suite=sig_suite,
        crypto_posture=crypto_posture,
    )

    payload = _signing_payload(
        version=version,
        alg=alg,
        back_link=back_link,
        outcome_derived=outcome_derived,
        receipt_asserted=receipt_asserted,
    )

    if alg == "HS256":
        if not isinstance(signing_material, (bytes, bytearray)):
            raise AttestationError("HS256 requires bytes shared_secret")
        signature_hex = sign_hs256(payload, shared_secret=bytes(signing_material))
    elif alg == "ES256":
        signature_hex = sign_es256(payload, private_key=signing_material)
    elif alg == "RS256":
        signature_hex = sign_rs256(payload, private_key=signing_material)
    else:
        raise AttestationError(f"unreachable alg: {alg!r}")

    return ExecutionReceipt(
        version=version,
        alg=alg,
        back_link=back_link,
        receipt_asserted=receipt_asserted,
        outcome_derived=outcome_derived,
        signature=signature_hex,
    )


def verify_receipt_signature(
    receipt: ExecutionReceipt,
    *,
    verifying_material: Any,
) -> bool:
    """Verify the receipt signature only.

    Returns True iff the signature matches the JCS-canonical encoding
    of the receipt blocks under ``verifying_material``. Back-link and
    result-commitment checks are composed separately by the caller via
    ``verify_back_link`` and ``verify_args_commitment``; a receipt is a
    durable record so there is no TTL to enforce.

    ``verifying_material`` is either a bytes shared secret (HS256) or a
    public-key object from ``cryptography.hazmat`` (ES256 / RS256).
    """
    payload = _signing_payload(
        version=receipt.version,
        alg=receipt.alg,
        back_link=receipt.back_link,
        outcome_derived=receipt.outcome_derived,
        receipt_asserted=receipt.receipt_asserted,
    )

    if receipt.alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        return verify_hs256(
            payload,
            signature_hex=receipt.signature,
            shared_secret=bytes(verifying_material),
        )
    if receipt.alg == "ES256":
        return verify_es256(
            payload,
            signature_hex=receipt.signature,
            public_key=verifying_material,
        )
    if receipt.alg == "RS256":
        return verify_rs256(
            payload,
            signature_hex=receipt.signature,
            public_key=verifying_material,
        )
    return False
