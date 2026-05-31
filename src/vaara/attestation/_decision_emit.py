"""Emit and verify-signature for decision-record envelopes.

Internal module. Public surface is in ``vaara.attestation.decision``.

Reuses the SEP-2787 canonicalization (RFC 8785 JCS) and signing stack
(HS256 / ES256 / RS256) unchanged. The only new wire shape is the
envelope layout; the cryptographic primitives are shared so a verifier
that already handles SEP-2787 signatures handles decision records with
no new crypto code.
"""

from __future__ import annotations

from typing import Any, Optional

from vaara.attestation._decision_types import (
    DecisionDerived,
    DecisionRecord,
    IssuerAsserted,
    decision_to_dict,
)
from vaara.attestation._receipt_types import BackLink, back_link_to_dict, receipt_asserted_to_dict
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
    decision_derived: DecisionDerived,
    issuer_asserted: IssuerAsserted,
) -> bytes:
    """JCS-canonical encoding of the decision blocks, signature excluded."""
    body = {
        "version": version,
        "alg": alg,
        "backLink": back_link_to_dict(back_link),
        "decisionDerived": decision_to_dict(decision_derived),
        "issuerAsserted": receipt_asserted_to_dict(issuer_asserted),
    }
    return canonical_json(body)


def emit_decision_record(
    *,
    back_link: BackLink,
    decision_derived: DecisionDerived,
    iss: str,
    sub: str,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    version: int = 1,
) -> DecisionRecord:
    """Build, JCS-canonicalize, and sign a DecisionRecord envelope.

    ``back_link`` joins the decision to the SEP-2787 attestation it
    governs (build it with ``make_back_link``). ``decision_derived``
    carries the verdict, its risk basis, and the decision time. Any
    float in the risk basis is rejected at the JCS boundary; the risk
    fields MUST be decimal strings.

    ``signing_material`` is either a bytes shared secret (HS256) or a
    private-key object from ``cryptography.hazmat`` (ES256 / RS256).
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if not back_link.attestation_digest.startswith("sha256:"):
        raise AttestationError(
            "backLink.attestationDigest MUST be a 'sha256:' digest"
        )
    if not back_link.attestation_nonce:
        raise AttestationError("backLink.attestationNonce MUST be non-empty")

    issuer_asserted = IssuerAsserted(
        iss=iss,
        sub=sub,
        iat=iat or now_iso8601(),
        nonce=nonce or new_nonce(),
        secret_version=secret_version,
        alg=alg,
    )

    payload = _signing_payload(
        version=version,
        alg=alg,
        back_link=back_link,
        decision_derived=decision_derived,
        issuer_asserted=issuer_asserted,
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

    return DecisionRecord(
        version=version,
        alg=alg,
        back_link=back_link,
        decision_derived=decision_derived,
        issuer_asserted=issuer_asserted,
        signature=signature_hex,
    )


def verify_decision_signature(
    record: DecisionRecord,
    *,
    verifying_material: Any,
) -> bool:
    """Verify the decision-record signature only.

    Returns True iff the signature matches the JCS-canonical encoding of
    the record blocks under ``verifying_material``. Back-link and pairing
    checks are composed separately via ``verify_decision_back_link`` and
    ``records_paired``; a decision record is a durable record so there is
    no TTL to enforce.

    ``verifying_material`` is either a bytes shared secret (HS256) or a
    public-key object from ``cryptography.hazmat`` (ES256 / RS256).
    """
    payload = _signing_payload(
        version=record.version,
        alg=record.alg,
        back_link=record.back_link,
        decision_derived=record.decision_derived,
        issuer_asserted=record.issuer_asserted,
    )

    if record.alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        return verify_hs256(
            payload,
            signature_hex=record.signature,
            shared_secret=bytes(verifying_material),
        )
    if record.alg == "ES256":
        return verify_es256(
            payload,
            signature_hex=record.signature,
            public_key=verifying_material,
        )
    if record.alg == "RS256":
        return verify_rs256(
            payload,
            signature_hex=record.signature,
            public_key=verifying_material,
        )
    return False
