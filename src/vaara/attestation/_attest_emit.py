# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit and verify for SEP-2787 envelopes.

Internal module. Public surface is in ``vaara.attestation.tool_call_attestation``.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from vaara.attestation._attest_canonical import (
    canonical_json,
    iso8601_to_epoch,
    new_nonce,
    now_iso8601,
)
from vaara.attestation._attest_signing import (
    sign_es256,
    sign_hs256,
    sign_rs256,
    verify_es256,
    verify_hs256,
    verify_rs256,
)
from vaara.attestation._attest_types import (
    VALID_ALGS,
    Algorithm,
    Attestation,
    AttestationError,
    IssuerAsserted,
    PayloadDerived,
    PlannerDeclared,
    issuer_to_dict,
    payload_to_dict,
    planner_to_dict,
)


def _signing_payload(
    *,
    version: int,
    alg: Algorithm,
    planner_declared: PlannerDeclared,
    issuer_asserted: IssuerAsserted,
    payload_derived: PayloadDerived,
) -> bytes:
    """JCS-canonical encoding of the four envelope blocks.

    The ``signature`` field is excluded; JCS sorts keys so dict order
    here is irrelevant.
    """
    body = {
        "version": version,
        "alg": alg,
        "plannerDeclared": planner_to_dict(planner_declared),
        "issuerAsserted": issuer_to_dict(issuer_asserted),
        "payloadDerived": payload_to_dict(payload_derived),
    }
    return canonical_json(body)


def emit_attestation(
    *,
    planner_declared: PlannerDeclared,
    payload_derived: PayloadDerived,
    iss: str,
    sub: str,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    exp_seconds: int = 300,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    version: int = 1,
) -> Attestation:
    """Build, JCS-canonicalize, and sign an Attestation envelope.

    ``planner_declared`` carries intent and an optional requested
    capability. ``payload_derived`` carries one or more tool-call
    bindings, each pointing at an args commitment derived from the
    request payload.

    ``signing_material`` is either a bytes shared secret (HS256) or a
    private-key object from ``cryptography.hazmat`` (ES256 / RS256).
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if not planner_declared.intent.strip():
        raise AttestationError("plannerDeclared.intent MUST be non-empty")
    if not payload_derived.tool_calls:
        raise AttestationError(
            "payloadDerived.toolCalls MUST contain at least one entry"
        )

    issuer_asserted = IssuerAsserted(
        iss=iss,
        sub=sub,
        iat=iat or now_iso8601(),
        exp_seconds=int(exp_seconds),
        nonce=nonce or new_nonce(),
        secret_version=secret_version,
        alg=alg,
    )

    payload = _signing_payload(
        version=version,
        alg=alg,
        planner_declared=planner_declared,
        issuer_asserted=issuer_asserted,
        payload_derived=payload_derived,
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

    return Attestation(
        version=version,
        alg=alg,
        planner_declared=planner_declared,
        issuer_asserted=issuer_asserted,
        payload_derived=payload_derived,
        signature=signature_hex,
    )


def verify_attestation(
    envelope: Attestation,
    *,
    verifying_material: Any,
    now: Optional[float] = None,
    clock_skew_seconds: int = 30,
) -> bool:
    """Verify signature and TTL.

    Returns True iff the signature matches the JCS-canonical encoding
    of the four envelope blocks under ``verifying_material`` AND the
    attestation is within its validity window:
    ``iat - clock_skew_seconds <= now <= iat + exp_seconds + clock_skew_seconds``.
    A future-dated ``iat`` (beyond the skew allowance) is rejected so the
    live window cannot be extended by stamping issuance ahead of now.

    Replay (nonce) tracking is stateful and belongs in the verifier's
    application layer; this function does NOT track nonces.

    ``verifying_material`` is either a bytes shared secret (HS256) or
    a public-key object from ``cryptography.hazmat`` (ES256 / RS256).
    """
    payload = _signing_payload(
        version=envelope.version,
        alg=envelope.alg,
        planner_declared=envelope.planner_declared,
        issuer_asserted=envelope.issuer_asserted,
        payload_derived=envelope.payload_derived,
    )

    if envelope.alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        ok = verify_hs256(
            payload,
            signature_hex=envelope.signature,
            shared_secret=bytes(verifying_material),
        )
    elif envelope.alg == "ES256":
        ok = verify_es256(
            payload,
            signature_hex=envelope.signature,
            public_key=verifying_material,
        )
    elif envelope.alg == "RS256":
        ok = verify_rs256(
            payload,
            signature_hex=envelope.signature,
            public_key=verifying_material,
        )
    else:
        return False

    if not ok:
        return False

    iat_epoch = iso8601_to_epoch(envelope.issuer_asserted.iat)
    if iat_epoch is None:
        return False
    current = now if now is not None else time.time()
    # Lower bound: an attestation issued in the future is not yet valid.
    # Without this, a forged or clock-wrong issuer could stamp iat far ahead
    # to extend the live window indefinitely (deadline = iat + exp + skew),
    # so the upper-bound TTL check alone never expires it. Allow only the
    # configured skew of forward drift.
    if iat_epoch > current + clock_skew_seconds:
        return False
    deadline = iat_epoch + envelope.issuer_asserted.exp_seconds + clock_skew_seconds
    return current <= deadline
