"""Emit and verify for SEP-2787 envelopes.

Internal module. Public surface is in ``vaara.attestation.sep2787``.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Optional

from vaara.attestation._sep2787_canonical import (
    canonical_json,
    iso8601_to_epoch,
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
    ArgsCommitment,
    Attestation,
    AttestationError,
    IssuerAsserted,
    PlannerDeclared,
    args_to_dict,
    planner_to_dict,
)


def _signing_payload(
    *,
    version: int,
    alg: Algorithm,
    planner_declared: PlannerDeclared,
    issuer_asserted: IssuerAsserted,
    payload_derived: tuple[ArgsCommitment, ...],
) -> bytes:
    """JCS-canonical encoding of the four envelope blocks.

    The ``signature`` field is excluded; JCS sorts keys so dict order
    here is irrelevant.
    """
    body = {
        "version": version,
        "alg": alg,
        "planner_declared": planner_to_dict(planner_declared),
        "issuer_asserted": asdict(issuer_asserted),
        "payload_derived": [args_to_dict(a) for a in payload_derived],
    }
    return canonical_json(body)


def emit_attestation(
    *,
    planner_declared: PlannerDeclared,
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

    ``signing_material`` is either a bytes shared secret (HS256) or a
    private-key object from ``cryptography.hazmat`` (ES256 / RS256).

    The ``payload_derived`` block is materialised from
    ``planner_declared.tool_calls[*].args`` in declaration order. The
    duplication is intentional: the planner declared the binding, the
    args commitment is the payload-derived projection of that binding.
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if not planner_declared.intent.strip():
        raise AttestationError("planner_declared.intent MUST be non-empty")
    if not planner_declared.tool_calls:
        raise AttestationError(
            "planner_declared.tool_calls MUST contain at least one entry"
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
    payload_derived = tuple(tc.args for tc in planner_declared.tool_calls)

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
    of the four envelope blocks under ``verifying_material`` AND
    ``iat + exp_seconds + clock_skew_seconds >= now``.

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
    deadline = iat_epoch + envelope.issuer_asserted.exp_seconds + clock_skew_seconds
    current = now if now is not None else time.time()
    return current <= deadline
