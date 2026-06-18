"""Emit and verify-signature for brokered-credential (grant) envelopes.

Internal module. Public surface is in ``vaara.credential``.

Reuses the SEP-2787 canonicalization (RFC 8785 JCS) and signing stack
(HS256 / ES256 / RS256) unchanged, so the grant key matches the
attestation/receipt key and a verifier that already handles SEP-2787
signatures handles grants with no new crypto. The only new wire shape is the
envelope layout (``_grant_types``).
"""

from __future__ import annotations

from typing import Any, Optional

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
from vaara.credential._grant_types import (
    BrokeredCredential,
    GrantAsserted,
    GrantBinding,
    GrantScope,
    asserted_to_dict,
    binding_to_dict,
    scope_to_dict,
)


def _signing_payload(
    *,
    version: int,
    alg: Algorithm,
    scope: GrantScope,
    binding: GrantBinding,
    asserted: GrantAsserted,
) -> bytes:
    """JCS-canonical encoding of the grant blocks, signature excluded."""
    body = {
        "version": version,
        "alg": alg,
        "scope": scope_to_dict(scope),
        "binding": binding_to_dict(binding),
        "asserted": asserted_to_dict(asserted),
    }
    return canonical_json(body)


def emit_grant(
    *,
    scope: GrantScope,
    binding: GrantBinding,
    iss: str,
    sub: str,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    exp_seconds: int = 60,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    version: int = 1,
) -> BrokeredCredential:
    """Build, JCS-canonicalize, and sign a BrokeredCredential envelope.

    ``binding`` pins the attestation instance the grant authorizes against
    (build it from ``attestation_digest`` + the attestation nonce).
    ``signing_material`` is either a bytes shared secret (HS256) or a
    private-key object from ``cryptography.hazmat`` (ES256 / RS256).
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if exp_seconds <= 0:
        raise AttestationError("exp_seconds must be a positive integer")
    if not binding.attestation_digest.startswith("sha256:"):
        raise AttestationError("binding.attestationDigest MUST be a 'sha256:' digest")
    if not binding.attestation_nonce:
        raise AttestationError("binding.attestationNonce MUST be non-empty")

    asserted = GrantAsserted(
        iss=iss,
        sub=sub,
        iat=iat or now_iso8601(),
        exp_seconds=exp_seconds,
        nonce=nonce or new_nonce(),
        secret_version=secret_version,
    )

    payload = _signing_payload(
        version=version, alg=alg, scope=scope, binding=binding, asserted=asserted
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

    return BrokeredCredential(
        version=version,
        alg=alg,
        scope=scope,
        binding=binding,
        asserted=asserted,
        signature=signature_hex,
    )


def verify_grant_signature(
    credential: BrokeredCredential,
    *,
    verifying_material: Any,
) -> bool:
    """Verify the grant signature only (scope/expiry/revocation are separate).

    Returns True iff the signature matches the JCS-canonical encoding of the
    grant blocks under ``verifying_material`` (bytes shared secret for HS256,
    public-key object for ES256 / RS256).
    """
    payload = _signing_payload(
        version=credential.version,
        alg=credential.alg,
        scope=credential.scope,
        binding=credential.binding,
        asserted=credential.asserted,
    )

    if credential.alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        return verify_hs256(
            payload,
            signature_hex=credential.signature,
            shared_secret=bytes(verifying_material),
        )
    if credential.alg == "ES256":
        return verify_es256(
            payload, signature_hex=credential.signature, public_key=verifying_material
        )
    if credential.alg == "RS256":
        return verify_rs256(
            payload, signature_hex=credential.signature, public_key=verifying_material
        )
    return False
