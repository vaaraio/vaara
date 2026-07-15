"""Vaara tool-call attestation envelope (v2 shape).

Vaara's own per-tool-call attestation: a signed JSON envelope carried in
MCP ``_meta`` binding a call's intent, issuer, and payload-derived facts
so a later execution receipt can pin the exact request it answers. Vaara
published the first implementation (v0.42.0-0.44.0, 29-30 May 2026) and
originated the trust-surface grouping and the four schema points the
community SEP-2787 draft later adopted; SEP-2787 references are
historical lineage, not a parent spec. The four grouping points:

1. **toolCalls under payloadDerived.** Tool bindings (name, server
   fingerprint, args commitment) are facts derived from the request
   payload, not planner declarations. ``plannerDeclared`` keeps intent
   and an optional requested-capability claim.
2. **argsProjection as JSON-stringified projection.** The ``projection``
   field is the JCS-canonical JSON encoding of the projection object,
   carried as a UTF-8 string; ``projectionDigest`` is computed over
   those bytes.
3. **No ``kind`` discriminator.** ``ArgsRef`` (ref + digest) and
   ``ArgsProjection`` (projection + projectionDigest) self-discriminate
   by which fields are present.
4. **Commitment-only audit via hash-only-identity projection.** When
   the payload must stay local, callers use ``make_args_digest`` which
   builds an ``ArgsProjection`` whose ``projection`` is the
   JCS-canonical encoding of ``{"digest": "sha256:..."}``. The verifier
   reconstructs the same digest from the runtime arguments and rejects
   on mismatch.

Signing modes: HS256 (HMAC-SHA256), ES256 (ECDSA
P-256 raw r||s, not DER), RS256 (RSASSA-PKCS1-v1_5). The signature is
computed over the JCS-canonical encoding of the four envelope blocks
``{version, alg, plannerDeclared, issuerAsserted, payloadDerived}``
and is excluded from its own input.

Install: ``pip install 'vaara[attestation]'``. Requires ``rfc8785`` for
canonicalization and ``cryptography`` for asymmetric signing.

Coexists with the OVERT 1.0 implementation in
``vaara.attestation.overt``. OVERT is the operator-side attestation
kernel emitting CBOR Base Envelopes for every action; this envelope is
per-tool-call JSON carried in MCP ``_meta``. See
``docs/attestation-overt-mapping.md`` for the field-level mapping
between them.

Verifier coverage: ``verify_attestation`` covers the signature and TTL
checks. The argument commitment is exposed separately as
``verify_args_commitment`` so it can be composed by the caller once the
runtime ``tools/call`` arguments are in hand. Nonce replay and tool-call
match are stateful in the runtime and remain the caller's
responsibility.
"""

from __future__ import annotations

from vaara.attestation._attest_canonical import (
    canonical_json,
    make_args_digest,
    make_args_projection,
)
from vaara.attestation._attest_emit import (
    emit_attestation,
    verify_attestation,
)
from vaara.attestation._attest_types import (
    Algorithm,
    ArgsCommitment,
    ArgsProjection,
    ArgsRef,
    Attestation,
    AttestationError,
    IssuerAsserted,
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    attestation_from_dict as parse_attestation,
)
from vaara.attestation._attest_verifier import (
    ArgsCommitmentResult,
    verify_args_commitment,
)

__all__ = [
    "Algorithm",
    "ArgsCommitment",
    "ArgsCommitmentResult",
    "ArgsProjection",
    "ArgsRef",
    "Attestation",
    "AttestationError",
    "IssuerAsserted",
    "PayloadDerived",
    "PlannerDeclared",
    "ToolCallBinding",
    "canonical_json",
    "emit_attestation",
    "make_args_digest",
    "make_args_projection",
    "parse_attestation",
    "verify_args_commitment",
    "verify_attestation",
]
