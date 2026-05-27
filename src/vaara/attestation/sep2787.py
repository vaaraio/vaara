"""SEP-2787 Tool Call Attestation envelope, v2 reference shape.

Implements the v2 envelope shape: trust-surface grouping incorporated
into the SEP draft via soup-oss commit ``dd030d5b``, plus the four
mechanical alignments Vaara committed to in
``modelcontextprotocol/modelcontextprotocol#2787``
(``issuecomment-4557017068``):

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

Signing modes follow the v1 draft: HS256 (HMAC-SHA256), ES256 (ECDSA
P-256 raw r||s, not DER), RS256 (RSASSA-PKCS1-v1_5). The signature is
computed over the JCS-canonical encoding of the four envelope blocks
``{version, alg, plannerDeclared, issuerAsserted, payloadDerived}``
and is excluded from its own input.

Install: ``pip install 'vaara[attestation]'``. Requires ``rfc8785`` for
canonicalization and ``cryptography`` for asymmetric signing.

Coexists with the existing OVERT 1.0 implementation in
``vaara.attestation.overt``. OVERT is the operator-side attestation
kernel emitting CBOR Base Envelopes for every action; SEP-2787 is a
per-tool-call JSON envelope carried in MCP ``_meta``. See
``docs/sep2787-overt-mapping.md`` for the field-level mapping between
them.

Verifier coverage: ``verify_attestation`` covers steps 1 (signature)
and 3 (TTL) of the SEP-2787 verification rules. Step 5 (argument
commitment) is exposed separately as ``verify_args_commitment`` so it
can be composed by the caller once the runtime ``tools/call``
arguments are in hand. Steps 2 (nonce replay) and 4 (tool call match)
are stateful in the runtime and remain the caller's responsibility.
"""

from __future__ import annotations

from vaara.attestation._sep2787_canonical import (
    canonical_json,
    make_args_digest,
    make_args_projection,
)
from vaara.attestation._sep2787_emit import (
    emit_attestation,
    verify_attestation,
)
from vaara.attestation._sep2787_types import (
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
from vaara.attestation._sep2787_verifier import (
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
