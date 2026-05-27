"""SEP-2787 Tool Call Attestation envelope, proposed-shape reference.

Implements the proposed SEP-2787 envelope shape with the four schema
changes Vaara raised in the v1 draft thread
(modelcontextprotocol/modelcontextprotocol#2787), not the v1 draft as
written:

1. **Fact-source labels.** Envelope fields are grouped under three
   named blocks by trust surface: ``plannerDeclared`` (intent, tool
   name and server bindings the agent planner claims),
   ``issuerAsserted`` (iss, sub, iat, expSeconds, nonce, secretVersion,
   alg, set by the attestation issuer at signing time), and
   ``payloadDerived`` (argsDigest / argsRef / argsProjection,
   deterministically derived from the request payload). The signature
   is the binding output and lives at the envelope root.
2. **Three-way args shape.** The v1 draft overloads a single
   ``args: string`` field with both inline JSON and a magic
   ``"resource: "`` prefix. This module replaces that with an explicit
   tagged union: ``ArgsDigest`` (commitment only, payload stays local),
   ``ArgsRef`` (URL plus digest), ``ArgsProjection`` (redacted /
   transformed projection plus its own digest).
3. **RFC 8785 (JCS) canonicalization.** The v1 "sorted keys, no
   whitespace" rule is replaced with a normative reference to RFC
   8785, plus an IEEE-754 float reject at the boundary (matching OVERT
   Protocol Profile 1.0 numeric discipline).
4. **Scope: request attestation only.** The v1 optional ``ack`` field
   crosses the pre-exec / post-exec boundary and is removed here.
   Execution receipts belong in a separate extension composed on top.

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
    ArgsDigest,
    ArgsProjection,
    ArgsRef,
    Attestation,
    AttestationError,
    IssuerAsserted,
    PlannerDeclared,
    ToolCallBinding,
)
from vaara.attestation._sep2787_verifier import (
    ArgsCommitmentResult,
    verify_args_commitment,
)

__all__ = [
    "Algorithm",
    "ArgsCommitment",
    "ArgsCommitmentResult",
    "ArgsDigest",
    "ArgsProjection",
    "ArgsRef",
    "Attestation",
    "AttestationError",
    "IssuerAsserted",
    "PlannerDeclared",
    "ToolCallBinding",
    "canonical_json",
    "emit_attestation",
    "make_args_digest",
    "make_args_projection",
    "verify_args_commitment",
    "verify_attestation",
]
