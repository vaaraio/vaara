"""SEP-2787 Tool Call Attestation envelope, proposed-shape reference.

Implements the proposed SEP-2787 envelope shape with the four schema
changes Vaara raised in the v1 draft thread
(modelcontextprotocol/modelcontextprotocol#2787), not the v1 draft as
written:

1. **Fact-source labels.** Envelope fields are grouped under three
   named blocks by trust surface: ``planner_declared`` (intent, tool
   name and server bindings the agent planner claims),
   ``issuer_asserted`` (iss, sub, iat, exp, nonce, secret_version,
   alg, set by the attestation issuer at signing time), and
   ``payload_derived`` (args_digest / args_ref / args_projection,
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
``{version, alg, planner_declared, issuer_asserted, payload_derived}``
and is excluded from its own input.

Install: ``pip install 'vaara[attestation]'``. Requires ``rfc8785`` for
canonicalization and ``cryptography`` for asymmetric signing.

Coexists with the existing OVERT 1.0 implementation in
``vaara.attestation.overt``. OVERT is the operator-side attestation
kernel emitting CBOR Base Envelopes for every action; SEP-2787 is a
per-tool-call JSON envelope carried in MCP ``_meta``. See
``docs/sep2787-overt-mapping.md`` for the field-level mapping between
them.
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

__all__ = [
    "Algorithm",
    "ArgsCommitment",
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
    "verify_attestation",
]
