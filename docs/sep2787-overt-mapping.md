# OVERT 1.0 Base Envelope ↔ SEP-2787 Attestation field mapping

This document maps fields between the OVERT 1.0 Protocol Profile 1.0
Base Envelope (`vaara.attestation.overt`) and the SEP-2787 Tool Call
Attestation envelope, v2 shape (`vaara.attestation.sep2787`).

The two envelopes coexist in Vaara. OVERT is the operator-side
attestation kernel, emitted per in-scope action and encoded as
deterministic CBOR. SEP-2787 is a per-tool-call JSON envelope carried
in MCP `_meta`. The fields below capture identity, intent, payload
commitment, time, and signature in both standards. Where the meanings
align, the mapping is direct. Where they diverge, the table notes the
reason.

## Trust-surface alignment

The SEP-2787 v2 shape groups envelope fields under three trust-surface
blocks: `plannerDeclared` (intent and requested capability),
`issuerAsserted` (iss, sub, iat, exp, nonce, key version, alg, set by
the attestation issuer at signing time), and `payloadDerived` (the
`toolCalls` array, each binding a tool name, server fingerprint, and
args commitment derived from the request payload). OVERT does not
group fields explicitly but applies the same separation through field
semantics. The mapping below names which block each OVERT field
corresponds to.

## Field-by-field mapping

| OVERT Base Envelope (CBOR) | SEP-2787 proposed shape (JSON) | Trust surface | Notes |
|---|---|---|---|
| `blinded_identifier` (32 bytes) | `issuerAsserted.nonce` (base64url) | issuer-asserted | Both serve replay-protection. OVERT uses 32 bytes per action; SEP-2787 uses 18 bytes base64url-encoded per attestation. |
| `request_commitment` (HMAC-SHA256, bytes) | `payloadDerived.toolCalls[*].args.projectionDigest` (`sha256:<hex>`), with `projection` carrying the JCS encoding of `{"digest": "sha256:<hex>"}` | payload-derived | Direct semantic match for commitment-only audit. OVERT uses keyed HMAC over a SHA-256 digest of the request content; SEP-2787 v2 expresses the same intent as a hash-only-identity projection (no separate `argsDigest` field in the spec). Both keep the raw payload local. |
| `encoder_binary_identity` (SHA-256, bytes) | (no direct equivalent) | n/a | OVERT pins the arbiter implementation + version + policy hash at signing time. SEP-2787 v2 has no equivalent. Vaara's reference implementation could attach an extension field for this. |
| `non_content_metadata` (CBOR map) | `plannerDeclared.requestedCapability` (string) | planner-declared | Partial match. OVERT carries structural classification fields (action class, severity, decision); SEP-2787 has only the capability string in the v2 shape. |
| `monotonic_counter` (uint64) | (no direct equivalent) | n/a | OVERT requires a strictly increasing per-arbiter sequence to detect gaps. SEP-2787 has no monotonic counter; replay protection is per-nonce within TTL. |
| `nanosecond_timestamp` (uint64) | `issuerAsserted.iat` (ISO 8601 string) | issuer-asserted | Same semantics, different encoding. OVERT uses uint64 nanoseconds for cross-language stability; SEP-2787 uses ISO 8601 strings for JSON-native consumption. |
| `key_identifier` (SHA-256 over public key, bytes) | `issuerAsserted.secretVersion` (string) | issuer-asserted | OVERT binds the verifying key cryptographically via SHA-256 fingerprint; SEP-2787 names it via opaque version string and leaves key lookup to the verifier. |
| `arbiter_instance_identifier` (16 bytes, UUID) | (no direct equivalent) | n/a | OVERT identifies the specific arbiter instance that produced the envelope. SEP-2787 conflates this with `iss`. |
| `signature` (Ed25519, 64 bytes) | `signature` (hex-encoded) | binding output | Different curve choices. OVERT mandates Ed25519. SEP-2787 supports HS256 (HMAC-SHA256), ES256 (ECDSA P-256), RS256 (RSASSA-PKCS1-v1_5). |
| (no equivalent) | `plannerDeclared.intent` (string) | planner-declared | Human-readable justification required by EU AI Act Article 12 audit reconstruction. OVERT can carry this in `non_content_metadata` but does not require it as a top-level field. |
| (no equivalent) | `payloadDerived.toolCalls[*].name` and `.serverFingerprint` | payload-derived | MCP-specific binding. OVERT is transport-agnostic and does not name a tool or server in the envelope. |

## Canonicalization

| OVERT | SEP-2787 v2 shape |
|---|---|
| Deterministic CBOR per RFC 8949 Section 4.2 (sorted keys, smallest int encoding, definite lengths) | JSON Canonicalization Scheme per RFC 8785 (sorted keys, ECMAScript number serialization, Unicode escaping) |
| IEEE-754 floats prohibited; rates and probabilities as decimal strings | IEEE-754 floats rejected at the boundary (same discipline applied to JSON) |

## Where the standards point at the same evidence

Both standards bind the same logical evidence: identity, intent,
payload commitment, time, and signature. The SEP-2787 v2 shape makes
the trust-surface separation explicit through named blocks; OVERT
achieves the same separation through field semantics and the Phase 1
/ Phase 2 / Phase 3 architecture. A deployment running OVERT Phase 2
Provisional Receipts can produce an equivalent SEP-2787 envelope by
projecting the Base Envelope fields into the three named blocks and
re-signing under the JSON-native algorithm of choice. The reverse
projection is also possible but loses OVERT's monotonic counter and
arbiter instance identifier, which have no SEP-2787 equivalent in the
v2 shape.

## See also

- OVERT 1.0 standard: <https://overt.is>
- SEP-2787 PR: <https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2787>
- [Vaara COMPLIANCE.md](../COMPLIANCE.md): section "Position relative to open runtime-attestation standards"
