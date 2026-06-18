# Receipt-bound credential broker (v0)

Status: Phase A, off by default. This is Vaara's authority layer: the move from
observability ("if it went through me I can prove what it was") to control
enforcement ("a tool cannot act unless it routed through me"). It separates
*intelligence* from *authority* the way OAuth separated identity from
authority. The model proposes a tool call; only Vaara grants the temporary,
scoped, receipt-bound capability to execute it.

```
LLM proposes -> Vaara mediates -> short-lived, scoped, bound credential -> tool
```

## A. Token shape

A brokered credential is a standalone signed envelope (`vaara.credential`),
not a field of the SEP-2787 attestation or the execution receipt. The
attestation is emitted before the upstream forward and the receipt after the
outcome, so a credential that must travel *with* the request cannot live in a
receipt that does not exist yet.

The signature covers the JCS (RFC 8785) encoding of
`{version, alg, scope, binding, asserted}` and does not cover itself:

- `scope`: `{toolName, argsCommitment, tenantId}`. `argsCommitment` is the
  attested arguments' `ArgsProjection.projectionDigest`.
- `binding`: `{attestationDigest, attestationNonce}`. The digest is
  `attestation_digest(att)`, the same value the receipt back-links, so an
  auditor can join grant to attestation to receipt offline.
- `asserted`: `{iss, sub, iat, expSeconds, nonce, secretVersion}`. `iss` is
  `vaara-mcp-proxy`; `sub` is `"{tenant}/{upstream}"`; `expSeconds` defaults
  to 60.

The signing stack is the SEP-2787 one (HS256 / ES256 / RS256) reused
unchanged, so the grant key equals the attestation/receipt key and a verifier
that already handles SEP-2787 signatures needs no new crypto. Each signed
block is a closed schema: an unrecognized wire key is a hard reject, keeping
the modeled preimage byte-exact to the wire.

## B. Where it is minted

`AttestPairEmitter.emit_grant` mints and persists the credential
(`{counter}-{nonce}-grant.json`) right after the attestation, inside
`mcp_proxy._handle_tools_call`, between the allow decision and the upstream
forward. The proxy injects it at `params._meta["vaara/credential"]`. The
upstream client serializes the whole payload, so `_meta` reaches the upstream
verbatim. Minting is gated on `proxy._mint_credentials` (default `False`):
observability is unchanged until an operator opts into enforcement.

If minting fails it is logged and swallowed and no credential is injected; a
gateway then fails closed on the missing credential rather than blocking
traffic on a broker error.

## C. Verification

`verify_grant` is the standalone check a gateway runs before letting a tool
execute. It gates on five facts and reports the first failing reason, in this
precedence, so an expired-but-authentic grant is never mislabeled:

1. `bad_signature` - signature does not match the grant blocks.
2. `expired` - now is past `iat + expSeconds` (or the grant is future-dated),
   using the same clock-skew window the SEP-2787 and inference verifiers use
   (30 s).
3. `scope_mismatch` - the runtime tool, tenant, or args do not match the
   committed scope. The args commitment is re-derived with `make_args_digest`,
   so a mutated argument after minting fails here.
4. `revoked` - the issuer or its bound key was revoked at or before issuance
   (`RevocationRegistry`).
5. `binding_unknown` - the bound attestation digest is not in the set of known
   mediation digests the verifier was given. Fail-closed when no set is
   supplied.

`CredentialGateway` is the reference shim. It reads the credential from
`params._meta`, parses it under the closed schema (`malformed` on failure),
loads the known digests by recomputing `attestation_digest` over every
`*-attest.json` in the proxy's receipts directory, and runs `verify_grant`. A
missing `_meta` returns `missing_credential`.

## D. Auditor reconciliation

For tools not behind the gateway, completeness is recovered after the fact, not
at call time. The gateway can fingerprint each used credential as
`sha256(JCS(credential))`. Vaara receipts each yield an
`(attestationDigest, attestationNonce)` via their back-link. The join: for each
used credential, the `binding.attestationDigest` must match a receipt's
`backLink.attestationDigest`. A used credential with no matching receipt, or a
provider action with no credential at all, is the signal of a defeated or
bypassed broker.

## E. Honest limits

This is detection of a defeated broker, not a mathematical-completeness claim.
What it does NOT cover:

- Operator root. Someone who controls the proxy host can mint or forge.
- Alternate credential sources. A tool that accepts some other auth is not
  constrained by this.
- Tool-side alternate auth and purely-local actions (filesystem, exec) with no
  mediation chokepoint to put a gateway in front of.
- `_meta` stripping by an intermediary. This degrades to a
  `missing_credential` refusal, which is fail-closed and acceptable.

The mint-before-receipt ordering means a true grant-to-receipt binding is
fully checkable only at reconciliation; the live shim checks
attestation-digest membership. This is honest and matches the
attestation/receipt split already in the stack.

## F. Conformance

`conformance/sep2828/credential_grant_v0/` ships vectors plus an independent
checker that re-derives every verdict with no `import vaara` (RFC 8785 +
standard library). Regenerate with
`scripts/build_credential_grant_vectors.py`.
