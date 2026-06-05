# Design spec: resolvable agent identity for execution receipts

Status: draft for v0.5. Companion to `docs/design/threshold-signing-spec.md`
and the SEP-2828 work in `src/vaara/attestation/`. Motivated by the
convergence in MCP discussions #2402 and #2704: an audit record that names
its issuer as an opaque string is self-reported, and a verifier cannot tell
whether the named agent actually signed it.

## Goal

Today `ReceiptAsserted.iss` (and `sub`) are opaque strings. The signature
proves the values were bound together at receipt time, but it proves nothing
about *who* `iss` is. A server can write `iss: "billing-agent"` into a record
it signed with its own key, and a later auditor has no way to check that the
string and the signing key belong to the same party.

The fix: let `iss` optionally be a resolvable identifier (a `did:web`
identity), so a verifier can resolve it to a public key and confirm the
receipt signature was made by a key that identity controls. This turns the
issuer field from a self-asserted label into a checkable claim, without
changing the envelope, the canonicalization, or any existing vector.

## Why did:web, not an opaque keyid or a new PKI

Three options were on the table:

- **did:web (chosen).** The identity is an HTTPS-resolvable DID document
  listing the agent's verification keys. No new trust root, no registry to
  run: resolution rides on the web PKI the deployer already operates. A
  regulator reads `did:web:agents.bank.example:billing` and can fetch the key
  set themselves. This matches the resolvable-keyid model the AlgoVoi and
  RFC 9421 work in #2704 settled on, expressed in Vaara's own envelope rather
  than adopting their transport-signature wire format.
- **Opaque keyid + out-of-band key map.** Lighter, but pushes key
  distribution onto every verifier and gives a regulator nothing resolvable.
  Rejected: it keeps the self-reported problem one layer down.
- **A dedicated agent registry / new PKI.** Maximum control, but it is a
  service to run, a single point to trust, and against the self-built,
  no-new-trust-root ethos. Rejected for v0.5.

did:web also keeps offline verification intact: a verifier that has already
pinned the DID document (or the key) verifies with no network call, the same
way pinned signing keys work today. Resolution is an upgrade to the trust of
the `iss` claim, not a new hard dependency for verifying the signature.

## Data model

`ReceiptAsserted` gains no required field. The change is a convention plus an
optional resolution hint:

- `iss` MAY be a `did:web:` identifier. When it is, a verifier MAY resolve it.
- An optional `iss_keyid` names which verification method in the resolved DID
  document signed this receipt, for documents that list more than one key.

```json
{
  "iss": "did:web:agents.bank.example:billing",
  "iss_keyid": "did:web:agents.bank.example:billing#key-2026",
  "sub": "did:web:agents.bank.example:billing",
  "...": "all existing ReceiptAsserted fields unchanged"
}
```

Both fields are inside the signed bytes, so the binding between the receipt
signature and the claimed identity cannot be altered after issuance without
invalidating the signature.

## Verification flow

Three levels, each strictly stronger than the last, all backward compatible:

1. **Opaque (today's behavior).** `iss` is any string. The signature is
   checked against a pinned key. Nothing about `iss` is resolved. Unchanged.
2. **Pinned-resolvable.** `iss` is a `did:web`, the verifier already holds
   the DID document, and it confirms the receipt signature matches a key the
   document lists. No network. This is the recommended regulator-export mode.
3. **Live-resolvable.** `iss` is a `did:web`, the verifier fetches the DID
   document over HTTPS at audit time and then performs the level-2 check. The
   fetch is recorded (URL, fetch time, document digest) so the resolution
   itself is auditable and reproducible.

A receipt that fails level-2/3 when the verifier opted into resolution is a
hard failure. A receipt with a plain-string `iss` is never failed for lack of
a DID: resolution is opt-in by the verifier, not mandatory on the record.

## Conformance-vector impact

None to existing vectors. Every current `decision_pairing_v0` vector keeps a
plain-string `iss` and verifies byte-for-byte as before. Resolvable identity
ships as a *new* vector family (`agent_identity_v0`) carrying:

- a receipt with a `did:web` `iss`, a captured DID document, and the expected
  level-2 verdict, so an independent implementation can reproduce the
  resolve-and-check without making a live network call;
- a negative vector where the receipt signature does not match any key in the
  resolved document, asserting a hard failure.

This keeps the property the #2402 thread cares about: the record verifies
against the spec and captured inputs, not against a live service or our code.

The vector family also covers level-3 revocation offline: `revoked.json`
carries a document that lists the signing key but marks it `revoked` before
the receipt's `iat`, with the expected verdict `bound` but not `trusted`.
That comparison needs no network, so it ships as a vector even though the
fetch itself does not.

## Implementation

Level 2 (`verify_receipt_identity`) takes a DID document the caller already
holds and returns `IdentityResult(resolved, bound, keyid, reason)`. It is
pure and offline.

Level 3 (`verify_receipt_identity_live`) wraps level 2 with resolution,
caching, deactivation, and revocation, and returns `LiveIdentityResult`:

- **Resolution record.** Each resolve produces `ResolutionMeta(did, url,
  fetched_at, document_digest, from_cache)`. `document_digest` is
  `sha256:<hex>` over the exact bytes fetched, so the resolution is
  reproducible: hand a second auditor the same document and they recompute
  the digest and the verdict. The default fetcher is a size-capped,
  HTTPS-only stdlib GET; a deployer injects their own for allowlisting,
  pinning, or proxy egress, or to verify offline against a captured document.
- **Caching.** `DidDocumentCache` is an in-memory TTL cache keyed by DID, so
  a verifier checking many receipts from one issuer resolves once per window.
  The clock is caller-supplied, so it is deterministic under test.
- **Revocation in time.** A verification method may carry a `revoked` ISO
  8601 instant; a document may carry `deactivated: true`. A key revoked at or
  before the receipt's `iat` yields `trusted=false` even when the signature
  matches; a key revoked afterwards still binds, because revocation is not
  retroactive. The result surfaces both `issued_at` and `revoked_at` so a
  verifier with a stronger time anchor than the self-asserted `iat` (the
  audit-trail hash chain) can re-decide.

`trusted` is the single overall verdict: resolved, bound, not deactivated,
and not revoked at or before issuance.

## Open questions

- Whether `sub` (the acting subject, distinct from the issuer) should resolve
  by the same mechanism, or stay opaque when the subject is an end user rather
  than an agent.
- Key rotation across long spans is partly addressed: the level-3 result
  records the fetch (URL, time, document digest) and surfaces the revocation
  instant against issuance, and the audit-trail hash chain supplies the
  external time anchor for which key was valid when. What remains is a
  convention for pinning the resolved document digest into the trail at
  verification time, so a re-verify years later can detect a since-rotated
  document rather than silently resolving the current one.
- Whether to register the receipt envelope's resolvable-identity convention
  as a follow-up to SEP-2828, so the MCP audit-context line and Vaara's
  envelope name identity the same way.
