# Cross-stack revocation (design)

Status: implemented in v0.55.0. Additive over v0.54.

## Problem

Revocation already works in exactly one place. The level-3 live identity
check (`verify_receipt_identity_live`, v0.53) fetches a `did:web` document,
reads each verification method's `revoked` instant, and applies the
revocation-in-time rule: a signing key revoked at or before a receipt was
issued no longer yields a trusted verdict, even when the signature still
verifies. A key revoked afterwards still binds, because revocation is not
retroactive.

That rule lives only in the identity lens. The same receipt, checked through
a different lens, ignores revocation entirely:

- The receipt verifier (`verify_receipt_signature` + `verify_back_link`)
  confirms the signature and the back-link. A receipt signed by a since
  revoked key passes both.
- The transparency log proves a receipt is included and that the log is
  append-only. Neither proof says anything about the issuer's standing.
- The Article-12 regulator export signs the audit trail. It carries no
  statement about which issuers were revoked at export time.

So a receipt whose issuer was revoked-in-time gets three different answers
depending on who is looking. That is the gap this closes: one revocation
rule, one implementation, consulted by every lens, so the verdict is the
same everywhere.

## The primitive: a revocation registry

A `RevocationRegistry` is a set of revocation entries, each one a single
fact:

- `scope`: `"key"` (a specific signing key, named by its keyid) or
  `"identity"` (a whole agent identity, named by its `did:web` issuer).
- `subject`: the keyid (key scope) or the issuer DID (identity scope).
- `revoked_at`: an ISO 8601 instant.

The registry exposes one predicate:

```
status(iss, issued_at, keyid=None) -> RevocationStatus
```

A receipt issued at `issued_at` by `iss` (optionally bound to `keyid`) is
**revoked-in-time** iff some entry matches it (an identity-scope entry whose
`subject == iss`, or a key-scope entry whose `subject == keyid`) and that
entry's `revoked_at` is at or before `issued_at`. An unparseable revocation
or issuance instant fails closed (treated as revoked). This is the exact
rule level-3 already applied, lifted out of the DID-document code so it has
no single home.

`RevocationStatus` reports `revoked`, the matching `revoked_at`, `matched_by`
(`"key"` or `"identity"`), and a human-readable `reason`. Both instants flow
through so a verifier holding a stronger time anchor than the receipt's
self-asserted `iat` (the audit-trail hash chain) can re-decide rather than
trust the receipt's own clock, the same escape hatch level 3 exposes.

## Why one registry makes the lenses agree

The registry is source-agnostic: entries can come from a DID document, from
an operator's out-of-band revocation list, or from revocations published in
the transparency log itself. Level 3 keeps reading the DID document, but the
revocation decision now goes through the shared `revoked_in_time` helper, so
`RevocationRegistry.from_did_document(doc, iss)` and the live identity check
agree on the same document by construction, not by coincidence.

Each lens consults the same registry:

- **Receipt verifier.** `check_receipt_revocation(receipt, registry)` reads
  the receipt's `iss` and `iat` and returns the `RevocationStatus`. No
  network, no DID fetch: the offline counterpart of the level-3 rule.
- **Transparency log.** `verify_logged_receipt(...)` checks the inclusion
  proof and the revocation status in one call, returning a verdict that is
  `ok` only when the receipt is both included and not revoked-in-time. A
  monitor reconstructing a registry from logged revocations reaches the same
  conclusion as the receipt verifier.
- **Article-12 export.** `export_signed(..., revocation=registry)` pins the
  registry into the signed manifest (`revocation.registry_sha256` plus a
  `revocation.json` member) so the exact revocation state at export time is
  part of the tamper-evident bundle. A regulator recomputes every receipt's
  revocation verdict against the registry the exporter actually used.

## Scope boundary

Deactivation (`deactivated: true` on a DID document) is identity existence,
not time-scoped revocation, so it stays a level-3 concern and is not
projected into the registry. The registry is exclusively about
revocation-in-time. This keeps `from_did_document` and the live check
consistent on the `revoked` dimension and keeps the registry rule a single,
testable comparison.

## Conformance

The `cross_stack_revocation_v0` vector set carries one receipt and a
registry, and asserts that the receipt-verifier lens, the transparency-log
lens, and the export-digest lens all produce the same revoked verdict. A
Vaara-free, standard-library-plus-`cryptography`-plus-`rfc8785` checker
reproduces every verdict, so the cross-stack guarantee is verifiable without
depending on Vaara.

## Compatibility

Purely additive. The receipt envelope, canonicalization, inclusion- and
consistency-proof formats, and signature verification are unchanged; the
envelope version stays 1. `export_signed` with no `revocation` argument
produces a byte-identical manifest to v0.54.
