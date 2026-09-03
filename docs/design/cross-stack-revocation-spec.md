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

## Reaching it from an export a regulator receives

Pinning only happens when the exporter supplies a registry, and every
shipped export path can now do that:

| Path | How |
|---|---|
| `vaara trail export` | `--revocations PATH` |
| `vaara trail export-threshold` | `--revocations PATH` |
| `vaara trail export-article12` | `--revocations PATH` |
| `vaara trail export-article50` | `--revocations PATH` |
| `export_signed`, `export_signed_threshold` | `revocation=` |
| `export_article12`, `export_article50`, `rotate` | `revocation=` |

An export without a registry carries no `revocation` key in its manifest
and no `revocation.json` in the zip. A reader who wants to recompute a
revocation-in-time verdict from that bundle has nothing to recompute
against, and the absence is visible in the bundle itself: a bundle that
pins an empty registry states that nothing was revoked as of a given
instant, and a bundle with no registry states nothing at all. Those are different
packages and a regulator can tell them apart.

A `--revocations` path that is supplied and unusable stops the export
instead of falling back to an unpinned bundle.

## Compatibility

Purely additive. The receipt envelope, canonicalization, inclusion- and
consistency-proof formats, and signature verification are unchanged; the
envelope version stays 1. `export_signed` with no `revocation` argument
produces a byte-identical manifest to v0.54, and so does every export
command invoked without `--revocations`.

In the threshold export the registry digest goes into the same manifest
every custodian signs, so `revocation.json` is covered transitively by all
k signatures rather than by a signature of its own.

## Freshness: what a clean answer is allowed to claim (v1.80.0)

The registry answers one question, and until v1.80.0 it answered it without
saying when it had last looked. `revoked=False` therefore read as "this key is
fine", when what the computation supports is "nothing in the entries I hold
revoked it, as of whenever I obtained them".

`draft-sirkkavaara-vaara-receipt-08` Section 10 states the limit directly:
offline verification is a computation over the parameters the consumer holds,
revocation is a property of the present, and where a decision depends on
revocation state the staleness a deployment accepts is an operational
parameter that deployment must state. The same shape appears in RPKI, where a
router validates against a locally held cache and the cache's refresh interval
is a stated operational parameter rather than something validation establishes.

### The model

`RevocationRegistry` gains an optional `as_of`, the instant the registry was
observed. `RevocationRegistry.status()` gains `now` and
`max_staleness_seconds`, which are the deployment's stated parameters, and
`RevocationStatus` gains `registry_as_of` and `freshness`.

`freshness` is `"fresh"`, `"stale"`, or `"unknown"`:

- `"fresh"` requires an `as_of`, a stated `max_staleness_seconds`, both
  instants parseable, and an age from zero to the bound inclusive.
- `"stale"` is an age beyond the bound.
- `"unknown"` is everything else, including a registry with no `as_of`, a
  caller who stated no bound, and an `as_of` later than `now`. A future
  observation instant means the clocks disagree, so the bound cannot be
  evaluated honestly and is not silently treated as satisfied.

`RevocationStatus.establishes_current` is true for exactly one combination:
not revoked and fresh. Every other combination is a statement about the past.

### The asymmetry, and why it is the part to protect

Staleness weakens the negative answer only. A revocation the verifier can see
is binding however old the registry is, because a revocation fact does not
expire: the key was revoked at that instant whether the list is an hour old or
a year old. An implementation that reasons "the registry is stale, so we know
nothing" would drop a revocation it is plainly holding, which is strictly
worse than the gap this change closes.

### Conformance

The `revocation_freshness_v0` vector set pins seven cases, six of them
negative. `revoked_stale` pins the asymmetry above. `future_as_of` pins the
clock-disagreement rule. `establishes_current` is true in exactly one row of
the table, which is the property the suite exists to defend.

The checker imports the standard library plus `rfc8785` and rebuilds both the
revocation-in-time predicate and the freshness rule from the text, so a second
implementation can confirm the rules from the committed bytes alone. That is
what the European Commission's Article 50 transparency guidelines describe at
paragraph 76 as detection "ideally locally executable on the digital device".

### Compatibility

Additive, and checkable rather than asserted. `as_of` is serialised only when
set, so a registry without one produces the bytes it produced before the field
existed. The `undated_clean` case in `revocation_freshness_v0` has registry
digest `sha256:a6a20076da005b27c9afc3a5d5b2457798c0ac817d1abc38b2fee4398ac3f133`,
byte-identical to the `clean` case in `cross_stack_revocation_v0`. No
previously issued digest moved. Callers that pass neither `now` nor
`max_staleness_seconds` get the previous `revoked` answer with
`freshness="unknown"` attached.
