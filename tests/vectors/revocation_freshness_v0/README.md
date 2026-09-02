# revocation_freshness_v0

What a clean revocation answer is allowed to claim.

## The rule these vectors hold

`draft-sirkkavaara-vaara-receipt-08` Section 10 states the limit:

> Offline verification is a computation over the parameters the consumer
> holds, while revocation is a property of the present, and this document
> defines no revocation mechanism and places no freshness requirement on key
> material. A consumer MUST NOT treat a signature that verifies as evidence
> that the signing key is still valid. Where a decision depends on revocation
> state, the key resolution path and the staleness a deployment accepts are
> operational parameters of that deployment and MUST be stated by it.

So a registry answer of "not revoked" is a statement about the entries the
verifier holds and the instant it holds them for. This suite pins the two
consequences.

**A clean answer speaks to the present only under three conditions.** The
registry carries an `as_of`, the caller states a `max_staleness_seconds`, and
the registry falls inside that bound. Missing any one of them gives
`freshness: "unknown"` and `establishes_current: false`. The absence of a
freshness parameter is never read as permission to treat the answer as
current.

**A revocation binds however stale the registry is.** A revocation fact does
not expire, so staleness weakens only the negative answer. `revoked_stale` is
the case that pins this asymmetry and it is the one most likely to regress,
because a naive "stale means we know nothing" implementation would drop a
revocation it can plainly see.

## The seven cases

| case | revoked | freshness | establishes_current |
|---|---|---|---|
| `fresh_clean` | false | fresh | **true** |
| `stale_clean` | false | stale | false |
| `undated_clean` | false | unknown | false |
| `unbounded_clean` | false | unknown | false |
| `revoked_fresh` | true | fresh | false |
| `revoked_stale` | true | stale | false |
| `future_as_of` | false | unknown | false |

`establishes_current` is true for exactly one row. That is the point of the
suite: the honest answer is narrow, and every other combination is a
statement about the past.

`future_as_of` is `unknown` rather than trivially fresh. An observation
instant after `now` means the two clocks disagree, and a bound cannot be
evaluated honestly across disagreeing clocks.

## Backward compatibility, checkable rather than asserted

`undated_clean` has registry digest
`sha256:a6a20076da005b27c9afc3a5d5b2457798c0ac817d1abc38b2fee4398ac3f133`,
which is byte-identical to the `clean` case in `cross_stack_revocation_v0`.
A registry with no `as_of` serialises to exactly the bytes it did before the
field existed, so no previously issued digest moved.

## Running it

```
python tests/vectors/revocation_freshness_v0/_check_independent.py
```

The checker imports only the standard library plus `rfc8785`. It does not
import Vaara, and it rebuilds both the revocation-in-time predicate and the
freshness rule from the text rather than calling into the implementation it
is checking. Exit code 0 means every case matched.

Regenerate with `python tests/vectors/revocation_freshness_v0/_generate.py`.
Every instant in the suite is a literal and no keys are involved, so
regeneration is byte-stable.
