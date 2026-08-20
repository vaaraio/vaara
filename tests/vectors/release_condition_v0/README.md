# release_condition_v0

Conformance vectors for the release condition: **a Vaara receipt gates a
payment, instead of a payment gating access.**

## What this pins

Everywhere else in this corpus a payment buys access and the settlement evidence
lands inside a receipt (`x402_settlement_v0`, SPEC.md Section 5.2). This suite is
the inversion. Money is held against a signed `vaara.release-condition/v0`
document that names exactly what must be proved. A receipt proving the authorised
action happened is what releases it.

Nothing in the profile holds a payer's key, signs a transaction, or touches a
chain. It answers one question about bytes; a settlement agent acts on the answer.

## Four states, and why they are four

| state | meaning |
|-------|---------|
| `released` | the authorised action is proved; the held value moves |
| `held` | the evidence is sound and does not satisfy the condition, or none has been presented yet |
| `expired` | the window the condition named has closed |
| `refused` | the presented artifact fails as evidence |

A verifier that proved nothing must never read as green, and must never read as
the same false as a genuine failure. `held` because no receipt arrived and
`refused` because a receipt was tampered with are different facts about the
world; one boolean for both throws away the difference between "not yet" and
"no". The reason space is partitioned in `REASON_STATE`, so a reason belongs to
exactly one state and no code path can file a forgery under a hold.

Two questions in order: is the artifact sound, and is it sufficient.

- Soundness failures are `refused`: a broken condition signature, a receipt under
  a key the condition does not pin, a broken receipt signature, or evidence that
  does not resolve to the digest the receipt signed.
- Sufficiency failures are `held`: nothing presented, another action, another
  authorization, another issuer, or a receipt that soundly proves a *refusal*.

Order is soundness, then the clock, then sufficiency. Soundness runs first so an
expired window cannot swallow a tampering finding. The clock runs before
sufficiency so a closed window is reported as the reason the money is not moving.

## Cases

| case | expected |
|------|----------|
| `pos_matching_receipt` | `released` / `receipt_matches` |
| `neg_absent_receipt` | `held` / `receipt_absent`, nothing presented, and absence is named as the reason |
| `neg_authorization_mismatch` | `held` / `authorization_mismatch`, a sound receipt under a different grant |
| `neg_other_action` | `held` / `action_digest_mismatch`, a sound receipt for a different action |
| `neg_blocked_decision` | `held` / `decision_not_accepted`, a sound receipt proving the action was blocked |
| `neg_expired_condition` | `expired` / `condition_expired`, and it does not read as refused |
| `neg_tampered_receipt` | `refused` / `receipt_signature_invalid`, `decidedAt` moved one second after signing |
| `neg_untrusted_key` | `refused` / `receipt_key_untrusted`, signed under a key the condition never pinned |

The negatives are what earn the suite. A corpus carrying only the positive case
would still pass with two of these states merged into one, so the checker also
asserts the reason-to-state mapping is a partition covering all four states.

## Fixture format

Each `cases/*.json` file contains:

```
condition          the signed vaara.release-condition/v0 document
condition_key      path to the issuer's Ed25519 public key, relative to this dir
receipt            the vaara.receipt/v1 envelope presented, or null
receipt_key        path to the key the receipt was presented under, or null
evidence           the record the receipt's evidenceRef pins, or null
now                pinned ISO 8601 UTC evaluation instant
expected_state     one of released / held / expired / refused
expected_reason    the closed-set reason
```

Keys travel beside the case rather than inside the condition, because a document
cannot vouch for the key that signed it. A relying party holds both out of band.

## Recomputation

`_check_independent.py` imports no Vaara. It needs only `rfc8785` and
`cryptography`, and recomputes every verdict from the case bytes.

Both signatures follow the same rule: canonicalize the document with its own
`signature` field removed, then sign.

- The condition is Ed25519 over `JCS(condition without "signature")`.
- The receipt is ES256 (raw `r||s`, hex) over
  `JCS({version, alg, backLink, decisionDerived, issuerAsserted})`, which is the
  wire record minus its signature.
- Content addresses are `sha256:` over JCS bytes: the evidence digest the receipt
  pins, the condition digest a decision names, and the receipt digest.

`notAfter` is inclusive: a receipt presented at exactly that instant releases.

The corpus keys are derived from fixed constants in `_generate.py` and are
corpus-only, never deployed. Only the public halves are committed.

## Regeneration

```
python3 tests/vectors/release_condition_v0/_generate.py
python3 tests/vectors/release_condition_v0/_check_independent.py
```

ECDSA signing is randomised, so regenerating changes the signature bytes. The
records under those signatures are pinned (`iat`, `decidedAt`, nonces) so nothing
else moves.
