# attribute_attestation_zk_v0

Conformance vectors for `vaara.attribute-attestation-zk/v0`: **an issuer that
commits to the value and hands the opening to the holder has nothing left to
sell.**

## What this pins

`attribute_attestation_v0` asks what a value is worth, and answers it by making
every attribute name its own source. This suite asks the question that follows:
what does the issuer have to *keep* in order to say it.

An attestation provider that vouches for an attribute has to hold the attribute.
Date of birth, mother's maiden name, images of an identity document, video of the
customer proving they are themselves. Anything held can be sold, subpoenaed,
breached or repurposed, and no policy statement changes what the holder is
capable of.

Commit at issuance and hand the opening to the holder, and the asset stops
existing. What is left is a Pedersen commitment and a signature over it. A
relying party still gets the two things it actually needed, and neither of them
is the value:

- that a **predicate** holds over the hidden value, proved rather than asserted
- **how strongly the value was sourced**, in the clear, on the same ladder

```
undeclared  <  operator_declared  <  measured  <  protocol_defined
```

The standing stays readable because judging the strength of evidence is the
relying party's job. Reading the value was never part of that job.

## The one field that changes

```
attribute_attestation_v0:      {name, value,      source, sourceDetail}
attribute_attestation_zk_v0:   {name, commitment, source, sourceDetail}
```

Everything else is deliberately identical: JCS canonicalization, Ed25519 over the
document with its own `signature` member removed, four states, the reason space
partitioned as data, and checks ordered soundness, clock, sufficiency.

## Four states

| state | meaning |
|-------|---------|
| `accepted` | the predicate was proved over a value sourced at or above the floor |
| `withheld` | the document is sound and does not answer what was asked |
| `expired` | now is outside `notBefore`..`notAfter` |
| `refused` | the document or the proof fails as evidence |

`proof_absent` withholds and `proof_invalid` refuses. Nothing proved is not the
same fact as something forged, and one boolean for both discards the difference
between "not yet" and "no". Two cases in this corpus exist only to keep those
apart.

## Cases

| case | expected |
|------|----------|
| `pos_at_least_holds` | `accepted` / `predicate_proven`, age over 18 without an age |
| `pos_in_range_holds` | `accepted` / `predicate_proven`, two directions, both blinds |
| `neg_proof_absent` | `withheld` / `proof_absent`, nothing was proved |
| `neg_attribute_absent` | `withheld` / `attribute_absent`, absence is named |
| `neg_source_below_floor` | `withheld` / `source_below_floor`, sound proof, the subject typed the value in |
| `neg_expired_window` | `expired` / `outside_validity_window` |
| `neg_predicate_false` | `refused` / `proof_invalid`, a proof of a statement that is untrue |
| `neg_proof_replayed` | `refused` / `proof_not_bound`, another document's proof presented here |
| `neg_tampered_commitment` | `refused` / `signature_invalid`, a commitment swapped after signing |
| `neg_tampered_standing` | `refused` / `signature_invalid`, `operator_declared` edited to `measured` |

`neg_predicate_false` is the one that cannot be produced honestly. The shipped
prover refuses to build it, because a predicate that does not hold has no
witness, so the generator calls the range argument directly on the false witness
the way a forger would have to. The bits of a negative witness reconstruct to a
different curve point, so the weighted sum misses the target and the proof does
not verify.

## The cryptography, stated so it can be rebuilt

The curve is NIST P-256, from the published domain parameters. A commitment is
`C = v*G + r*H`, where `G` is the standard base point and `H` is derived by
try-and-increment hash-to-curve from the label `vaara/zk/H/v0`, normalised to
even `y`. There is no trusted setup: recompute `H` from the label and check it.
Commitments are perfectly hiding and computationally binding.

A range proof shows that a commitment opens to a value in `[0, 2**32)`. It
publishes one commitment per bit, proves each opens to 0 or 1 with a Schnorr
OR-proof over base `H`, and the verifier checks that `sum(2**i * C_i)` equals the
target. Comparisons are the same argument over a shifted commitment, because
Pedersen commitments add:

```
value >= t    target  C - t*G     opens to (value - t) under blind  r
value <= t    target  t*G - C     opens to (t - value) under blind -r
a <= v <= b   both, in that order
```

Each proof's Fiat-Shamir transcript is seeded with the attestation digest, the
attribute name, the JCS of the predicate and the direction:

```
b"vaara/attribute-zk/v0" / <attestationDigest> / <name> / JCS(predicate) / <ge|le>
```

so a proof does not move to another document, another attribute or another
threshold. `neg_proof_replayed` is that property as a case.

## Fixture format

```
attestation        the signed vaara.attribute-attestation-zk/v0 document
proof              a vaara.attribute-predicate/v0 envelope, or null
issuer_key         path to the issuer's Ed25519 public key, relative to this dir
query              {name, predicate, minimum_source, subject_id, accepted_issuers}
now                pinned ISO 8601 UTC evaluation instant
expected_state     one of accepted / withheld / expired / refused
expected_reason    the closed-set reason
```

`subject_id` and `accepted_issuers` are `null` when the relying party does not
constrain them. `proof` is `null` when none was presented.

## Recomputation

`_check_independent.py` imports no Vaara. It needs only `rfc8785` and
`cryptography`, and it rebuilds the field arithmetic, the group law, the
hash-to-curve, the commitments, the OR-proofs and the range argument from the
parameters above.

Before grading any case it asserts six structural properties, because a corpus of
cases alone would still pass with two states merged, two standings tied, or a
generator nobody can recompute:

- the reason-to-state mapping covers all four states
- the standing ladder is a total order floored at `undeclared`
- `proof_absent` and `proof_invalid` land in different states
- `H` recomputes from its label, lies on the curve, and is not `G`
- commitments are additively homomorphic, which is what makes a shift a comparison
- the same value under two blinds gives two different commitments

## What this is not

**Not selective disclosure.** One signature covers every commitment in the
document, so a holder cannot present three attributes out of ten from a single
signed credential. That needs a signature scheme built for it (BBS+) and is not
here.

**Not qualified.** Not a qualified electronic attestation of attributes under
Regulation (EU) 910/2014, and it must not be described as one. Those terms are
tied to a supervised trusted-list entry that no cryptographic property
substitutes for.

**Not a claim that the issuer is honest.** Zero knowledge hides the value from
the relying party and from anyone the issuer might later sell to. It says nothing
about whether the issuer committed to the truth in the first place. That residual
is the same one documented in `docs/prove-what-an-ai-agent-did.md`.

**Bounded.** Every committed value and every predicate bound lies in
`[0, 2**32)`, because that is the interval the range proof argues over. Values
outside it are refused at issuance. String attributes are not carried; they would
need a membership proof against a committed set.

## Regeneration

```
python3 tests/vectors/attribute_attestation_zk_v0/_generate.py
python3 tests/vectors/attribute_attestation_zk_v0/_check_independent.py
```

Ed25519 signing is deterministic. The other two sources of randomness are pinned
in the generator so the committed bytes reproduce exactly: the commitment blinds
are derived from a fixed seed and passed in, and the prover's internal scalars
come from a seeded hash chain installed over `random_scalar` for the length of
that script. Neither substitution exists in the library, where a blind that is not
fresh is a linkage bug.
