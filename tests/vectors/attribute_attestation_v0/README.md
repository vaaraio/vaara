# attribute_attestation_v0

Conformance vectors for `vaara.attribute-attestation/v0`: **a signed value is
only worth what its source is worth.**

## What this pins

Any signed record can assert an attribute. The question a relying party has is
whether the assertion is evidence, and that depends on where the value came
from. So every attribute names its own source, drawn from a closed, ordered set,
and a relying party states the floor it will accept.

```
undeclared  <  operator_declared  <  measured  <  protocol_defined
```

`protocol_defined` outranks `measured` because a value fixed by a specification
cannot be wrong, while a measurement can come from a broken sensor. `undeclared`
is the floor and never converts upward.

The scenario in these vectors is a building alarm classified as an animal. The
same classification is evidence when a named model produced it and is not
evidence when the party paid to reduce alarms typed it in. Both are signed, both
verify, and the corpus grades the difference.

## Four states

| state | meaning |
|-------|---------|
| `accepted` | the attribute is present, in window, and at or above the floor |
| `withheld` | the attestation is sound and does not answer what was asked |
| `expired` | now is outside `notBefore`..`notAfter` |
| `refused` | the artifact fails as evidence |

A value that falls short of the floor is sound evidence of a claim and no
evidence of a fact. Reporting that the same way as a forgery throws away the
difference between "weaker than you asked for" and "someone edited this". The
reason space is partitioned so the two cannot collapse.

Order is soundness, then the clock, then sufficiency. Soundness runs first so an
expired window cannot swallow a broken signature.

## Cases

| case | expected |
|------|----------|
| `pos_measured_clears_floor` | `accepted` / `attribute_attested` |
| `neg_operator_declared_below` | `withheld` / `source_below_floor`, the supplier typed it in |
| `neg_attribute_absent` | `withheld` / `attribute_absent`, absence is named |
| `neg_subject_mismatch` | `withheld` / `subject_mismatch`, right attribute, wrong subject |
| `neg_issuer_not_accepted` | `withheld` / `issuer_not_accepted` |
| `neg_expired_window` | `expired` / `outside_validity_window` |
| `neg_tampered_value` | `refused` / `signature_invalid`, a value edited after signing |
| `neg_unknown_standing` | `refused` / `attestation_malformed`, a standing outside the closed set |

The last one matters more than it looks. A verifier that quietly floors a
standing it does not recognise hands a forger a way to introduce one, so an
unknown standing is malformed rather than treated as `undeclared`.

## Fixture format

```
attestation        the signed vaara.attribute-attestation/v0 document
issuer_key         path to the issuer's Ed25519 public key, relative to this dir
query              {name, minimum_source, subject_id, accepted_issuers}
now                pinned ISO 8601 UTC evaluation instant
expected_state     one of accepted / withheld / expired / refused
expected_reason    the closed-set reason
```

`subject_id` and `accepted_issuers` are `null` when the relying party does not
constrain them.

## Recomputation

`_check_independent.py` imports no Vaara. It needs only `rfc8785` and
`cryptography`.

The signature is Ed25519 over `JCS(attestation without "signature")`, the same
rule `release_condition_v0` and `data_locality_v0` use. Attributes are emitted
sorted by name so two issuers building the same statement produce the same bytes.
Both ends of the validity window are inclusive.

The checker also asserts two structural properties before grading any case: that
the reason-to-state mapping covers all four states, and that the standing ladder
is a total order floored at `undeclared`. A corpus of cases alone would still
pass with two states merged or two standings tied.

## What this is not

Not a qualified electronic attestation of attributes, and it must not be
described as one. Those terms are tied to a supervised trusted-list entry that no
amount of correct cryptography substitutes for. An attestation issued and signed
by the party it describes proves integrity, never independence.

## Regeneration

```
python3 tests/vectors/attribute_attestation_v0/_generate.py
python3 tests/vectors/attribute_attestation_v0/_check_independent.py
```

Ed25519 signing is deterministic, so regenerating reproduces the committed bytes
exactly.
