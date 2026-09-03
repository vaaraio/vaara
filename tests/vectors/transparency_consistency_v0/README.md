# transparency_consistency_v0 conformance vectors

Append-only consistency proofs for Vaara's Merkle transparency log, following
RFC 9162 (RFC 6962-bis) section 2.1.4. See
`docs/design/transparency-log-consistency-spec.md`.

A transparency log exists to be append-only: once an entry is logged, the
operator cannot quietly rewrite or drop earlier history. An *inclusion* proof
shows an entry is in the log; a *consistency* proof shows that the log at an
earlier size is a verifiable prefix of the log at a later size. A monitor that
pins the log's signed tree head over time, then checks a consistency proof
between consecutive heads, detects any fork or rewrite even if every
individual inclusion proof still verifies.

These vectors are pure SHA-256 Merkle hashing, no signatures, so they verify
with only the standard library:

- leaf hash: `SHA-256(0x00 || leaf)`
- internal node hash: `SHA-256(0x01 || left || right)`

## Files

- `log.json`: the committed log content (the ordered leaves) plus the hashing
  rule, so a verifier can recompute every root itself.
- `cases.json`: each case carries `first_size`, `second_size`, the two roots a
  verifier would hold at those sizes (`first_root`, `second_root`), and the
  `proof` hashes (hex).
- `expected.json`: the expected `verdict` per case, one of `consistent`,
  `inconsistent`, or `could_not_compare`.

## The verdict is three-valued

RFC 9162 section 2.1.4.2 bounds a consistency proof at `0 < m < n`. Asked
about sizes outside that range, a checker has nothing to compare, and a
two-valued answer has to lie in one direction or the other: `true` says the
log is append-only on the strength of a check that never ran, `false` says
the log was inconsistent on the same absence of evidence.

The third value comes with two rules that make it enforceable:

1. `could_not_compare` is decided first, before any comparison is attempted.
   Deciding invalid first would swallow it.
2. The verdict is falsy unless it is `consistent`, so a checker written
   against a boolean return fails closed on an input it could not compare.

Each case pins the exact verdict string. A case merely tagged as
out-of-range would be a row a harness could skip, and a checker answering
`true` on it would still pass. Requiring the checker to report that it did
not compare turns the case into a real negative.

## Cases

Positive cases (`consistent`) cover the sizes the algorithm most often gets
wrong: a power-of-two prefix (`1 to 12`, `8 to 12`), non-power-of-two
prefixes (`3 to 12`, `7 to 12`, `5 to 9`), and identical trees (`12 to 12`,
empty proof).

Negative cases (`inconsistent`) keep a genuine proof but corrupt one input,
so a checker that always answered yes is caught:

- `tampered_proof_hash_3_to_12`: one sibling hash in the proof is flipped.
- `forked_second_root_3_to_12`: the proof is checked against a `second_root`
  taken from an unrelated (forked) log, the rewrite a consistency proof exists
  to detect.

Out-of-range cases (`could_not_compare`) carry genuine roots and a
well-formed empty proof. Only the sizes are outside what the document
defines:

- `out_of_range_0_to_12`: an empty first tree, outside `0 < m`.
- `out_of_range_8_to_4`: a second tree smaller than the first, outside
  `m < n`.

### Changed in this revision

`out_of_range_0_to_12` was previously committed as **`consistent_0_to_12`,
expecting `true`**. That was wrong, and it was a verdict on a comparison that
never happened: the checker returned true for the empty-prefix case without
looking at either root, so it would have returned true for any pair of roots
at all as long as the proof list was empty. Blake Morrison found the
disagreement by running a second implementation against the pinned vectors,
and the fix is the shape he proposed on the SCITT list: put the third value
on what the checker answers rather than tagging the vector.

A runner pinning the old case name will not find it. The rename is
deliberate, so that a harness carrying the old expectation fails loudly
rather than skipping a row.

## Reproducing

Verify the committed vectors with no Vaara dependency:

```
python tests/vectors/transparency_consistency_v0/_check_independent.py
```

Regenerate them with Vaara (the committed JSON is the vector; the checker
verifies whatever is committed):

```
python tests/vectors/transparency_consistency_v0/_generate.py
```
