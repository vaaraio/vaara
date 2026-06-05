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
- `expected.json`: the expected `consistent` verdict per case.

## Cases

Positive cases (`consistent: true`) cover the sizes the algorithm most often
gets wrong: an empty prefix (`0 to 12`, empty proof), a power-of-two prefix
(`1 to 12`, `8 to 12`), non-power-of-two prefixes (`3 to 12`, `7 to 12`,
`5 to 9`), and identical trees (`12 to 12`, empty proof).

Negative cases (`consistent: false`) keep a genuine proof but corrupt one
input, so a checker that always returned `true` is caught:

- `tampered_proof_hash_3_to_12`: one sibling hash in the proof is flipped.
- `forked_second_root_3_to_12`: the proof is checked against a `second_root`
  taken from an unrelated (forked) log, the rewrite a consistency proof exists
  to detect.

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
