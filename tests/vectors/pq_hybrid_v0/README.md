# pq_hybrid_v0 — post-quantum hybrid signing vectors

Conformance vectors for the parallel ML-DSA-65 (FIPS 204) signature an execution
receipt carries alongside its classical ES256 / RS256 signature, and the
quantum-resistance verdict (`tier`) a verifier assigns. See
`docs/design/pq-hybrid-signing-spec.md`.

## The point

A receipt is a durable Article 12 record kept for years, so the live threat is
**trust now, forge later**: a classical signature is forgeable by a future quantum
adversary. v0 adds an ML-DSA-65 signature over the **same** JCS preimage. Both must
verify. `receiptAsserted.sigSuite` commits the hybrid intent *inside* the signed
preimage, so a stripped `pqSignature` is a detectable downgrade rather than a silent
loss of protection.

## Files

- `cases.json` — ten cases, each a wire receipt plus the DID document that lists its
  classical and AKP (ML-DSA) verification methods.
- `expected.json` — the verdict Vaara assigns each case (the non-normative `reason`
  is not part of the fixture).
- `_check_independent.py` — a standalone checker that reproduces every verdict using
  only the standard library, `cryptography`, `rfc8785`, and the one named
  post-quantum dependency `dilithium-py`. It does not import Vaara.
- `_generate.py` — regenerates the vectors. Signatures are randomized, so the
  committed files are the frozen fixtures; regenerate only when the envelope changes.

## The four tiers exercised

| tier | meaning | cases |
|---|---|---|
| `hybrid-verified` | committed suite, both signatures bind the same preimage; quantum- and downgrade-resistant | `es256_hybrid_clean`, `rs256_hybrid_clean` |
| `pqc-present` | a valid ML-DSA signature is attached but `sigSuite` was not committed, so it is strippable | `pqc_present` |
| `classical-only` | no committed suite, no PQC signature; verifiable today, not quantum-resistant | `classical_only` |
| `hybrid-downgraded` | fail-closed: a committed hybrid suite whose PQC member is absent, tampered, wrong-key, inconsistent, or unknown | `downgrade_stripped`, `tampered_pq_sig`, `tampered_body`, `unknown_suite`, `suite_alg_mismatch`, `pq_wrong_key` |

## Run

```
python tests/vectors/pq_hybrid_v0/_check_independent.py   # exit 0 = all matched
```

The Vaara side and this Vaara-free checker both reproduce `expected.json`; that two
independent implementations agree is the conformance claim.
