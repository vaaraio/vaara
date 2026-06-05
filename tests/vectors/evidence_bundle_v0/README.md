# evidence_bundle_v0 conformance vectors

One end-to-end `verify this evidence bundle` entrypoint. See
`docs/design/evidence-bundle-spec.md`.

The 0.52 to 0.55 line added each verification lens on its own: resolvable
identity, the receipt signature, the back-link to the request attestation,
transparency-log inclusion, append-only consistency, and cross-stack
revocation. Each was a separate call returning a separate verdict. These
vectors drive the single `verify_evidence_bundle` entrypoint that runs every
lens whose evidence is present and returns one verdict.

A lens with no evidence is **not applicable**, not a failure: a bundle
without a consistency proof is not rejected for lacking one. But `ok` is
fail-closed on authenticity. It is true only when the receipt signature was
established (the identity lens bound it to a document key, or the signature
lens verified it under supplied key material) **and** every applicable lens
passed. A receipt that is merely present in a log, with no signature ever
checked, is not `ok`.

## Files

- `cases.json`: one shared ES256 receipt (`did:web` issuer) and, per case,
  the evidence bundle: an optional DID document, supplied verifying key,
  request attestation, transparency-log inclusion proof and root, consistency
  proof and two tree heads, and revocation registry. The receipt sits at a
  non-trivial log index so the inclusion proof carries real siblings.
- `expected.json`: per case, the reference verdict: `ok`,
  `authenticity_established`, and per lens whether it applied and passed.

## Cases

- `all_lenses_pass`: every lens applies and passes. `ok` is true.
- `revoked_in_time`: the signing key was revoked before issuance. The
  revocation lens fails, so `ok` is false though every other lens passes.
- `tampered_inclusion`: the inclusion proof is checked against a tampered log
  root. The inclusion lens fails.
- `forked_consistency`: the consistency proof is checked against a forked
  second tree head. The consistency lens fails.
- `broken_back_link`: the bundle carries an attestation the receipt does not
  answer. The back-link lens fails.
- `signature_only`: no DID document, only a supplied public key.
  Authenticity is established by the signature lens alone and `ok` is true.
- `unauthenticated_in_log`: a valid inclusion proof and a clean registry, but
  no DID document and no verifying key. Authenticity is never established, so
  `ok` is false even though the applicable lenses pass. This is the
  fail-closed case.
- `wrong_signature_key`: the supplied public key is not the signer's. The
  signature lens fails and authenticity is not established.

## Reproduce

Independent checker (standard library plus `rfc8785` and `cryptography`, no
Vaara import):

```
python tests/vectors/evidence_bundle_v0/_check_independent.py
```

Exit code 0 means every case's recomputed verdict matched the reference.
Regenerate the cases (ECDSA signatures are randomized, so signatures change
but verdicts do not) with:

```
python tests/vectors/evidence_bundle_v0/_generate.py
```
