# bundle_doc_v0 conformance vectors

The on-disk evidence-bundle document: the single JSON file a holder hands to a
verifier, and the file `vaara verify-bundle` reads. Each file in `bundles/` is
one self-contained bundle; `expected.json` carries the reference
`verify_evidence_bundle` verdict for each, keyed by file name.

## Document shape

A bundle document is one JSON object. Only `receipt` is required; every other
key feeds one verification lens, and an absent key leaves that lens not
applicable.

| Key | Lens |
| --- | --- |
| `receipt` | (required) |
| `did_document`, `expected_keyid` | identity |
| `verifying_jwk` | signature |
| `attestation` | back-link |
| `inclusion`, `inclusion_leaf_hex` | inclusion |
| `consistency` | consistency |
| `registry` | revocation |

This is exactly the `bundle` object the `evidence_bundle_v0` vectors commit, so
those cases and these files are the same shape. The eight files here are
derived from those cases (`_generate.py` reads `../evidence_bundle_v0/cases.json`),
which is why they are not re-signed.

## The eight files

- `all_lenses_pass` — every lens applies and passes.
- `revoked_in_time` — the signing key was revoked before issuance.
- `tampered_inclusion` — inclusion proof against a tampered log root.
- `forked_consistency` — consistency proof against a forked second tree head.
- `broken_back_link` — back-link to an attestation the receipt does not answer.
- `signature_only` — no DID document; authenticity via a supplied JWK.
- `unauthenticated_in_log` — included and not revoked, but the signature is
  never checked, so `ok` is false (fail-closed on authenticity).
- `wrong_signature_key` — signature lens with the wrong public key.

## Independent verification

`_check_independent.py` reads each file and reproduces its verdict using only
the standard library plus `rfc8785` and `cryptography`, with no Vaara import.
It is the file-level twin of `evidence_bundle_v0/_check_independent.py`, which
parses the aggregate `cases.json`; this one parses the individual files a
verifier is actually handed, and reuses that checker's verdict logic.

Run:

    python tests/vectors/bundle_doc_v0/_check_independent.py

## Regenerate

    python tests/vectors/bundle_doc_v0/_generate.py
