# build_bundle_v0 conformance vectors

The issuer side of the evidence bundle: assembling the single on-disk document
a verifier reads from the separate pieces an issuer holds. This is the mirror
of `bundle_doc_v0` (the verifier reading that document), and the command under
test is `vaara build-bundle`.

Each `pieces/<name>/` directory holds one issuer's separate artifacts, each in
the conventional file `vaara build-bundle --from-dir` discovers. `documents/`
holds the expected assembled output per name, and `expected.json` carries the
reference `verify_evidence_bundle` verdict per name.

## Piece file conventions

| File | Bundle key | Lens |
| --- | --- | --- |
| `receipt.json` | `receipt` | (required) |
| `did_document.json`, `expected_keyid.txt` | `did_document`, `expected_keyid` | identity |
| `verifying_jwk.json` | `verifying_jwk` | signature |
| `attestation.json` | `attestation` | back-link |
| `inclusion.json`, `inclusion_leaf_hex.txt` | `inclusion`, `inclusion_leaf_hex` | inclusion |
| `consistency.json` | `consistency` | consistency |
| `registry.json` | `registry` | revocation |

The `.json` pieces are JSON objects; the `.txt` pieces are raw scalar strings.
A piece an issuer does not have is simply absent, which leaves its lens not
applicable. This is the same set of conventions `BUNDLE_PIECE_FILES` and
`BUNDLE_PIECE_SCALARS` define in `vaara.attestation._bundle_io`.

## The round-trip under test

Assembling a piece set produces, byte for byte, the `bundle_doc_v0` document a
verifier reads (sorted keys, two-space indent). So the issuer assembles exactly
the file the verifier checks: `build-bundle` then `verify-bundle` is one closed
loop over one file. The eight piece sets are derived from the eight
`bundle_doc_v0` documents (`_generate.py` splits each document back into its
pieces), so the two vector sets stay in lockstep.

## The eight piece sets

- `all_lenses_pass` — every lens applies and passes.
- `revoked_in_time` — the signing key was revoked before issuance.
- `tampered_inclusion` — inclusion proof against a tampered log root.
- `forked_consistency` — consistency proof against a forked second tree head.
- `broken_back_link` — back-link to an attestation the receipt does not answer.
- `signature_only` — no DID document; authenticity via a supplied JWK.
- `unauthenticated_in_log` — included and not revoked, but the signature is
  never checked, so `ok` is false (fail-closed on authenticity).
- `wrong_signature_key` — signature lens with the wrong public key.

A non-`ok` verdict here is the issuer faithfully assembling the evidence it
actually has: `build-bundle` reports the verdict, it does not refuse to write a
partial or failing bundle.

## Independent verification

`_check_independent.py` reads each piece set, assembles it, asserts the bytes
match `documents/<name>.json`, and reproduces the verdict, using only the
standard library plus `rfc8785` and `cryptography`, with no Vaara import. It
reuses the verdict logic from `evidence_bundle_v0/_check_independent.py`.

Run:

    python tests/vectors/build_bundle_v0/_check_independent.py

## Regenerate

    python tests/vectors/build_bundle_v0/_generate.py
