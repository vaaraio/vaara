# Evidence-bundle verification (design)

Status: implemented in v0.56.0. Additive over v0.55. The 0.6 trust-plane
capstone.

## Problem

The 0.52 to 0.55 line built six verification lenses, each on its own call:

- **identity** (`verify_receipt_identity`, level 2; `verify_receipt_identity_live`,
  level 3): the receipt's `did:web` issuer resolves to a document and the
  signature binds to a key it lists.
- **signature** (`verify_receipt_signature`): the signature verifies under
  given key material.
- **back-link** (`verify_back_link`): the receipt's `backLink` pins the
  SEP-2787 request attestation it answers.
- **inclusion** (`verify_inclusion`): the receipt is in the transparency log
  under a given root.
- **consistency** (`verify_consistency`): the log is append-only across two
  tree heads.
- **revocation** (`check_receipt_revocation`): the issuer or its key was not
  revoked at or before issuance.

Each returns its own verdict. A consumer holding a full evidence bundle had
to call all six, remember which applied to the evidence on hand, and combine
the answers itself. Worse, the combination has a sharp edge that is easy to
get wrong: proving a receipt is *in a log* and *not revoked* says nothing
about *who issued it* unless the signature was also checked. A naive
consumer that treats inclusion as sufficient accepts an unauthenticated
record.

This closes that gap: one entrypoint that runs every applicable lens, threads
the lenses together where they depend on each other, and returns one verdict
with the authenticity edge enforced.

## The bundle

An `EvidenceBundle` is one receipt plus whatever evidence the holder has.
Only the receipt is required; each remaining field feeds one lens, and a
field left `None` makes that lens **not applicable**.

| Field | Lens |
| --- | --- |
| `receipt` | (required) |
| `did_document`, `expected_keyid` | identity |
| `verifying_material` | signature |
| `attestation` | back-link |
| `inclusion`, `log_root`, `inclusion_leaf` | inclusion |
| `consistency`, `consistency_first_root`, `consistency_second_root` | consistency |
| `registry` | revocation |

`inclusion_leaf` defaults to the full canonical receipt bytes, matching what
the log would have appended.

## The verdict

`verify_evidence_bundle(bundle)` returns a `BundleVerdict`:

- `lenses`: one `LensResult` per lens, in a fixed order. Each carries
  `applicable` (did the bundle have the evidence), `ok` (applied and passed),
  and a human `reason`.
- `authenticity_established`: did the identity lens bind the signature, or did
  the signature lens verify it.
- `keyid`: the key the identity lens resolved, when one bound.
- `ok`: the single answer.

### The two rules

1. **A not-applicable lens does not count.** A bundle with no consistency
   proof is not rejected for lacking one; only lenses whose evidence is
   present can fail the verdict.

2. **`ok` is fail-closed on authenticity.** `ok` is true only when
   `authenticity_established` is true **and** every applicable lens passed.
   A bundle that proves inclusion and non-revocation but never verifies the
   signature is not `ok`. An unauthenticated record sitting in a log proves
   nothing about who issued it.

### Lens coupling

The lenses are not independent. Identity runs first because the keyid it
resolves sharpens revocation: a key-scope revocation entry can only match
once the keyid is known. So a bundle carrying a DID document and a key-scope
revocation gets the right answer, while the same bundle verified by signature
alone (no DID document, no keyid) correctly cannot match that key-scope
entry. The keyid falls back to `expected_keyid` when identity does not
resolve one.

The verdict composes the existing lens functions unchanged. It touches
neither the receipt envelope nor any canonicalization, so every existing
conformance vector verifies exactly as before.

## Conformance

`tests/vectors/evidence_bundle_v0/` carries eight bundles spanning every
outcome: all lenses passing, each lens failing in turn, signature-only
authenticity, and the fail-closed `unauthenticated_in_log` case. Each commits
the bundle and the reference verdict. The independent checker
(`_check_independent.py`, standard library plus `rfc8785` and `cryptography`,
no Vaara import) reproduces every verdict, so a second implementation can
consume the capstone without depending on Vaara.

## Scope

Additive. No envelope or canonicalization change; envelope version stays 1.
The entrypoint is a pure consumer of the existing lenses; it issues no new
crypto and defines no new wire shape. Live DID resolution stays out of the
core: the identity lens reads a document the caller already holds, so the
whole verdict is offline and reproducible.
