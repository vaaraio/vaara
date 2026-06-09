# cross_org_handoff_v0 conformance vectors

Hand one organisation's signed execution record to another organisation's
regulator, in a single self-contained file the regulator verifies offline,
years later, under a key that has since rotated out. These vectors pin the
`verify_handoff` verdict so an independent implementation can reproduce it.

Each case is a *handoff package*: `evidence` (the SEP-2828 record, the archived
DID document, the key history, revocations, and an optional eIDAS RFC 3161
anchor), a `manifest` of pinned component digests, the `manifest_digest`
fingerprint, and an optional `holder_attestation`. A case also carries the
verify inputs a regulator supplies: `anchoredTime` (a time it verified out of
band from the anchor token), `trustedDidDocument` (the identity it trusts as the
producer's), and `strict`.

## Files

- `cases.json`: the packages and their verify inputs.
- `expected.json`: the verdict fields Vaara produces, per case (the
  non-normative `reason` is not pinned).
- `_check_independent.py`: a checker that reproduces every verdict with only
  `cryptography` and `rfc8785`, no Vaara import.
- `_generate.py`: regenerates `cases.json` and `expected.json` from Vaara.

## What the cases cover

- `clean_no_anchor`, `anchored_not_verified`, `corroborated`: the verifiable
  tier, an anchor present but not yet cryptographically verified, and the
  corroborated tier once a verified anchor time is supplied.
- `pinned_corroborated`, `strict_pass`, `strict_unmet_no_anchor`: pinning the
  producer identity against a trusted document, and the regulator-grade strict
  gate (only a corroborated record with recorded windows, an affirmative
  revocation source, and a pinned identity passes).
- `anchor_over_different_record`: a genuine anchor whose imprint is **not**
  `sha256(jcs(record))`: it does not bind, so the record stays verifiable and is
  never silently corroborated.
- `tampered_did_document`, `tampered_manifest_producer`: a component edited
  after sealing: the recomputed digest disagrees with the manifest, integrity
  fails and names the drift.
- `holder_attested`, `holder_attestation_failed`: a self-supplied holder
  custody signature, valid and corrupted; custody is reported but never gates
  the record verdict.
- `noncanonical_key_history`: an override whose `keys` are in non-canonical
  order; the pinned digest is the **model** digest (canonicalised, re-sorted),
  so a raw-bytes hash of the supplied block would mismatch.
- `signed_after_retirement`: the record's `iat` is past the key's `validUntil`:
  the record is not verifiable even though the package is internally consistent.

## The one boundary

The checker reproduces every digest, the integrity check, the retained-record
arithmetic, the anchor-to-record byte binding, the holder attestation signature,
and the producer pin. It does **not** verify the RFC 3161 CMS token signature
itself, which needs the timeanchor extra and a trusted Time-Stamp Authority
chain; the attested time is taken pre-verified from the case, the same published
boundary as `key_rotation_v0`. The record-to-anchor binding is checked here in
full.

## Trust note

A green verdict proves the package is internally consistent and that the
producer's signature verifies against the **enclosed** DID document. It does not
prove that document is genuinely the producer's: a holder assembles the package
and controls both the components and their pinned digests. Authenticity rests on
establishing the producer's identity out of band (`producer_identity_basis`
stays `self_asserted_unpinned` until a `trustedDidDocument` pins it), and on the
eIDAS anchor, which a holder cannot forge.

## Regenerate

```
python tests/vectors/cross_org_handoff_v0/_generate.py
python tests/vectors/cross_org_handoff_v0/_check_independent.py
```

ECDSA signatures are randomized, so regenerating overwrites the cases with fresh
but equivalent vectors.
