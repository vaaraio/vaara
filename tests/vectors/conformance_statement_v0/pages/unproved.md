# SEP-2828 conformance statement

**Statement: UNPROVED**

At least one check could not be reached, so this statement does not establish the property either way. That is a different result from a check that ran and failed, and the sections below name which.

Checked against corpus `sep2828-execution-record-conformance` version 1.0.0 (corpusDigest `sha256:0baa437da95d6ffb6bda3185d657385b7738c79daba77819acc0c6280ac0ed7a`).

## Corpus integrity

Verified: all 32 fixture files match `MANIFEST.json` and the corpusDigest recomputes.

## Self-test

This implementation's keyless conformance check reproduced 17 of 17 recorded verdicts.

- `record_conformance_v0`: 10/10 reproduced
- `record_set_v0`: 7/7 reproduced

## Your records

2 records checked, 2 conform; your records are unproved: what was read conformed, but the set was not complete.

> Unproved: 1 file(s) could not be read, so the set check never saw them and claims nothing about them: broken.json

---

This statement is keyless and reproducible. Anyone holding the same corpus version can re-run `vaara conformance-statement` and reach the same verdict. It covers the wire schema, the record's self-proving digest, and the cross-record set properties; it is not signature verification, issuer trust, or time-anchor verification, which need external material and are checked separately.
