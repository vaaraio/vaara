# SEP-2828 conformance statement

**Statement: NON-CONFORMING**

Checked against corpus `sep2828-execution-record-conformance` version 1.0.1 (corpusDigest `sha256:5b46b5d51a024d6c9c7d6e48889091e561bad43e66730486c6a581ed9315bd08`).

## Corpus integrity

Verified: all 32 fixture files match `MANIFEST.json` and the corpusDigest recomputes.

## Self-test

This implementation's keyless conformance check reproduced 17 of 17 recorded verdicts.

- `record_conformance_v0`: 10/10 reproduced
- `record_set_v0`: 7/7 reproduced

## Your records

2 records checked, 1 conforms; your records do NOT conform.

Non-conforming records:

- `bad.json`: status_valid

---

This statement is keyless and reproducible. Anyone holding the same corpus version can re-run `vaara conformance-statement` and reach the same verdict. It covers the wire schema, the record's self-proving digest, and the cross-record set properties; it is not signature verification, issuer trust, or time-anchor verification, which need external material and are checked separately.
