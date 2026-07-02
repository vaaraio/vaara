# CycloneDX-CBOM crypto posture for execution receipts (v0)

Status: v0, additive. The classical receipt envelope, canonicalization, and every
existing conformance vector are untouched. A receipt carries a crypto-posture block
only when an emitter opts in.

## The problem: the record has to say how it is protected

A SEP-2828 execution receipt is a durable EU AI Act Article 12 record, kept across a
retention window measured in years. Whoever audits it years from now needs to answer,
from the signed bytes alone, a question the signature itself does not spell out: *what
cryptography protects this record, and is it quantum-resistant?* The `alg` field names
the classical suite, and `sigSuite` names a hybrid one, but neither states the security
posture in a vocabulary an auditor's tooling already reads.

A Cryptographic Bill of Materials (CBOM) is that vocabulary. CycloneDX 1.6 (ECMA-424)
models cryptographic assets with `cryptoProperties.algorithmProperties`, including a
`primitive`, and a `nistQuantumSecurityLevel` giving the NIST post-quantum security
category. This aligns with the CBOM direction in NIST SP 1800-38 and the migration
inventories the EU PQC roadmap (2035) and CNSA 2.0 (2030) will expect.

This is **not a scanner**. It records the posture of the one algorithm set that signed
this one record. It does not inventory a host's crypto.

## The block, and where it lives

`receiptAsserted.cryptoPosture` (optional object, **inside the signed preimage**):

```json
"cryptoPosture": {
  "assetType": "algorithm",
  "nistQuantumSecurityLevel": 3,
  "algorithms": [
    {"algorithm": "ES256",     "primitive": "signature", "nistQuantumSecurityLevel": 0},
    {"algorithm": "ML-DSA-65", "primitive": "signature", "nistQuantumSecurityLevel": 3}
  ]
}
```

It rides inside `receiptAsserted`, so it is covered by the JCS (RFC 8785) preimage the
classical (and, when present, the ML-DSA) signature covers, right next to `sigSuite`.
Absent means the record predates the block: the envelope is byte-for-byte what it was
before, and every existing vector verifies unchanged.

The top-level `nistQuantumSecurityLevel` is the **effective floor**: the max over
`algorithms`. A hybrid reaches its ML-DSA leg's category (3) because an attacker must
still forge that leg; a classical-only receipt reports 0.

## The algorithm-to-level table

A closed table. An unknown algorithm fails closed rather than defaulting to a reassuring
0, which would understate risk by omission.

| algorithm  | primitive | nistQuantumSecurityLevel |
|------------|-----------|--------------------------|
| HS256      | mac       | 0                        |
| ES256      | signature | 0                        |
| RS256      | signature | 0                        |
| Ed25519    | signature | 0                        |
| ML-DSA-65  | signature | 3                        |

ML-DSA-65 is FIPS 204, NIST security Category 3. The classical suites carry no quantum
resistance (level 0). HS256 is a keyed MAC, so its primitive is `mac`.

## Verification: pure recomputation, no keys

`verify_crypto_posture(receipt)` recomputes the expected posture from the receipt's own
`alg` + `sigSuite` and compares. No keying material, so an independent, Vaara-free
checker reproduces the verdict from the receipt bytes alone:

- `crypto_posture_ok`: the committed posture equals the recomputed one, and any claimed
  ML-DSA leg is backed by a present `pqSignature`.
- `crypto_posture_absent`: no block committed (a classical, pre-CBOM record). Not an
  error; the block is optional and its absence claims nothing.
- `crypto_posture_mismatch`: the committed posture does not match the recomputed one
  (a forged or drifted quantum-resistance claim, e.g. an ML-DSA leg asserted with no
  `sigSuite` to commit it). Because the block is inside the preimage, on a
  signature-verified record this only fires on an internally inconsistent claim, never
  on plain tampering, which breaks the signature first.
- `crypto_posture_downgrade`: the posture commits to an ML-DSA leg (level > 0) but
  `pqSignature` is absent: a signed claim of quantum resistance the record cannot back.

## Relationship to `sigSuite` and `pq_verdict`

`sigSuite` closes the strip-the-`pqSignature` downgrade at the signing layer;
`pq_verdict` grades quantum-resistance tiers against a DID document and its keys. The
crypto posture is the auditor-facing lens on the same facts: it restates them in the
CycloneDX CBOM vocabulary and checks them with zero keying material, so a CBOM consumer
reads the record's posture without a Vaara-specific parser or any key resolution. The
`crypto_posture_downgrade` verdict mirrors the `sigSuite` downgrade commitment in CBOM
terms.

## What v0 does not do

- No SLH-DSA (FIPS 205), ML-DSA-44/87, or classical-key-strength bits: the table covers
  only the suites the receipt path signs with today. Extending it is an explicit version
  bump, not a silent addition.
- No host or dependency inventory. One record, one algorithm set.
- No re-canonicalization or new crypto. The block is data inside the existing preimage.
