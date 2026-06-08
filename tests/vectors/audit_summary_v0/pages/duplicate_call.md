# Execution-record conformance summary

**Verdict: NON-CONFORMING**

2 records checked, 2 conform.

## What was checked

Each record was checked against the SEP-2828 execution-record schema for its type, decision or outcome, with no signing key: the wire shape and the record's own self-proving digest. Across the set, calls were checked for duplicates and decisions were paired with their outcomes.

## Records

- Outcomes: executed 1, refused 1

## Findings

Required (these gate conformance):

- **duplicate_call**: 2 outcome records pin the same call (attestationDigest sha256:6b23c0d5f35d1b11f9b683f0b0a617355deb11277d91ae091d399c655b87940d, nonce nonce-C); a call MUST be recorded once (r1.json, r2.json)

---

No signing key was used to produce this summary. Any party can reproduce every count and finding above from the records alone with `vaara verify-records`.
