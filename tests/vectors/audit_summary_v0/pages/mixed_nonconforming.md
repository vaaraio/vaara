# Execution-record conformance summary

**Verdict: NON-CONFORMING**

2 records checked, 1 conforms.

## What was checked

Each record was checked against the SEP-2828 execution-record schema for its type, decision or outcome, with no signing key: the wire shape and the record's own self-proving digest. Across the set, calls were checked for duplicates and decisions were paired with their outcomes.

## Records

- Outcomes: executed 1

## Findings

No cross-record findings.

## Non-conforming records

- `bad.json` (outcome): status_valid

---

No signing key was used to produce this summary. Any party can reproduce every count and finding above from the records alone with `vaara verify-records`.
