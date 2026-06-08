# Execution-record conformance summary

**Verdict: CONFORMS**

2 records checked, 2 conform.

## What was checked

Each record was checked against the SEP-2828 execution-record schema for its type, decision or outcome, with no signing key: the wire shape and the record's own self-proving digest. Across the set, calls were checked for duplicates and decisions were paired with their outcomes.

## Records

- Outcomes: executed 2

## Findings

Advisory (gaps that do not gate conformance):

- **executed_without_result_commitment**: 1 executed records carry no resultCommitment; an executed action SHOULD commit to its result (r1.json)

---

No signing key was used to produce this summary. Any party can reproduce every count and finding above from the records alone with `vaara verify-records`.
