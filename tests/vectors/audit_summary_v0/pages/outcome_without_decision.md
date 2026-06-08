# Execution-record conformance summary

**Verdict: CONFORMS**

3 records checked, 3 conform.

## What was checked

Each record was checked against the SEP-2828 execution-record schema for its type, decision or outcome, with no signing key: the wire shape and the record's own self-proving digest. Across the set, calls were checked for duplicates and decisions were paired with their outcomes.

## Records

- Decisions: allow 1
- Outcomes: executed 1, refused 1

## Findings

Advisory (gaps that do not gate conformance):

- **outcome_without_decision**: 1 outcome records have no matching decision record; a recorded action SHOULD trace to the decision that authorised it (outcome_b.json)

---

No signing key was used to produce this summary. Any party can reproduce every count and finding above from the records alone with `vaara verify-records`.
