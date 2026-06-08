# Execution-record conformance summary

**Verdict: CONFORMS**

3 records checked, 3 conform.

## What was checked

Each record was checked against the SEP-2828 execution-record schema for its type, decision or outcome, with no signing key: the wire shape and the record's own self-proving digest. Across the set, calls were checked for duplicates and decisions were paired with their outcomes.

## Records

- Decisions: allow 1, escalate 1
- Outcomes: executed 1

## Findings

Advisory (gaps that do not gate conformance):

- **decision_without_outcome**: 1 allow/escalate decisions have no matching outcome record; an authorised action SHOULD leave a recorded outcome (decision_b.json)

---

No signing key was used to produce this summary. Any party can reproduce every count and finding above from the records alone with `vaara verify-records`.
