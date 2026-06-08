# Record-set conformance vectors, v0

Fixtures for the set-level conformance check on a directory of SEP-2828
execution records: the receiving side an auditor works from. Where
`record_conformance_v0` asks "is this one record well-formed", these
vectors ask the question that only shows up across a pile of records,
possibly from more than one emitter: how many conform, and where are the
gaps in the chain? Like the per-record check, this needs no signing key.

A set is not all one shape. Each record is classified as a **decision**
(`decisionDerived`: allow / block / escalate) or an **outcome**
(`outcomeDerived`: executed / refused / errored) by which derived block it
carries, and checked against that type's schema; a record carrying neither
(or both) is unknown and cannot conform. The shared `backLink` pairs a
decision with its outcome.

Cross-record properties stack on top of per-record conformance:

- **Unique calls** (required). Two *outcome* records pinning the same
  `(attestationDigest, attestationNonce)` recorded the same call twice, a
  replay or a double-write, and this gates conformance. A decision and an
  outcome on one call are the expected *pair*, not a duplicate.
- **Pairing coverage** (advisory). In a set holding both kinds, an
  allow/escalate decision with no matching outcome
  (`decision_without_outcome`), or an outcome with no matching decision
  (`outcome_without_decision`): was every authorised action recorded, and
  was every recorded action authorised?
- **Outcome coverage** (advisory). An `executed` outcome carrying no
  `resultCommitment` is a gap: the action ran and left no committed
  evidence of its result, the EU AI Act Article 12 hole.

Set-level checks run only over records that individually conform, since
the linkage fields of a malformed record cannot be trusted. A set
conforms iff every record conforms and no required finding fired; the
advisory pairing gaps are reported but do not gate.

## Layout

```
sets/<case>/*.json     a directory of candidate records (one set)
expected.json          {case: {conforms, total, conforming, statusCounts,
                                verdictCounts, findings[]}}
_check_independent.py   stdlib only (hashlib + re + json), no Vaara import
```

Each `findings` entry is `{id, severity, records}` with `records` sorted,
so a case is order-independent. `statusCounts` tallies outcome statuses,
`verdictCounts` tallies decision verdicts.

## Cases

- `clean` — two conforming outcome records, distinct calls, no findings.
- `duplicate_call` — two conforming outcomes pinning the same call; one
  required `duplicate_call` finding, set does not conform.
- `executed_gap` — two executed outcomes, one without a result commitment;
  one advisory `executed_without_result_commitment` finding, set still
  conforms (advisory does not gate).
- `mixed_nonconforming` — one conforming record and one malformed; set
  does not conform because a record does not, and cross-record reasoning
  runs only over the conforming one.
- `proper_pair` — a decision and its outcome on the same call; the
  expected pair, so no `duplicate_call`, no pairing gap, set conforms.
- `decision_without_outcome` — a paired decision+outcome plus a lone
  `escalate` decision with no outcome; one advisory
  `decision_without_outcome` finding, set still conforms.
- `outcome_without_decision` — a paired decision+outcome plus a lone
  outcome with no decision; one advisory `outcome_without_decision`
  finding, set still conforms.
