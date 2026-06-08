# Record-set conformance vectors, v0

Fixtures for the set-level conformance check on a directory of SEP-2828
execution records: the receiving side an auditor works from. Where
`record_conformance_v0` asks "is this one record well-formed", these
vectors ask the question that only shows up across a pile of records,
possibly from more than one emitter: how many conform, and where are the
gaps in the chain? Like the per-record check, this needs no signing key.

Two properties stack on top of per-record conformance:

- **Unique calls** (required). Each record pins the attestation it
  answers through `backLink`. Two records pinning the same
  `(attestationDigest, attestationNonce)` recorded the same call twice, a
  replay or a double-write. A set that double-counts a call is not a
  faithful record, so this gates set conformance.
- **Outcome coverage** (advisory). An `executed` record carrying no
  `resultCommitment` is a gap: the action ran and left no committed
  evidence of its result. Reported, not gating, but exactly the hole a
  regulator looks for under EU AI Act Article 12.

Set-level checks run only over records that individually conform, since
the linkage fields of a malformed record cannot be trusted. A set
conforms iff every record conforms and no required finding fired.

## Layout

```
sets/<case>/*.json     a directory of candidate records (one set)
expected.json          {case: {conforms, total, conforming,
                                statusCounts, findings[]}}
_check_independent.py   stdlib only (hashlib + re + json), no Vaara import
```

Each `findings` entry is `{id, severity, records}` with `records` sorted,
so a case is order-independent.

## Cases

- `clean` — two conforming records, distinct calls, no findings.
- `duplicate_call` — two conforming records pinning the same call; one
  required `duplicate_call` finding, set does not conform.
- `executed_gap` — two executed records, one without a result commitment;
  one advisory `executed_without_result_commitment` finding, set still
  conforms (advisory does not gate).
- `mixed_nonconforming` — one conforming record and one malformed; set
  does not conform because a record does not, and cross-record reasoning
  runs only over the conforming one.
