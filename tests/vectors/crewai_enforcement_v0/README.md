# crewai_enforcement_v0

Conformance vectors for the enforcement claim behind the CrewAI
`GovernanceDecision` / `GovernanceOutcome` contract
([crewAIInc/crewAI#6030](https://github.com/crewAIInc/crewAI/pull/6030)): a
`deny` verdict counts as *enforced* only when a linked outcome attests the tool
did not execute. A `deny` paired with an outcome that shows execution is a
`violation`, the record says deny but the side effect happened anyway. That gap
is the P0 the reducer work has to close; this corpus makes it checkable from the
bytes instead of taken on trust.

`_check_independent.py` reproduces every verdict with no Vaara import, only
`rfc8785` (RFC 8785 JCS) and `cryptography`. A passing run is therefore a
property of the committed bytes, not of the generator. `_generate.py` rebuilds
the fixtures; the signing key is a fixed test scalar so the bytes are stable.

## Signed bytes

Each record is wrapped as `{"record": {...}, "signature": "<128 hex>"}`. The
signature is ES256 (P-256 + SHA-256) over `JCS(record)`; the 128 hex chars are
`r || s`, 32 bytes each. The public key is `keys/es256_public.pem`.

## What each case pins

`cases.json` holds two cases, both a `deny` decision over the same intent
(writing `/etc/passwd`):

- `enforced_deny`: the linked outcome carries `executed: false` and
  `status: "blocked"`. Verdict: `enforced`.
- `violated_deny`: the linked outcome carries `executed: true` and
  `status: "completed"`. Verdict: `violation`.

`expected.json` holds the verdict subset each case must reproduce.

## Derivations

Every digest is `sha256:` + `hex(SHA256(JCS(member_set)))`.

| field | preimage member set |
| --- | --- |
| `params_hash` | `JCS(params)` |
| `intent_digest` | `JCS({action_type, normalized_scope, params_hash})` |
| `intent_ref` | `JCS({schema, agent_id, action_type, normalized_scope, intent_digest})` |
| `receipt_ref` | `JCS({decision_id, intent_ref, seq})` |

The outcome links to the decision by `decision_id`, `intent_ref`, and
`receipt_ref`. An outcome that does not link recomputes to `unlinked`.

## What this does not establish

The corpus proves the enforcement verdict is recomputable from the records, not
that any particular runtime honored it. Binding the verdict to a real blocked
side effect at the tool boundary is the job of the reducer that consumes the
`GovernanceDecision`; these vectors are the reference it can check against.
