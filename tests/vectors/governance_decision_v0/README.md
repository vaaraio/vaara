# governance_decision_v0 conformance corpus

A reproducible vector set over the CrewAI `GovernanceDecision` / `GovernanceOutcome`
contract proposed in [crewAIInc/crewAI#6030](https://github.com/crewAIInc/crewAI/issues/6030),
plus the completeness layer that `vaara.integrations.crewai` (vaaraio/vaara#283)
emits across a crew run. The contract envelope is CrewAI's. What this corpus pins
is the part an auditor recomputes: the content derivations, the gap-evident
sequence, the terminal seal, and the fail-closed verdicts.

`_check_independent.py` reproduces every verdict with no Vaara import, only
`rfc8785` (RFC 8785 JCS) and `cryptography`. A passing run is therefore a
property of the committed bytes, not of the generator. `_generate.py` rebuilds
the fixtures; the keys are a fixed test scalar so the bytes are stable.

## Signed bytes

Each record is wrapped as `{"record": {...}, "signature": "<128 hex>"}`. The
signature is ES256 (P-256 + SHA-256) over `JCS(record)`; the 128 hex chars are
`r || s`, 32 bytes each. The public key is `keys/es256_public.pem`.

## Derivation preimages

Every digest is `sha256:` + `hex(SHA256(JCS(member_set)))`. JCS (RFC 8785) is the
canonicalizer, so the preimages below are exact and language-independent:

| field | preimage member set |
| --- | --- |
| `params_hash` | `JCS(params)` |
| `intent_digest` | `JCS({action_type, normalized_scope, params_hash})` |
| `intent_ref` | `JCS({schema, agent_id, action_type, normalized_scope, intent_digest})` |
| `target_state_digest` | `JCS(target_state)` |
| `decision_context_hash` | `JCS({policy_refs, target_state_digest, continuation_id, normalization_id})` |
| `receipt_ref` | `JCS({intent_ref, target_state_digest, continuation_id, seq, timestamp_ms, idempotency_key})` |

`intent_ref` carries no timestamp, so the same authorized intent recomputes to
the same identity on a retry. `receipt_ref` carries `seq` and `timestamp_ms`, so
it is unique per execution attempt. That is what makes a replayed outcome
detectable. `params_hash` and `target_state_digest` are the committed leaves the
record carries in place of the raw params / target state; the receipt commits to
a hash, not the cleartext, so the checker takes them as given and recomputes the
four refs derived from them.

## Sealed completeness

Each `stream/<case>/` holds `NNNN-decision.json` + `NNNN-outcome.json` per `seq`,
0-indexed, with `runningCount == seq + 1`. A terminal `GovernanceSeal` carries
the boundary `total`. The four cases isolate what the seal buys you:

| case | held seqs | seal | verdict |
| --- | --- | --- | --- |
| `complete` | 0,1,2,3 | yes | `ok: true` |
| `dropped` | 0,1,3 | yes | `ok: false`, `missingSeqs: [2]` (mid-gap, caught by the running count alone) |
| `tail_sealed` | 0,1,2 | yes | `ok: false`, `missingSeqs: [3]` (suffix drop, caught only because the seal pins the total) |
| `tail_unsealed` | 0,1,2 | no | `ok: true`, the irreducible residual (with no seal, a held prefix looks whole) |

`tail_unsealed` is the point of the seal. A mid-stream gap is self-evident from
the running count, but a dropped tail is invisible without a boundary total to
check against. That is why the seal is not optional.

## Fail-closed verdicts

Each `cases/<name>.json` carries the signed records and the verdict a verifier
reaches by recomputing the mismatch, never by trusting a `decision` field:

| case | recomputed condition | verdict |
| --- | --- | --- |
| `exact_intent_mismatch` | `candidate.intent_ref != approved.intent_ref` | `deny` |
| `target_state_drift` | `intent_ref` equal, `target_state_digest` differs | `revise` |
| `continuation_mismatch` | `intent_ref` equal, `continuation_id` and `decision_context_hash` differ | `deny` |
| `duplicate_outcome` | two outcomes share `receipt_ref` and `idempotency_key` | `deny` |

## Non-ASCII canonicalization

`cases/unicode_scope.json` has a `normalized_scope` of `merchant:café/order:Ünïcode-€`.
A producer that canonicalizes with `json.dumps(sort_keys=True)` escapes the
non-ASCII bytes to `\uXXXX` and computes a different `intent_ref`; RFC 8785 emits
raw UTF-8. The checker recomputes `intent_ref` over these bytes, so a non-JCS
producer fails the vector. This is the cheapest way to catch a canonicalizer that
is "sorted JSON" but not JCS.

## What this corpus claims

One file set covers: the six derivation preimages, completeness with a seal
(mid-gap, suffix drop, and the unsealed residual), four fail-closed verdicts, and
the Unicode trap, checked by a zero-import reproducer. It does **not** claim to
be byte-identical to any other party's governance vectors (e.g. algovoi / LS);
those preimages have not been verified here. The claim is completeness and
reproducibility from the bytes alone. A cross-check against another producer's
fixtures is welcome.
