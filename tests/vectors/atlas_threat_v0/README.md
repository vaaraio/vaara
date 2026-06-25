# atlas_threat_v0 conformance corpus

A reproducible vector set grounding `vaara.receipt/v1` against
[MITRE ATLAS](https://atlas.mitre.org/) AI agent threat patterns.
Each fixture proves that the signed receipt bytes are sufficient for a
verifier to detect the named attack — without trusting the agent, the
framework, or any Vaara library.

`_check_independent.py` reproduces every verdict with no Vaara import,
only `rfc8785` (RFC 8785 JCS) and stdlib. A passing run is a property
of the committed bytes, not of the generator.

## Threat mapping

| case | attack pattern | verdict |
| --- | --- | --- |
| `pos_clean_execution` | — (control) | `ok` |
| `neg_injected_args` | Prompt Injection — args mutated after authorization | `args_tampered` |
| `neg_tool_substitution` | Unauthorized tool use — actionType changed at runtime | `tool_mismatch` |
| `neg_replay` | Replay — valid receipt re-presented outside freshness window | `stale` |
| `neg_scope_escalation` | Privilege escalation — runtime scope exceeds authorized boundary | `scope_exceeded` |

## Receipt structure

Each `receipt` is signed with HMAC-SHA256 over the JCS (RFC 8785) canonical
form of all receipt fields excluding `signature`. The signing key is a
32-byte test constant (`y` × 32); corpus fixtures are not production credentials.

Fields bound by the signature:

| field | role |
| --- | --- |
| `agentId` | identity of the acting agent |
| `actionType` | the declared tool / action (e.g. `file.read`) |
| `argsCommitment` | SHA-256 over JCS of the authorized args (double-hash, see below) |
| `scope` | the authorized resource boundary |
| `timestampMs` | issuance time in milliseconds since Unix epoch |
| `seq` | monotonic per-agent sequence number |
| `iss`, `sub`, `schema`, `version` | envelope metadata |

## argsCommitment derivation

```
step1 = "sha256:" + hex(SHA256(JCS(args)))
commitment = "sha256:" + hex(SHA256(JCS({"digest": step1})))
```

JCS (RFC 8785) is the canonicalizer. The double-hash mirrors the
`make_args_digest` derivation in `vaara.attestation._sep2787_canonical`.

## Verification contract

A verifier receives a `receipt` and an `authorization` commitment. It:

1. Verifies the HMAC-SHA256 signature over JCS of the receipt fields.
2. Checks `receipt.actionType == authorization.actionType` — else `tool_mismatch`.
3. Checks `receipt.agentId == authorization.agentId` — else `agent_mismatch`.
4. Checks `receipt.argsCommitment == authorization.argsCommitment` — else `args_tampered`.
5. Checks `receipt.scope == authorization.scope` — else `scope_exceeded`.
6. Checks `abs(now_ms − receipt.timestampMs) ≤ freshness_window_ms` — else `stale`.

Checks are ordered: a tampered signature is caught before any field
comparison. Each negative vector isolates exactly one failing check so
the detection boundary is unambiguous.

## Pinned timestamps

| constant | value | meaning |
| --- | --- | --- |
| `IAT_MS` | `1779200000000` | issuance time (2026-05-19T14:13:20Z) |
| `NOW_MS` | `IAT_MS + 30 000` | verification clock for passing cases |
| `NOW_STALE_MS` | `IAT_MS + 120 000` | verification clock for `neg_replay` |
| `FRESHNESS_WINDOW_MS` | `60 000` | maximum age in milliseconds |

`neg_replay` uses the same `IAT_MS` receipt but a `now_ms` 120 s later,
which exceeds the 60 s window. The receipt is self-consistent and correctly
signed; only staleness condemns it.
