# `vaara-audit` — third-party auditor CLI

Vaara signs its audit trail with Ed25519 and chains every record with SHA-256. Regulators, internal reviewers, and external auditors need a way to inspect those trails without running the full Vaara agent stack. `vaara-audit` is that CLI.

```
vaara-audit verify     <trail.zip> [--pubkey KEY] [--json]
vaara-audit inspect    <trail.zip> [--agent ID] [--event-type T] [--since TS] [--until TS] [--limit N] [--json]
vaara-audit stats      <trail.zip> [--group-by agent|event_type|decision|category] [--json]
vaara-audit anomalies  <trail.zip> [--rules all|missing_completion|timestamp_regression|rate_burst|unknown_spike] [--json]
```

All subcommands operate on a signed trail zip produced by `vaara trail export` or programmatically via `vaara.audit.export.export_signed`. The CLI is strictly read-only.

## `verify`

Checks manifest schema, jsonl hash, manifest hash, Ed25519 signature, and walks the hash chain. Exit code `0` on pass, `1` on fail, `2` on IO errors.

```
$ vaara-audit verify trail.zip
trail:    trail.zip
status:   [PASS]
records:  143
signer:   f0df38b7b608de4a...
```

Pass `--pubkey signer.pub.pem` with a key received out-of-band to check the signer's identity, not just internal consistency. The zip ships its own public key for offline self-verification, but a malicious trail could ship any key, so an auditor should compare against an expected fingerprint.

## `inspect`

Streams records matching every filter (AND semantics). Default output is tabular. `--json` emits one payload per line for piping into `jq`.

```
$ vaara-audit inspect trail.zip --agent agent-0 --event-type decision_emitted --limit 5
seq    timestamp                  agent_id  event_type         decision  action_type
-----  -------------------------  --------  -----------------  --------  -----------
0      2026-04-22T11:30:01.123Z   agent-0   decision_emitted   allow     tx.transfer
0      2026-04-22T11:30:04.871Z   agent-0   decision_emitted   escalate  data.delete
...
matched 5 / 143 records
```

## `stats`

Aggregates by a grouping key. Emits counts, percentages, and mean risk scores where available.

```
$ vaara-audit stats trail.zip --group-by decision
group_by:  decision
total:     143
decision   n    pct     mean_risk
---------  ---  ------  ---------
allow      101  70.6%   0.082
escalate   31   21.7%   0.351
deny       11   7.7%    0.732
```

## `anomalies`

Four rule-based detectors:

- `missing_completion` — `action_requested` without a matching `decision_emitted`. Flags trail truncation, enforcement crashes, or gate bypasses. Skipped if the trail has no `decision_emitted` events at all (pure request logs are not flagged).
- `timestamp_regression` — a record with a timestamp earlier than the previous record for the same agent. Suggests clock skew, replay, or tampering that the signature check missed (it would not — but useful belt-and-braces).
- `rate_burst` — agent emits ≥20 records in ≤10 seconds. Default thresholds catch automated loops running away; tune in a follow-up PR if false positives appear on busy agents.
- `unknown_spike` — ≥25% of an agent's recent 50-record window has `action_type=unknown`. Suggests the agent is using tools Vaara has no registered classification for, which is a classification gap worth investigating.

All rules can run together (`--rules all`, default) or in isolation (`--rules rate_burst,unknown_spike`). Exit code `0` if no findings, `1` otherwise.

```
$ vaara-audit anomalies trail.zip
trail:    trail.zip
rules:    ['missing_completion', 'timestamp_regression', 'rate_burst', 'unknown_spike']
records:  143
findings: 2

[rate_burst] agent_id=bot-7, window_seconds=10, count=24, threshold=20, window_start=2026-04-22T11:41:03Z, window_end=2026-04-22T11:41:08Z
[missing_completion] agent_id=agent-0, action_id=a3f, tool_name=send_email, record_id=...
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed / no findings |
| 1 | Verification failed OR anomalies detected |
| 2 | Usage / IO error (missing file, bad zip) |

Every subcommand supports `--json` for pipeline integration.

## Relationship to `vaara` CLI

The existing `vaara trail verify` subcommand provides the same verify function wrapped in the main CLI. `vaara-audit` is the third-party entry point — a standalone name for the regulator-facing workflow, with richer `inspect`, `stats`, and `anomalies` capabilities beyond verify-only.
