---
name: vaara-governed-tool-call
description: >
  Put an EU AI Act Article 14 human-oversight checkpoint and an Article 12
  auditable record in front of a high-risk tool call. Use before an agent runs a
  consequential or irreversible action (writing to a clinical/genomic database,
  submitting a record to a regulator, moving money, deleting data, anything a
  human should be able to stop and a regulator should be able to verify later).
  Gates the call to allow / escalate / deny, routes escalations to a human review
  queue, and writes a hash-chained record you can export and verify offline.
---

# Governed tool call (EU AI Act Article 12 + 14)

## Overview

Capability skills let an agent *do* things. This skill records *what it was
allowed to do, on what basis, and what it actually did*, and inserts a human
where the law requires one. It wraps the `vaara` package (no server needed):

- **Article 14 (human oversight):** a call can be escalated to a person who must
  resolve it before the agent proceeds. The deployer can designate specific
  calls as always-escalate.
- **Article 12 (record-keeping):** every decision, escalation, resolution, and
  outcome is appended to a tamper-evident, hash-chained audit trail that exports
  to a signed regulator package and verifies offline.

The decision is Vaara's, governed by thresholds the operator sets; this skill
does not invent verdicts. Authenticity of the final package rests on the signing
key and the optional RFC 3161 time anchor, not on this skill.

## When to use

Use it before any tool call that is consequential, irreversible, or touches
regulated data, especially alongside capability skills (for example clinical or
genomic database skills). Do not use it for read-only, low-stakes calls; gate
the actions a human would want to be able to stop.

## Prerequisites

1. `pip install vaara` (the base install covers gate, escalate, record;
   `pip install 'vaara[attestation]'` is only needed for some export options).
2. Pick two SQLite paths and reuse them across calls: one for the audit trail
   (`--audit-db`), one for the review queue (`--queue-db`).

## Protocol

Run `scripts/governed_call.py` for each governed action.

### 1. Gate the call, before executing it

```
python scripts/governed_call.py gate \
  --agent-id <agent> --tool <tool_name> \
  --params-json '{"...": "..."}' \
  --audit-db audit.db --queue-db queue.db \
  [--mode strict|balanced|eco|performance] [--require-review]
```

Read the JSON verdict and the exit code:

- **`allow` (exit 0):** execute the call, then go to step 3.
- **`escalate` (exit 10):** **STOP. Do not execute.** The call is queued for a
  human. Surface the `action_id` to the user and wait. Proceed only after step 2
  resolves it `allow`.
- **`deny` (exit 20):** **do not execute.** Tell the user it was blocked by
  policy and why.

`--require-review` designates this call for mandatory human oversight: it always
escalates regardless of score. Use it for the operations the deployer has
decided a human must always sign off (this is the Article 14 designation). The
risk is still scored and recorded.

### 2. Human resolves an escalation (done by a person, not the agent)

A reviewer lists, then resolves:

```
vaara review list --db queue.db --status pending
vaara review resolve --db queue.db --queue-id <id> \
  --reviewer <who> --resolution allow|deny \
  --justification "..." --audit-db audit.db
```

`--audit-db` writes the `ESCALATION_RESOLVED` record (Article 14(4)(d) evidence)
into the same chain. On `allow`, the agent may now execute the call; on `deny`,
it must not.

### 3. Record the outcome, after an allowed call ran

```
python scripts/governed_call.py outcome \
  --action-id <from step 1> --agent-id <agent> --tool <tool_name> \
  --result-json '{"...": "..."}' [--severity 0.0] --audit-db audit.db
```

### 4. Export the Article 12 package, when a regulator asks

```
python scripts/governed_call.py export-jsonl --audit-db audit.db --out trail.jsonl
vaara trail export-article12 --trail trail.jsonl --key <ed25519.pem> --out pack.zip
vaara trail verify --zip pack.zip
```

The package carries the full lifecycle (request, decision, escalation,
resolution, execution, outcome), a human-readable report, and a verified hash
chain. Add `--anchor-tsa <url>` to fold in an RFC 3161 time anchor (Article 19
existence-in-time evidence).

## Trust model (read this)

- The verdict reflects Vaara's risk thresholds (`--mode`) and the deployer's
  `--require-review` designations. It is governance, not ground truth.
- The audit trail is tamper-evident by hash chain; tamper-*proof* comes from
  signing the export with a real key and anchoring it. The dev key from
  `vaara keygen --dev` is for evaluation only.
- This skill records and gates. It does not execute your tool or vouch for the
  tool's own correctness.
