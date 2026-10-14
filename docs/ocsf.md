# Vaara decisions in your SIEM: OCSF export

`vaara trail export-ocsf` writes every decision on a trail as an
[OCSF](https://schema.ocsf.io/) 1.9.0 API Activity event (class 6003), one
JSON object per line. Splunk, Microsoft Sentinel, Amazon Security Lake and any
other OCSF consumer can ingest them without a Vaara-specific parser.

```bash
# Straight from the audit database
vaara trail export-ocsf --db ~/.vaara/claude-code/audit.db --out vaara-ocsf.jsonl

# Or from a trail JSONL
vaara trail export-ocsf --trail trail.jsonl --out vaara-ocsf.jsonl --hostname build-01
```

`--hostname` names the host that made the decisions. The trail does not record
it, so by default the export uses the name of the host it runs on.

## What each event says

Each tool call an agent asked to make is an API call, and Vaara's decision on
it is the security control's outcome. The events use the `security_control`
and `ai_operation` profiles.

| Vaara | OCSF |
|---|---|
| tool name | `api.operation`, with `activity_id` 99 and `activity_name` "Tool Call" |
| agent | `ai_agent.uid` and `ai_agent.name` |
| session | `ai_agent.instance_uid` and `actor.session.uid` |
| allow | `action_id` 1 Allowed, `disposition_id` 1 Allowed |
| deny | `action_id` 2 Denied, `disposition_id` 2 Blocked, `status_id` 2 Failure, ground in `status_detail` |
| escalate (held for a person) | `action_id` 99 "Escalate", `disposition_id` 14 Delayed |
| risk score | `risk_score` from 0 to 100, and `severity_id` 1 to 4 by the score |
| policy | `policy.name` (the deny rule or policy id) and `policy.uid` (the decision's policy digest) |
| trail record | `metadata.uid`, `metadata.original_event_uid`; the action id is `metadata.correlation_uid` |

`is_alert` is true for deny and escalate. `unmapped.vaara` carries the fields
OCSF has no attribute for: the record hash and the hash before it, the
recorded ground, an input a check could not reach, and the earlier actions
that completed a sequence.

## Where integrity lives

An OCSF event in a SIEM is a copy. The events carry no `attestation_list`,
because an OCSF attestation is a signature over the OCSF event itself, and
Vaara signs its receipts and chains its trail instead. Each event names the
trail record it came from (`metadata.uid`, `unmapped.vaara.record_hash`), so
an analyst who finds a decision in the SIEM can check it against the signed
receipt or the trail, offline, without trusting the SIEM or Vaara. See
[verifying-evidence.md](verifying-evidence.md).
