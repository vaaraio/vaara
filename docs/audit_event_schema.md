# Vaara Audit Event Schema, v1.0

Versioned wire/storage contract for the audit events that flow through
the Vaara execution layer. This document is the schema; the
implementation in `src/vaara/audit/trail.py` is one conforming emitter.
The schema is versioned independently of the Vaara Python package so
downstream consumers (compliance combiners, regulatory exports,
third-party verifiers) can pin to a schema version without coupling to
a runtime version.

## Status and scope

- Schema version: **1.0**
- Status: stable
- Applies to: events appended to `trail.jsonl` and to the JSON shape
  returned by the audit HTTP API.
- Out of scope: the combiner that assembles per-Article evidence
  reports, the signer that wraps an exported trail, the verifier that
  walks the hash chain. Those consume conforming events and have their
  own contracts.

A conforming emitter MUST produce events that satisfy the field
requirements below. A conforming consumer MUST tolerate optional
fields and unknown additive fields under the rules in
[§ Forward compatibility](#forward-compatibility).

## Event envelope

Every audit event is a JSON object with the following fields.

| Field | Type | Required | Description |
|---|---|---|---|
| `record_id` | UUIDv4 string | yes | Identifier unique to this single event. |
| `action_id` | UUIDv4 string | yes | Groups every event that belongs to one action lifecycle (request, score, decision, execute/block, outcome). |
| `event_type` | string enum | yes | Lifecycle stage. Closed set in [§ Event types](#event-types). |
| `timestamp` | number | yes | Unix epoch seconds (UTC), IEEE-754 double. Finite (no NaN, no ±∞). |
| `agent_id` | string | yes | Identity of the agent that submitted the action. Free-form, ≤ 256 bytes. |
| `tool_name` | string | yes | Name of the tool or action under interception. ≤ 512 bytes. |
| `data` | object | no | Event-specific payload. Schema by `event_type`, see [§ Data payloads](#data-payloads). Default `{}`. |
| `regulatory_articles` | array of objects | no | Regulatory provenance of this event. See [§ Regulatory article objects](#regulatory-article-objects). Default `[]`. |
| `previous_hash` | hex string | yes | SHA-256 of the predecessor record's `record_hash`. Empty string for the first record. |
| `record_hash` | hex string | yes | SHA-256 over the canonical encoding of the hashed-fields subset of this record. See [§ Hash chain](#hash-chain). |
| `system_operation` | string | no | prEN ISO/IEC 12792 transparency axis: how the AI system operated at this event. Metadata, not hashed. |
| `data_usage` | string | no | prEN ISO/IEC 12792 transparency axis: what data was consumed. Metadata, not hashed. |
| `decision_making` | string | no | prEN ISO/IEC 12792 transparency axis: how the conclusion was reached. Metadata, not hashed. |
| `limitations` | string | no | prEN ISO/IEC 12792 transparency axis: known constraints. Usually carried out-of-band. Metadata, not hashed. |

## Event types

`event_type` is a closed enum at schema 1.0. Additive values may
appear in a minor version bump; see [§ Forward compatibility](#forward-compatibility).

| Value | Lifecycle position |
|---|---|
| `action_requested` | Agent submitted an action; recorded before processing. |
| `risk_scored` | Scorer produced a risk assessment with conformal prediction interval. |
| `decision_made` | Allow / escalate / deny decided. |
| `action_executed` | Action was actually executed downstream. |
| `action_blocked` | Action was blocked before execution. |
| `escalation_sent` | Action routed to human reviewer. |
| `escalation_resolved` | Human reviewer responded. |
| `outcome_recorded` | Post-execution outcome observed and recorded. |
| `policy_override` | Manual override of a prior automated decision. |

Each event for one action references a single shared `action_id`. The
canonical lifecycle is `action_requested` → `risk_scored` →
`decision_made` → (`action_executed` | `action_blocked` |
`escalation_sent` → `escalation_resolved`) → `outcome_recorded`.
`policy_override` may appear at any point after `decision_made`.

## Hash chain

The chain is SHA-256 over a canonical JSON encoding of a strict subset
of the record. Encoding: `json.dumps(content, sort_keys=True,
separators=(",", ":"), allow_nan=False)`.

Fields included in `content`: `record_id`, `action_id`, `event_type`
(as string), `timestamp`, `agent_id`, `tool_name`, `data`,
`regulatory_articles`, `previous_hash`.

Fields excluded: `system_operation`, `data_usage`, `decision_making`,
`limitations`. Rationale: the four prEN ISO/IEC 12792 transparency
annotations may evolve with the WG4 draft. Excluding them keeps
records hash-stable under re-emission and avoids coupling chain
integrity to a moving annotation schema. A future major schema may
introduce a separate signed-bundle mechanism for transparency tagging.

`previous_hash` of the first record is the empty string `""`.
`record_hash` is computed at append time and stable across
re-serialization.

## Regulatory article objects

`regulatory_articles` is an array of objects. Each object has:

| Field | Type | Required | Description |
|---|---|---|---|
| `domain` | string | yes | Regulatory regime: `EU_AI_ACT`, `DORA`, `NIS2`, `MiFID_II`, `GDPR`. |
| `article` | string | yes | Article reference, e.g. `Article 12(1)` or `Article 9(2)(a)`. |
| `requirement` | string | yes | What the article requires. |
| `how_satisfied` | string | yes | How this event satisfies the requirement. |

The combiner uses `regulatory_articles` to assemble per-Article
evidence reports. Including the regulatory provenance in the
hash-covered fields means that tampering with article attribution
after the fact breaks chain verification.

## Data payloads

`data` carries event-specific structured fields. The schema by
`event_type` is non-exhaustive at 1.0 (consumers MUST accept unknown
keys). Reserved top-level keys:

- `action_requested`: `action_request` (object), `context` (object, optional).
- `risk_scored`: `risk_score` (number in [0, 1]), `interval` (`[low, high]`), `classifier_version` (string), `contributing_signals` (array).
- `decision_made`: `decision` (`allow` | `escalate` | `deny`), `threshold_set` (string), `verdict_inputs` (array).
- `action_executed`: `executor` (string), `duration_ms` (number).
- `action_blocked`: `reason` (string), `blocking_policy` (string).
- `escalation_sent`: `reviewer_queue` (string), `priority` (string).
- `escalation_resolved`: `reviewer_id` (string), `decision` (enum), `reason` (string).
- `outcome_recorded`: `outcome` (`success` | `failure` | `partial` | `unknown`), `feedback` (object, optional).
- `policy_override`: `overrider_id` (string), `prior_decision` (enum), `new_decision` (enum), `reason` (string).

Caller-controlled strings in `data` (`agent_id`, `reason`,
`override_reason`) MUST be treated as untrusted at narrative-rendering
time. The hash chain still covers original values; the renderer
sanitizes for display only.

## Numeric and string discipline

- `timestamp` is IEEE-754 double Unix epoch seconds, UTC. NaN, +∞, -∞
  are rejected at the canonical-JSON boundary.
- Strings are UTF-8. ≤ 256 bytes for `agent_id`, ≤ 512 bytes for
  `tool_name`. `record_id` and `action_id` are UUIDv4 in canonical form.

## Wire and storage encodings

JSONL on disk: one event per line, sorted keys, no trailing
whitespace. The hash chain is computed and verified against this
encoding. JSON over HTTP: the audit API returns events as JSON
objects, optionally wrapped in a pagination envelope; the event
object itself is byte-identical to the JSONL line modulo whitespace.
Other encodings (CBOR, Protobuf) may be defined in sibling specs
without bumping this version, provided they round-trip to the
canonical JSON form.

## Signing and export

A trail is exported as a zip bundle: `trail.jsonl`, `manifest.json`,
`trail.sig`, `signer_pubkey`. Signed message:
`SHA-256(trail.jsonl || manifest.json)`. Reference signing algorithms:
Ed25519 (default), ML-DSA-65 (FIPS 204). This schema does not
constrain the signing algorithm beyond requiring that the export
bundle carry the public key in a form the verifier can consume.

## Forward compatibility

- **Minor (1.x).** Additive only: new `event_type` values, new
  optional fields, new entries inside `data`. Consumers MUST tolerate
  unknown fields.
- **Major (2.0+).** Breaking changes: field removal, renames, change
  of hash-input set, change of canonical encoding. Major bumps ship
  with a migration note and the prior schema remains a valid emission
  target for one release cycle.

Producers SHOULD include a schema version tag in the export manifest.
Events themselves do not carry a per-record schema field, since the
chain integrity guarantee covers re-emission only under the same
schema version.

## Relation to OVERT 1.0 and SEP-2787

OVERT 1.0 Base Envelopes (`vaara.attestation.overt`) are per-action
attestations encoded as deterministic CBOR. SEP-2787 v2 envelopes
(`vaara.attestation.sep2787`) are per-tool-call JSON envelopes carried
in MCP `_meta`. This audit-event schema is the full per-event
lifecycle log. The three coexist: an OVERT or SEP-2787 envelope can
back-link to the audit event that recorded the same action via
`record_id`. See `docs/sep2787-overt-mapping.md` for the
OVERT ↔ SEP-2787 field mapping.

## Reference implementation

- `src/vaara/audit/trail.py`, `signer.py`, `verify.py`, `export.py`.
- `src/vaara/server/schemas.py` (pydantic HTTP wire models).

The reference implementation pins this schema at version 1.0. A
conforming third-party emitter or consumer may target this document
without coupling to the Python implementation.
