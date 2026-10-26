# Digest-only capture

By default Vaara records the arguments of a governed call as the host sent
them: in the audit trail, in the review queue and in the approval request
file. If a call carries something you must not copy, such as patient data or an
account number, you can declare the field. Vaara then stores a digest and a
reference in those three places, and the value stays out of them.

## Turn it on

Create `~/.vaara/capture.json`, or point `VAARA_CAPTURE_POLICY` at another file:

```json
{
  "version": 1,
  "rules": [
    {"tool": "mcp__ehr__*", "fields": ["patient.id", "rows.*.note"], "reference": "ehr"},
    {"tool": "shell.exec", "fields": ["*"]}
  ]
}
```

- `tool` is a glob over the tool name. The default is `*`.
- `fields` are dotted paths through objects. A `*` segment takes every key of an
  object or every item of a list. `*` alone takes every top-level value and
  keeps the keys, so the record still says which fields were passed.
- `reference` is free text that says where the real value lives, for the person
  who has to find it later.

With no file, nothing changes. An edit applies to the next call, with no restart.

## What is stored

Each declared value becomes

```json
{"captured": "digest-only", "digest": "hmac-sha256:<hex>", "reference": "ehr"}
```

where

```
digest = HMAC-SHA256(key, "vaara.capture.v1\n" || path || "\n" || JCS(value))
```

`path` is the position of the value (`rows.2.note`) and `JCS` is RFC 8785. The
key is 32 random bytes, created on first use in `keys/capture-hmac.key` beside
the policy file with mode 0600. It is never written to the trail. Someone who
holds only the trail cannot test a guessed value against a digest. You hold the
key, so you can show that a given value is the one that was passed by
recomputing its digest.

Keep the policy file and the key where the governed agent cannot write them.

## Where it applies

The mask sits where the three stores are written, so each surface that writes
them is covered:

| Store | What changes |
|---|---|
| Audit trail | `parameters` of the `action_requested` record, and `modified_parameters` of a modify decision |
| Review queue | the `parameters` column |
| Approval request file | `parameters`, and the `parameters_sha256` that a signed approval binds, which is now the digest of the masked form |

The pipeline, the Claude Code hook, the MCP proxy and the shell proxy each have a
test. The engine and the calling agent still see the real arguments, so scoring
does not change.

A decision receipt never carried the arguments. Its `argumentsCommitment` covers
the arguments as the trail recorded them, so with capture on it covers the
masked form. `vaara trail verify` and `vaara trail erase` work on these records
as on any other.

## What it does not cover

- The reason text the engine writes, tool results, outcome feedback and the free
  text a reviewer types. Keep personal data out of those.
- Records written before the policy existed, and exports or archives made from
  them.
- A digest of a personal identifier can still be linked back to the person by
  whoever holds the key.
- The model proxy (`vaara proxy`) keeps a request and a response only as
  SHA-256 commitments in its signed receipts. When the proxy runs with a
  pipeline, the tool calls a reply asks for are recorded with their arguments
  and fall under this policy.

## If something is wrong

Failure goes toward not copying. A policy file that exists but cannot be read
or understood withholds every argument value, as `{"captured": "withheld"}`,
and logs a warning. A key that cannot be read or made withholds the value and
writes no digest. A value that cannot be canonicalised is withheld. The call
itself is not stopped.
