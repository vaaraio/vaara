# Vaara as an ACS Guardian

The OWASP [Agent Control Standard](https://github.com/GenAI-Security-Project/agent-control-standard)
(ACS) v0.1.0 is a wire protocol between an agent framework (the Observed Agent) and a separate
Guardian that allows or denies each step before it runs. `vaara.integrations.acs_guardian` is a
Guardian. A tool call that arrives over ACS gets the same decision it gets from Vaara's host hooks
or the MCP proxy: the same deny rules, the same scorer, the same trail.

```
vaara-acs-guardian --db vaara_audit.db --secret-file acs.secret
# Guardian listening at http://127.0.0.1:8787/acs
```

## What it answers

| Method | Answer |
|---|---|
| `handshake/hello` | ServerHello: version 0.1.0, `on_decision_failure: deny`, the profiles below |
| `steps/toolCallRequest` | Vaara's decision. allow is `allow`, deny is `deny` with the rule or ground in `reason_codes`, escalate is `deny` with `approver_unavailable` |
| `steps/toolCallResult` | `allow`, and the outcome is reported on the request's action |
| `steps/sessionStart`, `agentTrigger`, `turnStart`, `userMessage`, `agentResponse`, `subagentStart`, `subagentStop`, `turnEnd`, `sessionEnd` | `allow` |
| `system/ping` | `allow`, no signature needed, outside the chain |

Every tool-call response carries `policy_data.vaara` with the Vaara `action_id` and the
`record_hash` of the decision on Vaara's trail, and `policy_references` with the decision's
`policy_digest`. That record is signed and hash chained on Vaara's side, so a party who holds no
ACS key can check it offline (`vaara trail verify`, `vaara receipt verify`).

## The ACS chain and signatures

Each content step writes a ContextEntry by the ACS rule (spec 8.2: SHA-256 over the RFC 8785 form
of the entry without its two hashes, then the previous hash), with `request_hash` set on every
entry, and the response returns the new head as `chain_hash`.

With `--secret-file` (at least 32 bytes) every request must carry an HMAC-SHA256 signature over the
RFC 8785 form of the envelope without its `signature`, and every response is signed the same way,
so the published `chain_hash` is covered. The per-session key is HKDF-SHA256 over the secret. ACS
v0.1.0 does not fix the HKDF salt or info; this Guardian uses no salt and
`info = "acs/v0.1 hmac " + session_id`. A client must use the same derivation.

Requests outside a five-minute clock window, and a repeated `request_id` or `nonce` within a
session, are refused with the ACS error codes.

## Profiles

| Secret | `profiles_accepted` (of those the client offers) |
|---|---|
| none | `acs-audit` |
| set | `acs-core`, `acs-audit` |

Without a secret the Guardian declares no signature algorithm and does not claim `acs-core`, which
requires one. A non-loopback `--host` is refused without a secret.

## Limits of this version

- Sessions and their ACS chains live in the Guardian's memory; a restart starts new chains. Vaara's
  own trail persists as usual.
- `ASK` is not offered: a call Vaara would hold for a human is refused with `approver_unavailable`
  (spec 9.2), and nothing is lost from Vaara's record.
- `MODIFY`, `protocols/MCP/*` wrapping, AgBOM and Trace events are not implemented. Calls to methods
  this Guardian does not evaluate get `-32601`.
- HMAC-SHA256 only; the asymmetric ACS-Crypto algorithms are not offered.
