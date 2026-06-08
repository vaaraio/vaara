# @vaara/client

Typed JavaScript / TypeScript HTTP client for the [Vaara](https://github.com/vaaraio/vaara) v1 API.

Vaara is a runtime AI agent governance kernel: conformal risk scoring, hash-chained audit trail, EU AI Act article-evidence model, OVERT 1.0 attestation. This package is the JS/TS surface; the Python implementation runs the server.

## Install

```bash
npm install @vaara/client
```

Requires Node.js 18+ (global `fetch`). Works in modern browsers too. Pass your own `fetch` if you want to inject one explicitly.

## Quick start

```ts
import { VaaraClient } from "@vaara/client";

const vaara = new VaaraClient({ baseUrl: "http://localhost:8000" });

const result = await vaara.score({
  tool_name: "tx.transfer",
  agent_id: "agent-007",
  parameters: { to: "0x...", amount: 1000 },
  base_risk_score: 0.6,
});

if (result.decision === "deny") {
  throw new Error(`blocked: ${result.action_id}`);
}
if (result.decision === "escalate") {
  // hand off to human reviewer
}
// execute the tool, then report the outcome:
await vaara.reportOutcome({
  action_id: result.action_id,
  outcome_severity: 0.0,
});
```

## Surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `score(req)` | `POST /v1/score` | Conformal risk score + allow / escalate / deny verdict. |
| `reportOutcome(req)` | `POST /v1/score/outcome` | Feed back the post-execution outcome; drives MWU learning. |
| `appendAuditEvent(req)` | `POST /v1/audit/events` | Append a custom audit record. |
| `getActionChain(id)` | `GET /v1/audit/actions/{id}/chain` | All audit records bound to an action. |
| `verifyAuditChain()` | `POST /v1/audit/verify` | Full-chain hash verification. |
| `reloadPolicy(req)` | `POST /v1/policy/reload` | Atomic hot reload of the running policy (v0.13.0+). |
| `detectInjection(req)` | `POST /v1/detect/injection` | Score text for prompt injection. Backed by vaara-bench-v1 numbers. |
| `detectPII(req)` | `POST /v1/detect/pii` | Email / phone / SSN / IPv4 / credit_card / IBAN. |
| `serverInfo()` | `GET /v1/server` | Server identity and capabilities. |
| `health()` | `GET /v1/health` | Liveness probe. |

## Errors

```ts
import { VaaraClient, VaaraError, VaaraTransportError } from "@vaara/client";

try {
  await vaara.reloadPolicy({ body: badPolicy });
} catch (err) {
  if (err instanceof VaaraError) {
    // Server returned 4xx/5xx with a structured `{ error: { code, message } }`.
    console.error(`Vaara ${err.status} ${err.code}: ${err.message}`);
  } else if (err instanceof VaaraTransportError) {
    // Network failure / non-JSON response. Treat fail-closed, do not
    // assume the server saw the request.
    console.error(err);
  } else {
    throw err;
  }
}
```

`VaaraError.code` values map 1:1 to the Vaara HTTP API spec: `policy_invalid`, `policy_not_configured`, `invalid_request`, and the per-route HTTP error codes documented in [`docs/openapi.yaml`](https://github.com/vaaraio/vaara/blob/main/docs/openapi.yaml).

## Versioning

`@vaara/client` tracks the Vaara server's minor version. v0.15.x covers the v1 wire contract as of Vaara v0.15.0. Breaking wire changes will move the server major; the client follows.

## License

Apache-2.0. See the [LICENSE](https://github.com/vaaraio/vaara/blob/main/LICENSE) in the repository root.
