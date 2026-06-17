# Adapters and interfaces

How to put Vaara into an existing agent stack: native framework adapters, cloud and OSS guardrail signal adapters, the MCP proxy, the HTTP API, and the TypeScript client.

## Framework adapters

Native adapters in `src/vaara/integrations/` route the major Python agent frameworks through Vaara's pipeline. Each intercepts via the framework's own callback or hook surface, scores, gates, and emits the same audit events as a direct `pipeline.intercept()`. Frameworks are not hard dependencies (lazy import, duck typing), so audit records hash-chain together regardless of which one the action came through.

| Framework | Entry point | Use |
|---|---|---|
| LangChain | `VaaraCallbackHandler`, `vaara_wrap_tool` | Slots into `config={"callbacks": [...]}` or wraps per-tool |
| CrewAI | `VaaraCrewGovernance` | Wraps a crew so every agent action passes through scoring + audit |
| OpenAI Agents SDK | `VaaraToolGuardrail`, `vaara_wrap_function` | Function-tool wrap, compatible with Responses API and Agents-SDK tracing |
| MCP server | `vaara.integrations.mcp_server` | Exposes scoring, audit, policy reload as MCP tools |

## Upstream-signal adapters (cloud + OSS guardrails)

Adapters route findings from cloud and OSS guardrails into Vaara's audit trail and OVERT envelope. The filter runs in the deployer's environment; Vaara records the verdict, normalises 68 provider categories onto a shared vocabulary, and tags each finding against the relevant AI Act articles. Each adapter returns a `ContentSafetyFinding` the deployer routes into `pipeline.intercept(context=finding.to_audit_context())`. Article-by-article mapping in [COMPLIANCE.md](COMPLIANCE.md).

| Provider | Adapter | Extra | Wraps |
|---|---|---|---|
| AWS Bedrock Guardrails | `BedrockGuardrailsAdapter` | `vaara[bedrock]` | `ApplyGuardrail` across five Bedrock policy buckets |
| Azure AI Content Safety | `AzureContentSafetyAdapter` | `vaara[azure-content-safety]` | `analyze_text`, Prompt Shields, Protected Material, Groundedness |
| GCP Model Armor | `GcpModelArmorAdapter` | `vaara[gcp-model-armor]` | `sanitize_user_prompt`, `sanitize_model_response` |
| NVIDIA NeMo Guardrails | `NemoGuardrailsAdapter` | `vaara[nemo-guardrails]` | `GenerationResponse.log.activated_rails` (input / dialog / output / retrieval) |
| Guardrails AI | `GuardrailsAIAdapter` | `vaara[guardrails-ai]` | `ValidationOutcome.validation_summaries` from `Guard.parse` / `Guard.validate` |
| LLM Guard | `LLMGuardAdapter` | `vaara[llm-guard]` | `scan_prompt` / `scan_output` |
| Rebuff | `RebuffAdapter` | `vaara[rebuff]` | `DetectResponse` across heuristic, model, vector layers + canary-word leak check |

Mapping table at `src/vaara/integrations/_content_safety_articles.py`. Rationale in [COMPLIANCE.md](COMPLIANCE.md#cloud-guardrail-adapter-pattern).

## MCP proxy

`VaaraMCPProxy` sits between an MCP client (Claude Code, Cursor, any MCP host) and an upstream MCP server. Every `tools/call` routes through Vaara's pipeline before reaching the upstream: allowed calls forward transparently and report the outcome back to the scorer, blocked calls return an MCP `isError: true` with the reason. The handshake and `notifications/*` forward unchanged.

```bash
vaara-mcp-proxy \
  --upstream npx --upstream-arg -y --upstream-arg @sap/mdk-mcp-server \
  --db ./mcp_audit.db
```

Point your MCP client at the proxy instead of the upstream; the audit chain captures every call without changing client or upstream behavior. Upstreams can be local (`--upstream` launches a local stdio server) or remote (`--upstream-url NAME=URL` over Streamable HTTP). This is distinct from `mcp_server`, which exposes Vaara itself as a tool.

Worked examples: [`examples/github-mcp-proxy-demo/`](../examples/github-mcp-proxy-demo/) (Vaara in front of `github/github-mcp-server`, 42 tools) and [`examples/sap-mcp-proxy-demo/`](../examples/sap-mcp-proxy-demo/) (community SAP MCP servers).

### Fleet shape: one proxy, many upstreams, multi-tenant policy

`vaara-mcp-proxy` also runs over Streamable HTTP with fan-out, so one process can serve a fleet:

```bash
vaara-mcp-proxy \
  --transport http --http-host 127.0.0.1 --http-port 8765 \
  --upstream 'github=npx -y @github/mcp-server' \
  --upstream 'sap=npx -y @sap/mdk-mcp-server'
```

Each `POST /mcp` reads two headers: `X-Vaara-Upstream` picks the upstream slot, `X-Vaara-Tenant` scopes the policy, audit chain, and OVERT envelope. Single-upstream deployments keep the silent-default contract; multi-upstream deployments require `X-Vaara-Upstream` per call and return 400 with the slot list when it is missing. `vaara serve --policy-dir DIR` loads one policy per file (filename stem becomes `tenant_id`, `default.yaml` is the fallback) and hot-reloads per tenant.

### Operator perimeter and request attestation

Repeatable `--allow-tool` / `--deny-tool` flags (and the same for resources and prompts) filter the MCP surface. Filtered tools are dropped from `tools/list` before the client sees them and any matching call is rejected at the perimeter without contacting the upstream. Denylist wins on overlap; no flags means passthrough. Every allowed `resources/read` and `prompts/get` writes a request+decision audit pair so a regulator can reconstruct exactly what the agent read.

OVERT envelopes per interaction turn on with `--overt-signing-key`, `--overt-operator-key`, `--overt-receipts-dir`. SEP-2787 request attestation paired with an execution receipt turns on with `--attest-signing-key PATH` and `--attest-receipts-dir DIR`: each allowed call writes a pre-execution attestation and a post-execution receipt linked by a `backLink` digest. Key type auto-detects from the file (EC P-256 = ES256, RSA = RS256, raw bytes = HS256). Generate and verify offline:

```bash
vaara keygen --attest --out attest_key.pem
vaara attest verify  0000000001-ab12cd34-attest.json  --pubkey-file attest_key.pem.pub
vaara receipt verify 0000000001-ab12cd34-receipt.json --attestation 0000000001-ab12cd34-attest.json --pubkey-file attest_key.pem.pub
```

Both verifiers exit non-zero on any failed check, so they drop straight into CI. Format in [execution-receipts.md](execution-receipts.md), conformance surface in [sep2787-conformance.md](sep2787-conformance.md).

## HTTP API

The same scorer and audit trail are available over HTTP for non-Python agents and control planes that prefer a network boundary.

```bash
pip install 'vaara[server]'
vaara serve --host 0.0.0.0 --port 8000

curl -sX POST http://localhost:8000/v1/score \
  -H 'content-type: application/json' \
  -d '{"tool_name":"tx.transfer","agent_id":"agent-007","base_risk_score":0.5}'
```

Wire contract in [openapi.yaml](openapi.yaml). Operator endpoints include `POST /v1/policy/reload` (atomic hot policy swap) and named detectors `POST /v1/detect/injection` and `POST /v1/detect/pii`, with matching CLI subcommands that exit non-zero on detection for CI gating.

### TypeScript client

The first-party TypeScript client ships on npm as [`@vaara/client`](../clients/ts): typed wrappers over every v1 endpoint, Node 18+, ESM. JS/TS agents call Vaara without a Python sidecar.

```ts
import { VaaraClient } from "@vaara/client";
const vaara = new VaaraClient({ baseUrl: "http://localhost:8000" });
const r = await vaara.score({ tool_name: "tx.transfer", agent_id: "agent-007", base_risk_score: 0.6 });
if (r.decision === "deny") throw new Error("blocked");
```
