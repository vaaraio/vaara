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

### From Python

LangChain: one handler for the whole run, passed down in the config, or the tools that matter wrapped one by one so they check before they run wherever they are called from. A deny raises `ToolExecutionBlocked`; an escalation raises `ToolExecutionEscalated` when `block_on_escalate` is set and is otherwise recorded and allowed.

```python
from langchain_core.tools import tool

from vaara.integrations.langchain import ToolExecutionBlocked, VaaraCallbackHandler, vaara_wrap_tool
from vaara.pipeline import InterceptionPipeline

pipeline = InterceptionPipeline()
handler = VaaraCallbackHandler(pipeline, agent_id="support-bot", block_on_escalate=True)
# agent.invoke({"input": question}, config={"callbacks": [handler]})


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


safe_send = vaara_wrap_tool(send_email, pipeline, agent_id="support-bot")
try:
    print(safe_send.invoke({"to": "cfo@example.com", "body": "Q3 numbers attached"}))
except ToolExecutionBlocked as blocked:
    print("blocked:", blocked.reason, blocked.action_id)
```

CrewAI, two layers. `VaaraCrewGovernance` pre-screens a task from the tools it names (read-only, no trail record), wraps a crew's tools, and runs a crew with every call scored. `VaaraGovernance` with `register` hooks CrewAI's before and after tool-call hooks and records a receipt-backed decision stream per run, so a dropped record is a provable gap.

```python
from vaara.integrations.crewai import VaaraCrewGovernance, VaaraGovernance, register
from vaara.pipeline import InterceptionPipeline

pipeline = InterceptionPipeline()
crew_governance = VaaraCrewGovernance(pipeline, block_on_escalate=False)
screen = crew_governance.screen_task(
    task_description="Reconcile last month's invoices and pay the overdue ones",
    agent_role="finance assistant",
    tools=["data.read", "tx.transfer"],
)
print(screen["task_allowed"], screen["escalated_tools"], screen["blocked_tools"])
# governed_tools = crew_governance.wrap_tools(crew.tools, agent_id="finance-assistant")
# result = crew_governance.governed_kickoff(crew)

governance = VaaraGovernance(boundary_id_for=lambda context: "run-2026-10-11")
try:
    register(governance)            # CrewAI's before and after tool-call hooks
except ImportError as exc:
    print(exc)                      # CrewAI is not installed here
# crew.kickoff(...)
governance.finalize_run("run-2026-10-11")
print(governance.verify_run("run-2026-10-11").ok)
print(len(governance.decisions()), len(governance.outcomes()), len(governance.receipts()))
```

OpenAI Agents SDK and the Responses API: `vaara_wrap_function` wraps a plain function so a call checks first and raises `ToolCallBlocked` on a deny. `VaaraToolGuardrail` is an output guardrail: it reads the tool calls the agent plans and trips the wire on a deny.

```python
from vaara.integrations.openai_agents import ToolCallBlocked, VaaraToolGuardrail, vaara_wrap_function
from vaara.pipeline import InterceptionPipeline

pipeline = InterceptionPipeline()


@vaara_wrap_function(pipeline, agent_id="payments-bot")
def send_funds(to: str, amount: float) -> str:
    return f"sent {amount} to {to}"


try:
    print(send_funds(to="0xabc", amount=1000))
except ToolCallBlocked as blocked:
    print("blocked:", blocked, blocked.action_id)

guardrail = VaaraToolGuardrail(pipeline, agent_id="payments-bot", block_on_escalate=True)
# Agent(..., output_guardrails=[OutputGuardrail(guardrail_function=guardrail)])
```

## Upstream-signal adapters (cloud + OSS guardrails)

Adapters route findings from cloud and OSS guardrails into Vaara's audit trail and OVERT envelope. The filter runs in the deployer's environment; Vaara records the verdict, normalises 69 provider categories onto a shared vocabulary, and tags each finding against the relevant AI Act articles. Each adapter returns a `ContentSafetyFinding` the deployer routes into `pipeline.intercept(context=finding.to_audit_context())`. Article-by-article mapping in [COMPLIANCE.md](COMPLIANCE.md).

| Provider | Adapter | Extra | Wraps |
|---|---|---|---|
| AWS Bedrock Guardrails | `BedrockGuardrailsAdapter` | `vaara[bedrock]` | `ApplyGuardrail` across five Bedrock policy buckets |
| Azure AI Content Safety | `AzureContentSafetyAdapter` | `vaara[azure-content-safety]` | `analyze_text`, Prompt Shields, Protected Material, Groundedness |
| GCP Model Armor | `GcpModelArmorAdapter` | `vaara[gcp-model-armor]` | `sanitize_user_prompt`, `sanitize_model_response` |
| NVIDIA NeMo Guardrails | `NemoGuardrailsAdapter` | `vaara[nemo-guardrails]` | `GenerationResponse.log.activated_rails` (input / dialog / output / retrieval) |
| Guardrails AI | `GuardrailsAIAdapter` | `vaara[guardrails-ai]` | `ValidationOutcome.validation_summaries` from `Guard.parse` / `Guard.validate` |
| LLM Guard | `LLMGuardAdapter` | `vaara[llm-guard]` | `scan_prompt` / `scan_output` |
| Rebuff | `RebuffAdapter` | `vaara[rebuff]` | `DetectResponse` across heuristic, model, vector layers + canary-word leak check |

A finding's `verdict` is `allow`, `flag`, `block` or `unparsed`. `unparsed` means the reply lacked the field the adapter reads to know a scan ran (Bedrock `action`, Model Armor `filterMatchState`, NeMo `activated_rails`, Rebuff `injectionDetected`, a Guardrails AI outcome with neither `validation_passed` nor summaries, an LLM Guard result with no scanners, an Azure reply missing its analysis field), so the trail does not record it as a pass. A category the adapter did read still decides the verdict.

Mapping table at `src/vaara/integrations/_content_safety_articles.py`. Rationale in [COMPLIANCE.md](COMPLIANCE.md#cloud-guardrail-adapter-pattern).

### From Python

Every adapter has two doors: the class, which holds the provider's client and scans text, and a parse function for a response the deployer already has. The finding is the same either way, and `to_audit_context()` is what goes into the trail.

```python
from vaara.integrations.bedrock_guardrails import parse_apply_guardrail_response
from vaara.pipeline import InterceptionPipeline

response = {                                   # what boto3's apply_guardrail returned
    "action": "GUARDRAIL_INTERVENED",
    "assessments": [{"contentPolicy": {"filters": [
        {"type": "HATE", "confidence": "HIGH", "filterStrength": "HIGH",
         "action": "BLOCKED", "detected": True},
    ]}}],
}
finding = parse_apply_guardrail_response(response, scanned_role="prompt")
print(finding.verdict, finding.severity)                              # block high
print([c.vaara_category for c in finding.triggered_categories()])    # ['hate']
print(finding.ai_act_articles())                                      # the articles the row maps to

pipeline = InterceptionPipeline()
result = pipeline.intercept(agent_id="chat-bot", tool_name="llm.prompt",
                            context=finding.to_audit_context())
print(result.decision)
print(sorted(finding.to_overt_metadata()))                            # for an OVERT envelope
```

The clients the adapters wrap, one block per provider. Each needs its own extra, and `scan_prompt` / `scan_response` return the finding above.

```python
# needs: vaara[bedrock]
import boto3

from vaara.integrations.bedrock_guardrails import BedrockGuardrailsAdapter

adapter = BedrockGuardrailsAdapter(boto3.client("bedrock-runtime"),
                                   guardrail_id="gr-0a1b2c3d", guardrail_version="1")
finding = adapter.scan_prompt("user text")
print(finding.verdict)
```

```python
# needs: vaara[azure-content-safety]
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

from vaara.integrations.azure_content_safety import AzureContentSafetyAdapter

client = ContentSafetyClient("https://example.cognitiveservices.azure.com",
                             AzureKeyCredential("<key>"))
adapter = AzureContentSafetyAdapter(client, block_threshold=4)
finding = adapter.scan_prompt("user text", include={"analyze_text", "shield"})
print(finding.verdict)
```

```python
# needs: vaara[gcp-model-armor]
from google.cloud import modelarmor_v1

from vaara.integrations.gcp_model_armor import GcpModelArmorAdapter

adapter = GcpModelArmorAdapter(
    modelarmor_v1.ModelArmorClient(),
    template="projects/p/locations/europe-west4/templates/default",
)
finding = adapter.scan_response("model text")
print(finding.verdict)
```

```python
# needs: vaara[nemo-guardrails]
from nemoguardrails import LLMRails, RailsConfig

from vaara.integrations.nemo_guardrails import NemoGuardrailsAdapter

adapter = NemoGuardrailsAdapter(LLMRails(RailsConfig.from_path("./rails")))
answer, finding = adapter.generate([{"role": "user", "content": "user text"}])
print(finding.verdict, [c.provider_category for c in finding.triggered_categories()])
```

```python
# needs: vaara[guardrails-ai]
from guardrails import Guard
from guardrails.hub import ToxicLanguage

from vaara.integrations.guardrails_ai import GuardrailsAIAdapter

adapter = GuardrailsAIAdapter(Guard().use(ToxicLanguage, on_fail="exception"))
validated, finding = adapter.parse("model text")
print(finding.verdict)
```

```python
# needs: vaara[llm-guard]
from llm_guard.input_scanners import PromptInjection
from llm_guard.output_scanners import Sensitive

from vaara.integrations.llm_guard import LLMGuardAdapter

adapter = LLMGuardAdapter(input_scanners=[PromptInjection()], output_scanners=[Sensitive()])
finding = adapter.scan_prompt("user text")
print(finding.verdict)
finding = adapter.scan_response("model text", prompt="user text")
print(finding.verdict)
```

```python
# needs: vaara[rebuff]
from rebuff import RebuffSdk

from vaara.integrations.rebuff import RebuffAdapter

rebuff = RebuffSdk(openai_apikey="<key>", pinecone_apikey="<key>", pinecone_index="rebuff")
adapter = RebuffAdapter(rebuff)
finding = adapter.scan_prompt("user text")
print(finding.verdict)
canary = rebuff.add_canary_word("user text")      # (buffed prompt, canary word)
finding = adapter.scan_response("model text", prompt=canary[0], canary_word=canary[1])
print(finding.verdict)
```

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

The HTTP transport refuses any request carrying an `Origin` header from another site, which is what stops a web page you happen to visit from driving a loopback-bound proxy: a cross-origin `fetch` needs no preflight when it sends `Content-Type: text/plain`, and the page cannot read the reply but the tool call has already run. Native MCP clients send no `Origin` and are unaffected. For a browser-based client, allow its origin explicitly with `--allow-origin https://console.example` (repeatable, matched exactly) or `VAARA_PROXY_ALLOWED_ORIGINS`.

Each `POST /mcp` reads two headers: `X-Vaara-Upstream` picks the upstream slot, `X-Vaara-Tenant` scopes the policy, audit chain, and OVERT envelope. Single-upstream deployments keep the silent-default contract; multi-upstream deployments require `X-Vaara-Upstream` per call and return 400 with the slot list when it is missing. `vaara serve --policy-dir DIR` loads one policy per file (filename stem becomes `tenant_id`, `default.yaml` is the fallback) and hot-reloads per tenant.

### Operator perimeter and request attestation

Repeatable `--allow-tool` / `--deny-tool` flags (and the same for resources and prompts) filter the MCP surface. Filtered tools are dropped from `tools/list` before the client sees them and any matching call is rejected at the perimeter without contacting the upstream. Denylist wins on overlap; no flags means passthrough. Every allowed `resources/read` and `prompts/get` writes a request+decision audit pair so a regulator can reconstruct exactly what the agent read.

OVERT envelopes per interaction turn on with `--overt-signing-key`, `--overt-operator-key`, `--overt-receipts-dir`. SEP-2787 request attestation paired with an execution receipt turns on with `--attest-signing-key PATH` and `--attest-receipts-dir DIR`: each allowed call writes a pre-execution attestation and a post-execution receipt linked by a `backLink` digest. Key type auto-detects from the file (EC P-256 = ES256, RSA = RS256, raw bytes = HS256). Generate and verify offline:

```bash
vaara keygen --attest --out attest_key.pem
vaara attest verify  0000000001-ab12cd34-attest.json  --pubkey-file attest_key.pem.pub
vaara receipt verify 0000000001-ab12cd34-receipt.json --attestation 0000000001-ab12cd34-attest.json --pubkey-file attest_key.pem.pub
```

Both verifiers exit non-zero on any failed check, so they drop straight into CI. Format in [execution-receipts.md](execution-receipts.md), conformance surface in [attestation-conformance.md](attestation-conformance.md).

## HTTP API

The same scorer and audit trail are available over HTTP for non-Python agents and control planes that prefer a network boundary.

```bash
pip install 'vaara[server]'
vaara serve --host 0.0.0.0 --port 8000 --api-key "$VAARA_SERVER_API_KEY"

curl -sX POST http://localhost:8000/v1/score \
  -H 'content-type: application/json' \
  -H "Authorization: Bearer $VAARA_SERVER_API_KEY" \
  -d '{"tool_name":"tx.transfer","agent_id":"agent-007","base_risk_score":0.5}'
```

Authentication matters here: with no API key the server is loopback-only by default, and `vaara serve` refuses a non-loopback bind without `--api-key` (or the `VAARA_SERVER_API_KEY` env var) unless you pass `--allow-unauthenticated`. That refusal is deliberate: an unauthenticated reachable server lets any client swap the policy to allow-all (`POST /v1/policy/reload`), append forged audit events that still verify (`POST /v1/audit/events`), and poison calibration (`POST /v1/score/outcome`). When a key is configured, every endpoint except `GET /v1/health` requires `Authorization: Bearer <key>`.

Wire contract in [openapi.yaml](openapi.yaml). Operator endpoints include `POST /v1/policy/reload` (atomic hot policy swap) and named detectors `POST /v1/detect/injection` and `POST /v1/detect/pii`, with matching CLI subcommands that exit non-zero on detection for CI gating.

### TypeScript client

The first-party TypeScript client ships on npm as [`@vaara/client`](../clients/ts): typed wrappers over every v1 endpoint, Node 18+, ESM. JS/TS agents call Vaara without a Python sidecar.

```ts
import { VaaraClient } from "@vaara/client";
const vaara = new VaaraClient({ baseUrl: "http://localhost:8000" });
const r = await vaara.score({ tool_name: "tx.transfer", agent_id: "agent-007", base_risk_score: 0.6 });
if (r.decision === "deny") throw new Error("blocked");
```
