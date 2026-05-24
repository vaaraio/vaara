<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/vaara-wordmark-dark.png">
    <img src="docs/vaara-wordmark-light.png" alt="Vaara" width="900">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/vaara/"><img src="https://img.shields.io/pypi/v/vaara.svg" alt="PyPI"></a>
  <a href="https://github.com/vaaraio/vaara/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/vaara.svg" alt="License"></a>
  <a href="https://github.com/vaaraio/vaara/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/vaaraio/vaara/ci.yml?branch=main&label=tests" alt="CI"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/vaaraio/vaara"><img src="https://api.scorecard.dev/projects/github.com/vaaraio/vaara/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/12612"><img src="https://www.bestpractices.dev/projects/12612/badge" alt="OpenSSF Best Practices"></a>
  <a href="https://huggingface.co/spaces/vaaraio/vaara"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
</p>

Vaara is the runtime evidence layer for AI Act compliance. Open source, no SaaS, no telemetry.

Vaara intercepts agent tool calls, scores each one with a conformal risk interval, and writes a hash-chained audit record. Online learning across five expert signals via Multiplicative Weight Update. Distribution-free conformal coverage on the score. An external auditor can verify these properties without trusting your stack. Orchestration toolkits and identity layers (Microsoft Agent Governance Toolkit, others) sit on top.

## Numbers

97.1% attack recall on a held-out distribution-shift split. 140 µs p99 inference latency on commodity CPU. Zero attack success against a PAIR adaptive attacker over 25 attempts.

- 5,955-entry adversarial corpus (3,422 attack across 8 categories, 2,533 benign)
- 97.1% attack recall on held-out distribution-shift split, threshold 0.55
- PAIR adaptive-attacker calibration: ASR 0/25 against Qwen2.5-32B
- [vaara-bench-v1](bench/vaara-bench-v1.md): 77-trace synthetic-corpus benchmark with frozen methodology, 100% soft TPR, 0% hard FPR
- 140 µs / 210 µs p99 inference latency, commodity CPU
- Distribution-free conformal coverage on the score
- MWU regret bound O(sqrt(T log N))

Each figure is reproducible from the public corpus or the bench harness in `bench/`.

## Install

```bash
pip install vaara
```

Python 3.10+. Zero runtime deps. Optional XGBoost classifier: `pip install vaara[ml]`.

Releases ship with SLSA Build Level 3 provenance. Verify with `slsa-verifier verify-artifact`.

## Quick start

```python
from vaara.pipeline import InterceptionPipeline

pipeline = InterceptionPipeline()
result = pipeline.intercept(
    agent_id="agent-007",
    tool_name="fs.write_file",
    parameters={"path": "/etc/service.yaml", "content": "..."},
    agent_confidence=0.8,
)
if result.allowed:
    pipeline.report_outcome(result.action_id, outcome_severity=0.0)
else:
    print(result.reason)
```

`report_outcome` closes the loop. MWU reweights signals based on which ones predicted the outcome.

## Who reaches for Vaara

- **AI compliance teams** shipping high-risk systems under the EU AI Act — Article 9 risk management, Article 12 logging, Article 15 robustness, Article 61 post-market monitoring evidence.
- **ML platform teams** adding runtime governance to agentic stacks (LangChain, CrewAI, OpenAI Agents SDK, MCP-based hosts) without rewriting orchestration.
- **AI safety and red teams** calibrating scorers against adaptive attackers (PAIR, distribution-shift evals, custom corpora).
- **Notified Bodies and internal auditors** reading article-level evidence reports without trusting the deployer's stack.

## What evidence looks like

`vaara compliance report --format json` against a real audit trail produces an article-level evidence record an auditor can read directly. Status is reported honestly: articles without recorded events return `evidence_insufficient`, not a rubber-stamp.

```json
{
  "system_name": "Acme HR Assistant",
  "overall_status": "evidence_insufficient",
  "trail_integrity": {"size": 105, "chain_intact": true},
  "articles": [
    {"article": "Article 12(1)", "title": "Record-Keeping (Logging)",
     "status": "evidence_sufficient", "strength": "strong", "evidence_count": 105},
    {"article": "Article 9(2)(a)", "title": "Risk Identification and Analysis",
     "status": "evidence_sufficient", "strength": "strong", "evidence_count": 35},
    {"article": "Article 15(1)", "title": "Accuracy, Robustness and Cybersecurity",
     "status": "evidence_insufficient", "strength": "absent", "evidence_count": 0}
  ]
}
```

The same data renders as a styled PDF for Notified Bodies (`vaara compliance report --format pdf`, requires `pip install 'vaara[pdf]'`), a static HTML dashboard (`vaara compliance dashboard`), or a Sigstore-signed regulator-handoff envelope (`vaara trail export`, optional ML-DSA-65 / FIPS 204 post-quantum signer via `pip install 'vaara[pq]'`).

<details>
<summary>Per-article verdict drill-down</summary>

Each article in the report carries two extra surfaces a reviewer can read without re-running the engine. `verdict_inputs` lists the threshold-vs-observed snapshot the engine compared against (minimum record count, staleness window, strong-strength bounds, future-timestamp and chain-integrity flags) plus a `verdict_reasons` list of human-readable rationale lines explaining why the status and strength landed where they did. `contributing_events` lists the most recent qualifying audit records the verdict sits on (record ID, action ID, ISO timestamp, agent, tool, and a filtered `drill_down` dict of just the data fields that fed the risk/decision/outcome: point estimate, conformal interval, decision, reason, outcome severity). The drill-down renders in every output format: JSON, markdown, narrative, PDF, and the HTML dashboard. An auditor reading the report can trace `status → threshold delta → concrete event` in one sitting.
</details>

## Framework adapters

Native adapters in `src/vaara/integrations/` route the major Python agent frameworks through Vaara's pipeline. Each intercepts via the framework's own callback or hook surface, scores, gates, and emits the same audit events as a direct `pipeline.intercept()`. Frameworks are not hard dependencies (lazy import, duck typing).

| Framework | Entry point | Use |
|---|---|---|
| LangChain | `VaaraCallbackHandler`, `vaara_wrap_tool` | Slots into `config={"callbacks": [...]}` or wraps per-tool |
| CrewAI | `VaaraCrewGovernance` | Wraps a crew so every agent action passes through scoring + audit |
| OpenAI Agents SDK | `VaaraToolGuardrail`, `vaara_wrap_function` | Function-tool wrap, compatible with Responses API and Agents-SDK tracing |
| MCP server | `vaara.integrations.mcp_server` | Exposes scoring, audit, policy reload as MCP tools |

All four share the same in-process pipeline, so audit records hash-chain together regardless of which framework the action came through. For Vaara *in front of* an upstream MCP server, see the [MCP proxy](#mcp-proxy-vaara-as-a-transparent-governance-layer) section below.

## Upstream-signal adapters (cloud + OSS guardrails)

Adapters route findings from cloud and OSS guardrails into Vaara's audit trail and OVERT envelope with EU AI Act article tags. The filter runs in the deployer's environment as an upstream signal. Vaara records the verdict, normalises 68 provider categories onto a shared vocabulary, and tags each finding against Art. 5, 10, 13, 15, 53, and the CSAM-specific obligation from the Digital Omnibus political agreement of May 2026.

| Provider | Adapter | Extra | Wraps |
|---|---|---|---|
| AWS Bedrock Guardrails | `BedrockGuardrailsAdapter` | `vaara[bedrock]` | `ApplyGuardrail` across five Bedrock policy buckets |
| Azure AI Content Safety | `AzureContentSafetyAdapter` | `vaara[azure-content-safety]` | `analyze_text`, Prompt Shields, Protected Material, Groundedness |
| GCP Model Armor | `GcpModelArmorAdapter` | `vaara[gcp-model-armor]` | `sanitize_user_prompt`, `sanitize_model_response` |
| NVIDIA NeMo Guardrails | `NemoGuardrailsAdapter` | `vaara[nemo-guardrails]` | `GenerationResponse.log.activated_rails` (input / dialog / output / retrieval) |
| Guardrails AI | `GuardrailsAIAdapter` | `vaara[guardrails-ai]` | `ValidationOutcome.validation_summaries` from `Guard.parse` / `Guard.validate` |
| LLM Guard | `LLMGuardAdapter` | `vaara[llm-guard]` | `scan_prompt` / `scan_output`, parses `(sanitized, results_valid, results_score)` |
| Rebuff | `RebuffAdapter` | `vaara[rebuff]` | `DetectResponse` across heuristic, model, vector layers + canary-word leak check |

Each adapter returns a `ContentSafetyFinding` the deployer routes into `pipeline.intercept(context=finding.to_audit_context())`. The mapping table lives at `src/vaara/integrations/_content_safety_articles.py`. Article-level rationale in [COMPLIANCE.md](COMPLIANCE.md#cloud-guardrail-adapter-pattern) and [COMPLIANCE.md](COMPLIANCE.md#oss-guardrail-adapter-pattern).

## HTTP API

The same scorer and audit trail are available over HTTP for non-Python agents and for control planes that prefer a network boundary. Install with the `server` extra:

```
pip install 'vaara[server]'
vaara serve --host 0.0.0.0 --port 8000
```

```
curl -sX POST http://localhost:8000/v1/score \
  -H 'content-type: application/json' \
  -d '{"tool_name":"tx.transfer","agent_id":"agent-007","base_risk_score":0.5}'
```

The wire contract is in [docs/openapi.yaml](docs/openapi.yaml). Vaara defines the interface. Control-plane and orchestration vendors call it. Integration recipes for adopters live under `examples/recipes/`. Operator endpoints include `POST /v1/policy/reload` for atomic hot policy swap (start with `vaara serve --policy PATH` to enable), and `POST /v1/detect/injection` and `POST /v1/detect/pii` as named buyer-visible detectors with matching CLI subcommands that exit non-zero on detection for CI gating.

Vaara's scorer can be run alongside external scorers via `vaara.scorer.composition.ExternalScorer` and `vaara.scorer.composite.CompositeScorer`. Any service that implements the `/v1/score` wire contract (NeMo Guardrails, another Vaara instance) can be composed. The composite preserves the strongest decision across members.

### TypeScript client

The first-party TypeScript client lives at [`clients/ts`](clients/ts) and ships on npm as `@vaara/client`. Typed wrappers over every v1 endpoint, Node 18+, ESM, declarations shipped. JS/TS agents (LangChain.js, Vercel AI SDK, MCP, any Node service) can call Vaara without a Python sidecar.

```bash
npm install @vaara/client
```

```ts
import { VaaraClient } from "@vaara/client";
const vaara = new VaaraClient({ baseUrl: "http://localhost:8000" });
const r = await vaara.score({ tool_name: "tx.transfer", agent_id: "agent-007", base_risk_score: 0.6 });
if (r.decision === "deny") throw new Error("blocked");
```

## MCP proxy (Vaara as a transparent governance layer)

`vaara.integrations.mcp_proxy.VaaraMCPProxy` sits between an MCP client (Claude Code, Cursor, any MCP-capable host) and an upstream MCP server. Every `tools/call` from the client routes through Vaara's interception pipeline before reaching the upstream. Allowed calls forward transparently and report the upstream outcome back to the scorer. Blocked calls return an MCP `isError: true` response with the block reason. The initialization handshake and `notifications/*` forward unchanged. `tools/list`, `resources/list`, `resources/read`, `prompts/list`, and `prompts/get` route through the operator perimeter before reaching the client or upstream.

```bash
python -m vaara.integrations.mcp_proxy \
  --upstream npx --upstream-arg -y --upstream-arg @sap/mdk-mcp-server \
  --db ./mcp_audit.db
```

Point your MCP client at the proxy instead of the upstream. The audit chain captures every tool call without changing client or upstream behavior. Distinct from `mcp_server`, which exposes Vaara itself as an MCP server for agents that consult Vaara as a tool.

<details>
<summary>Operator perimeter: tool, resource, prompt filtering</summary>

The proxy accepts repeatable `--allow-tool NAME` / `--deny-tool NAME`, `--allow-resource URI` / `--deny-resource URI`, and `--allow-prompt NAME` / `--deny-prompt NAME` flags. Filtered tools are dropped from `tools/list` responses before the client sees them and any matching `tools/call` is rejected at the proxy perimeter without contacting the upstream. The same shape extends to `resources/list` + `resources/read` and `prompts/list` + `prompts/get`. Denylist wins on overlap with allowlist. No flags = passthrough. Every allowed `resources/read` and `prompts/get` writes a request+decision audit pair to the hash chain so a regulator can reconstruct exactly which resources the agent read and which prompts it retrieved. Read-oriented MCP surfaces do not run through the risk scorer. The operator perimeter is the gate, the audit chain is the evidence.
</details>

<details>
<summary>OVERT 1.0 envelopes per interaction</summary>

Off by default. When you pass `--overt-signing-key KEY.pem`, `--overt-operator-key OPKEY.bin`, and `--overt-receipts-dir DIR/`, the proxy writes one OVERT 1.0 Protocol Profile 1.0 Base Envelope (canonical CBOR, Ed25519, closed 9-field schema) per governed interaction into `DIR/{nanosecond_timestamp}-{counter:010d}.cbor`. Covers all four states: allowed `tools/call`, blocked `tools/call`, perimeter-filtered call, and perimeter-filtered `resources/read` / `prompts/get`. The arbiter public key is pinned alongside as `pubkey.bin`. Each envelope verifies offline under `vaara overt verify` against any conformant verifier.

```bash
# 1. Generate an Ed25519 signing key (evaluation/demo; for production use a KMS or HSM, see docs/signing-keys.md).
vaara keygen --dev --out signing.pem

# 2. Mint an operator HMAC key (>= 16 raw bytes). Used for request_commitment per OVERT Annex B.4.
head -c 32 /dev/urandom > op.key

# 3. Run the proxy with OVERT emission turned on.
python -m vaara.integrations.mcp_proxy \
  --upstream npx --upstream-arg -y --upstream-arg @sap/mdk-mcp-server \
  --overt-signing-key signing.pem \
  --overt-operator-key op.key \
  --overt-receipts-dir ./overt_receipts

# Each interaction now produces a Provisional Receipt:
vaara overt verify ./overt_receipts/1779309684224332669-0000000001.cbor \
  --pubkey-file ./overt_receipts/pubkey.bin
# → {"valid": true, "monotonic_counter": 1, ...}
```

`non_content_metadata` carries structural fields only (action class, tool/resource/prompt identifier, decision, reason, agent_id, action_id). The request content itself never leaves the operator environment; only its HMAC-SHA256 commitment crosses the trust boundary. The monotonic counter advances strictly across the whole proxy process so gaps are detectable. Emission failure is logged and swallowed: attestation problems must not block legitimate upstream traffic.
</details>

<details>
<summary>Streaming notifications inside the boundary</summary>

Long-running upstream tools emit `notifications/progress` and `notifications/message` over the lifetime of a `tools/call`. The proxy routes each notification through the same audit pair (request + decision) and, when OVERT is configured, emits a dedicated Base Envelope with action class `mcp.notification.progress` or `mcp.notification.message`. Progress events correlate to the originating call via the `_meta.progressToken` from the request, so a regulator reading the receipt directory can reconstruct what arrived between request and response. Notifications still forward to the client unchanged. Audit failures are logged and swallowed: observation never blocks streaming.
</details>

Worked examples with real upstream servers:

- [`examples/github-mcp-proxy-demo/`](examples/github-mcp-proxy-demo/). Vaara in front of [`github/github-mcp-server`](https://github.com/github/github-mcp-server) (GitHub's official MCP server, MIT-licensed). End-to-end verified: real subprocess, 42 tools advertised, hash-chained audit trail recorded.
- [`examples/sap-mcp-proxy-demo/`](examples/sap-mcp-proxy-demo/). Vaara in front of community SAP MCP servers ([`SAP/mdk-mcp-server`](https://github.com/SAP/mdk-mcp-server), [`mario-andreschak/mcp-abap-abap-adt-api`](https://github.com/mario-andreschak/mcp-abap-abap-adt-api), [`lemaiwo/btp-sap-odata-to-mcp-server`](https://github.com/lemaiwo/btp-sap-odata-to-mcp-server)).

The proxy is MCP-protocol-level, not vendor-specific. The same three-step recipe applies to any stdio-capable MCP server (Microsoft Graph MCP, Salesforce MCP, ServiceNow MCP, cloud-provider MCP servers, Databricks MCP, and so on).

## OVERT 1.0 attestation

**What.** OVERT 1.0 is an open standard for runtime trust in AI systems ([overt.is](https://overt.is/), authored by Glacis Technologies, published 25 March 2026). It defines a signed, schema-closed envelope a relying party can verify offline without trusting the emitter.

**Why.** A regulator, auditor, or customer can confirm that a runtime decision actually happened the way you say it did, without reading your code or trusting your stack.

**How Vaara emits it.** Vaara is the **Arbiter** in OVERT terms and ships Protocol Profile 1.0 Base Envelopes (canonical CBOR per RFC 8949, Ed25519 signatures, HMAC-SHA256 keyed commitments, closed 9-field schema, IEEE-754 float rejection) alongside every audit record when attestation is enabled.

```
pip install 'vaara[attestation]'
```

```python
from vaara.attestation.overt import emit_base_envelope, make_request_commitment, encoder_binary_identity

envelope = emit_base_envelope(
    signing_key=key,
    request_commitment=make_request_commitment(payload, operator_key=op_key),
    encoder_binary_identity=encoder_binary_identity(arbiter_version="vaara/0.26.0", policy_hash=ph),
    non_content_metadata={"action_class": "tx.transfer", "decision": "escalate"},
    monotonic_counter=42,
    arbiter_instance_identifier=uuid_bytes,
)
```

The reference Phase 3 IAP (`vaara.attestation.iap`) notary-signs the Provisional Receipt and anchors it in a transparency log. Production deployments can swap in sigstore Rekor or an equivalent independently-operated log at the same call sites. The OVERT S3P (MEA-2) emitter at `vaara.attestation.s3p` ships exact Clopper-Pearson confidence intervals (pure Python, no scipy) and a proposed Protocol Profile extension that reports aggregate statistics over per-action conformal prediction intervals alongside the standard binomial CI.

The `vaara overt verify RECEIPT.cbor --pubkey-file PUB.bin` CLI validates any canonical-CBOR Base Envelope against a supplied raw 32-byte Ed25519 public key. The verifier reads only the wire format and takes no dependency on Vaara's emitter, so any OVERT-conformant implementation can route its conformance check through it.

An experimental hardware TEE hook (`vaara.attestation.tee`) binds an OVERT envelope to an AMD SEV-SNP attestation report by placing `SHA-512(canonical_cbor(envelope))` in the report's 64-byte `REPORT_DATA` field. The envelope schema is unchanged (closed per spec). The TEE report is a sibling artefact: a relying party checks the Ed25519 envelope signature and the ECDSA P-384 report signature independently. `vaara tee parse` and `vaara tee verify` expose the verifier as a CLI.

See [COMPLIANCE.md](COMPLIANCE.md) "Position relative to open runtime-attestation standards" for the architectural framing and "OVERT 1.0 Part 3 (Agentic AI Controls) mapping" for the TOOL-*, MCP-*, MULTI-*, CAP-*, DISC-*, HITL-*, DRIFT-* control-by-control walk.

## Where things live

| Path | Contents |
|---|---|
| [docs/formal_specification.md](docs/formal_specification.md) | MWU regret bound, conformal coverage, security properties |
| [docs/conformal-prediction.md](docs/conformal-prediction.md) | Plain-language explainer for compliance reviewers and legal counsel |
| [COMPLIANCE.md](COMPLIANCE.md) | EU AI Act (Art. 9, 11 to 15, 61) and DORA (Art. 10, 12, 13) mapping, eval numbers, PAIR calibration |
| [VERDICTS.md](VERDICTS.md) | Per-article evidence sufficiency thresholds and decision tree |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version feature evolution |
| [PRIOR_ART.md](PRIOR_ART.md) | When each Vaara concept first shipped, and a neutral list of adjacent published work |
| [docs/signing-keys.md](docs/signing-keys.md) | Release signing and verification |
| [SECURITY.md](SECURITY.md) | Security policy and reporting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| `src/vaara/integrations/` | LangChain, OpenAI Agents SDK, CrewAI, MCP, Bedrock, Azure, GCP |
| `src/vaara/audit/` | Hash-chain trail, SQLite backend, append-only WAL |
| `src/vaara/policy/` | YAML / JSON policy schema, `vaara policy validate` and `vaara policy test` |
| `src/vaara/sandbox/` | Synthetic-trace cold-start calibration |

Acknowledgements:

- Vaara is listed in the industry acknowledgements of the [IMDA Model AI Governance Framework for Agentic AI v1.5](https://www.imda.gov.sg/-/media/imda/files/about/emerging-tech-and-research/artificial-intelligence/mgf-for-agentic-ai.pdf) (Singapore, 20 May 2026).
- The [AMD AI Developer Program](https://www.linkedin.com/posts/amd-developer_meet-henri-sirkkavaara-henri-created-vaara-activity-7459667676555132928-QFSd) ran a coordinated multi-channel developer testimonial of Vaara in May 2026.
- [Article 14 runtime: why oversight of agentic AI has to be evidenced as action, not model](https://futurium.ec.europa.eu/ga/apply-ai-alliance/community-content/article-14-runtime-why-oversight-agentic-ai-has-be-evidenced-action-not-model) is the position post on the EU Apply AI Alliance Futurium.

> Vaara helps deployers assemble evidence for their own conformity work. It does not certify compliance or constitute legal advice. Deployers own their obligations under the EU AI Act and other applicable law.

## License

Apache 2.0. See [LICENSE](LICENSE).
