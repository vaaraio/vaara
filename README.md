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

Vaara is the tamper-evident runtime evidence layer for AI systems. It covers EU AI Act compliance, and any other case where you need to prove what an agent actually did. Open source, no SaaS, no telemetry.

Vaara intercepts agent tool calls, scores each one with a conformal risk interval, and writes a hash-chained audit record. Online learning across five expert signals via Multiplicative Weight Update. Distribution-free conformal coverage on the score. An external auditor can verify these properties without trusting your stack. Orchestration toolkits and identity layers (Microsoft Agent Governance Toolkit, others) sit on top.

## Numbers

Held-out TEST recall 84.7% (95% Wilson [82.4, 86.7]) at FPR 4.1% [2.9, 5.7]. Phase 1 PAIR scale-up to n=300 per attacker family lands at 88.1% [85.8, 90.1]. Cross-model held-out recall, where no attacker model in the eval set was in TRAIN, is 66.8% [64.9, 68.7] over n=2,277; the weakest sub-cell is data_exfil against a closed-weight model at 38.9% [35.3, 42.5] (see [vaara-bench-v0.37](bench/vaara-bench-v0.37.md)). Under BIPIA-pressure context, false-positive rate on benign tool calls 1.2% [0.4, 3.6] across four agent backends (Claude Haiku 4.5, Llama-3.1-8B, Mistral-7B, Qwen-2.5-7B). Multi-attacker PAIR ASR 0/25 across three different attacker models with identical seeds. The rule scorer that runs in the hot path adds 140 µs mean / 210 µs p99 per call on commodity CPU; the MiniLM classifier is opt-in (`vaara[ml]`) and is not in that measured path. Every number reproducible end-to-end via `make bench`.

- 12,155-entry adversarial corpus (250 hand-curated + 11,905 LLM-generated), 70/15/15 split stratified by (category, source)
- Classifier v9 with 236 hand-features + 384-dim MiniLM embeddings at calibrated threshold 0.9150 on held-out TEST n=1,827: recall 84.7% [82.4, 86.7] at FPR 4.1% [2.9, 5.7]
- Multi-attacker PAIR robustness: 0/25 successes per attacker across Qwen2.5-32B, Qwen2.5-72B, Llama-3.3-70B hitting identical seed indices, Wilson upper 13.3%
- BIPIA-pressure FPR on benign tool calls 1.2% [0.4, 3.6] across four agent backends, n=244 benign tool calls under `context.source=injected_via_bipia_<class>`
- Cross-model held-out recall 66.8% [64.9, 68.7] over n=2,277 with no eval-set attacker model in TRAIN; data_exfil generalises unevenly, with a closed-weight sub-cell at 38.9% [35.3, 42.5]. This is the honest worst case; the in-distribution TEST number above is the easier denominator
- Chain of custody: corpus manifest SHA, split manifest SHA, training commit, bundle SHA, all locked and printed by every script
- 140 µs mean / 210 µs p99 for the hot-path rule scorer on commodity CPU; the MiniLM classifier is opt-in (`vaara[ml]`) and not in that path
- Distribution-free conformal coverage on the score
- MWU regret bound O(sqrt(T log N))
- [vaara-bench-v0.39](bench/vaara-bench-v0.39.md): current methodology, chain of custody, ship-gate record. v9 retrain on BIPIA-augmented corpus with follows upweighted (`--follow-weight 8.0`), calibrated to T=0.9150 at a 5% FPR target on v035 VAL. BIPIA-pressure FPR collapses from 35.2% on v8 to 1.2% on v9. In-distribution recall flat within Wilson intervals. Found-and-fixed in tree: auto-labeller `example.com` placeholder false-positive rule (42 to 14 true follows across four backends). Historical bench docs live under `bench/` for chain-of-custody continuity.
- [vaara-bench-v1](bench/vaara-bench-v1.md): 77-trace synthetic-corpus regression baseline with frozen methodology, 100% soft TPR, 0% hard FPR

Each figure is reproducible from the public corpus or the bench pipeline in `bench/`.

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

Each article verdict carries `verdict_inputs` (threshold-vs-observed snapshot), `verdict_reasons` (rationale lines), and `contributing_events` (the audit records the verdict sits on, with a `drill_down` of the data that fed the risk/decision/outcome). Reviewers can trace `status` to `threshold delta` to `concrete event` without re-running the engine.

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

Adapters route findings from cloud and OSS guardrails into Vaara's audit trail and OVERT envelope. The filter runs in the deployer's environment as an upstream signal. Vaara records the verdict, normalises 68 provider categories onto a shared vocabulary, and tags each finding against the relevant AI Act articles. Article-by-article mapping in [COMPLIANCE.md](COMPLIANCE.md).

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

The wire contract is in [docs/openapi.yaml](docs/openapi.yaml). Integration recipes under `examples/recipes/`. Operator endpoints include `POST /v1/policy/reload` for atomic hot policy swap, and `POST /v1/detect/injection` and `POST /v1/detect/pii` as named detectors with matching CLI subcommands that exit non-zero on detection for CI gating.

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
vaara-mcp-proxy \
  --upstream npx --upstream-arg -y --upstream-arg @sap/mdk-mcp-server \
  --db ./mcp_audit.db
```

Point your MCP client at the proxy instead of the upstream. The audit chain captures every tool call without changing client or upstream behavior. Distinct from `mcp_server`, which exposes Vaara itself as an MCP server for agents that consult Vaara as a tool.

Upstreams can be local or remote. `--upstream` launches a local stdio MCP server; `--upstream-url NAME=URL` connects to a remote MCP server over the Streamable HTTP transport, and a bare `--upstream-url URL` lands in the `default` slot. Each slot is one transport or the other, never both.

<details>
<summary>Fleet shape (v0.40): one proxy, many upstreams, multi-tenant policy</summary>

`vaara-mcp-proxy` also runs over Streamable HTTP with fan-out, so one process can serve a fleet of upstream MCP servers:

```bash
vaara-mcp-proxy \
  --transport http \
  --http-host 127.0.0.1 \
  --http-port 8765 \
  --upstream 'github=npx -y @github/mcp-server' \
  --upstream 'sap=npx -y @sap/mdk-mcp-server'
```

Each `POST /mcp` reads two headers. `X-Vaara-Upstream` picks the upstream slot. `X-Vaara-Tenant` scopes the policy, audit chain, and OVERT envelope for that call. Single-upstream deployments keep the v0.39 silent-default contract. Multi-upstream deployments require `X-Vaara-Upstream` per call and return 400 with the available slot list when the header is missing.

The reference HTTP API server (`vaara serve --policy-dir DIR`) loads one YAML or JSON policy per file in the directory (filename stem becomes the `tenant_id`, `default.yaml` lands in the fallback slot) and hot-reloads per tenant via `POST /v1/policy/reload` with a `tenant_id` body field or `X-Vaara-Tenant` header. The scorer dispatches allow and deny thresholds per call against the calling tenant's policy at `evaluate()` time.
</details>

<details>
<summary>Operator perimeter: tool, resource, prompt filtering</summary>

The proxy accepts repeatable `--allow-tool NAME` / `--deny-tool NAME`, `--allow-resource URI` / `--deny-resource URI`, and `--allow-prompt NAME` / `--deny-prompt NAME` flags. Filtered tools are dropped from `tools/list` responses before the client sees them and any matching `tools/call` is rejected at the proxy perimeter without contacting the upstream. The same shape extends to `resources/list` + `resources/read` and `prompts/list` + `prompts/get`. Denylist wins on overlap with allowlist. No flags = passthrough. Every allowed `resources/read` and `prompts/get` writes a request+decision audit pair to the hash chain so a regulator can reconstruct exactly which resources the agent read and which prompts it retrieved. Read-oriented MCP surfaces do not run through the risk scorer. The operator perimeter is the gate, the audit chain is the evidence.
</details>

OVERT envelopes per governed interaction turn on with `--overt-signing-key`, `--overt-operator-key`, `--overt-receipts-dir`. Wire format and verifier covered in the [OVERT 1.0 attestation](#overt-10-attestation) section below. Long-running tools' `notifications/progress` and `notifications/message` route through the same audit pair and OVERT envelope, correlated to the originating call via `_meta.progressToken`.

SEP-2787 request attestation paired with an execution receipt turns on with `--attest-signing-key PATH` and `--attest-receipts-dir DIR`. Each allowed `tools/call` writes a `{n}-attest.json` (pre-execution SEP-2787 envelope) and a `{n}-receipt.json` (post-execution outcome record with a `backLink` digest over the attestation). Key type is auto-detected from the file: EC P-256 PEM uses ES256, RSA PEM uses RS256, raw bytes uses HS256. An operator-supplied `X-Vaara-Intent` HTTP header overrides the derived `tools/call/{tool_name}` intent label. The `serverFingerprint` field in the attestation starts as a hash of the upstream command string and upgrades to a hash of the upstream's `tools/list` response on first use, binding the exact capability set the proxy presented. See [docs/execution-receipts.md](docs/execution-receipts.md) for the receipt format.

Generate the signing key and verify the output offline:

```
vaara keygen --attest --out attest_key.pem
vaara attest verify  0000000001-ab12cd34-attest.json  --pubkey-file attest_key.pem.pub
vaara receipt verify 0000000001-ab12cd34-receipt.json --attestation 0000000001-ab12cd34-attest.json --pubkey-file attest_key.pem.pub
```

`keygen --attest` emits an EC P-256 (ES256) key compatible with `--attest-signing-key`, replacing the `openssl ecparam | pkcs8` pipe. The two `verify` commands are reference verifiers: `attest verify` checks the attestation signature and reports TTL state (a saved attestation is durable evidence, so TTL is not enforced unless you pass `--enforce-ttl`); `receipt verify` checks the receipt signature, the attestation signature, and the `backLink` binding the two, plus an optional `--result` check when the receipt carries a result commitment. Both exit non-zero on any failed check, so they drop straight into CI or an audit script. The conformance surface they cover is documented in [docs/sep2787-conformance.md](docs/sep2787-conformance.md).

Worked examples:

- [`examples/github-mcp-proxy-demo/`](examples/github-mcp-proxy-demo/): Vaara in front of [`github/github-mcp-server`](https://github.com/github/github-mcp-server), 42 tools, hash-chained audit trail recorded end-to-end.
- [`examples/sap-mcp-proxy-demo/`](examples/sap-mcp-proxy-demo/): Vaara in front of community SAP MCP servers ([`SAP/mdk-mcp-server`](https://github.com/SAP/mdk-mcp-server), [`mario-andreschak/mcp-abap-abap-adt-api`](https://github.com/mario-andreschak/mcp-abap-abap-adt-api), [`lemaiwo/btp-sap-odata-to-mcp-server`](https://github.com/lemaiwo/btp-sap-odata-to-mcp-server)).

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
    encoder_binary_identity=encoder_binary_identity(arbiter_version=f"vaara/{vaara.__version__}", policy_hash=ph),
    non_content_metadata={"action_class": "tx.transfer", "decision": "escalate"},
    monotonic_counter=42,
    arbiter_instance_identifier=uuid_bytes,
)
```

`vaara overt verify RECEIPT.cbor --pubkey-file PUB.bin` validates any canonical-CBOR Base Envelope. The verifier reads only the wire format and takes no dependency on Vaara's emitter, so any conformant implementation can route through it.

Adjacent surfaces: a reference Phase 3 IAP (`vaara.attestation.iap`) notary-signs the Provisional Receipt and anchors it in a transparency log (sigstore Rekor swappable); an S3P emitter (`vaara.attestation.s3p`) ships Clopper-Pearson aggregate intervals; an experimental hardware TEE hook (`vaara.attestation.tee`) binds an envelope to an AMD SEV-SNP attestation report via `SHA-512(canonical_cbor(envelope))` in `REPORT_DATA`.

Architectural framing and the OVERT 1.0 Part 3 control walk in [COMPLIANCE.md](COMPLIANCE.md).

## Where things live

| Path | Contents |
|---|---|
| [docs/formal_specification.md](docs/formal_specification.md) | MWU regret bound, conformal coverage, security properties |
| [docs/conformal-prediction.md](docs/conformal-prediction.md) | Plain-language explainer for compliance reviewers and legal counsel |
| [docs/execution-receipts.md](docs/execution-receipts.md) | Execution receipts: the post-execution outcome record paired with SEP-2787 request attestation |
| [docs/sep2787-conformance.md](docs/sep2787-conformance.md) | SEP-2787 conformance surface: what `vaara attest verify` / `vaara receipt verify` check, keyed to the tracked spec revision |
| [COMPLIANCE.md](COMPLIANCE.md) | EU AI Act (Art. 9, 11 to 15, 61) and DORA (Art. 10, 12, 13) mapping, eval numbers, PAIR calibration |
| [VERDICTS.md](VERDICTS.md) | Per-article evidence sufficiency thresholds and decision tree |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version feature evolution |
| [PRIOR_ART.md](PRIOR_ART.md) | When each Vaara concept first shipped, and a neutral list of adjacent published work |
| [OWASP_AGENTIC.md](OWASP_AGENTIC.md) | Vaara mapping to OWASP Top 10 for Agentic Applications 2026 (ASI01 to ASI10) |
| [OVERT_CONTROLS.md](OVERT_CONTROLS.md) | Vaara mapping to OVERT 1.0 Part 3 Agentic AI Controls (TOOL-*, MCP-*, MULTI-*, CAP-*, DISC-*, HITL-*, DRIFT-*) |
| [docs/mit_ai_risk_repository_mapping.md](docs/mit_ai_risk_repository_mapping.md) | Vaara coverage map against the MIT AI Risk Repository v4 (1,835 risk-bearing entries across 7 domains) |
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

<!-- mcp-name: io.github.vaaraio/vaara -->
<!-- mcp-name: io.github.vaaraio/vaara-server -->
