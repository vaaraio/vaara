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

Vaara intercepts agent tool calls, scores each one with a conformal risk interval, and writes a hash-chained audit record. Online learning across five expert signals via Multiplicative Weight Update. Distribution-free conformal coverage on the score.

For broader agent governance (zero-trust identity, capability-based access control, multi-language SDKs) see Microsoft's [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit).

## Install

```bash
pip install vaara
```

Python 3.10+. Zero runtime deps. Optional XGBoost classifier: `pip install vaara[ml]`.

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

## Numbers

- 5,955-entry adversarial corpus (3,422 attack across 8 categories, 2,533 benign)
- 97.1% attack recall on held-out distribution-shift split, threshold 0.55
- PAIR adaptive-attacker calibration: ASR 0/25 against Qwen2.5-32B
- [vaara-bench-v1](bench/vaara-bench-v1.md): 77-trace synthetic-corpus benchmark with frozen methodology, 100% soft TPR, 0% hard FPR
- 140 µs / 210 µs p99 inference latency, commodity CPU
- Distribution-free conformal coverage on the score
- MWU regret bound O(sqrt(T log N))

## Framework integrations

Native adapters in `src/vaara/integrations/` route the major Python agent frameworks through Vaara's pipeline. Each adapter intercepts tool calls via the framework's own callback or hook surface, scores them, gates them, and emits the same audit events as a direct `pipeline.intercept()` call. Frameworks are not hard dependencies (lazy import, duck typing), so the base `pip install vaara` keeps a clean dependency tree.

- **LangChain** — `VaaraCallbackHandler` slots into `config={"callbacks": [...]}` and gates every tool invocation automatically. `vaara_wrap_tool(tool, pipeline)` is the per-tool variant for fine-grained control.
- **CrewAI** — `VaaraCrewGovernance` wraps a crew so every agent action passes through the same scoring and audit chain.
- **OpenAI Agents SDK** — `VaaraToolGuardrail` plus `vaara_wrap_function` wrap function-tool calls before they execute. Compatible with the Responses API and the Agents-SDK tracing model.
- **MCP server** — `vaara.integrations.mcp_server` exposes scoring, audit emission, and policy reload as MCP tools so any MCP-compatible agent can route through Vaara without a custom client.

All four framework adapters share the same in-process pipeline, so audit records hash-chain together regardless of which framework the action came through. Each adapter has its own docstring with the two integration patterns it supports.

### Cloud guardrails as upstream signals

Three adapters route findings from AWS Bedrock Guardrails, Azure AI Content Safety, and GCP Model Armor into Vaara's audit trail and OVERT envelope with EU AI Act article tags. The cloud filter runs in the deployer's environment as an upstream signal. Vaara records the verdict, normalises 27 provider categories onto a shared vocabulary, and tags each finding against Art. 5, 10, 13, 15, 53, and the CSAM-specific obligation from the Digital Omnibus political agreement of May 2026.

- **AWS Bedrock Guardrails** — `vaara.integrations.bedrock_guardrails.BedrockGuardrailsAdapter`. Wraps `ApplyGuardrail` across the five Bedrock policy buckets.
- **Azure AI Content Safety** — `vaara.integrations.azure_content_safety.AzureContentSafetyAdapter`. Wraps `analyze_text`, Prompt Shields, Protected Material, and Groundedness Detection.
- **GCP Model Armor** — `vaara.integrations.gcp_model_armor.GcpModelArmorAdapter`. Wraps `sanitize_user_prompt` and `sanitize_model_response`.

Each adapter returns a `ContentSafetyFinding` the deployer routes into `pipeline.intercept(context=finding.to_audit_context())`. Cloud SDKs are optional extras: `pip install 'vaara[bedrock]'`, `pip install 'vaara[azure-content-safety]'`, `pip install 'vaara[gcp-model-armor]'`. The category-to-article mapping table lives in `src/vaara/integrations/_content_safety_articles.py` and is the value the adapters wrap. Article-level rationale is in [COMPLIANCE.md](COMPLIANCE.md#cloud-guardrail-adapter-pattern).

### OSS guardrails as upstream signals

Four adapters route findings from NVIDIA NeMo Guardrails, Guardrails AI, LLM Guard, and Rebuff into the same audit trail and OVERT envelope path as the v0.19.0 cloud adapters. The OSS guardrail runs in the deployer's environment as an upstream signal. Vaara records the verdict, normalises 41 OSS provider categories onto the same shared vocabulary, and tags each finding against Art. 5, 10, 13, 15, and 53.

- **NVIDIA NeMo Guardrails**. `vaara.integrations.nemo_guardrails.NemoGuardrailsAdapter`. Parses `GenerationResponse.log.activated_rails` across input, dialog, output, and retrieval rail types.
- **Guardrails AI**. `vaara.integrations.guardrails_ai.GuardrailsAIAdapter`. Parses `ValidationOutcome.validation_summaries` from any `Guard.parse` or `Guard.validate` call.
- **LLM Guard**. `vaara.integrations.llm_guard.LLMGuardAdapter`. Wraps `scan_prompt` and `scan_output` and parses the `(sanitized, results_valid, results_score)` triple.
- **Rebuff**. `vaara.integrations.rebuff.RebuffAdapter`. Parses `DetectResponse` across the heuristic, model, and vector injection-detection layers, plus the canary-word leak check on responses.

Each adapter returns a `ContentSafetyFinding` the deployer routes into `pipeline.intercept(context=finding.to_audit_context())`. OSS SDKs are optional extras: `pip install 'vaara[nemo-guardrails]'`, `pip install 'vaara[guardrails-ai]'`, `pip install 'vaara[llm-guard]'`, `pip install 'vaara[rebuff]'`. The 41 new mapping rows extend the same table at `src/vaara/integrations/_content_safety_articles.py`. Article-level rationale is in [COMPLIANCE.md](COMPLIANCE.md#oss-guardrail-adapter-pattern).

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

## OVERT 1.0 attestation

Vaara implements the OVERT 1.0 ([overt.is](https://overt.is/)) Protocol Profile 1.0 Base Envelope. OVERT 1.0 is an open standard for runtime trust in AI systems, authored by Glacis Technologies and published 25 March 2026. Closed-schema 9-field structure at AAL-3 Phase 2 (Provisional Receipt), canonical CBOR (RFC 8949), Ed25519 signatures, HMAC-SHA256 keyed commitments, IEEE-754 float rejection.

```
pip install 'vaara[attestation]'
```

```python
from vaara.attestation.overt import emit_base_envelope, make_request_commitment, encoder_binary_identity

envelope = emit_base_envelope(
    signing_key=key,
    request_commitment=make_request_commitment(payload, operator_key=op_key),
    encoder_binary_identity=encoder_binary_identity(arbiter_version="vaara/0.15.0", policy_hash=ph),
    non_content_metadata={"action_class": "tx.transfer", "decision": "escalate"},
    monotonic_counter=42,
    arbiter_instance_identifier=uuid_bytes,
)
```

Vaara operates as the **Arbiter** in OVERT terms. The reference Phase 3 IAP (`vaara.attestation.iap`) notary-signs the Provisional Receipt and anchors it in a transparency log. Production deployments can swap in sigstore Rekor or an equivalent independently-operated log at the same call sites. The OVERT S3P (MEA-2) emitter at `vaara.attestation.s3p` ships exact Clopper-Pearson confidence intervals (pure Python, no scipy) and a proposed Protocol Profile extension that reports aggregate statistics over per-action conformal prediction intervals alongside the standard binomial CI.

The `vaara overt verify RECEIPT.cbor --pubkey-file PUB.bin` CLI validates any canonical-CBOR Base Envelope against a supplied raw 32-byte Ed25519 public key. The verifier reads only the wire format and takes no dependency on Vaara's emitter, so any OVERT-conformant implementation can route its conformance check through it.

An experimental hardware TEE hook (`vaara.attestation.tee`) binds an OVERT envelope to an AMD SEV-SNP attestation report by placing `SHA-512(canonical_cbor(envelope))` in the report's 64-byte `REPORT_DATA` field. The envelope schema is unchanged (closed per spec). The TEE report is a sibling artefact: a relying party checks the Ed25519 envelope signature and the ECDSA P-384 report signature independently. `vaara tee parse` and `vaara tee verify` expose the verifier as a CLI.

See [COMPLIANCE.md](COMPLIANCE.md) "Position relative to open runtime-attestation standards" for the architectural framing and "OVERT 1.0 Part 3 (Agentic AI Controls) mapping" for the TOOL-*, MCP-*, MULTI-*, CAP-*, DISC-*, HITL-*, DRIFT-* control-by-control walk.

## Where things live

| Path | Contents |
|---|---|
| [docs/formal_specification.md](docs/formal_specification.md) | MWU regret bound, conformal coverage, security properties |
| [COMPLIANCE.md](COMPLIANCE.md) | EU AI Act (Art. 9, 11 to 15, 61) and DORA (Art. 10, 12, 13) mapping, eval numbers, PAIR calibration |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version feature evolution |
| [docs/signing-keys.md](docs/signing-keys.md) | Release signing and verification |
| [SECURITY.md](SECURITY.md) | Security policy and reporting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| `src/vaara/integrations/` | LangChain, OpenAI Agents SDK, CrewAI, MCP, Bedrock, Azure, GCP |
| `src/vaara/audit/` | Hash-chain trail, SQLite backend, append-only WAL |
| `src/vaara/policy/` | YAML / JSON policy schema, `vaara policy validate` and `vaara policy test` |
| `src/vaara/sandbox/` | Synthetic-trace cold-start calibration |

[Article 14 runtime: why oversight of agentic AI has to be evidenced as action, not model](https://futurium.ec.europa.eu/ga/apply-ai-alliance/community-content/article-14-runtime-why-oversight-agentic-ai-has-be-evidenced-action-not-model) is the position post on the EU Apply AI Alliance Futurium.

> Vaara helps deployers assemble evidence for their own conformity work. It does not certify compliance or constitute legal advice. Deployers own their obligations under the EU AI Act and other applicable law.

## License

[LICENSE](LICENSE)
