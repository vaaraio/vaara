<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/vaara-wordmark-dark.png">
    <img src="docs/vaara-wordmark-light.png" alt="Vaara" width="900">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/vaara/"><img src="https://img.shields.io/pypi/v/vaara.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/vaara/"><img src="https://img.shields.io/pypi/pyversions/vaara.svg" alt="Python"></a>
  <a href="https://github.com/vaaraio/vaara/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/vaara.svg" alt="License"></a>
  <a href="https://github.com/vaaraio/vaara/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/vaaraio/vaara/ci.yml?branch=main&label=tests" alt="CI"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/vaaraio/vaara"><img src="https://api.scorecard.dev/projects/github.com/vaaraio/vaara/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/12612"><img src="https://www.bestpractices.dev/projects/12612/badge" alt="OpenSSF Best Practices"></a>
</p>

Vaara is the runtime evidence layer for AI Act compliance. Open source, no SaaS, no telemetry.

Vaara intercepts agent tool calls, scores each one with a conformal risk interval, and writes a hash-chained audit record. Online learning across five expert signals via Multiplicative Weight Update. Distribution-free conformal coverage on the score.

For broader agent governance (zero-trust identity, capability-based access control, multi-language SDKs) see Microsoft's [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit).

## Numbers

- 5,955-entry adversarial corpus (3,422 attack across 8 categories, 2,533 benign)
- 97.1% attack recall on held-out distribution-shift split, threshold 0.55
- PAIR adaptive-attacker calibration: ASR 0/25 against Qwen2.5-32B
- [vaara-bench-v1](bench/vaara-bench-v1.md): 77-trace synthetic-corpus benchmark with frozen methodology, 100% soft TPR, 0% hard FPR
- 140 µs / 210 µs p99 inference latency, commodity CPU
- Distribution-free conformal coverage on the score
- MWU regret bound O(sqrt(T log N))

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

## Framework integrations

Native adapters in `src/vaara/integrations/` route the major Python agent frameworks through Vaara's pipeline. Each adapter intercepts tool calls via the framework's own callback or hook surface, scores them, gates them, and emits the same audit events as a direct `pipeline.intercept()` call. Frameworks are not hard dependencies (lazy import, duck typing), so the base `pip install vaara` keeps a clean dependency tree.

- **LangChain** — `VaaraCallbackHandler` slots into `config={"callbacks": [...]}` and gates every tool invocation automatically. `vaara_wrap_tool(tool, pipeline)` is the per-tool variant for fine-grained control.
- **CrewAI** — `VaaraCrewGovernance` wraps a crew so every agent action passes through the same scoring and audit chain.
- **OpenAI Agents SDK** — `VaaraToolGuardrail` plus `vaara_wrap_function` wrap function-tool calls before they execute. Compatible with the Responses API and the Agents-SDK tracing model.
- **MCP server** — `vaara.integrations.mcp_server` exposes scoring, audit emission, and policy reload as MCP tools so any MCP-compatible agent can route through Vaara without a custom client.

All four adapters share the same in-process pipeline, so audit records hash-chain together regardless of which framework the action came through. Each adapter has its own docstring with the two integration patterns it supports.

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

The contract is in [docs/openapi.yaml](docs/openapi.yaml). Vaara defines the interface. Control-plane and orchestration vendors call it. Integration recipes for adopters live under `examples/recipes/`.

v0.13.0 adds three operator-facing endpoints. `POST /v1/policy/reload` atomically swaps the running policy without restarting the agent process (start with `vaara serve --policy PATH` to enable; in-flight requests keep the old thresholds, the next request sees the new ones). `POST /v1/detect/injection` and `POST /v1/detect/pii` expose Vaara's adversarial scorer and a zero-dependency PII extractor as named buyer-visible endpoints; the corresponding `vaara detect injection` and `vaara detect pii` CLI subcommands exit non-zero when the detector fires, so they slot into CI gates. `vaara compliance dashboard --db PATH --out site/` renders a single-file static HTML article-coverage page from the same evidence model as `vaara compliance report`.

v0.14.0 adds an optional ML-DSA-65 (FIPS 204) signer for the regulator-handoff export envelope (`pip install 'vaara[pq]'`), suitable for retention horizons that cross the credible quantum threshold. The same release adds `vaara.scorer.composition.ExternalScorer` and `vaara.scorer.composite.CompositeScorer` so Vaara's adaptive scorer can be run alongside external scorers (NeMo Guardrails, another Vaara instance, any service that implements the `/v1/score` wire contract); the composite preserves the strongest decision across members.

v0.15.0 ships the first-party TypeScript client at [`clients/ts`](clients/ts) and on npm as `@vaara/client`. Typed wrappers over every v1 endpoint, Node 18+, ESM, declarations shipped. JS/TS agents (LangChain.js, Vercel AI SDK, MCP, any Node service) can now call Vaara without a Python sidecar.

```bash
npm install @vaara/client
```

v0.16.0 adds a PDF render to the article-evidence report. `vaara compliance report --db PATH --format pdf --out report.pdf` writes a styled single-file PDF (per-domain article tables plus per-article detail sections) suitable for attaching to a conformity submission or internal-audit binder. Requires `pip install 'vaara[pdf]'`. Markdown, JSON, and narrative renders remain unchanged.

v0.17.0 adds an OVERT 1.0 Base Envelope verifier CLI. `vaara overt verify RECEIPT.cbor --pubkey-file PUB.bin` validates any canonical-CBOR Base Envelope (Annex B.6) against a supplied raw 32-byte Ed25519 public key. The schema is closed per the OVERT 1.0 spec, so envelopes carrying unknown fields are rejected. The verifier reads only the wire format and takes no dependency on Vaara's emitter, so any OVERT-conformant implementation can route its conformance check through it. Requires the `vaara[attestation]` extra.

```ts
import { VaaraClient } from "@vaara/client";
const vaara = new VaaraClient({ baseUrl: "http://localhost:8000" });
const r = await vaara.score({ tool_name: "tx.transfer", agent_id: "agent-007", base_risk_score: 0.6 });
if (r.decision === "deny") throw new Error("blocked");
```

## OVERT 1.0 attestation

Vaara implements the OVERT 1.0 ([overt.is](https://overt.is/)) Protocol Profile 1.0 Base Envelope. OVERT 1.0 is an open standard for runtime trust in AI systems, authored by Glacis Technologies and published 25 March 2026. Closed-schema 9-field structure at AAL-3 Phase 2 (Provisional Receipt), canonical CBOR (RFC 8949), Ed25519 signatures, HMAC-SHA256 keyed commitments, IEEE-754 float rejection. v0.13.0 adds a reference Phase 3 IAP (`vaara.attestation.iap`) that notary-signs the Provisional Receipt and anchors it in a transparency log; production deployments can swap in sigstore Rekor or an equivalent independently-operated log at the same call sites.

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

Vaara operates as the **Arbiter** in OVERT terms. See [COMPLIANCE.md](COMPLIANCE.md) "Position relative to open runtime-attestation standards" for the architectural framing.

v0.12.0 adds an OVERT S3P (MEA-2) emitter with exact Clopper-Pearson confidence intervals (pure Python, no scipy), plus a proposed Protocol Profile extension that reports aggregate statistics over Vaara's per-action conformal prediction intervals alongside the standard binomial CI. The agentic-controls mapping in [COMPLIANCE.md](COMPLIANCE.md) "OVERT 1.0 Part 3 (Agentic AI Controls) mapping" walks Vaara's coverage of TOOL-*, MCP-*, MULTI-*, CAP-*, DISC-*, HITL-*, and DRIFT-* control by control.

```python
from vaara.attestation.s3p import emit_s3p_attestation, ConformalExtension, make_epoch_nonce_commitment
```

## Where things live

- [docs/formal_specification.md](docs/formal_specification.md): math. MWU regret bound O(sqrt(T log N)), conformal coverage guarantees, security properties.
- [COMPLIANCE.md](COMPLIANCE.md): Article-level evidence mapping for EU AI Act (Articles 9, 11 to 15, 61) and DORA (Articles 10, 12, 13). Eval numbers, threshold sweeps, PAIR adversarial calibration.
- [Article 14 runtime: why oversight of agentic AI has to be evidenced as action, not model](https://futurium.ec.europa.eu/ga/apply-ai-alliance/community-content/article-14-runtime-why-oversight-agentic-ai-has-be-evidenced-action-not-model): why this exists. Posted on the EU Apply AI Alliance Futurium.
- `src/vaara/integrations/`: LangChain, OpenAI Agents SDK, CrewAI, MCP server.
- `src/vaara/audit/`: hash-chain trail, SQLite backend, append-only WAL.
- `src/vaara/policy/`: declarative YAML / JSON policy schema with `vaara policy validate` (semantic checks) and `vaara policy test` (Conftest-style cases-file runner) for reviewing the policy artifact in CI independently from agent code.
- `src/vaara/sandbox/`: synthetic-trace cold-start calibration.

> Vaara helps deployers assemble evidence for their own conformity work. It does not certify compliance or constitute legal advice. Deployers own their obligations under the EU AI Act and other applicable law.

## License

[LICENSE](LICENSE)
