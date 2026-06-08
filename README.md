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

Vaara is the open-source runtime evidence layer for AI agents under the EU AI Act. It sits in front of an agent's tool calls, gates each one against your policy, and writes a tamper-evident record an outside party can verify. When a regulator, an auditor, or a public-sector buyer needs proof of what your agent actually did and why, that record is the answer. Runs entirely in your own environment. No SaaS, no telemetry.

EU AI Act Article 12 record-keeping is the driver. The same trail answers any "show me exactly what the agent did" demand: procurement validation, incident reconstruction, SOC 2 evidence.

- Article-level EU AI Act evidence report, honest about the gaps instead of rubber-stamping them.
- Hash-chained, tamper-evident audit trail an outside party can verify without trusting your stack, with the chain head anchorable to an external trusted timestamp (RFC 3161 / eIDAS).
- Gate every agent tool call against your own policy: allow, block, or escalate.

## How it works

Every tool call an agent makes passes through Vaara before it runs:

1. **Intercept.** Vaara catches the call (`fs.write_file`, `tx.transfer`, an MCP `tools/call`, and so on) through your framework's own hook, or transparently as an MCP proxy in front of an upstream server.
2. **Score and decide.** Each call gets a risk score and an allow / block / escalate decision against your policy.
3. **Record.** The call, the score, the decision, and the real-world outcome are written to a hash-chained audit trail. An outside auditor can verify the chain is intact without trusting your stack or your word.

The scoring blends five expert signals and keeps adapting as outcomes come back, and each risk score carries a confidence interval with a coverage guarantee that holds regardless of the input distribution. Those are the properties an auditor can check independently; the math is in [Benchmarks](#benchmarks) and [docs/formal_specification.md](docs/formal_specification.md).

### External time anchor

The hash chain proves order and integrity but not *when* it existed: every timestamp comes from your own clock, so a compromised signing key could in principle be used to forge a backdated chain. Vaara can anchor the current chain head to an external RFC 3161 Time-Stamp Authority, the standard behind eIDAS qualified electronic timestamps. The authority signs the chain head and the time, so the chain's existence is provable against a clock you do not control. Verification is offline.

```bash
pip install 'vaara[timeanchor]'
```

```python
from vaara.audit.timeanchor import RFC3161TimeAnchorClient

# Periodically, or after a batch of high-risk actions:
trail.anchor_head(RFC3161TimeAnchorClient("https://freetsa.org/tsr"))
```

The anchor also folds into the one-command regulator package: `vaara trail export-article12 --anchor-tsa https://freetsa.org/tsr` writes the timestamp beside the signed trail as Article 19 existence-in-time evidence, and `vaara trail verify-anchor --zip <package>.zip` checks it offline.

## Install

```bash
pip install vaara
```

Python 3.10+. Zero runtime deps. Optional XGBoost classifier: `pip install vaara[ml]`. Releases ship with SLSA Build Level 3 provenance, verifiable via `slsa-verifier verify-artifact`.

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

`report_outcome` closes the loop: the signal weights reweight based on which ones predicted the outcome.

## What evidence looks like

`vaara compliance report --format json` against a real audit trail produces an article-level evidence record an auditor can read directly. Articles without recorded events return `evidence_insufficient`, not a rubber-stamp.

```json
{
  "system_name": "Acme HR Assistant",
  "overall_status": "evidence_insufficient",
  "trail_integrity": {"size": 105, "chain_intact": true},
  "articles": [
    {"article": "Article 12(1)", "title": "Record-Keeping (Logging)",
     "status": "evidence_sufficient", "strength": "strong", "evidence_count": 105},
    {"article": "Article 15(1)", "title": "Accuracy, Robustness and Cybersecurity",
     "status": "evidence_insufficient", "strength": "absent", "evidence_count": 0}
  ]
}
```

Each verdict carries the threshold-vs-observed snapshot, the rationale, and the underlying audit records, so a reviewer can trace `status` back to a concrete event without re-running the engine. The same data renders as a styled PDF for Notified Bodies (`--format pdf`, needs `vaara[pdf]`), a static HTML dashboard (`vaara compliance dashboard`), or a Sigstore-signed handoff envelope (`vaara trail export`, optional ML-DSA-65 / FIPS 204 post-quantum signer via `vaara[pq]`).

## Verify the evidence

Producing the trail is half the job. The other half is letting someone who does not trust you check it. `vaara verify-bundle` takes one evidence bundle and runs every check that applies to it, then prints a single verdict:

```bash
vaara verify-bundle evidence-bundle.json
```

No code to write, and no need to trust the tooling that produced the bundle. The command runs six lenses and is fail-closed on authenticity, so a record that is merely present in a log, with its signature never checked, does not pass:

- **Identity** resolves the signing key to a `did:web` the agent controls, so the receipt names who acted, not just that something signed it.
- **Signature** verifies the receipt under that key.
- **Back-link** checks that the receipt binds to the request attestation it answers and to the prior chain head.
- **Inclusion** checks that the record is in the transparency log.
- **Consistency** checks that the log is append-only, so an earlier verified head stays consistent with the current one and nothing was rewritten behind you.
- **Revocation** checks that no key or receipt in the chain has been revoked, across stacks.

`ok` is true only when the signature is actually established and every applicable lens passes. A bundle that proves inclusion and non-revocation but never verifies a signature is not `ok`. Each lens also ships as public conformance vectors with a standalone checker that imports no Vaara code, so an independent party reproduces every verdict offline. That property is the point of the standards work behind [SEP-2828](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2828): the evidence is verifiable by someone who runs none of your software.

### Build the bundle

`build-bundle` is the issuer side of the same file. Where `verify-bundle` checks a bundle, `build-bundle` produces it, from the receipt and whatever identity, signature, inclusion, consistency, and revocation material you hold:

```bash
vaara build-bundle --from-dir ./pieces --out evidence-bundle.json
vaara verify-bundle evidence-bundle.json
```

It writes the exact document `verify-bundle` reads, then loads it back and reports the verdict, so producing and checking the evidence is one closed loop over one file.

### Check any record

`verify-bundle` checks a bundle you assembled. `verify-record` checks the format itself: point it at any JSON that claims to be a SEP-2828 execution record, including one Vaara never produced, and it tells you whether the record is well formed and internally consistent.

```bash
vaara verify-record someone-elses-record.json
```

It needs no signing key and no attestation. The check is the wire schema plus the one binding a record proves about itself: the result commitment digest is the SHA-256 of the bytes it sits beside, so a verifier recomputes it with nothing but a hash function. Add `--attestation` to also check the back-link to the request the record answers, still without a key. The signature check, which does need the signer's key, stays in `vaara receipt verify`. This is the check an auditor, or a vendor whose software you do not run, can apply before trusting the producer or any key. The trust rests on the format, not on Vaara.

### The auditor's workbench

When the evidence is a folder of records or bundles rather than one file, each single-file command above has a set-level form that runs over a whole directory:

```bash
vaara verify-records ./records
vaara verify-bundles ./bundles
vaara audit-summary  ./records --out summary.md
```

- `verify-records` checks every record for SEP-2828 conformance, then checks the set as a whole: it flags a call recorded twice, an authorised decision with no matching outcome, and an executed action that committed no result. Keyless, like `verify-record`.
- `verify-bundles` runs the full six-lens `verify-bundle` over every bundle and reports per-lens pass counts and how many bundles authenticated.
- `audit-summary` renders the conformance verdict for a directory of records as a Markdown page an auditor reads directly. The page states what was checked and every count, and records that any party can reproduce it from the records alone.

## Benchmarks

Held-out test recall **84.7%** (95% Wilson [82.4, 86.7]) at a **4.1%** false-positive rate, and **1.2%** FPR on benign tool calls under live injection pressure. The hot-path rule scorer adds 140 µs mean / 210 µs p99 per call on commodity CPU. Every figure is reproducible end-to-end via `make bench`.

<details>
<summary>Full numbers, corpus, calibration, and chain of custody</summary>

- 12,155-entry adversarial corpus (250 hand-curated + 11,905 LLM-generated), 70/15/15 split stratified by (category, source)
- Classifier v9 (236 hand-features + 384-dim MiniLM embeddings) at calibrated threshold 0.9150 on held-out TEST n=1,827: recall 84.7% [82.4, 86.7] at FPR 4.1% [2.9, 5.7]. Phase 1 PAIR scale-up to n=300 per attacker family lands at 88.1% [85.8, 90.1]
- Cross-model held-out recall 66.8% [64.9, 68.7] over n=2,277 with no eval-set attacker model in TRAIN; the weakest sub-cell is data_exfil against a closed-weight model at 38.9% [35.3, 42.5]. This is the honest worst case; the in-distribution number above is the easier denominator
- BIPIA-pressure FPR on benign tool calls 1.2% [0.4, 3.6] across four agent backends (Claude Haiku 4.5, Llama-3.1-8B, Mistral-7B, Qwen-2.5-7B), n=244. Collapses from 35.2% on v8 to 1.2% on v9
- Multi-attacker PAIR robustness: 0/25 successes per attacker across Qwen2.5-32B, Qwen2.5-72B, Llama-3.3-70B on identical seeds, Wilson upper 13.3%
- 140 µs mean / 210 µs p99 for the hot-path rule scorer on commodity CPU; the MiniLM classifier is opt-in (`vaara[ml]`) and not in that path
- Distribution-free conformal coverage on the score; MWU regret bound O(sqrt(T log N))
- Chain of custody: corpus, split, training commit, and bundle SHAs locked and printed by every script
- Current methodology and ship-gate record in [vaara-bench-v0.39](bench/vaara-bench-v0.39.md); per-cell breakdown in [vaara-bench-v0.37](bench/vaara-bench-v0.37.md). Historical bench docs live under `bench/`

Each figure is reproducible from the public corpus or the bench pipeline in `bench/`.
</details>

## Framework adapters

Native adapters in `src/vaara/integrations/` route the major Python agent frameworks through Vaara's pipeline. Each intercepts via the framework's own callback or hook surface, scores, gates, and emits the same audit events as a direct `pipeline.intercept()`. Frameworks are not hard dependencies (lazy import, duck typing), so audit records hash-chain together regardless of which one the action came through.

| Framework | Entry point | Use |
|---|---|---|
| LangChain | `VaaraCallbackHandler`, `vaara_wrap_tool` | Slots into `config={"callbacks": [...]}` or wraps per-tool |
| CrewAI | `VaaraCrewGovernance` | Wraps a crew so every agent action passes through scoring + audit |
| OpenAI Agents SDK | `VaaraToolGuardrail`, `vaara_wrap_function` | Function-tool wrap, compatible with Responses API and Agents-SDK tracing |
| MCP server | `vaara.integrations.mcp_server` | Exposes scoring, audit, policy reload as MCP tools |

For Vaara *in front of* an upstream MCP server, see [MCP proxy](#mcp-proxy) below.

## Upstream-signal adapters (cloud + OSS guardrails)

Adapters route findings from cloud and OSS guardrails into Vaara's audit trail and OVERT envelope. The filter runs in the deployer's environment; Vaara records the verdict, normalises 68 provider categories onto a shared vocabulary, and tags each finding against the relevant AI Act articles. Each adapter returns a `ContentSafetyFinding` the deployer routes into `pipeline.intercept(context=finding.to_audit_context())`. Article-by-article mapping in [COMPLIANCE.md](docs/COMPLIANCE.md).

<details>
<summary>Seven cloud and OSS guardrails: Bedrock, Azure, GCP, NeMo, Guardrails AI, LLM Guard, Rebuff</summary>

| Provider | Adapter | Extra | Wraps |
|---|---|---|---|
| AWS Bedrock Guardrails | `BedrockGuardrailsAdapter` | `vaara[bedrock]` | `ApplyGuardrail` across five Bedrock policy buckets |
| Azure AI Content Safety | `AzureContentSafetyAdapter` | `vaara[azure-content-safety]` | `analyze_text`, Prompt Shields, Protected Material, Groundedness |
| GCP Model Armor | `GcpModelArmorAdapter` | `vaara[gcp-model-armor]` | `sanitize_user_prompt`, `sanitize_model_response` |
| NVIDIA NeMo Guardrails | `NemoGuardrailsAdapter` | `vaara[nemo-guardrails]` | `GenerationResponse.log.activated_rails` (input / dialog / output / retrieval) |
| Guardrails AI | `GuardrailsAIAdapter` | `vaara[guardrails-ai]` | `ValidationOutcome.validation_summaries` from `Guard.parse` / `Guard.validate` |
| LLM Guard | `LLMGuardAdapter` | `vaara[llm-guard]` | `scan_prompt` / `scan_output` |
| Rebuff | `RebuffAdapter` | `vaara[rebuff]` | `DetectResponse` across heuristic, model, vector layers + canary-word leak check |

Mapping table at `src/vaara/integrations/_content_safety_articles.py`. Rationale in [COMPLIANCE.md](docs/COMPLIANCE.md#cloud-guardrail-adapter-pattern).
</details>

## HTTP API

The same scorer and audit trail are available over HTTP for non-Python agents and control planes that prefer a network boundary.

```bash
pip install 'vaara[server]'
vaara serve --host 0.0.0.0 --port 8000

curl -sX POST http://localhost:8000/v1/score \
  -H 'content-type: application/json' \
  -d '{"tool_name":"tx.transfer","agent_id":"agent-007","base_risk_score":0.5}'
```

Wire contract in [docs/openapi.yaml](docs/openapi.yaml). Operator endpoints include `POST /v1/policy/reload` (atomic hot policy swap) and named detectors `POST /v1/detect/injection` and `POST /v1/detect/pii`, with matching CLI subcommands that exit non-zero on detection for CI gating.

The first-party TypeScript client ships on npm as [`@vaara/client`](clients/ts): typed wrappers over every v1 endpoint, Node 18+, ESM. JS/TS agents call Vaara without a Python sidecar.

```ts
import { VaaraClient } from "@vaara/client";
const vaara = new VaaraClient({ baseUrl: "http://localhost:8000" });
const r = await vaara.score({ tool_name: "tx.transfer", agent_id: "agent-007", base_risk_score: 0.6 });
if (r.decision === "deny") throw new Error("blocked");
```

## MCP proxy

`VaaraMCPProxy` sits between an MCP client (Claude Code, Cursor, any MCP host) and an upstream MCP server. Every `tools/call` routes through Vaara's pipeline before reaching the upstream: allowed calls forward transparently and report the outcome back to the scorer, blocked calls return an MCP `isError: true` with the reason. The handshake and `notifications/*` forward unchanged.

```bash
vaara-mcp-proxy \
  --upstream npx --upstream-arg -y --upstream-arg @sap/mdk-mcp-server \
  --db ./mcp_audit.db
```

Point your MCP client at the proxy instead of the upstream; the audit chain captures every call without changing client or upstream behavior. Upstreams can be local (`--upstream` launches a local stdio server) or remote (`--upstream-url NAME=URL` over Streamable HTTP). This is distinct from `mcp_server`, which exposes Vaara itself as a tool.

<details>
<summary>Fleet shape: one proxy, many upstreams, multi-tenant policy</summary>

`vaara-mcp-proxy` also runs over Streamable HTTP with fan-out, so one process can serve a fleet:

```bash
vaara-mcp-proxy \
  --transport http --http-host 127.0.0.1 --http-port 8765 \
  --upstream 'github=npx -y @github/mcp-server' \
  --upstream 'sap=npx -y @sap/mdk-mcp-server'
```

Each `POST /mcp` reads two headers: `X-Vaara-Upstream` picks the upstream slot, `X-Vaara-Tenant` scopes the policy, audit chain, and OVERT envelope. Single-upstream deployments keep the silent-default contract; multi-upstream deployments require `X-Vaara-Upstream` per call and return 400 with the slot list when it is missing. `vaara serve --policy-dir DIR` loads one policy per file (filename stem becomes `tenant_id`, `default.yaml` is the fallback) and hot-reloads per tenant.
</details>

<details>
<summary>Operator perimeter and request attestation</summary>

Repeatable `--allow-tool` / `--deny-tool` flags (and the same for resources and prompts) filter the MCP surface. Filtered tools are dropped from `tools/list` before the client sees them and any matching call is rejected at the perimeter without contacting the upstream. Denylist wins on overlap; no flags means passthrough. Every allowed `resources/read` and `prompts/get` writes a request+decision audit pair so a regulator can reconstruct exactly what the agent read.

OVERT envelopes per interaction turn on with `--overt-signing-key`, `--overt-operator-key`, `--overt-receipts-dir`. SEP-2787 request attestation paired with an execution receipt turns on with `--attest-signing-key PATH` and `--attest-receipts-dir DIR`: each allowed call writes a pre-execution attestation and a post-execution receipt linked by a `backLink` digest. Key type auto-detects from the file (EC P-256 = ES256, RSA = RS256, raw bytes = HS256). Generate and verify offline:

```
vaara keygen --attest --out attest_key.pem
vaara attest verify  0000000001-ab12cd34-attest.json  --pubkey-file attest_key.pem.pub
vaara receipt verify 0000000001-ab12cd34-receipt.json --attestation 0000000001-ab12cd34-attest.json --pubkey-file attest_key.pem.pub
```

Both verifiers exit non-zero on any failed check, so they drop straight into CI. Format in [docs/execution-receipts.md](docs/execution-receipts.md), conformance surface in [docs/sep2787-conformance.md](docs/sep2787-conformance.md).
</details>

Worked examples: [`examples/github-mcp-proxy-demo/`](examples/github-mcp-proxy-demo/) (Vaara in front of `github/github-mcp-server`, 42 tools) and [`examples/sap-mcp-proxy-demo/`](examples/sap-mcp-proxy-demo/) (community SAP MCP servers).

## OVERT 1.0 attestation

OVERT 1.0 is an open standard for runtime trust in AI systems ([overt.is](https://overt.is/), authored by Glacis Technologies, published 25 March 2026): a signed, schema-closed envelope a relying party can verify offline without trusting the emitter. Vaara is the **Arbiter** in OVERT terms and ships Protocol Profile 1.0 Base Envelopes (canonical CBOR per RFC 8949, Ed25519 signatures, HMAC-SHA256 commitments, closed 9-field schema) alongside every audit record when attestation is enabled.

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

`vaara overt verify RECEIPT.cbor --pubkey-file PUB.bin` validates any canonical-CBOR Base Envelope. The verifier reads only the wire format and takes no dependency on Vaara's emitter, so any conformant implementation can route through it. Adjacent surfaces (`vaara.attestation.iap` notary + transparency log, `vaara.attestation.s3p` aggregate intervals, an experimental AMD SEV-SNP TEE hook) and the OVERT 1.0 Part 3 control walk are in [COMPLIANCE.md](docs/COMPLIANCE.md).

## Where things live

| Path | Contents |
|---|---|
| [docs/formal_specification.md](docs/formal_specification.md) | MWU regret bound, conformal coverage, security properties |
| [docs/conformal-prediction.md](docs/conformal-prediction.md) | Plain-language explainer for compliance reviewers and legal counsel |
| [docs/execution-receipts.md](docs/execution-receipts.md) | Execution receipts paired with SEP-2787 request attestation |
| [docs/sep2787-conformance.md](docs/sep2787-conformance.md) | What `vaara attest verify` / `vaara receipt verify` check |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | EU AI Act (Art. 9, 11 to 15, 61) and DORA (Art. 10, 12, 13) mapping, eval numbers |
| [docs/VERDICTS.md](docs/VERDICTS.md) | Per-article evidence sufficiency thresholds and decision tree |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version feature evolution |
| [docs/PRIOR_ART.md](docs/PRIOR_ART.md) | When each Vaara concept first shipped, plus adjacent published work |
| [docs/OWASP_AGENTIC.md](docs/OWASP_AGENTIC.md) | Mapping to OWASP Top 10 for Agentic Applications 2026 |
| [docs/OVERT_CONTROLS.md](docs/OVERT_CONTROLS.md) | Mapping to OVERT 1.0 Part 3 Agentic AI Controls |
| [docs/mit_ai_risk_repository_mapping.md](docs/mit_ai_risk_repository_mapping.md) | Coverage map against the MIT AI Risk Repository v4 |
| [docs/signing-keys.md](docs/signing-keys.md) | Release signing and verification |
| [.github/SECURITY.md](.github/SECURITY.md) | Security policy and reporting |
| [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) | Contribution guidelines |

Acknowledgements:

- Vaara is listed in the industry acknowledgements of the [IMDA Model AI Governance Framework for Agentic AI v1.5](https://www.imda.gov.sg/-/media/imda/files/about/emerging-tech-and-research/artificial-intelligence/mgf-for-agentic-ai.pdf) (Singapore, 20 May 2026).
- The [AMD AI Developer Program](https://www.linkedin.com/posts/amd-developer_meet-henri-sirkkavaara-henri-created-vaara-activity-7459667676555132928-QFSd) ran a coordinated multi-channel developer testimonial of Vaara in May 2026.
- [Article 14 runtime: why oversight of agentic AI has to be evidenced as action, not model](https://futurium.ec.europa.eu/ga/apply-ai-alliance/community-content/article-14-runtime-why-oversight-agentic-ai-has-be-evidenced-action-not-model) is the position post on the EU Apply AI Alliance Futurium.

> Vaara helps deployers assemble evidence for their own conformity work. It does not certify compliance or constitute legal advice. Deployers own their obligations under the EU AI Act and other applicable law.

## License

Apache 2.0. See [LICENSE](LICENSE).

<!-- mcp-name: io.github.vaaraio/vaara -->
<!-- mcp-name: io.github.vaaraio/vaara-server -->
