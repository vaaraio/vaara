<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vaaraio/vaara/main/docs/vaara-wordmark-dark.png">
    <img src="https://raw.githubusercontent.com/vaaraio/vaara/main/docs/vaara-wordmark-light.png" alt="Vaara" width="900">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/vaara/"><img src="https://img.shields.io/pypi/v/vaara.svg" alt="PyPI"></a>
  <a href="https://github.com/vaaraio/vaara/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/vaara.svg" alt="License"></a>
  <a href="https://github.com/vaaraio/vaara/actions/workflows/ci.yml"><img src="https://github.com/vaaraio/vaara/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/vaaraio/vaara"><img src="https://github.com/vaaraio/vaara/actions/workflows/scorecard.yml/badge.svg" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/12612"><img src="https://www.bestpractices.dev/projects/12612/badge" alt="OpenSSF Best Practices"></a>
  <a href="https://huggingface.co/spaces/vaaraio/vaara"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
</p>

Your AI agent transferred the funds, wrote the file, called the tool. Later, someone who does not trust you asks you to prove exactly what it did and why. A regulator, an auditor, a customer after an incident. Your own logs will not settle it, because you could have edited them.

Vaara is an open-source evidence layer for AI governance. It checks every agent tool call against your policy, writes the call and its outcome into a hash-chained, signed record, and binds that record to your machine's own TPM 2.0 hardware root. An outside party can verify the whole trail offline, with no access to your system and none of your software. EU AI Act Article 12 record-keeping is what it was built for; it answers any "show me what the agent actually did" just as well.

It runs entirely in your own environment. No SaaS, no telemetry. Python 3.10+, zero runtime dependencies.

## Install and first call

```bash
pip install vaara
```

Releases ship SLSA Build Level 3 provenance, verifiable with `slsa-verifier verify-artifact`. Optional ML classifier: `pip install 'vaara[ml]'`.

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

Every call gets a risk score and an allow / block / escalate decision against your policy, then the call, the decision, and the real outcome are written to the audit trail. `report_outcome` closes the loop: the scorer reweights based on which signals actually predicted the outcome.

That is the whole loop. The rest of this page is what makes the record worth keeping.

## Verify it without trusting the producer

Writing a trail is the easy half. The half that matters is letting someone who does not trust you check it, with no key, no access, and none of your code. Every Vaara record is content-addressed and fail-closed on authenticity, and ships with public conformance vectors plus a standalone checker that imports no Vaara code, so an independent party reproduces every verdict offline.

```bash
vaara verify-bundle evidence-bundle.json
```

`ok` only when a signature is actually established, not merely present in a log. The same property drives the standards work behind [SEP-2828](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2828): evidence that holds up for someone who runs none of your software. The full verifier set, the trust model for each verb, and where trust comes from in each case are in [docs/verifying-evidence.md](docs/verifying-evidence.md).

## What the evidence looks like

`vaara compliance report --format json` against a real trail produces an article-level evidence record an auditor reads directly. Articles with no recorded events return `evidence_insufficient`, not a rubber stamp.

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

Each verdict carries the threshold-versus-observed snapshot, the rationale, and the underlying records, so a reviewer traces `status` back to a concrete event. The same data renders as a Notified-Body PDF, a static HTML dashboard, or a Sigstore-signed handoff envelope. See [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

## What you get

- **Gate every tool call** against your own policy: allow, block, or escalate.
- **A tamper-evident trail** an outside party verifies without trusting your stack, with the chain head anchorable to an external RFC 3161 / eIDAS timestamp so its existence is provable against a clock you do not control.
- **Article-level EU AI Act evidence**, honest about the gaps instead of papering over them.
- **Governance of the model call itself**, not only the tools around it: a hardware-rooted inference receipt that a second, different local model cross-checks. This is the sovereign inference harness, new in v1.0.

## Where it plugs in

Native adapters route the major Python agent frameworks through the same pipeline, each via the framework's own hook, emitting identical audit events:

| Framework | Entry point |
|---|---|
| LangChain | `VaaraCallbackHandler`, `vaara_wrap_tool` |
| CrewAI | `VaaraCrewGovernance` |
| OpenAI Agents SDK | `VaaraToolGuardrail`, `vaara_wrap_function` |
| MCP server | `vaara.integrations.mcp_server` |

To put Vaara **in front of** an MCP server, run it as a proxy. Every `tools/call` routes through the pipeline before reaching the upstream; allowed calls forward transparently, blocked calls return an MCP error.

```bash
vaara-mcp-proxy \
  --upstream npx --upstream-arg -y --upstream-arg @sap/mdk-mcp-server \
  --db ./mcp_audit.db
```

Point your MCP client (Claude Code, Cursor, any host) at the proxy instead of the upstream. There is also an HTTP API (`pip install 'vaara[server]'`, `vaara serve`) and a first-party TypeScript client on npm ([`@vaara/client`](clients/ts)) for non-Python agents. Framework details, the cloud and OSS guardrail adapters (Bedrock, Azure, GCP, NeMo, Guardrails AI, LLM Guard, Rebuff), and the multi-tenant proxy are in [docs/adapters.md](docs/adapters.md).

## How it scores

Each risk score blends five expert signals and keeps adapting as outcomes come back, and it carries a confidence interval with a coverage guarantee that holds regardless of the input distribution. On a held-out adversarial corpus the classifier reaches **84.7%** recall (95% Wilson [82.4, 86.7]) at a **4.1%** false-positive rate, and **1.2%** FPR on benign calls under live injection pressure. The hot-path rule scorer adds 140 µs mean per call on commodity CPU; the ML classifier is opt-in (`vaara[ml]`) and off that path. Every figure is reproducible via `make bench`.

<details>
<summary>Full numbers, corpus, calibration, and chain of custody</summary>

- 12,155-entry adversarial corpus (250 hand-curated + 11,905 LLM-generated), 70/15/15 split stratified by (category, source).
- Classifier v9 (236 hand-features + 384-dim MiniLM embeddings) at calibrated threshold 0.9150 on held-out TEST n=1,827: recall 84.7% [82.4, 86.7] at FPR 4.1% [2.9, 5.7].
- Cross-model held-out recall 66.8% [64.9, 68.7] over n=2,277 with no eval-set attacker model in TRAIN; the weakest sub-cell is data_exfil against a closed-weight model at 38.9%. This is the honest worst case; the in-distribution number above is the easier denominator.
- BIPIA-pressure FPR on benign tool calls 1.2% [0.4, 3.6] across four agent backends (Claude Haiku 4.5, Llama-3.1-8B, Mistral-7B, Qwen-2.5-7B). Down from 35.2% on v8.
- Multi-attacker PAIR robustness: 0/25 successes per attacker across Qwen2.5-32B, Qwen2.5-72B, Llama-3.3-70B on identical seeds, Wilson upper 13.3%.
- Distribution-free conformal coverage on the score; MWU regret bound O(sqrt(T log N)).
- Chain of custody: corpus, split, training commit, and bundle SHAs locked and printed by every script.

Method and per-cell breakdown: [docs/architecture.md](docs/architecture.md) and [bench/](bench/).
</details>

## Standards and attestation

- **[SEP-2828](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2828)** signed execution records and **SEP-2787** request-attestation test vectors, in the MCP standards process. A second independent implementation has reproduced the SEP-2828 conformance vectors from a clean checkout with no shared code.
- **OVERT 1.0** ([overt.is](https://overt.is/)): Vaara is the Arbiter and emits Protocol Profile 1.0 Base Envelopes (canonical CBOR, Ed25519) alongside every record when attestation is on.
- **Post-quantum**: an optional parallel ML-DSA-65 / FIPS 204 signature over the same preimage, so a stripped post-quantum signature is a detectable downgrade rather than a silent loss.
- **Sovereign inference harness** (v1.0): a local model behind a signing proxy that emits a hardware-rooted inference receipt a second local model cross-checks. Developed privately, published here under AGPL-3.0.

Details and the offline checkers for each: [docs/standards.md](docs/standards.md).

## Docs

| Path | Contents |
|---|---|
| [docs/verifying-evidence.md](docs/verifying-evidence.md) | Every verifier and its trust model |
| [docs/architecture.md](docs/architecture.md) | Scoring, conformal coverage, time anchor, formal properties |
| [docs/standards.md](docs/standards.md) | SEP-2828, SEP-2787, OVERT, the sovereign inference harness |
| [docs/adapters.md](docs/adapters.md) | Framework and cloud/OSS guardrail adapters, multi-tenant proxy |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | EU AI Act and DORA article mapping, eval numbers |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version evolution |
| [docs/PRIOR_ART.md](docs/PRIOR_ART.md) | When each concept first shipped, plus adjacent work |

## Acknowledgements

- Listed in the industry acknowledgements of the [IMDA Model AI Governance Framework for Agentic AI v1.5](https://www.imda.gov.sg/-/media/imda/files/about/emerging-tech-and-research/artificial-intelligence/mgf-for-agentic-ai.pdf) (Singapore, 20 May 2026).
- The [AMD AI Developer Program](https://www.linkedin.com/posts/amd-developer_meet-henri-sirkkavaara-henri-created-vaara-activity-7459667676555132928-QFSd) ran a developer testimonial of Vaara in May 2026.
- [Article 14 runtime: why oversight of agentic AI has to be evidenced as action, not model](https://futurium.ec.europa.eu/ga/apply-ai-alliance/community-content/article-14-runtime-why-oversight-agentic-ai-has-be-evidenced-action-not-model), the position post on the EU Apply AI Alliance Futurium.

> Vaara helps deployers assemble evidence for their own conformity work. It does not certify compliance or constitute legal advice. Deployers own their obligations under the EU AI Act and other applicable law.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

<!-- mcp-name: io.github.vaaraio/vaara -->
<!-- mcp-name: io.github.vaaraio/vaara-server -->
