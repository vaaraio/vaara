<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vaaraio/vaara/main/docs/vaara-wordmark-dark.png">
    <img src="https://raw.githubusercontent.com/vaaraio/vaara/main/docs/vaara-wordmark-light.png" alt="Vaara" width="900">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/vaara/"><img src="https://raw.githubusercontent.com/vaaraio/vaara/badges/pypi.svg" alt="PyPI"></a>
  <a href="https://github.com/vaaraio/vaara/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/vaara.svg" alt="License"></a>
  <a href="https://github.com/vaaraio/vaara/actions/workflows/ci.yml"><img src="https://github.com/vaaraio/vaara/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/vaaraio/vaara"><img src="https://github.com/vaaraio/vaara/actions/workflows/scorecard.yml/badge.svg" alt="OpenSSF Scorecard"></a>
  <a href="https://www.bestpractices.dev/projects/12612"><img src="https://www.bestpractices.dev/projects/12612/badge" alt="OpenSSF Best Practices"></a>
  <a href="https://huggingface.co/spaces/vaaraio/vaara"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/vaaraio/vaara/badges/downloads.svg" alt="Downloads">
</p>

<p align="center"><b>Accountable Autonomy.</b></p>

<p align="center">A verifiable receipt for every action your AI agents take, checkable by anyone.</p>

Your AI agent transferred the funds, wrote the file, called the tool. Later, someone who does not trust you asks you to prove exactly what it did and why: a regulator, an auditor, a customer after an incident. Your own logs will not settle it, because you could have edited them.

Vaara checks every agent tool call against your policy and writes the call and its outcome into a signed, hash-chained record an outside party can verify offline, with no access to your system and none of your software. It needs no special hardware, and binds to your machine's TPM 2.0 or confidential-VM root when you have one. It runs entirely in your own environment. No SaaS, no telemetry. It answers "show me what the agent actually did" wherever that question lands: after an incident, in procurement, in a dispute. And when EU AI Act record-keeping obligations reach your systems, the same trail is the Article 12 evidence, already running.

<p align="center">
  <a href="https://github.com/vaaraio/vaara/releases/tag/v1.50.0"><img src="https://raw.githubusercontent.com/vaaraio/vaara/main/docs/vaara-v150-launch.gif" width="720" alt="Vaara launch demo"></a>
</p>

<p align="center"><a href="https://github.com/vaaraio/vaara/releases/tag/v1.50.0">Watch the full video with sound</a> &middot; Vaara for macOS is in public beta</p>

## Quick start

```bash
pip install vaara
```

```python
import vaara

@vaara.govern
def transfer_funds(to: str, amount: float) -> str:
    ...
```

That is the whole thing. Every call to a governed function is risk-scored and decided against your policy before the body runs. A blocked call raises `vaara.Blocked`; an allowed call runs, and the decision, the call, and the outcome land in a signed record anyone can verify offline. Python 3.10+, zero runtime dependencies.

Other ways in: Homebrew installs the CLI (`brew tap vaaraio/tap && brew install vaara`; newer brew asks you to `brew trust vaaraio/tap` first), and [`@vaara/client`](https://www.npmjs.com/package/@vaara/client) on npm is the TypeScript client for the HTTP API. The MCP proxy and server ship with the Python package.

<details>
<summary><b>Prefer the explicit pipeline?</b></summary>

The decorator drives the same engine you can call directly when you want the decision object in hand.

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

Every call gets a risk score and an allow / block / escalate decision against your policy, then the call, the decision, and the real outcome are written to the audit trail. `report_outcome` closes the loop: the scorer reweights based on which signals actually predicted the outcome. Releases ship SLSA Build Level 3 provenance, verifiable with `slsa-verifier verify-artifact`. Optional ML classifier: `pip install 'vaara[ml]'`.
</details>

<details>
<summary><b>Verify it without trusting the producer</b></summary>

Writing a trail is the easy half. The half that matters is letting someone who does not trust you check it, with no key, no access, and none of your code. Every Vaara record is content-addressed and fail-closed on authenticity, and ships with public conformance vectors plus a standalone checker that imports no Vaara code, so an independent party reproduces every verdict offline.

```bash
vaara verify-bundle evidence-bundle.json
```

`ok` only when a signature is actually established, not merely present in a log. The same property drives the standards work behind [the Vaara Receipt Internet-Draft](https://datatracker.ietf.org/doc/draft-sirkkavaara-vaara-receipt/): evidence that holds up for someone who runs none of your software. The full verifier set, the trust model for each verb, and where trust comes from in each case are in [docs/verifying-evidence.md](docs/verifying-evidence.md).

To check that claim yourself, without installing Vaara, run the standalone checker against the published vectors. Its only dependencies are `cryptography` and `rfc8785`:

```bash
git clone https://github.com/vaaraio/vaara
cd vaara
pip install cryptography rfc8785        # the checker's only dependencies
python tests/vectors/external_evidence_v0/_check_independent.py
```

It re-derives every verdict from the receipt bytes and the public key alone. The output shows the property the trail is built for: a receipt dropped from inside a declared boundary is a provable gap from the held set, with no issuer access and no external witness.

For the whole loop in one runnable file, produce a signed record, verify it yourself, then watch a single forged byte get caught, see [examples/prove-it-yourself/](examples/prove-it-yourself/). The logs-versus-evidence argument behind it is in [docs/logs-vs-evidence.md](docs/logs-vs-evidence.md).

The aggregate runner grades every suite at once, and grades another implementation's vectors the same way:

```bash
python scripts/conformance_runner.py                                 # grade the reference corpus
python scripts/conformance_runner.py --vectors-dir ./your_vectors    # grade your own
```

The named, versioned rule set, what a pass does and does not establish, and the full suite list are in [docs/conformance-profile.md](docs/conformance-profile.md).
</details>

<details>
<summary><b>What the evidence looks like</b></summary>

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
</details>

<details>
<summary><b>What you get</b></summary>

- **Gate every tool call** against your own policy: allow, block, or escalate.
- **A tamper-evident trail** an outside party verifies without trusting your stack, with the chain head anchorable to an external RFC 3161 / eIDAS timestamp so its existence is provable against a clock you do not control.
- **Article-level EU AI Act evidence**, honest about the gaps instead of papering over them.
- **Governance of the model call itself**, not only the tools around it: a hardware-rooted inference receipt that a second, different local model cross-checks. This is the sovereign inference harness, new in v1.0.
- **Enforcement, not only a record** (v1.1.0): a credential broker mints a signed, short-lived credential bound to the attestation digest and scoped to one tool, its argument commitment, and tenant, with typed capability scopes that bound what a call may do. A gateway in front of a protected tool refuses any call without a valid, attestation-bound grant, so a bypass stops being silent. Off by default.
- **Gap-evident completeness** (v1.4.0): each authorization receipt can carry a signed per-boundary sequence and running count, so a dropped receipt inside a declared boundary is a provable gap from the held receipts alone, with no issuer access and no external witness (`vaara verify-contiguity`). Off by default.
- **Independent re-mint** (v1.14.0): a second generator in each public vector set reproduces the signed carrier byte-exact from the declared canonicalization alone, with no Vaara import. A verifier that has never run Vaara software reproduces the same bytes.
</details>

<details>
<summary><b>Where it plugs in</b></summary>

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
  --db ./mcp_audit.db --shadow
```

Start with `--shadow`: every call is classified, scored, and recorded, nothing is blocked. After a few days, `vaara trail shadow-report --db ./mcp_audit.db` shows what enforcement would have done; then drop the flag and enforce, starting from a ready-made perimeter for common MCP servers in [examples/policies/mcp-starters/](examples/policies/mcp-starters/README.md). Point your MCP client (Claude Code, Cursor, any host) at the proxy instead of the upstream. There is also an HTTP API (`pip install 'vaara[server]'`, `vaara serve`) and a first-party TypeScript client on npm ([`@vaara/client`](clients/ts)) for non-Python agents. Framework details, the cloud and OSS guardrail adapters (Bedrock, Azure, GCP, NeMo, Guardrails AI, LLM Guard, Rebuff), and the multi-tenant proxy are in [docs/adapters.md](docs/adapters.md).
</details>

<details>
<summary><b>How it scores</b></summary>

Each risk score blends five expert signals and keeps adapting as outcomes come back, and it carries a confidence interval with a coverage guarantee that holds regardless of the input distribution. On a held-out adversarial corpus the classifier reaches **84.7%** recall (95% Wilson [82.4, 86.7]) at a **4.1%** false-positive rate, and **1.2%** FPR on benign calls under live injection pressure. The hot-path rule scorer adds 140 µs mean per call on commodity CPU; the ML classifier is opt-in (`vaara[ml]`) and off that path. Every figure is reproducible via `make bench`.

- 12,155-entry adversarial corpus (250 hand-curated + 11,905 LLM-generated), 70/15/15 split stratified by (category, source).
- Classifier v9 (236 hand-features + 384-dim MiniLM embeddings) at calibrated threshold 0.9150 on held-out TEST n=1,827: recall 84.7% [82.4, 86.7] at FPR 4.1% [2.9, 5.7].
- Cross-model held-out recall 66.8% [64.9, 68.7] over n=2,277 with no eval-set attacker model in TRAIN; the weakest sub-cell is data_exfil against a closed-weight model at 38.9%. This is the honest worst case; the in-distribution number above is the easier denominator.
- BIPIA-pressure FPR on benign tool calls 1.2% [0.4, 3.6] across four agent backends (Claude Haiku 4.5, Llama-3.1-8B, Mistral-7B, Qwen-2.5-7B). Down from 35.2% on v8.
- Multi-attacker PAIR robustness: 0/25 successes per attacker across Qwen2.5-32B, Qwen2.5-72B, Llama-3.3-70B on identical seeds, Wilson upper 13.3%.
- Distribution-free conformal coverage on the score; MWU regret bound O(sqrt(T log N)).
- Chain of custody: corpus, split, training commit, and bundle SHAs locked and printed by every script.

Method and per-cell breakdown: [docs/architecture.md](docs/architecture.md) and [bench/](bench/).
</details>

<details>
<summary><b>Standards and attestation</b></summary>

- **[vaara.receipt/v1](SPEC.md)** is the canonical parent spec for the signed receipt format: hash-chained, canonicalized with JCS (RFC 8785), verifiable offline from a public key. The x402 settlement binding and an eIDAS qualified-timestamp profile are downstream profiles that pin to it rather than competing formats. Receipts can carry a self-hosted RFC 3161 timestamp that Vaara mints offline.
- **[SEP-2828](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2828)** signed execution records and **SEP-2787** request-attestation test vectors, in the MCP standards process. A second independent implementation has reproduced the SEP-2828 conformance vectors from a clean checkout with no shared code. Published corpora with independent checkers cover the fallback binding path ([`conformance/sep2828/fallback_projection_v0/`](conformance/sep2828/fallback_projection_v0/)) and CrewAI governance decisions ([`tests/vectors/governance_decision_v0/`](tests/vectors/governance_decision_v0/)).
- **OVERT 1.0** ([overt.is](https://overt.is/)): Vaara is the Arbiter and emits Protocol Profile 1.0 Base Envelopes (canonical CBOR, Ed25519) alongside every record when attestation is on.
- **Post-quantum**: an optional parallel ML-DSA-65 / FIPS 204 signature over the same preimage, so a stripped post-quantum signature is a detectable downgrade rather than a silent loss.
- **Root-agnostic evidence**: the same Article 12 record is provable with or without a hardware TEE and re-expressible as an IETF RATS EAR (AR4SI vector), whether rooted in a TPM 2.0 host, an AMD SEV-SNP confidential VM, or no TEE at all.
- **Sovereign inference harness** (v1.0): a local model behind a signing proxy that emits a hardware-rooted inference receipt a second local model cross-checks. Developed privately, published here under AGPL-3.0.

Details and the offline checkers for each: [docs/standards.md](docs/standards.md).
</details>

<details>
<summary><b>Surface stability</b></summary>

The public surface is fixed: the signed envelope (`vaara.receipt/v1`), capability constraints, the credential grant and gateway, and the `@vaara.govern` entry point. No new primitives are planned. New behavior ships as profiles that pin to `vaara.receipt/v1`, not as new core types, and no new format bindings will be added (the last was v1.13.0). From here the work is hardening and subtraction within this surface, so anyone building on it has a stable target.
</details>

<details>
<summary><b>Docs</b></summary>

| Path | Contents |
|---|---|
| [docs/verifying-evidence.md](docs/verifying-evidence.md) | Every verifier and its trust model |
| [docs/logs-vs-evidence.md](docs/logs-vs-evidence.md) | Logs vs evidence: proving what an agent did, and what the AI Act actually requires |
| [docs/prove-what-an-ai-agent-did.md](docs/prove-what-an-ai-agent-did.md) | The four properties a provable record of agent actions needs |
| [docs/eu-ai-act-article-12.md](docs/eu-ai-act-article-12.md) | Article 12 record-keeping: what it requires, what it does not, what to demand from tooling |
| [docs/tamper-evident-audit-trail.md](docs/tamper-evident-audit-trail.md) | How the trail works, its honest limits, and what it costs |
| [docs/vaara-vs-observability-vs-grc.md](docs/vaara-vs-observability-vs-grc.md) | Vaara vs Datadog/Splunk vs Vanta/Drata: three different questions |
| [docs/dogfood/](docs/dogfood/README.md) | Our marketing runs under this gate; the signed trail and key to verify it |
| [docs/architecture.md](docs/architecture.md) | Scoring, conformal coverage, time anchor, formal properties |
| [SPEC.md](SPEC.md) | The canonical vaara.receipt/v1 receipt format spec |
| [docs/standards.md](docs/standards.md) | SEP-2828, SEP-2787, OVERT, the sovereign inference harness |
| [docs/adapters.md](docs/adapters.md) | Framework and cloud/OSS guardrail adapters, multi-tenant proxy |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | EU AI Act and DORA article mapping, eval numbers |
| [docs/multi-replica-deployment.md](docs/multi-replica-deployment.md) | Scaling past one proxy process: per-replica chains, rotation, archive index |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version evolution |
| [docs/PRIOR_ART.md](docs/PRIOR_ART.md) | When each concept first shipped, plus adjacent work |
</details>

Vaara helps deployers assemble evidence for their own conformity work. It does not certify compliance or constitute legal advice. Deployers own their obligations under the EU AI Act and other applicable law.

Commercial license and paid pilots available: see [vaara.io](https://vaara.io/#pilots) or contact [hello@vaara.io](mailto:hello@vaara.io). Licensing terms are in [LICENSING.md](LICENSING.md).

## Acknowledgements

- Listed in the industry acknowledgements of the [IMDA Model AI Governance Framework for Agentic AI v1.5](https://www.imda.gov.sg/-/media/imda/files/about/emerging-tech-and-research/artificial-intelligence/mgf-for-agentic-ai.pdf) (Singapore, 20 May 2026).
- The [AMD AI Developer Program](https://www.linkedin.com/posts/amd-developer_meet-henri-sirkkavaara-henri-created-vaara-activity-7459667676555132928-QFSd) ran a developer testimonial of Vaara in May 2026.
- [Article 14 runtime: why oversight of agentic AI has to be evidenced as action, not model](https://futurium.ec.europa.eu/ga/apply-ai-alliance/community-content/article-14-runtime-why-oversight-agentic-ai-has-be-evidenced-action-not-model), the position post on the EU Apply AI Alliance Futurium.
- [Article 12 and the difference between a log and evidence](https://futurium.ec.europa.eu/en/apply-ai-alliance/community-content/article-12-and-difference-between-log-and-evidence), the companion position post on the EU Apply AI Alliance Futurium.
- [Sovereign proof for the AI Act](https://futurium.ec.europa.eu/en/apply-ai-alliance/community-content/sovereign-proof-ai-act), the latest position post on the EU Apply AI Alliance Futurium.

## Citation

If you build on Vaara or its receipt format, cite the repository (see [CITATION.cff](CITATION.cff)) and the specification it implements:

Henri Sirkkavaara. *The Vaara Receipt: A Recomputable Receipt Format for Decisions About Agent Actions.* IETF Internet-Draft [draft-sirkkavaara-vaara-receipt](https://datatracker.ietf.org/doc/draft-sirkkavaara-vaara-receipt/).

## License

Copyright © 2026 Henri Sirkkavaara. Licensed under AGPL-3.0-or-later. See [LICENSE](LICENSE).

<!-- mcp-name: io.github.vaaraio/vaara -->
<!-- mcp-name: io.github.vaaraio/vaara-server -->
