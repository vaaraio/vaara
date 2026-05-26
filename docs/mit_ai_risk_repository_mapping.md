# Vaara mapping to the MIT AI Risk Repository

This page maps Vaara's runtime governance surface against the [MIT AI Risk Repository v4](https://airisk.mit.edu/) (MIT FutureTech, Slattery et al., updated 2025-12-03, [arxiv:2408.12622](https://arxiv.org/abs/2408.12622), CC BY 4.0). The repository is a taxonomy of 1,835 risk-bearing entries (570 categories plus 1,265 sub-categories) drawn from 74 source papers, organised into seven top-level domains.

The mapping is deliberately honest. Vaara is a runtime evidence layer at the agent tool-call boundary. It addresses risks that surface at that boundary and does not address risks that live elsewhere in the AI lifecycle (training data, model design, content scoring, environmental footprint). Naming the gap is the point.

## How the MIT repository is structured

Each risk-bearing entry carries two orthogonal taxonomies:

**Causal taxonomy** (three axes on every risk):
- Entity: Human, AI, Other
- Intent: Intentional, Unintentional, Other
- Timing: Pre-deployment, Post-deployment, Other

**Domain taxonomy** (seven top-level domains, with sub-domains):

| # | Domain | Entries |
|---|---|---|
| 1 | Discrimination & Toxicity | 224 |
| 2 | Privacy & Security | 199 |
| 3 | Misinformation | 80 |
| 4 | Malicious Actors & Misuse | 268 |
| 5 | Human-Computer Interaction | 106 |
| 6 | Socioeconomic and Environmental | 299 |
| 7 | AI System Safety, Failures & Limitations | 421 |

## What Vaara addresses

The sub-domains below have direct runtime evidence shape against Vaara's primitives. The Vaara columns name the specific feature that produces the evidence.

### Domain 2: Privacy & Security (199 entries)

| Sub-domain | n | Vaara mechanism |
|---|---|---|
| 2.1 Compromise of privacy by leaking or correctly inferring sensitive information | 80 | OVERT 1.0 envelope keeps request content local. Only HMAC-SHA256 commitment crosses the trust boundary. The audit chain records every tool call that touched sensitive parameters. `vaara compliance report` surfaces leak-shaped events. |
| 2.2 AI system security vulnerabilities and attacks | 112 | MCP proxy enforces allow/deny at the protocol perimeter. Adversarial classifier trained on 12,155-entry corpus scores `tools/call` actions for data exfil, privilege escalation, tool misuse. PAIR robustness measured against three attacker families. Cross-model held-out evaluation on Mixtral and Claude Sonnet 4.6. |

Direct coverage: 192 of 199 in domain 2.

### Domain 4: Malicious Actors & Misuse (268 entries)

| Sub-domain | n | Vaara mechanism |
|---|---|---|
| 4.0 Malicious use | 24 | Policy gate at the MCP boundary. DENY/ESCALATE on configured tool patterns. Hash-chained record of every blocked attempt. |
| 4.1 Disinformation, surveillance, and influence at scale | 84 | Per-action audit chain on tool calls that produce or exfiltrate content. Operator perimeter filters resources and prompts. |
| 4.2 Cyberattacks, weapon development or use, and mass harm | 82 | Adversarial classifier specifically targets agent actions in this lane. The v0.34-v0.36 corpus extensions added cross-model adversarial samples on `tool_misuse`, `privilege_escalation`, `data_exfil`. |
| 4.3 Fraud, scams, and targeted manipulation | 77 | Per-action audit with conformal risk interval. Policy gate routes high-score calls to human review. |

Direct coverage: 267 of 268 in domain 4.

### Domain 5: Human-Computer Interaction (106 entries)

| Sub-domain | n | Vaara mechanism |
|---|---|---|
| 5.1 Overreliance and unsafe use | 60 | `evidence_insufficient` status returned honestly when an article has no recorded events. Compliance reports do not rubber-stamp. Per-article verdict drill-down lets a reviewer trace status to concrete events. |
| 5.2 Loss of human agency and autonomy | 46 | Article 14 human-in-loop escalation queue. ESCALATE decision routes the action to a reviewer before execution. The reviewer's outcome is recorded back into the chain. |

Direct coverage: 106 of 106 in domain 5.

### Domain 6: Socioeconomic and Environmental (299 entries)

| Sub-domain | n | Vaara mechanism |
|---|---|---|
| 6.5 Governance failure | 61 | Regulator-handoff envelope via `vaara trail export` (Sigstore-signed, optional ML-DSA-65 / FIPS 204 post-quantum signer). Per-article evidence reports aligned to EU AI Act Articles 9, 11-15, 17, 61 and DORA Articles 10, 12, 13. |

Direct coverage: 61 of 299 in domain 6. The remaining 238 entries in domain 6 (power centralisation, inequality, employment, devaluation, competitive dynamics, environmental harm) sit outside Vaara's scope by design.

### Domain 7: AI System Safety, Failures & Limitations (421 entries)

| Sub-domain | n | Vaara mechanism |
|---|---|---|
| 7.0 AI system safety, failures, & limitations | 19 | Vaara is the runtime governance substrate for actions an AI system takes. |
| 7.4 Lack of transparency or interpretability | 42 | Hash-chained audit trail (SHA-256, optional Ed25519, optional ML-DSA-65). Per-article verdict drill-down with `verdict_inputs` and `contributing_events`. OVERT 1.0 signed envelopes per interaction. Addresses organisational transparency, not mechanistic model interpretability. |
| 7.6 Multi-agent risks | 53 | Per-agent action attribution via `agent_id`. Composable audit chains across multiple agents through a single Vaara instance or a fleet. Multi-tenancy is the v0.29+ scope. |

Direct coverage: 114 of 421 in domain 7. The remaining 307 entries (goal conflict, dangerous capabilities, capability/robustness baseline, AI welfare) are model-side or capability-side risks that a runtime substrate does not address.

## Summary numbers

| Domain | Total | Vaara direct | % of domain |
|---|---|---|---|
| 1. Discrimination & Toxicity | 224 | 0 | 0% |
| 2. Privacy & Security | 199 | 192 | 96% |
| 3. Misinformation | 80 | 0 | 0% |
| 4. Malicious Actors & Misuse | 268 | 267 | 100% |
| 5. Human-Computer Interaction | 106 | 106 | 100% |
| 6. Socioeconomic and Environmental | 299 | 61 | 20% |
| 7. AI System Safety, Failures & Limitations | 421 | 114 | 27% |
| **Total** | **1,597** | **740** | **46%** |

The repository's 1,835 total entries include 238 umbrella-only and Excluded entries (`X.x > Excluded`, `1.0 > Discrimination & Toxicity`, etc.) that do not name a specific risk. The 1,597 in the table above are the sub-domain-tagged entries that map cleanly.

## What Vaara explicitly does not address

Naming the boundary is part of the honesty. Vaara is not the right primitive for:

- **Domain 1 (Discrimination & Toxicity).** Training data, fairness metrics, and toxicity classification live at the model and dataset layer. Vaara records that an action happened, not whether the model was fair when it chose the action.
- **Domain 3 (Misinformation).** Content-level fact-checking and information-ecosystem analysis. Vaara records the agent's tool calls, not the truth value of the content the agent produced.
- **Domain 6 sub-domains 6.1, 6.2, 6.3, 6.4, 6.6.** Structural and macroeconomic risks (power centralisation, inequality, devaluation, competitive dynamics, environmental harm) are policy and design problems, not protocol-boundary problems.
- **Domain 7 sub-domains 7.1, 7.2, 7.3, 7.5.** Goal misalignment, dangerous-capability emergence, capability and robustness baseline, and AI welfare are model-side concerns. A runtime evidence layer can record what a system did, not what its underlying capabilities are.

## How this composes with other governance work

The MIT taxonomy is descriptive. Vaara is one of several primitives a deployer can compose to address the entries Vaara covers. Other primitives sit at adjacent layers:

- Content-safety filters (Bedrock Guardrails, Azure Content Safety, GCP Model Armor, NeMo Guardrails, Guardrails AI, LLM Guard, Rebuff) for domains 1 and 3. Vaara integrates them as upstream signals into the same audit chain.
- Model evaluation harnesses (HELM, BIG-bench, capability evals) for domain 7 sub-domains on capability and robustness.
- Mechanistic interpretability research for the technical-transparency side of 7.4.
- Organisational and societal policy work for domain 6 sub-domains outside 6.5.

Vaara is the runtime evidence layer. The other layers feed in. The audit chain is where their findings become evidence.

## Citation and attribution

The MIT AI Risk Repository is published under CC BY 4.0. The work cited here is:

> Slattery, P., Saeri, A. K., Grundy, E. A. C., Graham, J., Noetel, M., Uuk, R., Dao, J., Pour, S., Casper, S., Thompson, N. (2024). "The AI Risk Repository: A Comprehensive Meta-Review, Database, and Taxonomy of Risks From Artificial Intelligence." arXiv:2408.12622.

Database last updated 2025-12-03. Local copies of the v4 database and the companion AI Risk Mitigations sheet are tracked under `research/external/` for reproducibility.
