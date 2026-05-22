# Compliance Evidence Mapping

This document maps what Vaara produces at runtime to the specific article
references a deployer needs to assemble conformity evidence under the
EU AI Act and DORA.

## Scope

Vaara is a tool. It records, scores, and gates agent actions, and it
produces evidence artefacts that map to named article obligations. It
does not perform conformity assessments, does not issue certifications,
and is not legal advice. The deployer owns the conformity decision,
together with a Notified Body where one is required.

What Vaara gives you is structured, article-tagged evidence that a
compliance team or auditor can ingest without having to reconstruct it
from application logs after the fact.

## EU AI Act Article Mapping

The table below maps each article Vaara addresses to the runtime event
types that populate evidence for it. The list matches the default
`ComplianceEngine` requirements in `vaara.compliance.engine`. For the
thresholds the engine uses to decide whether the evidence is sufficient,
partial, or insufficient, see [VERDICTS.md](VERDICTS.md).

| Article | Requirement | Evidence Vaara produces |
|---|---|---|
| **9(1)** | Risk Management System | Every intercepted action is scored and the score is recorded with inputs. `RISK_SCORED` events. |
| **9(2)(a)** | Risk Identification and Analysis | `RISK_SCORED` plus `ACTION_BLOCKED` records show which risks were detected and which were blocked. |
| **9(4)(a)** | Risk Mitigation Measures | `ACTION_BLOCKED` and `DECISION_MADE` records show mitigation applied per action. |
| **9(7)** | Testing Procedures | `RISK_SCORED` plus `OUTCOME_RECORDED` pairs form the test signal. Conformal intervals give distribution-free calibration metrics. |
| **11(1)** | Technical Documentation | Checked outside the audit trail. Vaara does not replace the Annex IV technical file. |
| **12(1)** | Record-Keeping (Logging) | Every `ACTION_REQUESTED`, `RISK_SCORED`, and `DECISION_MADE` is written to a hash-chained, tamper-evident trail. See "Audit trail integrity" below. |
| **13(1)** | Transparency and Provision of Information | `RISK_SCORED` and `DECISION_MADE` records carry the risk score, the interval, the decision, and the reason string shown to the operator. |
| **14(1)** | Human Oversight -- Design | `ESCALATION_SENT` and `ESCALATION_RESOLVED` events prove the oversight path exists and was exercised. The `vaara.audit.review_queue` storage layer turns `escalate` into a substantive queued-for-review step rather than a fire-and-forget log line. The `vaara review` CLI is the operator surface. |
| **14(4)(d)** | Human Oversight -- Override Capability | `ESCALATION_RESOLVED` and `POLICY_OVERRIDE` events prove a human can decide not to proceed or can override Vaara's decision. The `vaara review resolve --audit-db PATH` CLI writes the `ESCALATION_RESOLVED` row directly from an operator action, so the override is a single recorded transaction rather than an out-of-band promise. |
| **15(1)** | Accuracy, Robustness and Cybersecurity | `OUTCOME_RECORDED` events feed the adaptive scorer. Recency is tracked (default weekly calibration window). |
| **61(1)** | Post-Market Monitoring | `OUTCOME_RECORDED` events form the post-market signal, tied back to the original action via `action_id`. |
| **73(1-7)** | Serious-Incident Reporting | `vaara trail export-incident` builds a standalone JSON report referencing the trigger audit record by `record_id`. INTERIM format pending the Commission template promised by paragraph 7. Reporting deadline is derived from the Article 3(49) sub-category (general 15 days, death 10 days, Article 3(49)(b) widespread or serious 2 days). |

### Article 14 in particular

The risk interval is what makes Article 14 oversight substantive rather
than cosmetic. A point score of 0.6 tells a reviewer nothing about
whether the model is confident. A conformal interval of [0.58, 0.62]
versus [0.2, 0.95] tells them whether to trust the number. Vaara
surfaces both on every escalation.

### Policy artifact review

The Vaara policy is a declarative YAML / JSON document loaded via
`vaara.policy.from_yaml()` or `from_json()`. As of v0.9, two CLI
surfaces let a compliance team review the policy artifact
independently from the agent code that uses it:

- `vaara policy validate POLICY_PATH` runs structured semantic checks
  (parse errors plus warnings for narrow threshold bands, dangling
  per-class overrides, unreachable escalation routes, sequence steps
  not naming a declared action class, missing default escalation
  route). Exit code 0 if no errors. Warnings print without flipping
  the exit code.
- `vaara policy test POLICY_PATH --cases CASES_PATH` runs a YAML/JSON
  cases file against the policy (Conftest analog for Vaara). Each
  case names an action class, a risk score, any sequence patterns to
  treat as matched, and an expected verdict and route. Exit code 0
  if every case passes.

Both commands carry a `--json` flag so CI pipelines can consume the
output directly. The policy document and its cases file live in the
deployer's source-control tree, version-controlled and diffable,
alongside any other policy-as-code artifacts (Rego, Cedar, Casbin)
used in the same governance stack. Worked example at
`examples/policies/test_cases.yaml` exercises
`examples/policies/full.yaml`.

### Article 26 (deployer obligations)

Article 26 obligations sit on the deployer, not on Vaara. The evidence
Vaara produces is the feedstock a deployer uses to satisfy 26(1)
("use the system in accordance with the instructions for use"),
26(5) ("monitor operation"), and 26(6) ("keep logs"). Deployer conduct
outside the Vaara pipeline is not in scope.

## EU AI Act Annex IV evidence sections

Annex IV defines nine technical documentation sections required under
Article 11. Vaara fills three of those sections directly, contributes
to four, and stays out of two.

| Annex IV section | What it asks for | Vaara contribution |
|---|---|---|
| §1 General description | Purpose, intended use, versions, provider info | Out of scope. Provider supplies. |
| §2 Elements and development process | Architecture, datasets, training choices | Contributes a description of the runtime governance layer. Vaara docs and configuration are an Annex IV §2 input. |
| §3 Monitoring, functioning and control | How the system is monitored at runtime | **Direct fill.** Hash-chained audit trail, per-action risk score and reason, decision records. |
| §4 Performance metrics appropriateness | Metric choice and justification | Contributes runtime metrics: allow / deny / escalation rate, score distribution, calibration window. |
| §5 Risk management system per Article 9 | Risk identification, assessment, mitigation | **Direct fill.** `RISK_SCORED`, `ACTION_BLOCKED`, `DECISION_MADE` events with article tags. |
| §6 Relevant changes during lifecycle | Versioned change history of the system | Contributes the timestamped audit trail showing runtime config and threshold changes. Provider tracks model and code changes separately. |
| §7 List of harmonised standards applied | Named CEN-CENELEC standards | Vaara aligns with several JTC21 drafts (see next section). Once those finalize, deployers list them here. |
| §8 Copy of EU declaration of conformity | The DoC document itself | Out of scope. Provider drafts and signs. |
| §9 Post-market performance evaluation system | Mechanism for monitoring AI performance after deployment | **Direct fill.** `OUTCOME_RECORDED` events tied back to `action_id`, feeding the adaptive scorer. |

Direct-fill sections (§3, §5, §9) are populated automatically by the
`vaara trail export` handoff zip plus the `run_compliance_assessment`
report. Contributing sections (§2, §4, §6, §7) need a deployer to
combine Vaara output with their own provider-side documentation.
Out-of-scope sections (§1, §8) are the deployer's or provider's domain.

## DORA Article Mapping

Relevant for financial entities only. The default `ComplianceEngine`
also ships with a DORA bundle:

| Article | Requirement | Evidence Vaara produces |
|---|---|---|
| **10(1)** | ICT Risk Management -- Protection and Prevention | `ACTION_BLOCKED` and `DECISION_MADE` records. |
| **12(1)** | ICT Incident Detection | `ACTION_REQUESTED` and `ACTION_BLOCKED` records, with risk score and reason. |
| **13(1)** | ICT Incident Response and Learning | `OUTCOME_RECORDED` events close the loop and feed the adaptive scorer. |

## Cloud guardrail adapter pattern

Adapters from v0.19.0 take findings from AWS Bedrock Guardrails, Azure
AI Content Safety, and GCP Model Armor and route them through Vaara's
audit trail. Each cloud filter speaks its own category vocabulary
(`topicPolicy` / `Hate` / `responsible_ai.hate_speech`). The adapters
normalise those onto a single Vaara vocabulary and a published
article-mapping table at
`src/vaara/integrations/_content_safety_articles.py`.

The adapter is thin. The mapping is the artefact. A deployer can read
the table, dispute a row, and override mappings without touching
adapter code. 27 rows total across the three vendors as of v0.19.0.

### Category to article mapping

| Vaara category | Provider categories | AI Act article | OWASP LLM |
|---|---|---|---|
| `prohibited_topic` | Bedrock `topicPolicy` | Art. 5 | LLM08 |
| `hate` | Bedrock `contentPolicy.HATE` / `INSULTS` · Azure `Hate` · GCP `responsible_ai.hate_speech` / `harassment` | Art. 5 | LLM05 |
| `sexual` | Bedrock `contentPolicy.SEXUAL` · Azure `Sexual` · GCP `responsible_ai.sexually_explicit` | Art. 5 | LLM05 |
| `violence` | Bedrock `contentPolicy.VIOLENCE` · Azure `Violence` · GCP `responsible_ai.dangerous` | Art. 5 | LLM05 |
| `self_harm` | Azure `SelfHarm` | Art. 5 | LLM05 |
| `misconduct` | Bedrock `contentPolicy.MISCONDUCT` | Art. 5 | LLM05 |
| `word_block` | Bedrock `wordPolicy` | Art. 5 | LLM05 |
| `pii` | Bedrock `sensitiveInformationPolicy` · GCP `sdp` | Art. 10 | LLM02 |
| `protected_material` | Azure `ProtectedMaterial.Text` / `Code` | Art. 53 | LLM02 |
| `grounding` | Bedrock `contextualGroundingPolicy` · Azure `Groundedness` | Art. 13, Art. 15 | LLM09 |
| `adversarial` | Bedrock `contentPolicy.PROMPT_ATTACK` · Azure `PromptShield.UserPrompt` / `Documents` · GCP `pi_and_jailbreak` | Art. 15 | LLM01 |
| `malicious_uri` | GCP `malicious_uris` | Art. 15 | LLM05 |
| `csam` | GCP `csam` | Art. 5 + Digital Omnibus CSAM (effective 2 Dec 2026) | — |

### Where the finding lands

Adapter output is a `ContentSafetyFinding` with two helpers:

- `to_audit_context()` returns a dict the deployer passes into
  `pipeline.intercept(context=...)`. The finding lands on the
  hash-chained audit trail with the article tags above, satisfying
  Article 12 logging and feeding Article 9 risk-management evidence.
- `to_overt_metadata()` returns a dict suitable for the OVERT envelope
  `non_content_metadata` field. Severities are decimal strings per
  Protocol Profile 1.0 Section B.3 (IEEE-754 floats prohibited).

### What this is not

The adapters do not run the cloud filter. The deployer's runtime makes
the cloud API call with the deployer's own credentials. Vaara observes
and records the finding. Vaara does not replace the cloud filter and
does not claim conformity assessment of the cloud vendor's filter
performance. The deployer's choice of guardrail vendor and confidence
threshold is a policy decision recorded in Vaara's audit trail, not a
decision Vaara makes.

## OSS guardrail adapter pattern

Adapters from v0.20.0 take findings from NVIDIA NeMo Guardrails,
Guardrails AI, LLM Guard, and Rebuff and route them through the same
audit-and-OVERT path as the v0.19.0 cloud adapters. Each OSS guardrail
speaks its own category vocabulary (`input_rails.jailbreak` /
`DetectPII` / `PromptInjection` / `heuristicScore`). The adapters
normalise those onto the same Vaara vocabulary and the same published
article-mapping table at
`src/vaara/integrations/_content_safety_articles.py`, extended with
OSS provider rows.

41 OSS rows total across the four vendors (7 NeMo, 10 Guardrails AI,
20 LLM Guard, 4 Rebuff) as of v0.20.0. Combined with v0.19.0's 27
cloud rows, the published table covers 68 provider categories across
seven upstream guardrails.

### Category to article mapping (OSS providers)

| Vaara category | Provider categories | AI Act article | OWASP LLM |
|---|---|---|---|
| `adversarial` | NeMo `input_rails.jailbreak` / `self_check` · Guardrails AI `DetectPromptInjection` · LLM Guard `PromptInjection` / `InvisibleText` · Rebuff `heuristic_injection` / `model_injection` / `vector_injection` | Art. 15 | LLM01 |
| `prohibited_topic` | NeMo `dialog_rails.topic` · Guardrails AI `MentionsDrugs` · LLM Guard `BanCode` / `BanCompetitors` / `BanTopics` | Art. 5 | LLM08 |
| `hate` | Guardrails AI `ToxicLanguage` / `ProfanityFree` · LLM Guard `Toxicity` | Art. 5 | LLM05 |
| `pii` | NeMo `output_rails.sensitive_data` · Guardrails AI `DetectPII` · LLM Guard `Anonymize` / `Deanonymize` / `Sensitive` | Art. 10 | LLM02 |
| `secrets_leak` | Guardrails AI `SecretsPresent` · LLM Guard `Secrets` · Rebuff `canary_leak` | Art. 15 | LLM02 |
| `word_block` | LLM Guard `BanSubstrings` / `Regex` | Art. 5 | LLM05 |
| `bias` | Guardrails AI `BiasCheck` · LLM Guard `Bias` | Art. 10 | LLM05 |
| `schema_violation` | Guardrails AI `ValidJSON` / `RegexMatch` / `ValidLength` · LLM Guard `JSON` | Art. 15 | LLM05 |
| `output_validation` | NeMo `output_rails.self_check` · LLM Guard `NoRefusal` | Art. 13 | — |
| `grounding` | NeMo `output_rails.fact_check` / `retrieval_rails.relevance` · LLM Guard `Relevance` | Art. 13, Art. 15 | LLM09 |
| `malicious_uri` | LLM Guard `MaliciousURLs` | Art. 15 | LLM05 |
| `language` | LLM Guard `Language` | Art. 13 | — |
| `sentiment` | LLM Guard `Sentiment` | — | — |
| `resource_limit` | LLM Guard `TokenLimit` | Art. 15 | — |

### Where the finding lands

Same flow as cloud adapters. OSS-adapter output is a
`ContentSafetyFinding` with `to_audit_context()` and
`to_overt_metadata()` helpers that route into
`pipeline.intercept(context=...)` and OVERT `non_content_metadata`
respectively.

### What this is not

The adapters do not run the OSS guardrail. The deployer's runtime
loads NeMo, Guardrails AI, LLM Guard, or Rebuff and invokes them with
the deployer's configuration. Vaara observes and records the finding.
Vaara does not replace the upstream guardrail and does not claim
conformity assessment of the guardrail vendor's detection performance.
The deployer's choice of guardrails, scanners, validators, and
thresholds is a policy decision recorded in Vaara's audit trail, not a
decision Vaara makes.

## CEN-CENELEC harmonised standards alignment

The harmonised standards under EU AI Act Article 40 are being drafted
by CEN-CENELEC JTC21. Most are still in draft or public-consultation
phase. The table below maps Vaara's current state to the relevant
JTC21 work items so deployers can track alignment as standards
finalize. Status as of April 2026.

| Standard | WG | Status | Vaara alignment |
|---|---|---|---|
| **ISO/IEC 42001** AI Management System | WG2 | Final ballot for European adoption | Vaara is a tool that fits inside an Article 17 / 42001 AIMS. Vaara does not implement the AIMS itself. |
| **prEN 18286** European AI QMS for Regulatory Purposes | WG2 | Public consultation closed 22 Jan 2026 | Vaara feeds Article 72 ongoing-surveillance obligations and supports Annex VI / Annex VII evidence requirements. The QMS is the deployer's. |
| **prEN 18228** European AI Risk Management Standard | WG2 | Drafting | Vaara contributes the ongoing-monitoring signal called for in the AI Act risk-category integration sections. |
| **ISO/IEC 42006** Requirements for AI Management System Auditors | WG2 | DIS Stage 40 | Vaara's hash-chained trail is the artefact 42006-qualified auditors examine for surveillance evidence. |
| **prEN ISO/IEC 24970** AI System Logging | WG3 | Stage 30.2 (comment resolution) | Vaara aligns with the tamper-resistance, decision-factor logging, and audit-system integration requirements. Field-level alignment pending the published version. |
| **prEN 18229-1** Trustworthiness Framework Pt 1 (logging, transparency, human oversight) | WG4 | Public enquiry | Implements AI Act Articles 12-14, which Vaara already maps to in the article table above. Field-level alignment pending the published version. |
| **prEN ISO/IEC 12792** Transparency Taxonomy of AI Systems | WG4 | Stage 40 (final vote) | v0.6 ships per-action audit records tagged against the four-axis model (System Operation, Data Usage, Decision Making, Limitations) via four optional `AuditRecord` fields. Default classification heuristic by event type. Per-record override available. NOT tamper-evident in v0.6 - fields are metadata annotations excluded from `record_hash` so pre-v0.6 chains stay valid. |

**What "alignment" means here.** Most of these standards have not
published. The mapping above is pre-compliance positioning: Vaara is
built so that when the finals drop, the gap to certified alignment is
small. It is not a claim of certified compliance. Once a standard
publishes, expect a v0.6 or v0.7 alignment audit and an updated entry
in this table.

**What the deployer does with this table.** When listing harmonised
standards applied (Annex IV §7), the deployer cites the published
ones. Where Vaara's runtime behaviour aligns with a draft, that is
useful context for an auditor or notified body but not a substitute
for the published version.

**Independent legal-architecture analysis.** Nannini et al. (2026),
*AI Agents Under EU Law: A Compliance Architecture for AI Providers*
(arXiv:2604.04604), proposes a twelve-layer compliance architecture
across the AI Act essential requirements, M/613 harmonised standards
(prEN 18286, 18228, 18229-1/2, 18282, 18284, 18283), GPAI obligations,
and parallel instruments. Section 6.4 (Emergent behavioural drift)
identifies an open infrastructure gap: *"providers cannot currently
acquire the evaluation infrastructure necessary to demonstrate
conformity for behavioral drift in multi-agent systems, because that
infrastructure does not yet exist in published form"* (footnote 19).
Vaara implements pieces of that infrastructure: hash-chained
operational-state versioning, distribution-free conformal interval
over a behavioral metric, and MWU-bounded online policy update with
regret guarantee O(sqrt(T log N)) (see [docs/formal_specification.md](docs/formal_specification.md)).

## What Vaara produces

Three artefact classes, all tied to a single `action_id`:

**1. The hash-chained audit trail.** Every event is SHA-256 chained to
its predecessor. Any insertion, deletion, or edit breaks the chain and
is detected by `vaara trail verify`.

**2. The conformity report.** A structured per-article rollup. Each
article gets an `EvidenceStatus` (sufficient, insufficient, stale,
error) and an `EvidenceStrength` (strong, moderate, weak) based on
how many qualifying events exist and how recent they are.

**3. The signed regulator handoff zip.** Produced by `vaara trail
export`. Contains the trail, a manifest, and an Ed25519 signature.
The regulator verifies with `vaara trail verify` and the deployer's
public key.

## What the deployer still owns

The line is deliberate. Vaara does not do, and will not claim to do,
any of the following:

- **Conformity assessment.** Vaara produces evidence. The assessment is
  a decision by the deployer, the provider, and where applicable a
  Notified Body.
- **The Annex IV technical file.** Article 11(1) requires technical
  documentation drawn up before the system is placed on the market.
  Vaara does not generate it.
- **Quality Management System.** Article 17 QMS is organisational, not
  runtime. Out of scope.
- **Legal interpretation.** Which articles apply, whether the system
  is high-risk under Annex III, whether a general-purpose AI model
  threshold is triggered. That is a legal call for the deployer.
- **Model-level evaluations.** Vaara governs actions, not models. If
  you need model safety evaluations (bias, accuracy on held-out sets),
  that is a separate workstream.

If a vendor tells you a drop-in tool makes you EU AI Act compliant,
they are either misunderstanding the regulation or selling you a
liability.

## Vaara is not an eIDAS trust service

Vaara's hash-chained audit trail and Article 12 commit-prove receipts
produce **technical evidence**. They are not Qualified Electronic
Signatures, Qualified Electronic Seals, or Qualified Electronic
Attestations of Attributes (QEAA) under Regulation (EU) 910/2014
(eIDAS), Articles 3(12), 3(25), 3(45), or 3(46). Vaara does not
operate as a qualified trust service under Article 3(16).

If your conformity workstream requires evidence to be backed by a
qualified trust service, layer that on top of Vaara's output. Vaara's
hash chain and receipt formats are designed to be wrapped by a
qualified seal or signature without changing the underlying evidence.

## Position relative to open runtime-attestation standards

The runtime-attestation space is converging on the principle that
**self-attestation is not sufficient** - the entity attesting to
governance should be structurally independent of the entity being
governed. OVERT 1.0 (Glacis Technologies, overt.is) makes this
explicit through its four-tier Attestation Assurance Level model,
where AAL-4 mandates an Independent Attestation Provider (IAP)
operating notary infrastructure that the AI operator does not
control.

Vaara's position in this picture:

- **Vaara is structurally independent of the agent it governs.**
  Vaara is a third-party runtime kernel that intercepts the agent's
  actions, scores risk, and writes the audit trail. The agent does
  not produce its own evidence about itself. This satisfies the
  underlying *agent-vs-governor independence* principle.
- **Vaara is not, by itself, structurally independent of the
  operator.** The operator deploys Vaara in their own environment.
  In OVERT 1.0 terms this maps to AAL-3 (automated monitoring with
  operator-controlled infrastructure), not AAL-4. Reaching AAL-4
  requires layering an external IAP - a notary service that the
  operator does not control - on top of Vaara's emitted evidence.
- **Vaara's design admits an external IAP layer without internal
  change.** The hash chain, the commit-prove receipt pair, and the
  HTTP API surface all produce structured, signable artefacts that
  an IAP can co-sign, batch into a transparency log, or seal under
  eIDAS qualified trust services.
- **Vaara ships a reference IAP from v0.13.0
  (`vaara.attestation.iap`).** `Phase3Attestation` wraps an AAL-3
  `BaseEnvelope` with a notary Ed25519 signature over canonical CBOR
  of all nine envelope fields (the inner Arbiter signature bound by
  reference) plus a transparency-log inclusion proof. Structural
  independence between the Arbiter key and the notary key is
  enforced at both emit and verify. `InProcessTransparencyLog` is an
  RFC 6962-style binary Merkle log; production deployments swap in
  sigstore Rekor or an equivalent independently-operated log at the
  same call sites. Operators MAY run the reference IAP themselves;
  an *independent* IAP for AAL-4 still requires a separate operator
  controlling the notary keys.

This positioning is deliberate. Vaara does not claim AAL-4
conformance and does not market a self-attestation pattern.
Operators who need AAL-4 should pair Vaara with an independent
attestation provider. The Vaara-emitted evidence is the input to
that provider, not a replacement for it.

### Hardware TEE attestation hook (experimental, v0.18.0)

Beyond the software-signed attestation chain described above, Vaara
since v0.18.0 supports binding an OVERT envelope to a hardware-rooted
attestation report from a Trusted Execution Environment. The initial
backend is **AMD SEV-SNP**, the natural fit for the confidential-VM
deployment model used in agent runtimes. Intel TDX and Intel SGX are
tracked for later releases.

The OVERT 1.0 Base Envelope schema is closed (9 fields). Hardware TEE
attestation does NOT extend the envelope. Instead, Vaara emits a sibling
SEV-SNP attestation report and binds it to the envelope by placing
`SHA-512(canonical_cbor(envelope))` into the report's 64-byte
`REPORT_DATA` field. A relying party therefore checks two signatures
independently:

1. The envelope's Ed25519 (or ML-DSA-65) signature over the 8 signable
   fields, as before.
2. The SEV-SNP report's ECDSA P-384 signature against the AMD VCEK
   (Versioned Chip Endorsement Key), and that `REPORT_DATA` equals
   `SHA-512(canonical_cbor(envelope))`.

If both hold, the attestation says "this OVERT envelope was emitted by
an arbiter running inside an AMD SEV-SNP confidential VM at the
measured launch state recorded in the report."

What ships in v0.18.0: report parser, binding helper, signature verifier
against a caller-supplied VCEK, deterministic mock attester for tests,
and a `vaara tee parse|verify` CLI. What does NOT ship in v0.18.0: full
VCEK chain validation against AMD's Key Distribution Service (tracked
for v0.19+) and the `/dev/sev-guest` ioctl emitter (tracked for v0.19+
once a tested SEV-SNP guest is available). The hook is marked
experimental until both land.

## OVERT 1.0 Part 3 (Agentic AI Controls) mapping

OVERT 1.0 Part 3 (Sections 11-16) defines the agentic-specific
execution controls: tool-call governance, MCP server trust, multi-
agent boundaries, capability mediation, agent disclosure, human-in-
the-loop attestation, and behavioural drift governance. The mapping
below states, control by control, whether Vaara satisfies the
requirement today (✅), partially satisfies it (◐), or leaves it as
explicit gap-to-deployer or future-work (◯). This mapping does not
establish legal compliance with any regulation. It records technical
correspondence.

### Section 11 - Tool-Call Governance

- **TOOL-1.1** (intercept all tool calls before execution) - ✅.
  `InterceptionPipeline.intercept()` is the enforcement boundary. No
  tool call proceeds without a governance decision.
- **TOOL-1.2** (evaluate against capability policy) - ✅. The policy
  DSL declares permitted tools, parameter ranges, destinations, and
  approval gates. `policy.evaluate` returns the verdict carried in
  the per-call receipt.
- **TOOL-1.3** (denial receipt with policy reference and violation
  type) - ✅. Denials emit a `DENY` event on the hash chain with
  policy id and violation reason.
- **TOOL-1.4** (provisional receipt before execution, upgrade to full
  attestation after notary validation) - ✅ structurally at AAL-3,
  with the AAL-3 → AAL-4 path now implementable in-tree. The Article
  12 commit-prove receipt pair (shipped v0.10.0) is the Phase 2
  Provisional Receipt; the v0.11.0 OVERT Base Envelope is the
  attested form. v0.13.0 ships a reference Phase 3 IAP
  (`vaara.attestation.iap.emit_phase3_attestation`) that notary-signs
  the Provisional Receipt and anchors it in a transparency log.
  Reaching AAL-4 still requires the notary keys to live with an
  independent operator.
- **TOOL-2.1** (explicit function allowlist with hash in policy
  attestation) - ✅. Policy hash flows into `encoder_binary_identity`
  in the Base Envelope (v0.11.0).
- **TOOL-2.2** (parameter schema validation before execution) - ✅
  for declared parameter shapes. ◐ for arbitrary deep schemas (the
  policy DSL is intentionally bounded).
- **TOOL-2.3** (rejection receipt with parameter violation detail) -
  ✅.
- **TOOL-3.1** (per-tool rate limits with attested enforcement) - ◐.
  The adaptive scorer applies velocity-aware risk signals. Explicit
  per-tool calls-per-epoch counters are not yet emitted as
  standalone receipts.
- **TOOL-3.2** (per-session / per-user velocity caps) - ◐ via the
  agent profile in `scorer/adaptive.py`.
- **TOOL-3.3** (circuit breakers on error / violation rate) - ◐ in
  policy DSL. Circuit-breaker receipt not yet a first-class artefact.
- **TOOL-3.4** (recursion-depth termination per trace_id) - ◯.
  Not implemented. Agent-loop termination is currently the deployer's
  responsibility.
- **TOOL-4** (human approval gates) - ◐. The SQLite-backed review
  queue (`vaara.audit.review_queue`) routes `ESCALATE` verdicts to
  human reviewers and records `ESCALATION_RESOLVED` events with
  reviewer identity, timestamp, and decision. TOOL-4.4 approval-
  velocity caps are not enforced.
- **TOOL-5** (tamper-evident tool-call log with epoch attestation) -
  ✅ for TOOL-5.1 and TOOL-5.2 (hash-chained `AuditTrail`,
  Article 12 commit-prove receipt pair). TOOL-5.3 epoch notary
  attestation is satisfied by the v0.13.0 reference IAP
  (`vaara.attestation.iap`) paired with the in-process transparency
  log; a sigstore Rekor-backed log can substitute at the same call
  sites.

### Section 11.5 - MCP Server Trust Governance

Vaara ships an MCP server (`vaara.integrations.mcp_server`) that
exposes governance tools to MCP clients. It does not currently act
as an MCP *client* governing tools hosted on third-party MCP
servers. The MCP-1/2/3 control set therefore applies to Vaara only
in the **custom (operator-hosted)** mode (MCP-2): the operator runs
the Vaara MCP server in their own environment.

- **MCP-2.1** (server binary identity in co-epoch binding) - ◐ at
  v0.12.0: arbiter binary identity is captured in
  `encoder_binary_identity`. A dedicated MCP-server binary identity
  field is future work.
- **MCP-2.2** (network topology attestation) - ◯. Deployer concern.
  Vaara does not measure its own network position.
- **MCP-2.3** (per-call authorization at the MCP server boundary) -
  ✅. Every MCP tool invocation passes through `intercept()`.
- **MCP-2.4** (configuration change detection within an epoch) - ◯.
  Future work.
- **MCP-1** and **MCP-3** (managed-vendor and external-third-party
  MCP servers) - outside Vaara's current surface. An operator using
  Vaara as the *governor in front of* a third-party MCP server would
  need adapter work. The architecture admits it but no implementation
  ships today.

### Section 12 - Multi-Agent System Controls

- **MULTI-1** (inter-agent trust boundaries) - ◯. Per-agent policy
  evaluation works today (each `intercept()` call carries an
  `agent_id`), but agent-vs-agent trust separation is not
  enforced beyond what the deployment policy declares.
- **MULTI-2** (agent composition / topology attestation) - ◯.
  Deployer-side documentation. No Vaara-emitted topology receipt.

### Section 13 - Capability-Based Access Control

- **CAP-1** (data provenance tracking) - ◐. The taxonomy and policy
  DSL accept provenance tags on actions. Transformation propagation
  (CAP-1.2) is the deployer's responsibility because Vaara intercepts
  tool calls, not arbitrary data transformations inside the agent
  process.
- **CAP-2** (architectural separation of planning from untrusted
  data) - ◯. AAL-2 documentation at most. This is a deployer-side
  architecture choice that Vaara records but does not enforce.

### Section 14 - Agent Disclosure and Transparency

- **DISC-1.1** (capability documentation) - ◐ via the deployer's
  policy file + `vaara compliance report`.
- **DISC-1.2** (AIBOM in CycloneDX-AI or SPDX 3.0) - ◯. Future
  work. The auditor-facing evidence export (v0.10.0) is a candidate
  surface to embed AIBOM references.
- **DISC-1.3** (attestation summary with coverage ratio, S3P
  signals, override frequency) - ◐ from v0.12.0: Vaara now emits
  S3P attestations (`vaara.attestation.s3p`) carrying coverage
  ratio and binomial CI. The deployer aggregates these for
  disclosure.

### Section 15 - Human-in-the-Loop Attestation

- **HITL-1** (consent attestation) - ◯. Deployer-side concern.
  Vaara does not collect end-user consent.
- **HITL-2** (human review attestation) - ◐. Review-queue resolution
  events on the audit chain carry reviewer identity (when supplied
  by the deployer), timestamp, decision, and reference to the
  original `ESCALATE` verdict by `action_id`. AAL-4 identity
  binding is the deployer's responsibility.
- **HITL-3** (human correction and override) - ◐ via
  `report_outcome` and the review-queue resolution event.
- **HITL-4** (policy and configuration approval with separation of
  duties) - ◯ at the receipt level. Policy-change approval is
  currently a git-history artefact, not an attested OVERT event.
- **SESS-1..5** (session-scoped attestation) - ◯.
- **STATE-1, STATE-2** (durable state sealing and prompt artifact
  binding) - ◯.
- **IDENT-1** (federated identity / token provenance chain) - ◐.
  `vaara.auth` accepts authenticated caller identity into the audit
  record. Full delegation-chain attestation per IDENT-1.2 is future
  work.

### Section 16 - Behavioural Drift Governance

- **DRIFT-1** (baseline intent declaration) - ◯. Future work. The
  policy DSL is the candidate surface for machine-readable behavioural
  bounds.
- **DRIFT-2** and downstream drift controls - ◐ in spirit. The
  adaptive scorer tracks coverage error via FACI (`scorer/adaptive.py`)
  and emits drift signals through audit events, but these are not yet
  packaged as DRIFT-* receipts.

### S3P (Section 9, MEA-2) statistical safety signal

S3P sits in Domain 5 (MEASURE), not Part 3, but it is the agentic-
relevant measurement primitive that ties everything above together.

- **MEA-1** (deterministic sampling infrastructure) - ◯. Vaara
  evaluates every intercepted action. Sampling-rate-based
  measurement is opt-in. A deployer who wants S3P sampling provides
  the PRF tag and threshold.
- **MEA-2.1** (epoch nonce commitment) - ✅ via
  `vaara.attestation.s3p.make_epoch_nonce_commitment`.
- **MEA-2.4** (exact binomial CI) - ✅. Pure-Python Clopper-Pearson
  via the regularized incomplete beta function. No scipy dependency.
- **MEA-2.6** (closed-schema S3P attestation, Ed25519-signed,
  canonical CBOR per Protocol Profile 1.0) - ✅ via
  `emit_s3p_attestation`.
- **Vaara conformal extension (proposed Protocol Profile
  extension):** the `ConformalExtension` field reports aggregate
  statistics over Vaara's per-action conformal prediction intervals
  alongside the standard Clopper-Pearson CI. The conformal
  aggregates carry the same non-parametric coverage guarantee with
  no distributional assumption - exactly the property MEA-2.4
  requires from a method offered as an alternative to (or
  complement of) Clopper-Pearson. The extension rides in a single
  field in the signed metadata. Standard OVERT verifiers ignore it.

## EU Product Liability Directive 2024/2853

Directive (EU) 2024/2853 of 23 October 2024 on liability for defective
products treats software - including AI systems - as a product within
scope of strict product-liability rules. Member State transposition
deadline is **9 December 2026** (Article 22). The provisions that
matter for runtime evidence:

- **Article 9 (Burden of proof, rebuttable presumptions).** A
  national court SHALL presume the defectiveness of a product, or
  the causal link between defectiveness and damage, where the
  claimant faces excessive difficulties proving the technical
  facts - in particular due to the technical complexity of the
  product (Article 9(4)). The defendant rebuts the presumption by
  showing the product was not defective.
- **Article 7 (Defectiveness assessment).** Defectiveness is
  assessed having regard to, among other factors, the effect on
  the product of any ability to continue to learn or acquire new
  features after deployment, the foreseeable use, and the specific
  requirements of safety regulation applicable to the product.
- **Article 8 (Disclosure of evidence).** Where a claimant
  presents facts and evidence sufficient to support plausibility,
  national courts may order disclosure of relevant evidence held by
  the defendant. Failure to disclose triggers an Article 9
  presumption.

How Vaara fits:

- The hash-chained audit trail, the per-action commit-prove receipt
  pair, the auditor-facing evidence report (`vaara compliance
  report`), and the OVERT 1.0 Base Envelope + S3P attestations
  together constitute the **technical record of foreseeable use,
  governance decisions, and risk signal evolution** that a defendant
  needs to rebut the Article 9 presumption.
- The hash-chain integrity, Ed25519 signatures, and Article 12
  receipt pair give the evidence the tamper-evident shape that
  national courts will expect from contemporaneous records.
- Vaara does not generate liability defences. It produces the
  technical evidence those defences are built from. Legal strategy,
  expert witness work, and the substantive risk-management policy
  remain with the deployer's counsel.

This is forward-looking documentation: transposition statutes will
land between now and 2026-12-09 and the exact procedural shape of
Article 9 presumptions will be a Member State implementation detail.
The intent of this section is to mark Vaara's evidence surface as
**designed to be usable** under the Directive, not to claim
sufficiency in advance of any specific transposition.

## Pulling the evidence

### From Python

```python
from vaara.pipeline import InterceptionPipeline

pipeline = InterceptionPipeline()

# Normal runtime interception
result = pipeline.intercept(
    agent_id="agent-007",
    tool_name="fs.write_file",
    parameters={"path": "/etc/service.yaml", "content": "..."},
    agent_confidence=0.8,
)

# result.allowed       -> bool
# result.action_id     -> str, use for report_outcome
# result.decision      -> "allow" | "deny" | "escalate"
# result.risk_score    -> float, point estimate
# result.risk_interval -> (lower, upper), conformal interval
# result.reason        -> human-readable rationale

if result.allowed:
    pipeline.report_outcome(result.action_id, outcome_severity=0.0)

# Article-by-article conformity report
report = pipeline.run_compliance_assessment(
    system_name="My Agent System",
    system_version="1.0.0",
)

for article in report.articles:
    print(article.requirement.article, article.status.value)
    # Article 9(1): evidence_sufficient
    # Article 12(1): evidence_sufficient
    # Article 14(1): evidence_insufficient
    # ...
```

### From the CLI

```bash
# Signed regulator-handoff zip (Ed25519)
vaara keygen --out-dir ./keys
vaara trail export \
    --db ./vaara_audit.db \
    --key ./keys/signing_key.pem \
    --out ./handoff-2026-04.zip

# Regulator or auditor verifies
vaara trail verify \
    --zip ./handoff-2026-04.zip \
    --public-key ./keys/signing_key.pub
```

## Audit trail integrity

Properties the hash chain guarantees:

- **Append-only.** Events are chained by SHA-256. Any mutation of a
  prior event changes every downstream hash.
- **Tamper-evident.** `vaara trail verify` detects chain breaks,
  signature mismatches, and manifest divergence.
- **Regulator-portable.** The handoff zip is self-contained. The
  regulator verifies against the deployer's public key without needing
  live access to the Vaara instance.

Two things the chain does not give you, which are the deployer's
problem:

- **Content authenticity of the inputs.** Vaara records what the agent
  submitted. It does not attest that the tool arguments were not
  tampered with before reaching Vaara. Run Vaara inside a trust
  boundary you control.
- **Retention policy.** Article 12(2) allows log retention periods set
  in accordance with the intended purpose and applicable law. The
  deployer picks the period. Vaara enforces it via
  `vaara trail purge --db PATH --retention-days N` (or
  `SQLiteAuditBackend.purge_older_than(seconds)` from Python). A
  `--dry-run` flag reports the count without modifying the DB.

  **Hash-chain seam at the retention boundary.** Surviving records
  still reference deleted predecessors via `previous_hash`, so
  `vaara trail verify` will report a chain break at the boundary.
  Intended workflow: export a signed handoff zip BEFORE purging,
  archive the zip externally for long-tail audit history, then purge
  the live DB. The signed zip remains self-consistent forever. The
  live DB chain has a documented seam at the retention boundary.

## Current limits

Honest about the edges:

- The DORA bundle ships today but the EU AI Act bundle is the
  better-calibrated one. DORA mapping will be refined in the 0.5.x
  series with input from deployers in scope.
- `min_evidence_count` thresholds on the default requirements are
  conservative starting points. For production, tune them against your
  own traffic volume and risk tolerance through `ComplianceEngine.
  add_requirement`.
- The Article 11 technical documentation requirement is checked as a
  presence flag only. Drafting the Annex IV file is outside Vaara's
  scope and will stay that way.
- **Distribution-shift split (v0.6 measurement of v0.5.3 stack).** The
  `AdversarialClassifier` (opt-in via `vaara[ml]`) was retrained on a
  corpus that mixes hand-curated and LLM-generated entries. v0.6
  measures the per-source full-stack performance:

  | Source                                | Attack recall | Benign FPR |
  |---------------------------------------|--------------:|-----------:|
  | Hand-curated (held-out, 250 entries)  |        97.1% |      70.0% |
  | LLM-generated (in-sample, 5,705)      |        95.2% |      87.5% |

  Reading: full-stack = heuristic `ESCALATE`/`DENY` preserved + classifier
  upgrades on heuristic `ALLOW`. Hand-curated entries are held-out (not
  in classifier training). LLM-generated entries WERE in training, so
  their numbers are in-sample fit, not generalization.

  The 1.9pp recall gap (97.1% > 95.2%) is small but goes against the
  expected direction. The 18pp benign-FPR gap (70.0% < 87.5%) is the
  dominant distribution-shift signal: the stack is much more confused
  about LLM-generated benigns than hand-curated ones.

  Note on FPR vs CHANGELOG headline: the CHANGELOG quotes "global benign
  FPR 21.0%" which is classifier-alone 5-fold CV OOF. The full-stack
  numbers above are dominated by the heuristic - most benign escalations
  come from the heuristic `ESCALATE` branch, not from classifier upgrades
  on heuristic-`ALLOW`ed entries.

  Detailed per-source/per-class breakdown: `tests/adversarial/distribution_shift_v0_5_3.json`.
  Reproducible via `scripts/eval_distribution_shift.py`. A proper OOF
  split for the LLM-generated portion (re-running held-out per fold) is
  a v0.7 follow-up if the gap demands it.
- **Stack composition (v0.6 measurement).** The full-stack numbers above
  decompose into independent layer contributions. `heuristic_only` recall
  is 35% / 63% (hand-curated / LLM-generated). `classifier_only` recall
  is 94% / 86%. Layers are not redundant - heuristic catches a small set
  of attacks the classifier misses, justifying the ensemble. Most of the
  full-stack benign FPR comes from heuristic ESCALATEs, not classifier
  upgrades. Detailed breakdown: `tests/adversarial/stack_ablation_v0_5_3.json`.
  Reproducible via `scripts/eval_stack_ablation.py`.
- **Adaptive-attacker calibration (v0.6 measurement of v0.5.3 stack).**
  PAIR (Chao et al. 2023) iterative attacker against the full Vaara
  stack:
  - Attacker + judge model: Qwen2.5-32B-Instruct (Apache 2.0)
  - Seed corpus: 25 hand-curated jailbreak entries (`tests/adversarial/jailbreak.jsonl`)
  - Max iterations per seed: 5
  - Total LLM calls: 125 attacker iterations across 25 seeds, plus
    judge confirmations on heuristic-ALLOW outcomes
  - **ASR: 0.0% (0/25)**. Across 125 candidate prompts, Vaara
    escalated 124 and allowed 1. The judge ruled the allowed candidate
    not a successful jailbreak.

  Reading: Vaara stack catches DAN-roleplay, "hypothetical scenario",
  and "security drill" -style jailbreak attempts at this attacker
  capability level. NOT a claim of imperviousness to all adaptive attackers
  - a stronger attacker model (70B+), longer iteration budgets, or
  different strategies (multi-turn drift, language-switch, obfuscation)
  might produce non-zero ASR. v0.7 follow-up: re-run with 70B+ attacker
  + judge if a compliance audience requires the harder calibration.

  Detailed per-seed breakdown: `tests/adversarial/pair_v0_5_3.json`.
  Reproducible via `scripts/eval_pair_attack.py`.

## Questions

Issues, mistakes in article mapping, regulator feedback: open an
issue at
[github.com/vaaraio/vaara/issues](https://github.com/vaaraio/vaara/issues)
or email compliance@vaara.io.
