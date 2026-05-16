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
`ComplianceEngine` requirements in `vaara.compliance.engine`.

| Article | Requirement | Evidence Vaara produces |
|---|---|---|
| **9(1)** | Risk Management System | Every intercepted action is scored and the score is recorded with inputs. `RISK_SCORED` events. |
| **9(2)(a)** | Risk Identification and Analysis | `RISK_SCORED` plus `ACTION_BLOCKED` records show which risks were detected and which were blocked. |
| **9(4)(a)** | Risk Mitigation Measures | `ACTION_BLOCKED` and `DECISION_MADE` records show mitigation applied per action. |
| **9(7)** | Testing Procedures | `RISK_SCORED` plus `OUTCOME_RECORDED` pairs form the test signal. Conformal intervals give distribution-free calibration metrics. |
| **11(1)** | Technical Documentation | Checked outside the audit trail. Vaara does not replace the Annex IV technical file. |
| **12(1)** | Record-Keeping (Logging) | Every `ACTION_REQUESTED`, `RISK_SCORED`, and `DECISION_MADE` is written to a hash-chained, tamper-evident trail. See "Audit trail integrity" below. |
| **13(1)** | Transparency and Provision of Information | `RISK_SCORED` and `DECISION_MADE` records carry the risk score, the interval, the decision, and the reason string shown to the operator. |
| **14(1)** | Human Oversight -- Design | `ESCALATION_SENT` and `ESCALATION_RESOLVED` events prove the oversight path exists and was exercised. The `vaara.audit.review_queue` storage layer turns `escalate` into a substantive queued-for-review step rather than a fire-and-forget log line; the `vaara review` CLI is the operator surface. |
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
  route). Exit code 0 if no errors; warnings print without flipping
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
| **prEN ISO/IEC 12792** Transparency Taxonomy of AI Systems | WG4 | Stage 40 (final vote) | v0.6 ships per-action audit records tagged against the four-axis model (System Operation, Data Usage, Decision Making, Limitations) via four optional `AuditRecord` fields. Default classification heuristic by event type; per-record override available. NOT tamper-evident in v0.6 — fields are metadata annotations excluded from `record_hash` so pre-v0.6 chains stay valid. |

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
**self-attestation is not sufficient** — the entity attesting to
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
  requires layering an external IAP — a notary service that the
  operator does not control — on top of Vaara's emitted evidence.
- **Vaara's design admits an external IAP layer without internal
  change.** The hash chain, the commit-prove receipt pair, and the
  HTTP API surface all produce structured, signable artefacts that
  an IAP can co-sign, batch into a transparency log, or seal under
  eIDAS qualified trust services. Future work: a documented IAP
  adapter interface.

This positioning is deliberate. Vaara does not claim AAL-4
conformance and does not market a self-attestation pattern.
Operators who need AAL-4 should pair Vaara with an independent
attestation provider; the Vaara-emitted evidence is the input to
that provider, not a replacement for it.

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
  the live DB. The signed zip remains self-consistent forever; the
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
  numbers above are dominated by the heuristic — most benign escalations
  come from the heuristic `ESCALATE` branch, not from classifier upgrades
  on heuristic-`ALLOW`ed entries.

  Detailed per-source/per-class breakdown: `tests/adversarial/distribution_shift_v0_5_3.json`.
  Reproducible via `scripts/eval_distribution_shift.py`. A proper OOF
  split for the LLM-generated portion (re-running held-out per fold) is
  a v0.7 follow-up if the gap demands it.
- **Stack composition (v0.6 measurement).** The full-stack numbers above
  decompose into independent layer contributions. `heuristic_only` recall
  is 35% / 63% (hand-curated / LLM-generated); `classifier_only` recall
  is 94% / 86%. Layers are not redundant — heuristic catches a small set
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
    escalated 124 and allowed 1; the judge ruled the allowed candidate
    not a successful jailbreak.

  Reading: Vaara stack catches DAN-roleplay, "hypothetical scenario",
  and "security drill" -style jailbreak attempts at this attacker
  capability level. NOT a claim of imperviousness to all adaptive attackers
  — a stronger attacker model (70B+), longer iteration budgets, or
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
