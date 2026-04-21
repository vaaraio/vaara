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
| **14(1)** | Human Oversight -- Design | `ESCALATION_SENT` and `ESCALATION_RESOLVED` events prove the oversight path exists and was exercised. |
| **14(4)(d)** | Human Oversight -- Override Capability | `ESCALATION_RESOLVED` and `POLICY_OVERRIDE` events prove a human can decide not to proceed or can override Vaara's decision. |
| **15(1)** | Accuracy, Robustness and Cybersecurity | `OUTCOME_RECORDED` events feed the adaptive scorer. Recency is tracked (default weekly calibration window). |
| **61(1)** | Post-Market Monitoring | `OUTCOME_RECORDED` events form the post-market signal, tied back to the original action via `action_id`. |

### Article 14 in particular

The risk interval is what makes Article 14 oversight substantive rather
than cosmetic. A point score of 0.6 tells a reviewer nothing about
whether the model is confident. A conformal interval of [0.58, 0.62]
versus [0.2, 0.95] tells them whether to trust the number. Vaara
surfaces both on every escalation.

### Article 26 (deployer obligations)

Article 26 obligations sit on the deployer, not on Vaara. The evidence
Vaara produces is the feedstock a deployer uses to satisfy 26(1)
("use the system in accordance with the instructions for use"),
26(5) ("monitor operation"), and 26(6) ("keep logs"). Deployer conduct
outside the Vaara pipeline is not in scope.

## DORA Article Mapping

Relevant for financial entities only. The default `ComplianceEngine`
also ships with a DORA bundle:

| Article | Requirement | Evidence Vaara produces |
|---|---|---|
| **10(1)** | ICT Risk Management -- Protection and Prevention | `ACTION_BLOCKED` and `DECISION_MADE` records. |
| **12(1)** | ICT Incident Detection | `ACTION_REQUESTED` and `ACTION_BLOCKED` records, with risk score and reason. |
| **13(1)** | ICT Incident Response and Learning | `OUTCOME_RECORDED` events close the loop and feed the adaptive scorer. |

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
  in accordance with the intended purpose and applicable law. Vaara
  does not purge on your behalf. Wire a retention job to your policy.

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

## Questions

Issues, mistakes in article mapping, regulator feedback: open an
issue at
[github.com/vaaraio/vaara/issues](https://github.com/vaaraio/vaara/issues)
or email compliance@vaara.io.
