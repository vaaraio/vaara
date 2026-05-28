# Evidence Sufficiency Rules

This document is the public reference for how Vaara's compliance engine
decides whether the evidence collected for each regulatory article is
sufficient, partial, or insufficient. It is the companion document to
[COMPLIANCE.md](COMPLIANCE.md), which maps articles to the event types
that feed them. COMPLIANCE.md answers "what runtime evidence does Vaara
collect for Article X?" This document answers "given that evidence,
what does Vaara conclude, and why?"

The rules described here are the same rules executed by
`vaara.compliance.engine.ComplianceEngine.assess_conformity`. Every
report produced by `vaara compliance report` surfaces the inputs and
the rationale per article in the `verdict_inputs` block.

## Status and strength taxonomy

A single article carries two orthogonal labels in a report:

**Status.** Does the body of evidence meet the threshold?

| Value | Meaning |
|---|---|
| `evidence_sufficient` | Records present, count at or above threshold, freshest record within the staleness window, no clock-skew or chain integrity issues. |
| `evidence_partial` | Records present at or above the count threshold, but at least one freshness or integrity check flagged a gap. |
| `evidence_insufficient` | Record count below threshold, or no qualifying records at all. |
| `not_applicable` | Article does not apply to this system. |

**Strength.** How good is the evidence beyond the minimum?

| Value | Meaning |
|---|---|
| `strong` | Count at or above twice the minimum, freshest record younger than one quarter of the staleness window. |
| `moderate` | Count at or above the minimum and freshest record inside the staleness window, but not at strong levels. |
| `weak` | Records present, but either count below the minimum or freshest record beyond the staleness window. |
| `absent` | No qualifying records, or chain integrity broken. |

The status and strength labels are independent dimensions: a `weak`
article can still be `evidence_partial` (e.g. enough records but stale),
and a `strong` article is always `evidence_sufficient`.

## Decision tree

For each requirement, the engine collects records of the event types
listed for that article, then evaluates in this order:

1. **External-evidence requirements.** Articles like Article 11(1)
   carry no runtime event types. They return `evidence_partial`,
   `weak`, with the note "Requires manual verification (documentation,
   design docs)." The Annex IV technical file is the deployer's
   artefact. Vaara cannot synthesise it.
2. **Chain integrity check.** If `AuditTrail.verify_chain()` returns
   an error, every article is pinned to `evidence_insufficient` and
   `absent`, regardless of per-article counts. `verdict_inputs.chain_intact`
   flips to `false` and a chain-integrity warning is prepended to
   `verdict_reasons`. A dashboard cannot render a green cell over a
   broken chain.
3. **Gap detection.** Three gap classes are checked independently:
   - Count gap: fewer records than `min_evidence_count`.
   - Freshness gap: freshest record older than `staleness_hours`.
   - Clock-skew gap: at least one record carries a timestamp in the future.
4. **Status assignment.** If no gaps, `evidence_sufficient`. Else if
   the count threshold is met (only freshness or integrity gaps remain),
   `evidence_partial`. Otherwise `evidence_insufficient`.
5. **Strength assignment.** Count at or above `2 × min_evidence_count`
   AND freshest age below `staleness_hours / 4` produces `strong`. Count
   at or above the minimum AND freshest age inside the staleness window
   produces `moderate`. Any qualifying records below those bars are
   `weak`. No records are `absent`.
6. **Future-timestamp downgrade.** If any record carries a future
   timestamp, strength drops one tier (`strong` to `moderate`,
   `moderate` to `weak`). The freshness signal cannot be trusted when
   the clock cannot be trusted.

## EU AI Act per-article thresholds

The defaults below are the values shipped in
`EU_AI_ACT_REQUIREMENTS` in `vaara/compliance/engine.py`. A deployer
can override them by registering a custom `RegulatoryRequirement` with
the engine.

| Article | Title | Min count | Staleness window | Strong-count | Strong-freshness | Critical |
|---|---|---|---|---|---|---|
| 9(1) | Risk Management System | 10 | 720 h (30 d) | 20 | 180 h (7.5 d) | yes |
| 9(2)(a) | Risk Identification and Analysis | 5 | 720 h | 10 | 180 h | yes |
| 9(4)(a) | Risk Mitigation Measures | 3 | 720 h | 6 | 180 h | yes |
| 9(7) | Testing Procedures | 10 | 720 h | 20 | 180 h | yes |
| 11(1) | Technical Documentation | n/a | n/a | n/a | n/a | yes (external) |
| 12(1) | Record-Keeping (Logging) | 20 | 720 h | 40 | 180 h | yes |
| 13(1) | Transparency and Provision of Information | 5 | 720 h | 10 | 180 h | yes |
| 14(1) | Human Oversight: Design | 1 | 720 h | 2 | 180 h | yes |
| 14(4)(d) | Human Oversight: Override Capability | 1 | 720 h | 2 | 180 h | no |
| 15(1) | Accuracy, Robustness and Cybersecurity | 10 | 168 h (7 d) | 20 | 42 h | yes |
| 61(1) | Post-Market Monitoring | 20 | 720 h | 40 | 180 h | yes |

Article 11(1) is the external-evidence row. It does not consume runtime
events. The threshold table is intentionally blank.

Article 15(1) is the only EU AI Act row with a non-default staleness
window. It defaults to 168 hours (one week) because accuracy and
robustness signals are expected to be refreshed on a roughly weekly
calibration cadence, not a monthly one.

A requirement marked `is_critical=True` participates in the
overall-status roll-up. The overall report status is the worst status
across all critical articles: any critical `evidence_insufficient`
collapses the report to `evidence_insufficient` overall.

## DORA per-article thresholds

| Article | Title | Min count | Staleness window | Strong-count | Strong-freshness | Critical |
|---|---|---|---|---|---|---|
| 10(1) | ICT Risk Management: Protection and Prevention | 5 | 720 h | 10 | 180 h | yes |
| 12(1) | ICT Incident Detection | 10 | 720 h | 20 | 180 h | yes |
| 13(1) | ICT Incident Response and Learning | 5 | 720 h | 10 | 180 h | no |

## What an auditor sees

Every `ArticleEvidence` entry in a report carries a `verdict_inputs`
block with the threshold values the engine compared against and the
actual values it observed. Example (abbreviated) for Article 12(1):

```json
{
  "article": "Article 12(1)",
  "status": "evidence_sufficient",
  "strength": "strong",
  "evidence_count": 47,
  "freshest_evidence_age_hours": 1.3,
  "oldest_evidence_age_hours": 168.7,
  "verdict_inputs": {
    "min_evidence_count": 20,
    "staleness_hours": 720.0,
    "evidence_count_observed": 47,
    "freshest_evidence_age_hours": 1.3,
    "future_timestamp_count": 0,
    "chain_intact": true,
    "strength_thresholds": {
      "strong_min_count": 40,
      "strong_max_age_hours": 180.0,
      "moderate_min_count": 20,
      "moderate_max_age_hours": 720.0
    },
    "verdict_reasons": [
      "Status SUFFICIENT: 47 record(s) >= threshold 20, freshest 1.3h within 720h staleness window, no clock-skew indicators.",
      "Strength STRONG: 47 >= 2x threshold (40) and freshness 1.3h < staleness/4 (180.0h)."
    ]
  }
}
```

The same article rendered after a broken chain produces:

```json
{
  "article": "Article 12(1)",
  "status": "evidence_insufficient",
  "strength": "absent",
  "verdict_inputs": {
    "chain_intact": false,
    "verdict_reasons": [
      "Audit chain integrity compromised; status pinned to INSUFFICIENT and strength to ABSENT regardless of per-article evidence count.",
      "Status SUFFICIENT: 47 record(s) >= threshold 20, freshest 1.3h within 720h staleness window, no clock-skew indicators.",
      "Strength STRONG: 47 >= 2x threshold (40) and freshness 1.3h < staleness/4 (180.0h)."
    ]
  }
}
```

The second and third lines are preserved so an auditor can see what the
verdict would have been on the underlying evidence had the chain held.
The first line is the pin.

## What this document is not

This is an evidence-sufficiency reference, not a conformity
determination. The engine produces structured evidence with rationale.
A conformity verdict under the EU AI Act is made by the deployer (and,
for high-risk systems, a Notified Body), not by a library. The
deployer's auditor reads the per-article thresholds in this document
alongside the rationale lines in `verdict_inputs.verdict_reasons` and
forms their own judgement.
