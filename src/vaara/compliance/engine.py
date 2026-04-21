"""Compliance engine — collects article-by-article evidence from the audit trail.

This module takes the audit trail as input and produces:

1. **Article-by-article evidence mapping**: For each relevant regulatory article,
   which audit records provide evidence relevant to the article's requirements.
2. **Coverage analysis**: Which articles have evidence vs which have gaps.
3. **Evidence reports**: Structured reports that a deployer can submit as
   input to their own conformity work (internal audit, Notified-Body
   review, regulatory filing). The report is evidence — not a conformity
   determination; that decision is reserved to the deployer.

Supported regulatory frameworks:
- EU AI Act (Articles 9, 11-15, 17, 61) — high-risk AI system requirements
- DORA (Articles 10, 12, 13) — ICT risk management for financial entities
- NIS2 — network and information security
- GDPR (data protection impact assessments)
- MiFID II (algorithmic trading compliance)

The engine is declarative — it doesn't know what "good" looks like in absolute
terms.  Instead, it maps evidence to requirements and flags where evidence
is missing, insufficient, or stale.

EU AI Act enforcement for high-risk systems: August 2, 2026.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from vaara.audit.trail import AuditTrail, EventType
from vaara.taxonomy.actions import RegulatoryDomain

import logging

logger = logging.getLogger(__name__)


# ── Evidence status ───────────────────────────────────────────────────────
#
# NOTE: Vaara records the *evidence status* for each article — whether the
# runtime has captured artefacts that a deployer (and their auditor) could
# use to argue the article is satisfied. It is NOT a legal determination
# of compliance. A conformity verdict under the EU AI Act is made by the
# deployer (and, for high-risk systems, a Notified Body), not by a library.


class EvidenceStatus(str, Enum):
    """Status of evidence collected for a specific regulatory article."""
    EVIDENCE_SUFFICIENT = "evidence_sufficient"      # Evidence present and within freshness window
    EVIDENCE_PARTIAL = "evidence_partial"            # Some evidence, gaps remain
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"  # No evidence or critical gaps
    NOT_APPLICABLE = "not_applicable"                # Article doesn't apply to this system


# Backwards-compatible alias for pre-0.3.1 imports.
ComplianceStatus = EvidenceStatus


class EvidenceStrength(str, Enum):
    """How strong is the evidence for a compliance claim."""
    STRONG = "strong"         # Continuous, automated, verifiable
    MODERATE = "moderate"     # Present but manual or intermittent
    WEAK = "weak"             # Indirect or insufficient
    ABSENT = "absent"         # No evidence


# ── Regulatory requirement definitions ────────────────────────────────────

@dataclass(frozen=True)
class RegulatoryRequirement:
    """A specific regulatory requirement to check compliance against."""
    domain: RegulatoryDomain
    article: str
    title: str
    description: str
    evidence_event_types: tuple[EventType, ...]  # Which events provide evidence
    min_evidence_count: int = 1                  # Minimum records needed
    staleness_hours: float = 720.0               # Evidence older than this = stale (30 days)
    is_critical: bool = False                    # True = blocks conformity if missing


# EU AI Act requirements for high-risk AI systems
EU_AI_ACT_REQUIREMENTS = [
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 9(1)",
        "Risk Management System",
        "A risk management system shall be established, implemented, documented and maintained",
        (EventType.RISK_SCORED,),
        min_evidence_count=10,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 9(2)(a)",
        "Risk Identification and Analysis",
        "Identification and analysis of known and reasonably foreseeable risks",
        (EventType.RISK_SCORED, EventType.ACTION_BLOCKED),
        min_evidence_count=5,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 9(4)(a)",
        "Risk Mitigation Measures",
        "Elimination or reduction of risks as far as possible through adequate design and development",
        (EventType.ACTION_BLOCKED, EventType.DECISION_MADE),
        min_evidence_count=3,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 9(7)",
        "Testing Procedures",
        "Testing shall be suitable to identify relevant risks and use appropriately precise metrics",
        (EventType.RISK_SCORED, EventType.OUTCOME_RECORDED),
        min_evidence_count=10,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 11(1)",
        "Technical Documentation",
        "Technical documentation shall be drawn up before the system is placed on the market",
        (),  # Checked separately — docs exist outside audit trail
        min_evidence_count=0,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 12(1)",
        "Record-Keeping (Logging)",
        "High-risk AI systems shall technically allow for the automatic recording of events",
        (EventType.ACTION_REQUESTED, EventType.RISK_SCORED, EventType.DECISION_MADE),
        min_evidence_count=20,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 13(1)",
        "Transparency and Provision of Information",
        "High-risk AI systems shall be designed and developed to ensure their operation is sufficiently transparent",
        (EventType.RISK_SCORED, EventType.DECISION_MADE),
        min_evidence_count=5,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 14(1)",
        "Human Oversight — Design",
        "High-risk AI systems shall be designed to be effectively overseen by natural persons",
        (EventType.ESCALATION_SENT, EventType.ESCALATION_RESOLVED),
        min_evidence_count=1,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 14(4)(d)",
        "Human Oversight — Override Capability",
        "The human shall be able to decide not to use the system or to override the output",
        (EventType.ESCALATION_RESOLVED, EventType.POLICY_OVERRIDE),
        min_evidence_count=1,
        is_critical=False,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 15(1)",
        "Accuracy, Robustness and Cybersecurity",
        "High-risk AI systems shall be designed to achieve appropriate levels of accuracy",
        (EventType.OUTCOME_RECORDED,),
        min_evidence_count=10,
        staleness_hours=168.0,  # Weekly calibration
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.EU_AI_ACT,
        "Article 61(1)",
        "Post-Market Monitoring",
        "Providers shall establish a post-market monitoring system proportionate to the nature of the AI",
        (EventType.OUTCOME_RECORDED,),
        min_evidence_count=20,
        is_critical=True,
    ),
]

DORA_REQUIREMENTS = [
    RegulatoryRequirement(
        RegulatoryDomain.DORA,
        "Article 10(1)",
        "ICT Risk Management — Protection and Prevention",
        "Financial entities shall have ICT security policies and mechanisms for protection and prevention",
        (EventType.ACTION_BLOCKED, EventType.DECISION_MADE),
        min_evidence_count=5,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.DORA,
        "Article 12(1)",
        "ICT Incident Detection",
        "Financial entities shall have ICT-related incident detection capabilities with automated alert mechanisms",
        (EventType.ACTION_REQUESTED, EventType.ACTION_BLOCKED),
        min_evidence_count=10,
        is_critical=True,
    ),
    RegulatoryRequirement(
        RegulatoryDomain.DORA,
        "Article 13(1)",
        "ICT Incident Response and Learning",
        "Financial entities shall put in place response and recovery plans that include learning and evolving",
        (EventType.OUTCOME_RECORDED,),
        min_evidence_count=5,
        is_critical=False,
    ),
]


# ── Evidence assessment ───────────────────────────────────────────────────

@dataclass
class ArticleEvidence:
    """Evidence assessment for a single regulatory article."""
    requirement: RegulatoryRequirement
    status: ComplianceStatus
    strength: EvidenceStrength
    evidence_count: int
    freshest_evidence_age_hours: float
    oldest_evidence_age_hours: float
    sample_record_ids: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # When evidence_count==0 the freshness ages are float('inf') sentinel
        # values. Leaving them in the dict breaks any strict-JSON consumer —
        # json.dumps(allow_nan=False), Go encoding/json, Rust serde_json all
        # reject Infinity. Coerce to None at the dict boundary so the report
        # stays RFC 8259-portable regardless of how the caller serializes it.
        # strict_json_dumps in MCP already scrubs, but direct callers using
        # stock json.dumps or consuming via cross-language tooling don't.
        import math as _math
        def _finite_or_none(v: float) -> Optional[float]:
            return round(v, 1) if _math.isfinite(v) else None

        return {
            "domain": self.requirement.domain.value,
            "article": self.requirement.article,
            "title": self.requirement.title,
            "status": self.status.value,
            "strength": self.strength.value,
            "evidence_count": self.evidence_count,
            "freshest_evidence_age_hours": _finite_or_none(self.freshest_evidence_age_hours),
            "oldest_evidence_age_hours": _finite_or_none(self.oldest_evidence_age_hours),
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "sample_record_ids": self.sample_record_ids[:5],
        }


@dataclass
class ConformityReport:
    """Article-by-article evidence report for use in a deployer's conformity work.

    This is an evidence artefact. It does not by itself constitute a
    conformity determination or legal compliance verdict.
    """
    generated_at: float
    system_name: str
    system_version: str
    overall_status: ComplianceStatus
    critical_gaps: list[str]
    articles: list[ArticleEvidence]
    summary: str
    trail_size: int
    trail_chain_intact: bool

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "generated_at_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.generated_at)
            ),
            "system_name": self.system_name,
            "system_version": self.system_version,
            "overall_status": self.overall_status.value,
            "critical_gaps": self.critical_gaps,
            "trail_integrity": {
                "size": self.trail_size,
                "chain_intact": self.trail_chain_intact,
            },
            "articles": [a.to_dict() for a in self.articles],
            "summary": self.summary,
        }

    @property
    def narrative(self) -> str:
        """Human-readable evidence report narrative.

        Not a conformity determination — deployer and (where required)
        Notified Body own that decision.
        """
        ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self.generated_at))
        lines = [
            f"ARTICLE-LEVEL EVIDENCE REPORT",
            f"System: {self.system_name} v{self.system_version}",
            f"Generated: {ts}",
            f"Overall Evidence Status: {self.overall_status.value.upper()}",
            f"Audit Trail: {self.trail_size} records, "
            f"chain {'INTACT' if self.trail_chain_intact else 'BROKEN'}",
            "",
        ]

        if self.critical_gaps:
            lines.append("CRITICAL GAPS:")
            for gap in self.critical_gaps:
                lines.append(f"  - {gap}")
            lines.append("")

        # Group by domain
        by_domain: dict[str, list[ArticleEvidence]] = {}
        for article in self.articles:
            domain = article.requirement.domain.value
            by_domain.setdefault(domain, []).append(article)

        for domain, articles in sorted(by_domain.items()):
            lines.append(f"--- {domain.upper()} ---")
            for art in articles:
                status_icon = {
                    EvidenceStatus.EVIDENCE_SUFFICIENT: "[OK]",
                    EvidenceStatus.EVIDENCE_PARTIAL: "[!!]",
                    EvidenceStatus.EVIDENCE_INSUFFICIENT: "[XX]",
                    EvidenceStatus.NOT_APPLICABLE: "[--]",
                }[art.status]
                lines.append(
                    f"  {status_icon} {art.requirement.article}: "
                    f"{art.requirement.title}"
                )
                lines.append(
                    f"       Evidence: {art.evidence_count} records, "
                    f"strength={art.strength.value}"
                )
                if art.gaps:
                    for gap in art.gaps:
                        lines.append(f"       Gap: {gap}")
                if art.recommendations:
                    for rec in art.recommendations:
                        lines.append(f"       Rec: {rec}")
            lines.append("")

        lines.append(f"SUMMARY: {self.summary}")
        return "\n".join(lines)


# ── Compliance Engine ─────────────────────────────────────────────────────

class ComplianceEngine:
    """Evaluates audit trail evidence against regulatory requirements.

    The engine is stateless — it reads the audit trail and produces
    a point-in-time article-level evidence report.  Run it periodically
    or on-demand for reporting.  The report is an evidence artefact for
    a deployer's own conformity work; it is not itself a conformity
    determination.
    """

    def __init__(
        self,
        requirements: Optional[list[RegulatoryRequirement]] = None,
    ) -> None:
        """
        Args:
            requirements: Regulatory requirements to check.
                          Defaults to EU AI Act + DORA.
        """
        self._requirements = list(
            requirements or (EU_AI_ACT_REQUIREMENTS + DORA_REQUIREMENTS)
        )
        self._lock = threading.Lock()

    def assess(
        self,
        trail: AuditTrail,
        system_name: str = "Vaara Execution Layer",
        system_version: str = "0.1.0",
    ) -> ConformityReport:
        """Assemble an article-level evidence report from the audit trail.

        Returns a structured report with per-article evidence mapping,
        gap analysis, and recommendations.  This is evidence input to a
        deployer's own conformity work — not a conformity determination.
        """
        now = time.time()
        articles: list[ArticleEvidence] = []
        critical_gaps: list[str] = []
        with self._lock:
            requirements_snapshot = list(self._requirements)

        # Verify chain integrity first. A broken chain means every record
        # below is potentially tampered, so per-article SUFFICIENT/STRONG
        # tags would mislead an auditor filtering by status. Downgrade
        # every article to EVIDENCE_INSUFFICIENT + ABSENT with an explicit
        # gap, so dashboards cannot render green cells over broken evidence.
        chain_error = trail.verify_chain()
        chain_broken = chain_error is not None

        for req in requirements_snapshot:
            evidence = self._assess_article(trail, req, now)
            if chain_broken:
                evidence.status = EvidenceStatus.EVIDENCE_INSUFFICIENT
                evidence.strength = EvidenceStrength.ABSENT
                evidence.gaps.insert(
                    0,
                    f"Audit chain integrity compromised: {chain_error}; "
                    f"evidence cannot be trusted until chain is "
                    f"reconstructed or re-verified",
                )
                evidence.recommendations.insert(
                    0,
                    "Investigate chain break, restore from verified "
                    "backup, and re-run conformity assessment",
                )
            articles.append(evidence)

            if (
                req.is_critical
                and evidence.status == EvidenceStatus.EVIDENCE_INSUFFICIENT
            ):
                critical_gaps.append(
                    f"{req.article} ({req.title}): {'; '.join(evidence.gaps)}"
                )

        # Overall status: worst of critical articles. Broken chain already
        # forced every critical article to INSUFFICIENT above, so this
        # naturally lands on EVIDENCE_INSUFFICIENT in that case.
        critical_statuses = [
            a.status for a in articles if a.requirement.is_critical
        ]
        if EvidenceStatus.EVIDENCE_INSUFFICIENT in critical_statuses:
            overall = EvidenceStatus.EVIDENCE_INSUFFICIENT
        elif EvidenceStatus.EVIDENCE_PARTIAL in critical_statuses:
            overall = EvidenceStatus.EVIDENCE_PARTIAL
        elif critical_statuses:
            overall = EvidenceStatus.EVIDENCE_SUFFICIENT
        else:
            overall = EvidenceStatus.NOT_APPLICABLE

        # Generate summary
        total = len(articles)
        sufficient = sum(1 for a in articles if a.status == EvidenceStatus.EVIDENCE_SUFFICIENT)
        partial = sum(1 for a in articles if a.status == EvidenceStatus.EVIDENCE_PARTIAL)
        insufficient = sum(1 for a in articles if a.status == EvidenceStatus.EVIDENCE_INSUFFICIENT)

        summary = (
            f"{sufficient}/{total} articles with sufficient evidence, "
            f"{partial} partial, {insufficient} insufficient. "
            f"{len(critical_gaps)} critical gaps."
        )

        return ConformityReport(
            generated_at=now,
            system_name=system_name,
            system_version=system_version,
            overall_status=overall,
            critical_gaps=critical_gaps,
            articles=articles,
            summary=summary,
            trail_size=trail.size,
            trail_chain_intact=not chain_broken,
        )

    def _assess_article(
        self,
        trail: AuditTrail,
        req: RegulatoryRequirement,
        now: float,
    ) -> ArticleEvidence:
        """Assess evidence for a single regulatory article."""
        # Special case: requirements that need external evidence (like docs)
        if not req.evidence_event_types:
            return ArticleEvidence(
                requirement=req,
                status=EvidenceStatus.EVIDENCE_PARTIAL,
                strength=EvidenceStrength.WEAK,
                evidence_count=0,
                freshest_evidence_age_hours=float("inf"),
                oldest_evidence_age_hours=float("inf"),
                gaps=["Requires manual verification (documentation, design docs)"],
                recommendations=[
                    "Provide technical documentation as per Annex IV"
                ],
            )

        # Collect evidence records
        evidence_records = []
        for event_type in req.evidence_event_types:
            evidence_records.extend(
                trail.get_records_by_type(
                    event_type, limit=max(500, req.min_evidence_count)
                )
            )

        evidence_count = len(evidence_records)

        if evidence_count == 0:
            return ArticleEvidence(
                requirement=req,
                status=EvidenceStatus.EVIDENCE_INSUFFICIENT,
                strength=EvidenceStrength.ABSENT,
                evidence_count=0,
                freshest_evidence_age_hours=float("inf"),
                oldest_evidence_age_hours=float("inf"),
                gaps=[f"No audit evidence found for {req.article}"],
                recommendations=[
                    f"Enable {', '.join(et.value for et in req.evidence_event_types)} "
                    f"recording in the audit trail"
                ],
            )

        # Analyze freshness. Drop NaN/inf timestamps so a single bad row
        # can't poison the report; clamp negative ages (future timestamps
        # from clock skew or spoofed clocks) to 0 so evidence doesn't
        # spuriously flip to STRONG via the `age < staleness/4` check.
        # Surface the clock-skew case as an explicit gap — a regulator
        # reading "STRONG evidence" should not see that status implicitly
        # granted by timestamps from the future.
        import math
        valid_timestamps = [
            r.timestamp for r in evidence_records
            if isinstance(r.timestamp, (int, float))
            and math.isfinite(r.timestamp)
        ]
        future_timestamp_count = sum(1 for t in valid_timestamps if t > now)
        if not valid_timestamps:
            freshest_age_hours = float("inf")
            oldest_age_hours = float("inf")
        else:
            freshest_age_hours = max(0.0, (now - max(valid_timestamps)) / 3600)
            oldest_age_hours = max(0.0, (now - min(valid_timestamps)) / 3600)

        # Determine evidence strength
        gaps = []
        recommendations = []

        if evidence_count < req.min_evidence_count:
            gaps.append(
                f"Insufficient evidence: {evidence_count}/{req.min_evidence_count} "
                f"records (need {req.min_evidence_count - evidence_count} more)"
            )
            recommendations.append(
                f"Continue operating to accumulate at least "
                f"{req.min_evidence_count} records"
            )

        if freshest_age_hours > req.staleness_hours:
            gaps.append(
                f"Evidence is stale: most recent is "
                f"{freshest_age_hours:.0f}h old (threshold: {req.staleness_hours:.0f}h)"
            )
            recommendations.append(
                "Ensure the system is actively processing and logging actions"
            )

        if future_timestamp_count > 0:
            gaps.append(
                f"{future_timestamp_count} evidence record(s) have "
                f"timestamps in the future — possible clock skew or "
                f"tampering; freshness may be overstated"
            )
            recommendations.append(
                "Check system clock synchronisation (NTP); verify audit "
                "chain integrity for tampering"
            )

        # Assess strength
        if evidence_count >= req.min_evidence_count * 2 and freshest_age_hours < req.staleness_hours / 4:
            strength = EvidenceStrength.STRONG
        elif evidence_count >= req.min_evidence_count and freshest_age_hours < req.staleness_hours:
            strength = EvidenceStrength.MODERATE
        elif evidence_count > 0:
            strength = EvidenceStrength.WEAK
        else:
            strength = EvidenceStrength.ABSENT

        # If any record has a future timestamp the freshness signal is
        # untrustworthy — clamping made it look ideal, so STRONG would
        # otherwise be granted by an unverifiable clock. Downgrade one
        # tier so the strength field on its own (consumed by dashboards
        # that ignore gaps) does not imply confidence an auditor
        # shouldn't have.
        if future_timestamp_count > 0:
            if strength == EvidenceStrength.STRONG:
                strength = EvidenceStrength.MODERATE
            elif strength == EvidenceStrength.MODERATE:
                strength = EvidenceStrength.WEAK

        # Determine status
        if not gaps:
            status = EvidenceStatus.EVIDENCE_SUFFICIENT
        elif evidence_count >= req.min_evidence_count:
            status = EvidenceStatus.EVIDENCE_PARTIAL  # Enough records but maybe stale
        else:
            status = EvidenceStatus.EVIDENCE_INSUFFICIENT

        # Sample record IDs for reference
        sample_ids = [r.record_id for r in evidence_records[:5]]

        return ArticleEvidence(
            requirement=req,
            status=status,
            strength=strength,
            evidence_count=evidence_count,
            freshest_evidence_age_hours=freshest_age_hours,
            oldest_evidence_age_hours=oldest_age_hours,
            sample_record_ids=sample_ids,
            gaps=gaps,
            recommendations=recommendations,
        )

    # ── Utilities ─────────────────────────────────────────────────

    def add_requirement(self, requirement: RegulatoryRequirement) -> None:
        """Add a custom regulatory requirement.

        Idempotent on (domain, article): a repeat registration replaces
        the prior entry instead of appending a duplicate. Without this,
        hot reload, config-driven registration, or a deployer customising
        a built-in article (e.g. overriding Article 12(1) description)
        silently produces N duplicate rows in the ConformityReport — a
        regulator auditing the report sees the same article N times with
        slightly different text and cannot reconcile. Same pattern as
        AdaptiveScorer.add_sequence_pattern for the same reason.
        """
        key = (requirement.domain, requirement.article)
        with self._lock:
            self._requirements = [
                r for r in self._requirements
                if (r.domain, r.article) != key
            ]
            self._requirements.append(requirement)

    @property
    def requirements(self) -> list[RegulatoryRequirement]:
        with self._lock:
            return list(self._requirements)

    def requirements_by_domain(
        self, domain: RegulatoryDomain
    ) -> list[RegulatoryRequirement]:
        with self._lock:
            return [r for r in self._requirements if r.domain == domain]


# ── Factory ───────────────────────────────────────────────────────────────

def create_default_engine(**kwargs: Any) -> ComplianceEngine:
    """Create a compliance engine with production defaults."""
    return ComplianceEngine(**kwargs)
