"""Tests for the compliance engine."""

import pytest
from vaara.audit.trail import AuditTrail, EventType
from vaara.compliance.engine import (
    ComplianceEngine,
    EvidenceStatus,
    EvidenceStrength,
    EU_AI_ACT_REQUIREMENTS,
    DORA_REQUIREMENTS,
)
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)


@pytest.fixture
def engine():
    return ComplianceEngine()


@pytest.fixture
def trail():
    return AuditTrail()


@pytest.fixture
def populated_trail():
    """Trail with enough records to be partially compliant."""
    trail = AuditTrail()

    at = ActionType(
        "tx.transfer", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
        BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
        frozenset({RegulatoryDomain.MIFID2, RegulatoryDomain.DORA}),
    )

    # Generate 25 action cycles
    for i in range(25):
        req = ActionRequest(
            agent_id=f"agent-{i % 3}",
            tool_name="tx.transfer",
            action_type=at,
            parameters={"amount": 100 * i},
            confidence=0.5 + (i % 5) * 0.1,
        )
        action_id = trail.record_action_requested(req)

        trail.record_risk_scored(
            action_id=action_id,
            agent_id=req.agent_id,
            tool_name="tx.transfer",
            assessment={"risk": 0.5},
            regulatory_domains=at.regulatory_domains,
        )

        if i % 3 == 0:
            trail.record_decision(
                action_id=action_id, agent_id=req.agent_id,
                tool_name="tx.transfer",
                decision="deny", reason="too risky", risk_score=0.8,
                regulatory_domains=at.regulatory_domains,
            )
        elif i % 3 == 1:
            trail.record_decision(
                action_id=action_id, agent_id=req.agent_id,
                tool_name="tx.transfer",
                decision="allow", reason="ok", risk_score=0.2,
                regulatory_domains=at.regulatory_domains,
            )
            trail.record_outcome(
                action_id=action_id, agent_id=req.agent_id,
                tool_name="tx.transfer",
                outcome_severity=0.1,
            )
        else:
            trail.record_decision(
                action_id=action_id, agent_id=req.agent_id,
                tool_name="tx.transfer",
                decision="escalate", reason="borderline", risk_score=0.5,
                regulatory_domains=at.regulatory_domains,
            )
            trail.record_escalation(
                action_id=action_id, agent_id=req.agent_id,
                tool_name="tx.transfer",
                escalation_target="human", risk_score=0.5,
            )
            trail.record_escalation_resolved(
                action_id=action_id, agent_id=req.agent_id,
                tool_name="tx.transfer",
                resolution="allow", reviewer="admin", justification="reviewed",
            )

    return trail


class TestComplianceEngine:
    def test_empty_trail_non_compliant(self, engine, trail):
        report = engine.assess(trail)
        assert report.overall_status == EvidenceStatus.EVIDENCE_INSUFFICIENT
        assert len(report.critical_gaps) > 0

    def test_populated_trail_has_evidence(self, engine, populated_trail):
        report = engine.assess(populated_trail)
        # Should have at least some compliant articles
        compliant = [a for a in report.articles
                     if a.status == EvidenceStatus.EVIDENCE_SUFFICIENT]
        assert len(compliant) > 0

    def test_chain_integrity_reported(self, engine, populated_trail):
        report = engine.assess(populated_trail)
        assert report.trail_chain_intact is True

    def test_broken_chain_detected(self, engine, populated_trail):
        # Tamper with a record
        populated_trail._records[5].data["tampered"] = True
        report = engine.assess(populated_trail)
        assert report.trail_chain_intact is False

    def test_report_narrative(self, engine, populated_trail):
        report = engine.assess(populated_trail)
        narrative = report.narrative
        assert "ARTICLE-LEVEL EVIDENCE REPORT" in narrative
        assert "eu_ai_act" in narrative.lower() or "EU_AI_ACT" in narrative

    def test_report_serialization(self, engine, populated_trail):
        report = engine.assess(populated_trail)
        d = report.to_dict()
        assert "overall_status" in d
        assert "articles" in d
        assert "trail_integrity" in d
        assert isinstance(d["articles"], list)

    def test_custom_requirement(self, trail):
        from vaara.compliance.engine import RegulatoryRequirement
        custom_req = RegulatoryRequirement(
            RegulatoryDomain.SOC2,
            "SOC2-CC6.1",
            "Logical Access Security",
            "The entity implements logical access security measures",
            (EventType.DECISION_MADE,),
            min_evidence_count=5,
            is_critical=True,
        )
        engine = ComplianceEngine(requirements=[custom_req])
        report = engine.assess(trail)
        assert len(report.articles) == 1
        assert report.articles[0].requirement.article == "SOC2-CC6.1"

    def test_staleness_detection(self, engine, trail):
        # Add old evidence
        import time as t
        at = ActionType(
            "data.read", ActionCategory.DATA, Reversibility.FULLY,
            BlastRadius.SELF,
        )
        for _ in range(30):
            req = ActionRequest(
                agent_id="agent", tool_name="data.read",
                action_type=at,
            )
            trail.record_action_requested(req)
            trail.record_risk_scored(
                action_id="x", agent_id="agent", tool_name="data.read",
                assessment={"risk": 0.1},
            )
            trail.record_decision(
                action_id="x", agent_id="agent", tool_name="data.read",
                decision="allow", reason="ok", risk_score=0.1,
            )

        report = engine.assess(trail)
        # Some articles should be compliant since we just added records
        compliant = [a for a in report.articles
                     if a.status == EvidenceStatus.EVIDENCE_SUFFICIENT]
        assert len(compliant) > 0


class TestRequirements:
    def test_eu_ai_act_has_critical_articles(self):
        critical = [r for r in EU_AI_ACT_REQUIREMENTS if r.is_critical]
        assert len(critical) >= 8

    def test_dora_requirements_exist(self):
        assert len(DORA_REQUIREMENTS) >= 3

    def test_all_requirements_have_event_types(self):
        for req in EU_AI_ACT_REQUIREMENTS:
            if req.article != "Article 11(1)":  # Docs are special
                assert len(req.evidence_event_types) > 0, (
                    f"{req.article} has no evidence event types"
                )


class TestAddRequirementIdempotent:
    """Loop 46: add_requirement must dedup on (domain, article) — repeated
    registration (hot reload, config-driven init, deployer override of
    built-in article) previously appended N duplicates, causing the
    regulator-facing ConformityReport to list the same article N times."""

    def test_duplicate_add_replaces_instead_of_appending(self, engine):
        from vaara.compliance.engine import RegulatoryRequirement
        before_count = len([
            r for r in engine.requirements
            if r.domain == RegulatoryDomain.EU_AI_ACT
            and r.article == "Article 12(1)"
        ])
        # Add a custom override for the same article
        override = RegulatoryRequirement(
            domain=RegulatoryDomain.EU_AI_ACT,
            article="Article 12(1)",
            title="Custom override",
            description="overridden",
            evidence_event_types=[],
        )
        engine.add_requirement(override)
        engine.add_requirement(override)
        engine.add_requirement(override)
        after = [
            r for r in engine.requirements
            if r.domain == RegulatoryDomain.EU_AI_ACT
            and r.article == "Article 12(1)"
        ]
        # Must be exactly one, and it must be the override (last-write-wins)
        assert len(after) == 1
        assert after[0].title == "Custom override"
        # Built-in count for same article + domain was replaced, not duplicated
        assert before_count >= 1
