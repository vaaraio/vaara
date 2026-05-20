"""Tests for the compliance engine."""

import pytest

from vaara.audit.trail import AuditTrail, EventType
from vaara.compliance.engine import (
    DORA_REQUIREMENTS,
    EU_AI_ACT_REQUIREMENTS,
    ComplianceEngine,
    EvidenceStatus,
    RegulatoryRequirement,
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


class TestVerdictDrillDown:
    """v0.26: ConformityReport articles carry verdict_inputs (threshold-vs-actual
    + rationale) and contributing_events (the specific audit records the
    verdict sits on). An auditor reading the report can trace status ->
    threshold delta -> concrete event without re-running the engine."""

    def test_verdict_inputs_present_for_populated_article(
        self, engine, populated_trail
    ):
        report = engine.assess(populated_trail)
        # Find an article that has runtime evidence (skip Article 11(1) docs).
        runtime_articles = [
            a for a in report.articles
            if a.requirement.evidence_event_types
        ]
        assert runtime_articles, "populated trail should yield runtime articles"
        for art in runtime_articles:
            vi = art.verdict_inputs
            assert vi, f"{art.requirement.article} missing verdict_inputs"
            assert "min_evidence_count" in vi
            assert "staleness_hours" in vi
            assert "evidence_count_observed" in vi
            assert "strength_thresholds" in vi
            assert "verdict_reasons" in vi
            assert isinstance(vi["verdict_reasons"], list)
            assert vi["verdict_reasons"], "verdict_reasons must explain the verdict"

    def test_contributing_events_populated_for_runtime_article(
        self, engine, populated_trail
    ):
        report = engine.assess(populated_trail)
        # Pick an article with sufficient/partial evidence — must list at
        # least one contributing event.
        for art in report.articles:
            if art.evidence_count > 0:
                assert art.contributing_events, (
                    f"{art.requirement.article} has {art.evidence_count} "
                    "records but no contributing_events"
                )
                ev = art.contributing_events[0]
                assert "record_id" in ev
                assert "action_id" in ev
                assert "event_type" in ev
                assert "timestamp_iso" in ev
                assert "narrative" in ev
                assert "drill_down" in ev
                assert isinstance(ev["drill_down"], dict)
                return
        raise AssertionError("expected at least one article with evidence")

    def test_contributing_events_are_most_recent_first(
        self, engine, populated_trail
    ):
        report = engine.assess(populated_trail)
        for art in report.articles:
            events = art.contributing_events
            if len(events) < 2:
                continue
            ages = [
                e["age_hours"] for e in events
                if isinstance(e.get("age_hours"), (int, float))
            ]
            assert ages == sorted(ages), (
                f"{art.requirement.article} contributing_events not "
                "ordered most-recent-first"
            )

    def test_drill_down_filters_to_known_keys(self, engine):
        # Build a trail with a single RISK_SCORED event whose data carries
        # both a known reasoning key (point_estimate) and a free-form key
        # that must NOT leak into the regulator-facing drill_down.
        trail = AuditTrail()
        at = ActionType(
            "x", ActionCategory.DATA, Reversibility.FULLY, BlastRadius.SELF,
        )
        req = ActionRequest(agent_id="a", tool_name="x", action_type=at)
        action_id = trail.record_action_requested(req)
        trail.record_risk_scored(
            action_id=action_id, agent_id="a", tool_name="x",
            assessment={
                "risk": 0.4,
                "point_estimate": 0.42,
                "conformal_lower": 0.3,
                "conformal_upper": 0.55,
                "secret_payload": "should-not-appear",
            },
        )
        custom = RegulatoryRequirement(
            RegulatoryDomain.EU_AI_ACT, "Article 9(1)", "RMS", "...",
            (EventType.RISK_SCORED,), min_evidence_count=1,
        )
        engine = ComplianceEngine(requirements=[custom])
        report = engine.assess(trail)
        ev = report.articles[0].contributing_events[0]
        drill = ev["drill_down"]
        assert drill.get("point_estimate") == 0.42
        assert drill.get("conformal_lower") == 0.3
        assert drill.get("conformal_upper") == 0.55
        assert "secret_payload" not in drill, (
            "free-form data fields must not leak into drill_down"
        )

    def test_empty_trail_verdict_inputs_pin_to_insufficient(
        self, engine, trail
    ):
        report = engine.assess(trail)
        runtime = [
            a for a in report.articles
            if a.requirement.evidence_event_types
        ]
        for art in runtime:
            vi = art.verdict_inputs
            assert vi["evidence_count_observed"] == 0
            reasons = " ".join(vi["verdict_reasons"]).upper()
            assert "INSUFFICIENT" in reasons
            assert art.contributing_events == []

    def test_broken_chain_marks_verdict_inputs(
        self, engine, populated_trail
    ):
        populated_trail._records[5].data["tampered"] = True
        report = engine.assess(populated_trail)
        runtime = [
            a for a in report.articles
            if a.requirement.evidence_event_types
        ]
        for art in runtime:
            vi = art.verdict_inputs
            assert vi.get("chain_intact") is False
            joined = " ".join(vi.get("verdict_reasons", [])).lower()
            assert "chain integrity compromised" in joined

    def test_to_dict_includes_verdict_drill_down(
        self, engine, populated_trail
    ):
        import json
        report = engine.assess(populated_trail)
        d = report.to_dict()
        # JSON round-trip with strict mode — no inf/NaN must leak through.
        text = json.dumps(d, allow_nan=False)
        assert "verdict_inputs" in text
        assert "contributing_events" in text
        # Every article carries the new keys.
        for art in d["articles"]:
            assert "verdict_inputs" in art
            assert "contributing_events" in art


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
