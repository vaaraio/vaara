"""Tests for the interception pipeline — end-to-end integration."""

import pytest
from vaara.pipeline import InterceptionPipeline
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionType,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)


@pytest.fixture
def pipeline():
    return InterceptionPipeline()


class TestInterceptionPipeline:
    def test_intercept_low_risk(self, pipeline):
        result = pipeline.intercept(
            agent_id="trusted-agent",
            tool_name="data.read",
            parameters={"table": "users"},
            agent_confidence=0.95,
        )
        assert result.action_id  # Non-empty
        assert result.decision in ("allow", "escalate", "deny")
        assert 0.0 <= result.risk_score <= 1.0
        assert result.evaluation_ms > 0

    def test_intercept_high_risk(self, pipeline):
        # Register the tool mapping so it classifies correctly
        pipeline.registry.map_tool("phy.safety_override", "phy.safety_override")
        result = pipeline.intercept(
            agent_id="unknown-agent",
            tool_name="phy.safety_override",
            parameters={"system": "brakes"},
        )
        # Safety override with unknown agent should be denied or escalated
        assert result.decision in ("deny", "escalate")
        # Taxonomy base risk is 0.875, but MWU dilutes across 5 experts.
        # The conformal upper bound drives the decision, not point estimate.
        assert result.risk_interval[1] > 0.3  # Upper bound should be elevated

    def test_audit_trail_populated(self, pipeline):
        result = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.sign",
            parameters={"hash": "0xabc"},
        )
        trail = pipeline.trail.get_action_trail(result.action_id)
        # Should have: requested, scored, decision (at minimum)
        assert len(trail) >= 3

    def test_outcome_feedback_loop(self, pipeline):
        result = pipeline.intercept(
            agent_id="agent-1",
            tool_name="data.read",
            agent_confidence=0.9,
        )
        # Report safe outcome
        pipeline.report_outcome(
            action_id=result.action_id,
            outcome_severity=0.0,
            description="Completed safely",
        )
        # Check outcome recorded in audit
        trail = pipeline.trail.get_action_trail(result.action_id)
        event_types = [r.event_type.value for r in trail]
        assert "outcome_recorded" in event_types

        # Check scorer updated
        assert pipeline.scorer.calibration_size >= 1

    def test_escalation_resolution(self, pipeline):
        result = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 1000000},
        )
        # Resolve it regardless of actual decision
        pipeline.resolve_escalation(
            action_id=result.action_id,
            resolution="allow",
            reviewer="admin@company.com",
            justification="Reviewed and approved",
        )
        trail = pipeline.trail.get_action_trail(result.action_id)
        event_types = [r.event_type.value for r in trail]
        assert "escalation_resolved" in event_types

    def test_multiple_agents(self, pipeline):
        for i in range(5):
            pipeline.intercept(
                agent_id=f"agent-{i}",
                tool_name="data.read",
                agent_confidence=0.8,
            )
        # All agents should have profiles
        for i in range(5):
            profile = pipeline.scorer.get_agent_profile(f"agent-{i}")
            assert profile is not None
            assert profile.total_actions == 1

    def test_sequence_across_actions(self, pipeline):
        # Build a data exfiltration sequence
        pipeline.intercept(
            agent_id="suspect",
            tool_name="data.read",
            parameters={"table": "sensitive"},
        )
        result = pipeline.intercept(
            agent_id="suspect",
            tool_name="data.export",
            parameters={"destination": "external"},
        )
        # Second action should have elevated sequence risk
        assert result.signals.get("sequence_pattern", 0) > 0

    def test_compliance_assessment(self, pipeline):
        # Generate some activity
        for i in range(15):
            result = pipeline.intercept(
                agent_id="agent-1",
                tool_name="data.read" if i % 2 == 0 else "tx.transfer",
                agent_confidence=0.7,
            )
            pipeline.report_outcome(result.action_id, 0.0)

        report = pipeline.run_compliance_assessment()
        assert report.trail_chain_intact
        assert report.summary  # Non-empty

    def test_status_snapshot(self, pipeline):
        status = pipeline.status()
        assert "scorer" in status
        assert "trail_size" in status
        assert "trail_chain_intact" in status
        assert status["trail_size"] == 0  # Empty initially

    def test_unknown_tool_handled(self, pipeline):
        result = pipeline.intercept(
            agent_id="agent",
            tool_name="completely.unknown.tool",
        )
        # Should still work — classified as unknown
        assert result.action_id
        assert result.action_type.name == "unknown"

    def test_custom_registry_integration(self, pipeline):
        # Register a custom action type
        custom = ActionType(
            "defi.flashloan",
            ActionCategory.FINANCIAL,
            Reversibility.IRREVERSIBLE,
            BlastRadius.GLOBAL,
            UrgencyClass.IRREVOCABLE,
            frozenset({RegulatoryDomain.DORA}),
            "Execute flash loan",
        )
        pipeline.registry.register(custom)
        pipeline.registry.map_tool("defi.flashloan", "defi.flashloan")

        result = pipeline.intercept(
            agent_id="defi-bot",
            tool_name="defi.flashloan",
            parameters={"amount": 1000000, "protocol": "aave"},
        )
        assert result.action_type.name == "defi.flashloan"
        assert result.action_type.category == ActionCategory.FINANCIAL


class TestIngressLengthCaps:
    """Loop 47: caller-supplied strings and dicts at the pipeline boundary
    must be capped so a 50MB tool_name / params blob cannot poison the
    hash chain, audit export, and compliance report (4x amplification
    across action_requested, risk_scored, decision_made, escalation_sent).
    """

    def test_oversized_tool_name_is_capped(self, pipeline):
        huge = "A" * (10 * 1024 * 1024)
        result = pipeline.intercept(
            agent_id="attacker", tool_name=huge, parameters={},
        )
        records = pipeline.trail.get_action_trail(result.action_id)
        # Every record's tool_name must be bounded
        for r in records:
            assert len(r.tool_name) <= 1024, (
                f"tool_name should be capped; got {len(r.tool_name)}"
            )

    def test_oversized_agent_id_is_capped(self, pipeline):
        huge = "A" * (1024 * 1024)
        result = pipeline.intercept(
            agent_id=huge, tool_name="tx.transfer", parameters={},
        )
        records = pipeline.trail.get_action_trail(result.action_id)
        for r in records:
            assert len(r.agent_id) <= 512

    def test_oversized_parameters_replaced_with_marker(self, pipeline):
        huge_value = "A" * (5 * 1024 * 1024)
        result = pipeline.intercept(
            agent_id="a", tool_name="tx.transfer",
            parameters={"blob": huge_value, "amount": 100},
        )
        records = pipeline.trail.get_action_trail(result.action_id)
        # action_requested record stores parameters dict
        req_rec = records[0]
        params = req_rec.data.get("parameters", {})
        assert params.get("_truncated") is True
        assert params.get("_original_bytes") > 5 * 1024 * 1024


class TestPostDecisionIngressLengthCaps:
    """Loop 50: resolve_escalation, report_outcome, and record_policy_override
    are public post-decision ingress points that also land on the hash
    chain. Same Loop 47 amplification risk, different API surface.
    """

    def test_resolve_escalation_caps_justification(self, pipeline):
        # Force an escalation first by triggering a risky action that the
        # default thresholds will escalate.
        result = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 1_000_000},
        )
        huge = "A" * (2 * 1024 * 1024)
        pipeline.resolve_escalation(
            action_id=result.action_id,
            resolution="allow",
            reviewer="admin",
            justification=huge,
        )
        trail = pipeline.trail.get_action_trail(result.action_id)
        resolved = [r for r in trail if r.event_type.value == "escalation_resolved"]
        if not resolved:
            import pytest
            pytest.skip("action was not escalated")
        just = resolved[0].data.get("justification", "")
        assert len(just) <= 10_000

    def test_resolve_escalation_caps_reviewer(self, pipeline):
        result = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 1_000_000},
        )
        huge = "R" * (1024 * 1024)
        pipeline.resolve_escalation(
            action_id=result.action_id,
            resolution="deny",
            reviewer=huge,
            justification="ok",
        )
        trail = pipeline.trail.get_action_trail(result.action_id)
        resolved = [r for r in trail if r.event_type.value == "escalation_resolved"]
        if not resolved:
            import pytest
            pytest.skip("action was not escalated")
        assert len(resolved[0].data.get("reviewer", "")) <= 512

    def test_report_outcome_caps_description(self, pipeline):
        result = pipeline.intercept(
            agent_id="agent-1",
            tool_name="data.read",
            agent_confidence=0.9,
        )
        huge = "D" * (5 * 1024 * 1024)
        pipeline.report_outcome(
            action_id=result.action_id,
            outcome_severity=0.0,
            description=huge,
        )
        trail = pipeline.trail.get_action_trail(result.action_id)
        outcomes = [r for r in trail if r.event_type.value == "outcome_recorded"]
        assert outcomes, "outcome_recorded missing"
        assert len(outcomes[0].data.get("description", "")) <= 8192

    def test_policy_override_caps_reason_and_overrider(self, pipeline):
        # Seed a record first so the override has a parent action
        result = pipeline.intercept(
            agent_id="agent-1",
            tool_name="data.read",
            agent_confidence=0.9,
        )
        huge = "X" * (10 * 1024 * 1024)
        pipeline.trail.record_policy_override(
            action_id=result.action_id,
            agent_id="agent-1",
            tool_name="data.read",
            override_reason=huge,
            overrider=huge,
            original_decision="escalate",
            new_decision="allow",
        )
        trail = pipeline.trail.get_action_trail(result.action_id)
        overrides = [r for r in trail if r.event_type.value == "policy_override"]
        assert overrides
        d = overrides[0].data
        assert len(d["override_reason"]) <= 10_000
        assert len(d["overrider"]) <= 512


class TestScorerReasonCap:
    """Loop 52: scorer-supplied `reason` is a third ingress class (after
    intercept args and post-decision args) — a plugin scorer returning a
    huge reason, or an exception with a huge str(), must not balloon the
    audit record.
    """

    def test_custom_scorer_huge_reason_is_capped(self, pipeline):
        class HugeReasonScorer:
            name = "huge"

            def evaluate(self, context):
                return {
                    "action": "allow",
                    "reason": "R" * (5 * 1024 * 1024),
                    "point_estimate": 0.1,
                    "conformal_interval": [0.0, 0.2],
                    "signals": {},
                }

            def record_outcome(self, **kwargs):
                pass

            def status(self):
                return {}

        pipeline.scorer = HugeReasonScorer()
        result = pipeline.intercept(
            agent_id="agent", tool_name="data.read",
        )
        trail = pipeline.trail.get_action_trail(result.action_id)
        decision = [
            r for r in trail if r.event_type.value in ("decision_made", "action_blocked")
        ][0]
        assert len(decision.data["reason"]) <= 8192

    def test_scorer_exception_huge_message_is_capped(self, pipeline):
        class ExplodingScorer:
            name = "boom"

            def evaluate(self, context):
                raise RuntimeError("B" * (3 * 1024 * 1024))

            def record_outcome(self, **kwargs):
                pass

            def status(self):
                return {}

        pipeline.scorer = ExplodingScorer()
        try:
            pipeline.intercept(agent_id="agent", tool_name="data.read")
        except RuntimeError:
            pass
        # A fail-closed deny record should exist with a capped reason
        trail_all = pipeline.trail._records
        # Find the most recent action_blocked
        blocked = [
            r for r in trail_all if r.event_type.value == "action_blocked"
        ]
        assert blocked
        assert len(blocked[-1].data["reason"]) <= 8192
