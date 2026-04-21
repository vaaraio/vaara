"""Tests for the adaptive risk scorer."""

import pytest
from vaara.scorer.adaptive import (
    AdaptiveScorer,
    ConformalCalibrator,
    Decision,
    MWUExperts,
    SequencePattern,
    BUILTIN_SEQUENCES,
)


class TestMWUExperts:
    def test_initial_weights_uniform(self):
        mwu = MWUExperts(["a", "b", "c"])
        weights = mwu.weights
        assert len(weights) == 3
        assert abs(weights["a"] - 1/3) < 0.01

    def test_predict_weighted_average(self):
        mwu = MWUExperts(["a", "b"])
        # Uniform weights → simple average
        result = mwu.predict({"a": 0.2, "b": 0.8})
        assert abs(result - 0.5) < 0.01

    def test_update_shifts_weights(self):
        mwu = MWUExperts(["good", "bad"], eta=0.5)
        # "good" predicts 0.8, "bad" predicts 0.2, actual was 0.8
        for _ in range(10):
            mwu.update({"good": 0.8, "bad": 0.2}, outcome=0.8)
        # "good" should have higher weight
        assert mwu.weights["good"] > mwu.weights["bad"]

    def test_predict_clamped_0_1(self):
        mwu = MWUExperts(["x"])
        assert 0.0 <= mwu.predict({"x": -5.0}) <= 1.0
        assert 0.0 <= mwu.predict({"x": 5.0}) <= 1.0

    def test_min_weight_floor(self):
        mwu = MWUExperts(["a", "b"], eta=2.0, min_weight=0.05)
        # Extreme updates
        for _ in range(50):
            mwu.update({"a": 0.0, "b": 1.0}, outcome=0.0)
        # "b" is bad but shouldn't die completely
        assert mwu.weights["b"] >= 0.01  # At least close to floor after normalization


class TestConformalCalibrator:
    def test_not_calibrated_initially(self):
        cc = ConformalCalibrator(min_calibration=30)
        assert not cc.is_calibrated
        assert cc.calibration_size == 0

    def test_conservative_interval_before_calibration(self):
        cc = ConformalCalibrator(min_calibration=30)
        lower, upper = cc.predict_interval(0.5)
        # Before calibration: wide interval
        assert lower < 0.5
        assert upper > 0.5
        assert upper - lower >= 0.5  # At least 0.5 wide

    def test_calibration_reduces_interval(self):
        cc = ConformalCalibrator(min_calibration=10, alpha=0.10)
        # Add tight calibration points (predictions close to actuals)
        for i in range(50):
            predicted = 0.5
            actual = 0.5 + (i % 3 - 1) * 0.02  # ±0.02 noise
            cc.add_calibration_point(predicted, actual)

        assert cc.is_calibrated
        lower, upper = cc.predict_interval(0.5)
        # Should be tighter than the uncalibrated interval
        assert upper - lower < 0.5

    def test_miscoverage_widens_interval(self):
        cc = ConformalCalibrator(min_calibration=10, alpha=0.10, gamma=0.05)
        # Calibrate with tight residuals first
        for _ in range(20):
            cc.add_calibration_point(0.5, 0.5)

        _, upper_tight = cc.predict_interval(0.5)

        # Now add points where predictions are way off
        for _ in range(20):
            cc.add_calibration_point(0.5, 0.9)

        _, upper_wide = cc.predict_interval(0.5)
        assert upper_wide > upper_tight


class TestAdaptiveScorer:
    def test_ms_toolkit_protocol(self):
        scorer = AdaptiveScorer()
        assert scorer.name == "vaara_adaptive"
        result = scorer.evaluate({"tool_name": "data.read", "agent_id": "test"})
        assert "allowed" in result
        assert "action" in result
        assert "reason" in result
        assert "backend" in result
        assert result["backend"] == "vaara_adaptive"

    def test_low_risk_action_allowed(self):
        scorer = AdaptiveScorer(threshold_allow=0.5)
        result = scorer.evaluate({
            "tool_name": "data.read",
            "agent_id": "trusted",
            "base_risk_score": 0.1,
            "agent_confidence": 0.9,
        })
        assert result["allowed"] is True

    def test_high_risk_action_denied(self):
        scorer = AdaptiveScorer(threshold_deny=0.5)
        result = scorer.evaluate({
            "tool_name": "phy.safety_override",
            "agent_id": "unknown",
            "base_risk_score": 0.9,
            "agent_confidence": None,
        })
        # High base risk + unknown agent + no confidence → should be denied or escalated
        assert result["action"] in ("deny", "escalate")

    def test_sequence_detection(self):
        scorer = AdaptiveScorer(threshold_deny=0.8)
        # Build up a data exfiltration pattern
        scorer.evaluate({
            "tool_name": "data.read", "agent_id": "suspect",
            "base_risk_score": 0.1,
        })
        result = scorer.evaluate({
            "tool_name": "data.export", "agent_id": "suspect",
            "base_risk_score": 0.4,
        })
        # Sequence pattern should boost the risk
        raw = result.get("raw_result", {})
        assert raw.get("sequence_risk", 0) > 0

    def test_outcome_learning(self):
        scorer = AdaptiveScorer()
        initial_weights = dict(scorer.mwu_weights)

        # Simulate some actions with outcomes
        signals = {
            "taxonomy_base": 0.5,
            "agent_history": 0.3,
            "sequence_pattern": 0.0,
            "action_frequency": 0.0,
            "confidence_gap": 0.2,
        }
        scorer.record_outcome("agent-1", "test.tool", 0.5, 0.0, signals)
        scorer.record_outcome("agent-1", "test.tool", 0.5, 0.0, signals)

        assert scorer.mwu_update_count == 2
        # Weights should have shifted
        assert scorer.mwu_weights != initial_weights

    def test_status_snapshot(self):
        scorer = AdaptiveScorer()
        status = scorer.status()
        assert "calibrated" in status
        assert "mwu_weights" in status
        assert "thresholds" in status
        assert status["calibrated"] is False  # No data yet

    def test_agent_profile_builds(self):
        scorer = AdaptiveScorer()
        scorer.evaluate({
            "tool_name": "data.read", "agent_id": "agent-x",
            "base_risk_score": 0.1,
        })
        scorer.evaluate({
            "tool_name": "data.write", "agent_id": "agent-x",
            "base_risk_score": 0.3,
        })
        profile = scorer.get_agent_profile("agent-x")
        assert profile is not None
        assert profile.total_actions == 2

    def test_custom_sequence_pattern(self):
        custom = SequencePattern(
            "custom_danger", ("step1", "step2"), risk_boost=0.5, window_size=5,
        )
        scorer = AdaptiveScorer(sequence_patterns=[custom])
        scorer.evaluate({
            "tool_name": "step1", "agent_id": "a", "base_risk_score": 0.1,
        })
        result = scorer.evaluate({
            "tool_name": "step2", "agent_id": "a", "base_risk_score": 0.1,
        })
        raw = result.get("raw_result", {})
        assert raw.get("sequence_risk", 0) > 0

    def test_builtin_sequences_loaded(self):
        assert len(BUILTIN_SEQUENCES) >= 5
        names = {p.name for p in BUILTIN_SEQUENCES}
        assert "data_exfiltration" in names
        assert "financial_drain" in names


class TestLRUAgentCap:
    """Loop 48: unbounded `_agents` dict is a multi-tenant DoS surface.
    100K unique agent_ids × ~37KB AgentProfile ≈ 3.7GB memory. LRU cap
    keeps the profile table bounded and also prunes per-(agent, pattern)
    sequence match state on eviction.
    """

    def test_agent_dict_capped_at_max_tracked(self):
        scorer = AdaptiveScorer(max_tracked_agents=50)
        for i in range(200):
            scorer.evaluate({
                "tool_name": "data.read",
                "agent_id": f"attacker-{i}",
                "base_risk_score": 0.3,
            })
        assert len(scorer._agents) == 50

    def test_lru_evicts_oldest_first(self):
        scorer = AdaptiveScorer(max_tracked_agents=3)
        for aid in ("a", "b", "c"):
            scorer.evaluate({"tool_name": "data.read", "agent_id": aid})
        # Touch "a" to make it most-recent
        scorer.evaluate({"tool_name": "data.read", "agent_id": "a"})
        # Inserting "d" should evict "b" (now oldest), not "a"
        scorer.evaluate({"tool_name": "data.read", "agent_id": "d"})
        assert "a" in scorer._agents
        assert "b" not in scorer._agents
        assert "c" in scorer._agents
        assert "d" in scorer._agents

    def test_seq_match_state_pruned_on_eviction(self):
        scorer = AdaptiveScorer(max_tracked_agents=2)
        # Inject some pattern state for agents that will get evicted
        scorer._seq_match_state[("ghost-1", "data_exfiltration")] = True
        scorer._seq_match_state[("live", "data_exfiltration")] = True
        scorer.evaluate({"tool_name": "data.read", "agent_id": "live"})
        scorer.evaluate({"tool_name": "data.read", "agent_id": "new-1"})
        scorer.evaluate({"tool_name": "data.read", "agent_id": "new-2"})
        # "ghost-1" was never in _agents so its _seq_match_state entry
        # persists; the real test is that live agents' state survives
        # as long as the agent does, and evicted agents' state goes.
        # Re-run with a cleaner setup:
        scorer2 = AdaptiveScorer(max_tracked_agents=1)
        scorer2.evaluate({"tool_name": "data.read", "agent_id": "victim"})
        scorer2._seq_match_state[("victim", "data_exfiltration")] = True
        scorer2.evaluate({"tool_name": "data.read", "agent_id": "survivor"})
        # victim was evicted, its match state must be gone
        assert "victim" not in scorer2._agents
        assert ("victim", "data_exfiltration") not in scorer2._seq_match_state


class TestRecordOutcomeAfterEviction:
    """Loop 49: a late record_outcome for an evicted agent must not
    resurrect the agent (re-create zero-history profile AND evict an
    active one via LRU). MWU/conformal are global and legitimately
    updated; per-agent counters are skipped when the agent is gone.
    """

    def test_outcome_for_evicted_agent_does_not_resurrect(self):
        scorer = AdaptiveScorer(max_tracked_agents=2)
        scorer.evaluate({"tool_name": "data.read", "agent_id": "alice"})
        scorer.evaluate({"tool_name": "data.read", "agent_id": "bob"})
        # carol evicts alice
        scorer.evaluate({"tool_name": "data.read", "agent_id": "carol"})
        assert "alice" not in scorer._agents

        # Late outcome for alice
        scorer.record_outcome(
            agent_id="alice",
            tool_name="data.read",
            predicted_risk=0.2,
            actual_outcome=0.9,
            signals={},
        )
        # alice must NOT be re-inserted
        assert "alice" not in scorer._agents
        # bob and carol stay
        assert "bob" in scorer._agents
        assert "carol" in scorer._agents

    def test_outcome_updates_mwu_and_conformal_for_evicted_agent(self):
        scorer = AdaptiveScorer(max_tracked_agents=1)
        scorer.evaluate({"tool_name": "data.read", "agent_id": "ghost"})
        scorer.evaluate({"tool_name": "data.read", "agent_id": "live"})
        assert "ghost" not in scorer._agents

        before = scorer._conformal.calibration_size
        scorer.record_outcome(
            agent_id="ghost",
            tool_name="data.read",
            predicted_risk=0.3,
            actual_outcome=0.8,
            signals={"taxonomy": 0.3},
        )
        # Global calibration updated even though profile was gone
        assert scorer._conformal.calibration_size == before + 1

    def test_outcome_for_live_agent_moves_to_end(self):
        scorer = AdaptiveScorer(max_tracked_agents=3)
        for aid in ("a", "b", "c"):
            scorer.evaluate({"tool_name": "data.read", "agent_id": aid})
        # a is now oldest. Outcome on a should move it to end.
        scorer.record_outcome(
            agent_id="a", tool_name="data.read",
            predicted_risk=0.2, actual_outcome=0.0, signals={},
        )
        # Inserting "d" should now evict "b", not "a"
        scorer.evaluate({"tool_name": "data.read", "agent_id": "d"})
        assert "a" in scorer._agents
        assert "b" not in scorer._agents
