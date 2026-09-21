"""Tests for the TrainedGateScorer — wraps a frozen ActionGate bundle.

The bundle comes from the ``trained_gate_bundle`` fixture, which builds one in
the real format (see tests/gate_bundle_factory.py). Set
VAARA_TRAINED_GATE_BUNDLE to score a production bundle instead.
"""

from __future__ import annotations

import pytest

from tests.gate_bundle_factory import BENIGN_CONTEXT, RISKY_CONTEXT

# The backend is only importable with the ml extra installed; without it
# vaara.scorer does not export TrainedGateScorer at all.
pytest.importorskip("sklearn", reason="trained gate needs the ml extra")
pytest.importorskip("joblib", reason="trained gate needs the ml extra")


@pytest.fixture(scope="module")
def scorer(trained_gate_bundle):
    from vaara.scorer import TrainedGateScorer

    return TrainedGateScorer(bundle_path=str(trained_gate_bundle))


class TestTrainedGateScorer:
    def test_import_and_construct(self, scorer):
        assert scorer.name == "vaara_trained_gate"
        assert scorer.ensemble_size > 0
        assert 0.0 <= scorer.q_hat <= 1.0

    def test_evaluate_shape(self, scorer):
        context = {
            "tool_name": "edit_file",
            "agent_id": "test-agent",
            "history_pairs": [
                ("I'll open the file.", "file contents here"),
                ("```\nls\n```", "README.md setup.py"),
            ],
            "proposed_action_text": "```\nedit file.py\n+ return 42\n```",
        }
        result = scorer.evaluate(context)
        # Backend decision dict shape
        assert "allowed" in result
        assert "action" in result
        assert result["action"] in {"allow", "deny", "escalate"}
        assert "reason" in result
        assert "backend" in result
        assert result["backend"] == "vaara_trained_gate"
        assert "evaluation_ms" in result
        raw = result["raw_result"]
        assert "point_estimate" in raw
        assert "conformal_interval" in raw
        assert 0.0 <= raw["point_estimate"] <= 1.0
        lower, upper = raw["conformal_interval"]
        assert 0.0 <= lower <= upper <= 1.0
        assert "verdict" in raw
        assert raw["verdict"] in {"execute", "block", "flag_review"}
        assert "inside_ok_set" in raw
        assert "inside_error_set" in raw
        assert "top_risk_factor" in raw

    def test_verdict_decision_mapping(self, scorer):
        # Not a behavioral test — just that decision field is consistent
        # with verdict after evaluate().
        context = {
            "tool_name": "read_file",
            "agent_id": "t2",
            "history_pairs": [],
            "proposed_action_text": "```\ncat README.md\n```",
        }
        result = scorer.evaluate(context)
        verdict = result["raw_result"]["verdict"]
        action = result["action"]
        if verdict == "execute":
            assert action == "allow"
        elif verdict == "block":
            assert action == "deny"
        else:
            assert action == "escalate"

    def test_empty_history(self, scorer):
        # Fresh agent with no history — should still produce a valid decision.
        context = {
            "tool_name": "list_dir",
            "agent_id": "new",
            "history_pairs": [],
            "proposed_action_text": "```\nls /\n```",
        }
        result = scorer.evaluate(context)
        assert result["action"] in {"allow", "deny", "escalate"}

    def test_a_benign_read_and_a_risky_write_land_on_opposite_verdicts(self, scorer):
        # Shape assertions pass on a scorer that returns a constant. This one
        # does not: the two contexts have to separate, and in the right
        # direction, or the wrapper is not carrying the model's output through.
        benign = scorer.evaluate(BENIGN_CONTEXT)
        risky = scorer.evaluate(RISKY_CONTEXT)
        assert benign["raw_result"]["point_estimate"] < risky["raw_result"]["point_estimate"]
        assert benign["raw_result"]["verdict"] == "execute"
        assert benign["action"] == "allow"
        assert risky["raw_result"]["verdict"] == "block"
        assert risky["action"] == "deny"

    def test_conformal_interval_is_q_hat_wide_and_clipped(self, scorer):
        result = scorer.evaluate(RISKY_CONTEXT)
        raw = result["raw_result"]
        lower, upper = raw["conformal_interval"]
        point = raw["point_estimate"]
        assert lower == pytest.approx(max(0.0, point - scorer.q_hat))
        assert upper == pytest.approx(min(1.0, point + scorer.q_hat))

    def test_risk_attribution_names_the_error_history(self, scorer):
        # RISKY_CONTEXT's history is three failing steps, so hist_error_rate
        # clears the 0.3 branch in ActionGate.evaluate ahead of the others.
        raw = scorer.evaluate(RISKY_CONTEXT)["raw_result"]
        assert "error rate" in raw["top_risk_factor"]

    def test_bundle_path_is_reported_back(self, scorer, trained_gate_bundle):
        raw = scorer.evaluate(BENIGN_CONTEXT)["raw_result"]
        assert raw["bundle"] == str(trained_gate_bundle)


def test_missing_bundle_raises_with_the_path_in_the_message(tmp_path):
    from vaara.scorer import TrainedGateScorer

    missing = tmp_path / "not_here.joblib"
    with pytest.raises(FileNotFoundError) as excinfo:
        TrainedGateScorer(bundle_path=str(missing))
    assert str(missing) in str(excinfo.value)


def test_factory_builds_the_same_backend(trained_gate_bundle):
    from vaara.scorer import create_trained_gate_scorer

    scorer = create_trained_gate_scorer(bundle_path=str(trained_gate_bundle))
    assert scorer.name == "vaara_trained_gate"
    assert scorer.ensemble_size > 0
