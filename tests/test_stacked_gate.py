"""Tests for StackedGateScorer — composes GBM + MC dropout via trained LR.

The bundle comes from the ``stacked_gate_bundle`` fixture, which writes an LR
stack over the other two fixture bundles (see tests/gate_bundle_factory.py).
Set VAARA_STACKED_GATE_BUNDLE to score a production bundle instead.
"""

from __future__ import annotations

import math

import pytest

from tests.gate_bundle_factory import BENIGN_CONTEXT, RISKY_CONTEXT

# The stack loads both backends, so it needs both dependency sets.
pytest.importorskip("sklearn", reason="stacked gate needs the ml extra")
pytest.importorskip("joblib", reason="stacked gate needs the ml extra")
pytest.importorskip("torch", reason="stacked gate needs torch")


@pytest.fixture(scope="module")
def scorer(stacked_gate_bundle):
    from vaara.scorer import StackedGateScorer

    return StackedGateScorer(bundle_path=str(stacked_gate_bundle))


class TestStackedGateScorer:
    def test_import_and_construct(self, scorer):
        assert scorer.name == "vaara_stacked_gate"
        assert 0.0 <= scorer.q_hat <= 1.0

    def test_evaluate_shape(self, scorer):
        context = {
            "tool_name": "edit_file",
            "agent_id": "test",
            "history_pairs": [
                ("I'll open the file.", "file contents"),
                ("```\nls\n```", "README.md setup.py"),
            ],
            "proposed_action_text": "```\nedit file.py\n```",
        }
        result = scorer.evaluate(context)
        assert result["backend"] == "vaara_stacked_gate"
        raw = result["raw_result"]
        assert 0.0 <= raw["point_estimate"] <= 1.0
        assert 0.0 <= raw["gbm_p"] <= 1.0
        assert 0.0 <= raw["mc_p"] <= 1.0
        assert raw["backend_disagree"] >= 0.0
        assert raw["verdict"] in {"execute", "block", "flag_review"}

    def test_verdict_decision_mapping(self, scorer):
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
        context = {
            "tool_name": "list_dir",
            "agent_id": "new",
            "history_pairs": [],
            "proposed_action_text": "```\nls /\n```",
        }
        result = scorer.evaluate(context)
        assert result["action"] in {"allow", "deny", "escalate"}

    def test_point_estimate_is_the_logistic_stack_of_the_two_backends(self, scorer):
        # The whole reason this backend exists is the combination step. Recompute
        # it from the two component estimates the result already reports: a stack
        # that quietly returned one backend's number would pass every other test
        # in this file.
        raw = scorer.evaluate(RISKY_CONTEXT)["raw_result"]
        z = (
            scorer._w_gbm * raw["gbm_p"]
            + scorer._w_mc * raw["mc_p"]
            + scorer._intercept
        )
        assert raw["point_estimate"] == pytest.approx(1.0 / (1.0 + math.exp(-z)))

    def test_backend_disagreement_is_the_absolute_gap(self, scorer):
        raw = scorer.evaluate(BENIGN_CONTEXT)["raw_result"]
        assert raw["backend_disagree"] == pytest.approx(abs(raw["gbm_p"] - raw["mc_p"]))

    def test_a_benign_read_and_a_risky_write_land_on_opposite_verdicts(self, scorer):
        benign = scorer.evaluate(BENIGN_CONTEXT)
        risky = scorer.evaluate(RISKY_CONTEXT)
        assert benign["raw_result"]["point_estimate"] < risky["raw_result"]["point_estimate"]
        assert benign["action"] == "allow"
        assert risky["action"] == "deny"

    def test_conformal_interval_is_q_hat_wide_and_clipped(self, scorer):
        raw = scorer.evaluate(RISKY_CONTEXT)["raw_result"]
        lower, upper = raw["conformal_interval"]
        point = raw["point_estimate"]
        assert lower == pytest.approx(max(0.0, point - scorer.q_hat))
        assert upper == pytest.approx(min(1.0, point + scorer.q_hat))


def test_missing_bundle_raises_with_the_path_in_the_message(tmp_path):
    from vaara.scorer import StackedGateScorer

    missing = tmp_path / "not_here.joblib"
    with pytest.raises(FileNotFoundError) as excinfo:
        StackedGateScorer(bundle_path=str(missing))
    assert str(missing) in str(excinfo.value)


def test_a_missing_component_bundle_fails_loudly(tmp_path, mc_dropout_gate_bundle):
    # The stack holds paths to two other files. If one of them has moved, the
    # stack must refuse to load rather than come up on one backend.
    import joblib

    from vaara.scorer import StackedGateScorer

    broken = tmp_path / "stacked.joblib"
    joblib.dump(
        {
            "variant": "stacked",
            "stack_coef": [4.0, 4.0],
            "stack_intercept": -4.0,
            "q_hat": 0.35,
            "gbm_bundle_path": str(tmp_path / "gone.joblib"),
            "mc_bundle_path": str(mc_dropout_gate_bundle),
        },
        str(broken),
    )
    with pytest.raises(FileNotFoundError):
        StackedGateScorer(bundle_path=str(broken))


def test_factory_builds_the_same_backend(stacked_gate_bundle):
    from vaara.scorer import create_stacked_scorer

    scorer = create_stacked_scorer(bundle_path=str(stacked_gate_bundle))
    assert scorer.name == "vaara_stacked_gate"
