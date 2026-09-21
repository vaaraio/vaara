"""Tests for the MCDropoutGateScorer — wraps a frozen MC dropout NN bundle.

The bundle comes from the ``mc_dropout_gate_bundle`` fixture, which trains the
shipped architecture on synthetic steps (see tests/gate_bundle_factory.py).
Set VAARA_MC_DROPOUT_BUNDLE to score a production bundle instead.
"""

from __future__ import annotations

import pytest

from tests.gate_bundle_factory import BENIGN_CONTEXT, RISKY_CONTEXT

# torch is not declared in requirements-dev.txt and is not in the ml extra
# either, so the backend is unimportable on a default install.
pytest.importorskip("torch", reason="mc dropout gate needs torch")


@pytest.fixture(scope="module")
def scorer(mc_dropout_gate_bundle):
    from vaara.scorer import MCDropoutGateScorer

    return MCDropoutGateScorer(bundle_path=str(mc_dropout_gate_bundle))


class TestMCDropoutGateScorer:
    def test_import_and_construct(self, scorer):
        assert scorer.name == "vaara_mc_dropout_gate"
        assert 0.0 <= scorer.q_hat <= 1.0
        assert scorer.variant in {"behavioral", "combined"}

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
        assert "allowed" in result
        assert "action" in result
        assert result["action"] in {"allow", "deny", "escalate"}
        assert "reason" in result
        assert "backend" in result
        assert result["backend"] == "vaara_mc_dropout_gate"
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
        assert "mc_samples" in raw
        assert "mc_std" in raw
        assert raw["mc_samples"] >= 1
        assert raw["mc_std"] >= 0.0

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

    def test_a_benign_read_and_a_risky_write_separate(self, scorer):
        import torch

        # Dropout stays on at inference, so fix the seed: the point of this
        # test is the direction of the gap, not a particular sample.
        torch.manual_seed(0)
        benign = scorer.evaluate(BENIGN_CONTEXT)["raw_result"]["point_estimate"]
        torch.manual_seed(0)
        risky = scorer.evaluate(RISKY_CONTEXT)["raw_result"]["point_estimate"]
        assert benign < risky

    def test_mc_sampling_is_stochastic_across_calls(self, scorer):
        # _model.train() is deliberately left on so dropout masks resample.
        # If a future change flips it to eval() the UQ signal dies silently,
        # and mc_std pinned at exactly zero is what that looks like.
        spread = {
            scorer.evaluate(RISKY_CONTEXT)["raw_result"]["point_estimate"]
            for _ in range(5)
        }
        assert len(spread) > 1

    def test_conformal_interval_is_q_hat_wide_and_clipped(self, scorer):
        raw = scorer.evaluate(BENIGN_CONTEXT)["raw_result"]
        lower, upper = raw["conformal_interval"]
        point = raw["point_estimate"]
        assert lower == pytest.approx(max(0.0, point - scorer.q_hat))
        assert upper == pytest.approx(min(1.0, point + scorer.q_hat))

    def test_variant_and_bundle_path_come_from_the_file(self, scorer, mc_dropout_gate_bundle):
        raw = scorer.evaluate(BENIGN_CONTEXT)["raw_result"]
        assert raw["variant"] == scorer.variant
        assert raw["bundle"] == str(mc_dropout_gate_bundle)


def test_missing_bundle_raises_with_the_path_in_the_message(tmp_path):
    from vaara.scorer import MCDropoutGateScorer

    missing = tmp_path / "not_here.joblib"
    with pytest.raises(FileNotFoundError) as excinfo:
        MCDropoutGateScorer(bundle_path=str(missing))
    assert str(missing) in str(excinfo.value)


def test_non_finite_samples_fall_back_to_a_half(scorer, monkeypatch):
    import numpy as np

    # Corrupt weights make the net emit NaN. The guard in evaluate() turns
    # that into 0.5 and a warning rather than an interval of NaNs that every
    # comparison downstream reads as False.
    monkeypatch.setattr(
        scorer, "_mc_sample", lambda feat: np.full(4, np.nan, dtype=np.float32)
    )
    raw = scorer.evaluate(BENIGN_CONTEXT)["raw_result"]
    assert raw["point_estimate"] == 0.5
    assert raw["mc_std"] == 0.5


def test_factory_builds_the_same_backend(mc_dropout_gate_bundle):
    from vaara.scorer import create_mc_dropout_scorer

    scorer = create_mc_dropout_scorer(bundle_path=str(mc_dropout_gate_bundle))
    assert scorer.name == "vaara_mc_dropout_gate"
