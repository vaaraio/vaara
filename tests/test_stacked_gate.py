"""Tests for StackedGateScorer — composes GBM + MC dropout via trained LR."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_BUNDLE = Path(
    os.environ.get(
        "VAARA_STACKED_GATE_BUNDLE",
        str(Path.home() / ".vaara" / "cache" / "stacked_gate_bundle.joblib"),
    )
)


pytestmark = pytest.mark.skipif(
    not _BUNDLE.exists(),
    reason=f"Stacked gate bundle not present at {_BUNDLE}",
)


@pytest.fixture(scope="module")
def scorer():
    from vaara.scorer import StackedGateScorer

    return StackedGateScorer(bundle_path=str(_BUNDLE))


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
