"""Tests for the MCDropoutGateScorer — wraps a frozen MC dropout NN bundle.

Tests are skipped if the bundle doesn't exist yet (mc_dropout_gate.py hasn't run).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_BUNDLE = Path(
    os.environ.get(
        "VAARA_MC_DROPOUT_BUNDLE",
        str(Path.home() / ".vaara" / "cache" / "mc_dropout_gate_bundle.joblib"),
    )
)


pytestmark = pytest.mark.skipif(
    not _BUNDLE.exists(),
    reason=f"MC dropout gate bundle not present at {_BUNDLE}",
)


@pytest.fixture(scope="module")
def scorer():
    from vaara.scorer import MCDropoutGateScorer

    return MCDropoutGateScorer(bundle_path=str(_BUNDLE))


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
