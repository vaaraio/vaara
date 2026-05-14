"""Tests for vaara.policy.test_cases — evaluator + Conftest-analog test runner."""

from __future__ import annotations

import pytest

from vaara.policy import SCHEMA_VERSION, from_dict
from vaara.policy.test_cases import (
    EvaluationResult,
    PolicyTestCase,
    evaluate,
    run_test_cases,
)


def _policy() -> dict:
    return {
        "version": SCHEMA_VERSION,
        "domains": ["eu_ai_act", "dora"],
        "action_classes": {
            "fs.write_file": {
                "category": "data", "reversibility": "partially_reversible",
                "blast_radius": "local", "urgency": "timely",
                "regulatory": ["aiact:9", "aiact:12"],
            },
            "tx.sign": {
                "category": "financial", "reversibility": "irreversible",
                "blast_radius": "shared", "urgency": "irrevocable",
                "regulatory": ["aiact:14", "dora:10"],
            },
        },
        "thresholds": {
            "default": {"escalate": 0.55, "deny": 0.85},
            "fs.write_file": {"deny": 0.75},
            "tx.sign": {"escalate": 0.40, "deny": 0.65},
        },
        "sequences": {
            "config_then_signal": {
                "pattern": ["config.write", "tx.sign"],
                "risk_boost": 0.3, "window_seconds": 60,
                "regulatory": ["aiact:14"],
            },
        },
        "escalation": {
            "routes": [
                {"if": ["aiact:14"], "operator_group": "ai_oversight_team"},
                {"if": ["dora:10"], "operator_group": "ict_risk_team"},
                {"default": "on_call"},
            ],
        },
    }


class TestEvaluate:
    def test_allow_below_escalate(self) -> None:
        p = from_dict(_policy())
        r = evaluate(p, "fs.write_file", 0.30)
        assert r == EvaluationResult(verdict="allow", boosted_risk=0.30, route=None)

    def test_escalate_at_threshold(self) -> None:
        p = from_dict(_policy())
        r = evaluate(p, "fs.write_file", 0.55)
        assert r.verdict == "escalate"
        assert r.route == "on_call"

    def test_deny_via_per_class_override(self) -> None:
        p = from_dict(_policy())
        r = evaluate(p, "fs.write_file", 0.80)
        assert r.verdict == "deny"

    def test_tx_sign_escalate_routes_aiact14(self) -> None:
        p = from_dict(_policy())
        r = evaluate(p, "tx.sign", 0.50)
        assert r.verdict == "escalate"
        assert r.route == "ai_oversight_team"

    def test_sequence_boost_lifts_to_deny(self) -> None:
        p = from_dict(_policy())
        r = evaluate(p, "tx.sign", 0.40, ["config_then_signal"])
        assert r.verdict == "deny"
        assert r.boosted_risk == pytest.approx(0.70)

    def test_unknown_action_class_raises(self) -> None:
        p = from_dict(_policy())
        with pytest.raises(ValueError, match="not declared"):
            evaluate(p, "nope", 0.5)

    def test_unknown_sequence_raises(self) -> None:
        p = from_dict(_policy())
        with pytest.raises(ValueError, match="unknown pattern"):
            evaluate(p, "tx.sign", 0.4, ["ghost_pattern"])

    def test_out_of_range_risk_raises(self) -> None:
        p = from_dict(_policy())
        with pytest.raises(ValueError, match="risk_score"):
            evaluate(p, "tx.sign", 1.5)

    def test_boosted_risk_caps_at_one(self) -> None:
        p = from_dict(_policy())
        r = evaluate(p, "tx.sign", 0.90, ["config_then_signal"])
        assert r.boosted_risk == 1.0
        assert r.verdict == "deny"


class TestRunCases:
    def test_all_pass(self) -> None:
        p = from_dict(_policy())
        cases = [
            PolicyTestCase("low", "fs.write_file", 0.30, expected_verdict="allow"),
            PolicyTestCase(
                "high", "tx.sign", 0.50,
                expected_verdict="escalate", expected_route="ai_oversight_team",
            ),
        ]
        results = run_test_cases(p, cases)
        assert all(r.passed for r in results)
        assert results[0].diagnostic == "ok"

    def test_verdict_mismatch_fails(self) -> None:
        p = from_dict(_policy())
        results = run_test_cases(p, [
            PolicyTestCase("wrong", "fs.write_file", 0.30, expected_verdict="deny"),
        ])
        assert not results[0].passed
        assert "verdict" in results[0].diagnostic

    def test_route_mismatch_fails(self) -> None:
        p = from_dict(_policy())
        results = run_test_cases(p, [
            PolicyTestCase(
                "wrong_route", "tx.sign", 0.50,
                expected_verdict="escalate", expected_route="ghost_team",
            ),
        ])
        assert not results[0].passed
        assert "route" in results[0].diagnostic

    def test_evaluation_error_is_captured_not_raised(self) -> None:
        p = from_dict(_policy())
        results = run_test_cases(p, [
            PolicyTestCase("bad_class", "ghost.tool", 0.5),
        ])
        assert not results[0].passed
        assert results[0].actual is None
        assert "evaluation error" in results[0].diagnostic


class TestCaseValidation:
    def test_bad_verdict_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="expected_verdict"):
            PolicyTestCase("x", "tx.sign", 0.5, expected_verdict="banana")

    def test_route_without_escalate_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected_route"):
            PolicyTestCase(
                "x", "tx.sign", 0.5,
                expected_verdict="allow", expected_route="team",
            )
