"""Tests for the empirical-coverage additions in scripts/eval_adversarial.py.

The eval script lives outside the package so we load it via importlib and
exercise summarize() / overall() / _actual_risk_from_expected() / _interval_from_result()
against a hand-computed synthetic corpus.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_adversarial.py"


@pytest.fixture(scope="module")
def eval_mod():
    spec = importlib.util.spec_from_file_location("eval_adversarial", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_rows():
    """Five-row corpus with hand-computed expected aggregates.

    Cat A (3 rows):
        coverage: rows 1 and 3 covered, row 2 not (actual_risk=1.0, upper=0.5)
            to 2/3 = 0.6667
        mean_interval_width: (0.6 + 0.3 + 0.3) / 3 = 0.4
        allow_leakage: row 2 only (actual=ALLOW, expected DENY)
        mean_risk: (0.7 + 0.3 + 0.1) / 3 = 0.3667
    Cat B (2 rows):
        coverage: row 5 covered, row 4 not (actual_risk=1.0, upper=0.9)
            to 1/2 = 0.5
        mean_interval_width: (0.8 + 1.0) / 2 = 0.9
        allow_leakage: 0
        mean_risk: (0.5 + 0.0) / 2 = 0.25
    Overall:
        coverage: 3/5 = 0.6
        mean_interval_width: (0.6 + 0.3 + 0.3 + 0.8 + 1.0) / 5 = 0.6
    """
    return [
        {
            "id": "a1", "category": "A", "tool_name": "t",
            "expected_set": {"DENY"}, "actual": "DENY", "risk": 0.7,
            "lower": 0.4, "upper": 1.0, "actual_risk": 1.0,
        },
        {
            "id": "a2", "category": "A", "tool_name": "t",
            "expected_set": {"DENY"}, "actual": "ALLOW", "risk": 0.3,
            "lower": 0.2, "upper": 0.5, "actual_risk": 1.0,
        },
        {
            "id": "a3", "category": "A", "tool_name": "t",
            "expected_set": {"ALLOW"}, "actual": "ALLOW", "risk": 0.1,
            "lower": 0.0, "upper": 0.3, "actual_risk": 0.0,
        },
        {
            "id": "b1", "category": "B", "tool_name": "t",
            "expected_set": {"ESCALATE"}, "actual": "ESCALATE", "risk": 0.5,
            "lower": 0.1, "upper": 0.9, "actual_risk": 1.0,
        },
        {
            "id": "b2", "category": "B", "tool_name": "t",
            "expected_set": {"ALLOW"}, "actual": "ALLOW", "risk": 0.0,
            "lower": 0.0, "upper": 1.0, "actual_risk": 0.0,
        },
    ]


class TestActualRiskFromExpected:
    def test_allow_only_is_zero(self, eval_mod):
        assert eval_mod._actual_risk_from_expected({"ALLOW"}) == 0.0

    def test_deny_is_one(self, eval_mod):
        assert eval_mod._actual_risk_from_expected({"DENY"}) == 1.0

    def test_escalate_is_one(self, eval_mod):
        assert eval_mod._actual_risk_from_expected({"ESCALATE"}) == 1.0

    def test_deny_or_escalate_is_one(self, eval_mod):
        assert eval_mod._actual_risk_from_expected({"DENY", "ESCALATE"}) == 1.0

    def test_empty_set_is_zero(self, eval_mod):
        assert eval_mod._actual_risk_from_expected(set()) == 0.0


class TestIntervalFromResult:
    def test_missing_attribute_yields_trivial(self, eval_mod):
        class Bare:
            pass
        assert eval_mod._interval_from_result(Bare()) == (0.0, 1.0)

    def test_none_interval_yields_trivial(self, eval_mod):
        class R:
            risk_interval = None
        assert eval_mod._interval_from_result(R()) == (0.0, 1.0)

    def test_well_formed_interval(self, eval_mod):
        class R:
            risk_interval = (0.2, 0.7)
        assert eval_mod._interval_from_result(R()) == (0.2, 0.7)

    def test_swapped_bounds_get_reordered(self, eval_mod):
        class R:
            risk_interval = (0.8, 0.3)
        assert eval_mod._interval_from_result(R()) == (0.3, 0.8)

    def test_out_of_range_clamps_to_unit(self, eval_mod):
        class R:
            risk_interval = (-0.5, 1.5)
        assert eval_mod._interval_from_result(R()) == (0.0, 1.0)

    def test_non_numeric_falls_back(self, eval_mod):
        class R:
            risk_interval = ("low", "high")
        assert eval_mod._interval_from_result(R()) == (0.0, 1.0)

    def test_short_tuple_falls_back(self, eval_mod):
        class R:
            risk_interval = (0.3,)
        assert eval_mod._interval_from_result(R()) == (0.0, 1.0)


class TestSummarize:
    def test_per_category_aggregates(self, eval_mod, synthetic_rows):
        summary = eval_mod.summarize(synthetic_rows)
        assert set(summary.keys()) == {"A", "B"}

        a = summary["A"]
        assert a["n"] == 3
        assert a["denied"] == 1
        assert a["escalated"] == 0
        assert a["allowed"] == 2
        assert a["allow_leakage"] == 1
        assert math.isclose(a["allow_leakage_rate"], 0.3333, abs_tol=1e-4)
        assert math.isclose(a["mean_risk"], 0.3667, abs_tol=1e-4)
        assert math.isclose(a["coverage"], 0.6667, abs_tol=1e-4)
        assert math.isclose(a["mean_interval_width"], 0.4, abs_tol=1e-4)

        b = summary["B"]
        assert b["n"] == 2
        assert b["denied"] == 0
        assert b["escalated"] == 1
        assert b["allowed"] == 1
        assert b["allow_leakage"] == 0
        assert math.isclose(b["mean_risk"], 0.25, abs_tol=1e-4)
        assert math.isclose(b["coverage"], 0.5, abs_tol=1e-4)
        assert math.isclose(b["mean_interval_width"], 0.9, abs_tol=1e-4)

    def test_empty_rows_yields_empty_summary(self, eval_mod):
        assert eval_mod.summarize([]) == {}


class TestOverall:
    def test_aggregates_across_categories(self, eval_mod, synthetic_rows):
        block = eval_mod.overall(synthetic_rows)
        assert block["n"] == 5
        assert math.isclose(block["coverage"], 0.6, abs_tol=1e-4)
        assert math.isclose(block["mean_interval_width"], 0.6, abs_tol=1e-4)

    def test_empty_rows_yields_zero_block(self, eval_mod):
        block = eval_mod.overall([])
        assert block == {"n": 0, "coverage": 0.0, "mean_interval_width": 0.0}

    def test_boundary_inclusive(self, eval_mod):
        rows = [
            {
                "id": "x", "category": "C", "tool_name": "t",
                "expected_set": {"DENY"}, "actual": "DENY", "risk": 0.5,
                "lower": 1.0, "upper": 1.0, "actual_risk": 1.0,
            },
            {
                "id": "y", "category": "C", "tool_name": "t",
                "expected_set": {"ALLOW"}, "actual": "ALLOW", "risk": 0.0,
                "lower": 0.0, "upper": 0.0, "actual_risk": 0.0,
            },
        ]
        assert eval_mod.overall(rows)["coverage"] == 1.0
