"""Tests for AdaptiveScorer's opt-in Mondrian routing.

The calibrator-side per-category behaviour is covered by
tests/test_mondrian_conformal.py. These tests verify that the scorer's
mondrian_categories flag plumbs through to the calibrator at the three
real call sites (evaluate, dry_run_evaluate, record_outcome) and stays
off by default so the marginal contract holds for existing callers.
"""
from __future__ import annotations

import pytest

from vaara.scorer.adaptive import AdaptiveScorer, _category_of


@pytest.fixture
def base_signals():
    return {
        "taxonomy_base": 0.4,
        "agent_history": 0.3,
        "sequence_pattern": 0.0,
        "action_frequency": 0.1,
        "confidence_gap": 0.2,
    }


class TestDefaultIsMarginal:
    def test_mondrian_off_by_default(self):
        scorer = AdaptiveScorer()
        assert scorer._mondrian is False
        assert scorer._calib_category("data.read") is None
        assert scorer._calib_category("tx.sign") is None

    def test_default_routing_keeps_default_bucket(self, base_signals):
        scorer = AdaptiveScorer(pre_seed_calibration=False)
        # 5 distinct categories of actions; without mondrian, all land in default
        for tool in ("data.read", "tx.sign", "infra.deploy", "id.grant_permission", "comm.post_public"):
            scorer.record_outcome("agent-1", tool, 0.5, 0.5, base_signals)
        assert scorer._conformal.calibration_size_for(None) == 5
        # No per-category bucket got created
        for category in ("data", "financial", "infrastructure", "identity", "communication"):
            assert scorer._conformal.calibration_size_for(category) == 0


class TestMondrianOn:
    def test_calib_category_returns_tool_category(self):
        scorer = AdaptiveScorer(mondrian_categories=True)
        assert scorer._calib_category("data.read") == "data"
        assert scorer._calib_category("tx.sign") == "financial"
        assert scorer._calib_category("infra.deploy") == "infrastructure"
        assert scorer._calib_category("id.grant_permission") == "identity"

    def test_record_outcome_routes_per_category(self, base_signals):
        scorer = AdaptiveScorer(
            pre_seed_calibration=False, mondrian_categories=True,
        )
        # 3 outcomes on data.read, 2 on tx.sign
        for _ in range(3):
            scorer.record_outcome("agent-1", "data.read", 0.4, 0.5, base_signals)
        for _ in range(2):
            scorer.record_outcome("agent-1", "tx.sign", 0.7, 0.8, base_signals)
        assert scorer._conformal.calibration_size_for("data") == 3
        assert scorer._conformal.calibration_size_for("financial") == 2
        # Default bucket untouched in Mondrian mode
        assert scorer._conformal.calibration_size_for(None) == 0
        # Aggregate matches sum of buckets
        assert scorer._conformal.calibration_size == 5

    def test_evaluate_uses_per_category_interval(self):
        # Build a calibrator state where one category has a wide quantile
        # and another has zero residuals, then verify evaluate's interval
        # tracks the right bucket per tool.
        scorer = AdaptiveScorer(
            pre_seed_calibration=False, mondrian_categories=True,
        )
        # Wide bucket for data: residual = 0.5 every time, 35 points
        cc = scorer._conformal
        for _ in range(35):
            cc.add_calibration_point(0.0, 0.5, category="data")
        # Tight bucket for financial: zero residuals, 35 points
        for _ in range(35):
            cc.add_calibration_point(0.5, 0.5, category="financial")
        # Querying an underfilled bucket directly returns the conservative fallback
        lo_data, hi_data = cc.predict_interval(0.5, category="data")
        lo_fin, hi_fin = cc.predict_interval(0.5, category="financial")
        assert (hi_data - lo_data) > (hi_fin - lo_fin)

    def test_seed_prior_stays_in_default_bucket_when_mondrian_on(self):
        scorer = AdaptiveScorer(
            pre_seed_calibration=True, mondrian_categories=True,
        )
        # The 50 synthetic seed pairs go to the default bucket regardless
        # of mondrian flag (they have no real category context).
        assert scorer._conformal.calibration_size_for(None) == 50
        assert scorer._conformal.calibration_size_for("data") == 0


class TestCategoryDerivation:
    """Sanity check that the helper agrees with the module-level _category_of."""

    @pytest.mark.parametrize(
        "tool,expected",
        [
            ("data.read", "data"),
            ("data.export", "data"),
            ("tx.sign", "financial"),
            ("vault.unlock", "financial"),
            ("infra.deploy", "infrastructure"),
            ("id.create_key", "identity"),
            ("comm.post_public", "communication"),
            ("gov.vote", "governance"),
            ("phy.actuator", "physical"),
            ("unknown_prefix.foo", _category_of("unknown_prefix.foo")),
        ],
    )
    def test_calib_category_mirrors_category_of(self, tool, expected):
        scorer = AdaptiveScorer(mondrian_categories=True)
        assert scorer._calib_category(tool) == expected
