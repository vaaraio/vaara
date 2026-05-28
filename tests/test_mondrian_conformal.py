"""Tests for the Mondrian (class-conditional) extension to ConformalCalibrator.

The marginal-mode behaviour is covered by tests/test_scorer.py against
ConformalCalibrator's pre-existing surface. This file verifies the new
opt-in per-category mode and that it leaves the marginal mode bit-for-bit
unchanged.
"""
from __future__ import annotations

import math

import pytest

from vaara.scorer.adaptive import ConformalCalibrator


@pytest.fixture
def calibrator():
    # Small min_calibration keeps test runs fast while still exercising
    # the FACI update path (which only triggers at min_calibration and beyond).
    return ConformalCalibrator(
        alpha=0.10, min_calibration=10, max_calibration=100, gamma=0.005,
    )


class TestBackwardsCompat:
    def test_no_category_arg_lands_in_default_bucket(self, calibrator):
        for _ in range(10):
            calibrator.add_calibration_point(0.5, 0.5)
        assert calibrator.calibration_size == 10
        assert calibrator.calibration_size_for(None) == 10
        assert calibrator.is_calibrated

    def test_predict_interval_marginal_path_unchanged(self, calibrator):
        # Below min_calibration to conservative ±0.3 fallback
        assert calibrator.predict_interval(0.5) == (0.2, 0.8)
        for _ in range(10):
            calibrator.add_calibration_point(0.5, 0.5)
        # All zero residuals to tight interval at the point estimate
        assert calibrator.predict_interval(0.5) == (0.5, 0.5)

    def test_effective_alpha_starts_at_configured(self, calibrator):
        assert calibrator.effective_alpha == 0.10
        assert calibrator.effective_alpha_for(None) == 0.10
        assert calibrator.effective_alpha_for("never_seen") == 0.10


class TestPerCategoryIsolation:
    def test_residuals_isolated_by_category(self, calibrator):
        for _ in range(10):
            calibrator.add_calibration_point(0.0, 0.5, category="A")
            calibrator.add_calibration_point(0.0, 0.0, category="B")
        assert calibrator.calibration_size_for("A") == 10
        assert calibrator.calibration_size_for("B") == 10
        assert calibrator.calibration_size_for(None) == 0
        assert calibrator.calibration_size == 20

    def test_quantile_diverges_per_category(self, calibrator):
        # A: residual = 0.5 every time. B: residual = 0.05 every time.
        for _ in range(15):
            calibrator.add_calibration_point(0.0, 0.5, category="A")
            calibrator.add_calibration_point(0.05, 0.0, category="B")
        lo_a, hi_a = calibrator.predict_interval(0.5, category="A")
        lo_b, hi_b = calibrator.predict_interval(0.5, category="B")
        assert (hi_a - lo_a) > (hi_b - lo_b)
        assert (hi_b - lo_b) < 0.2

    def test_default_bucket_unaffected_by_categorized_calls(self, calibrator):
        for _ in range(20):
            calibrator.add_calibration_point(0.0, 0.7, category="X")
        assert not calibrator.is_calibrated
        assert calibrator.predict_interval(0.5) == (0.2, 0.8)

    def test_alpha_t_isolated_per_bucket(self, calibrator):
        # 15 points with residual 1.0 to 5 FACI updates, alpha_t drifts
        for _ in range(15):
            calibrator.add_calibration_point(0.0, 1.0, category="A")
        assert calibrator.effective_alpha_for("A") != 0.10
        assert calibrator.effective_alpha_for(None) == 0.10
        assert calibrator.effective_alpha == 0.10


class TestReadiness:
    def test_is_calibrated_for_per_bucket(self, calibrator):
        for _ in range(10):
            calibrator.add_calibration_point(0.5, 0.5, category="A")
        for _ in range(5):
            calibrator.add_calibration_point(0.5, 0.5, category="B")
        assert calibrator.is_calibrated_for("A") is True
        assert calibrator.is_calibrated_for("B") is False
        assert calibrator.is_calibrated_for(None) is False

    def test_unseen_category_returns_zero_size(self, calibrator):
        assert calibrator.calibration_size_for("never_seen") == 0
        assert not calibrator.is_calibrated_for("never_seen")


class TestPredictIntervalPerCategory:
    def test_underfilled_category_conservative_fallback(self, calibrator):
        for _ in range(15):
            calibrator.add_calibration_point(0.5, 0.5, category="A")
        # Querying an underfilled bucket falls back regardless of A's state
        assert calibrator.predict_interval(0.5, category="B") == (0.2, 0.8)

    def test_categorized_predict_uses_bucket_quantile(self, calibrator):
        for _ in range(15):
            calibrator.add_calibration_point(0.5, 0.5, category="A")
        assert calibrator.predict_interval(0.5, category="A") == (0.5, 0.5)

    def test_clamping_preserved_per_category(self, calibrator):
        for _ in range(15):
            calibrator.add_calibration_point(0.0, 1.0, category="A")
        lo, hi = calibrator.predict_interval(0.5, category="A")
        assert lo == 0.0
        assert hi == 1.0


class TestRobustness:
    def test_nan_input_dropped_per_bucket(self, calibrator):
        calibrator.add_calibration_point(0.5, float("nan"), category="A")
        calibrator.add_calibration_point(float("inf"), 0.5, category="A")
        calibrator.add_calibration_point(0.5, math.nan, category="A")
        assert calibrator.calibration_size_for("A") == 0

    def test_non_numeric_input_dropped(self, calibrator):
        calibrator.add_calibration_point("oops", 0.5, category="A")
        calibrator.add_calibration_point(0.5, None, category="A")
        assert calibrator.calibration_size_for("A") == 0

    def test_empty_string_category_routes_to_default(self, calibrator):
        # Empty string is falsy. Routing it to the default bucket avoids
        # phantom "" buckets when callers accidentally pass an empty value.
        calibrator.add_calibration_point(0.5, 0.5, category="")
        assert calibrator.calibration_size_for(None) == 1
        assert calibrator.calibration_size_for("") == 1
        assert calibrator.calibration_size == 1
