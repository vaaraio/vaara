"""Tests for ExternalScorer + CompositeScorer."""

from __future__ import annotations

from typing import Any

import pytest

from vaara.scorer.composite import CompositeScorer
from vaara.scorer.composition import ExternalScorer, _coerce_external_result


class _Stub:
    def __init__(self, *, score: float, decision: str, name: str = "stub") -> None:
        self._score = score
        self._decision = decision
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, _context: dict[str, Any]) -> dict[str, Any]:
        return {
            "point_estimate": self._score,
            "decision": self._decision,
            "backend": self._name,
            "raw_result": {
                "point_estimate": self._score,
                "conformal_interval": [self._score, self._score],
            },
        }


# ── ExternalScorer ──────────────────────────────────────────────────


def test_external_scorer_requires_url():
    with pytest.raises(ValueError):
        ExternalScorer("")


def test_external_scorer_fail_closed_on_transport_error():
    sc = ExternalScorer("http://127.0.0.1:1/score", timeout=0.05)
    r = sc.evaluate({"tool_name": "x"})
    assert r["decision"] == "deny"
    assert r["point_estimate"] == 1.0
    assert "fail_closed_reason" in r["raw_result"]


def test_coerce_external_result_decodes_vaara_score_shape():
    r = _coerce_external_result(
        "peer",
        {
            "risk": {"point": 0.62, "lower": 0.55, "upper": 0.71},
            "decision": "escalate",
        },
    )
    assert r["point_estimate"] == 0.62
    assert r["decision"] == "escalate"
    assert r["backend"] == "peer"
    assert r["raw_result"]["conformal_interval"] == [0.55, 0.71]


def test_coerce_external_result_handles_non_object_body():
    r = _coerce_external_result("peer", "not-an-object")
    assert r["decision"] == "deny"
    assert "fail_closed_reason" in r["raw_result"]


def test_coerce_external_result_clamps_out_of_range_score():
    r = _coerce_external_result("peer", {"risk": {"point": 1.7}})
    assert 0.0 <= r["point_estimate"] <= 1.0


# ── CompositeScorer ─────────────────────────────────────────────────


def test_composite_requires_at_least_one_scorer():
    with pytest.raises(ValueError):
        CompositeScorer([])


def test_composite_max_picks_highest_score_member():
    a = _Stub(score=0.2, decision="allow", name="vaara")
    b = _Stub(score=0.81, decision="deny", name="peer")
    c = _Stub(score=0.4, decision="escalate", name="other")
    composite = CompositeScorer([a, b, c], combine="max")
    out = composite.evaluate({"tool_name": "tx.transfer"})
    assert out["backend"] == "composite"
    assert out["point_estimate"] == 0.81
    assert out["decision"] == "deny"
    assert out["composition"]["members"] == ["vaara", "peer", "other"]
    assert out["composition"]["mode"] == "max"


def test_composite_mean_averages_scores_and_keeps_strongest_decision():
    a = _Stub(score=0.2, decision="allow", name="a")
    b = _Stub(score=0.8, decision="escalate", name="b")
    composite = CompositeScorer([a, b], combine="mean")
    out = composite.evaluate({})
    assert out["point_estimate"] == pytest.approx(0.5)
    assert out["decision"] == "escalate"


def test_composite_callable_combine_runs():
    seen: list[list[dict[str, Any]]] = []

    def merge(results: list[dict[str, Any]]) -> dict[str, Any]:
        seen.append(results)
        return {"point_estimate": 0.99, "decision": "deny"}

    a = _Stub(score=0.1, decision="allow", name="a")
    composite = CompositeScorer([a], combine=merge)
    out = composite.evaluate({})
    assert seen and len(seen[0]) == 1
    assert out["composition"]["mode"] == "callable"
    assert out["point_estimate"] == 0.99


def test_composite_rejects_unknown_combine_string():
    with pytest.raises(ValueError):
        CompositeScorer([_Stub(score=0.1, decision="allow")], combine="median")
