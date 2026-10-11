# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CompositeScorer — run Vaara alongside external scorers, combine results.

The composite implements the same ``evaluate(context) -> dict`` shape
that ``InterceptionPipeline`` expects, so it drops into existing
pipelines as a direct replacement for ``AdaptiveScorer``.

Combine modes (v1):

- ``"max"`` (default): highest point estimate wins; decision is the
  strongest of any member. Conservative — any scorer firing high
  produces an escalation.
- ``"mean"``: arithmetic mean across point estimates; decision is the
  strongest of any member.
- A callable ``Callable[[list[dict]], dict]`` for custom merge logic.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, Union


class ScorerBackend(Protocol):
    """Anything that maps an action context to a Vaara-shaped scorer dict."""

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]: ...

    @property
    def name(self) -> str: ...


CombineFn = Callable[[list[dict[str, Any]]], dict[str, Any]]

_DECISION_RANK = {"allow": 0, "escalate": 1, "deny": 2}
_RANK_DECISION = {v: k for k, v in _DECISION_RANK.items()}


class CompositeScorer:
    """Run Vaara's scorer alongside one or more peers; combine results."""

    def __init__(
        self,
        scorers: Iterable[ScorerBackend],
        *,
        combine: Union[str, CombineFn] = "max",
        name: str = "composite",
    ) -> None:
        self._scorers = list(scorers)
        if not self._scorers:
            raise ValueError("CompositeScorer requires at least one scorer")
        if isinstance(combine, str) and combine not in ("max", "mean"):
            raise ValueError(
                f"unknown combine mode: {combine!r}; use 'max', 'mean', or a callable"
            )
        self._combine = combine
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        results = [s.evaluate(context) for s in self._scorers]
        combine = self._combine
        if isinstance(combine, str):
            merged = _combine_max(results) if combine == "max" else _combine_mean(results)
        else:
            merged = combine(results)
        merged["backend"] = self._name
        # The pipeline reads ``action``; members speak ``decision``.
        merged["action"] = merged.get("decision", merged.get("action"))
        # A member that never answered refused without judging. If the merged
        # verdict is a deny, it must not read as a threshold the action crossed.
        failed = [r for r in results if r.get("unavailable_input")]
        if failed and merged["action"] == "deny":
            merged["violation_type"] = failed[0].get("violation_type", "scorer_unavailable")
            merged["unavailable_input"] = ", ".join(
                str(r["unavailable_input"]) for r in failed)
        merged["composition"] = {
            "members": [r.get("backend", "?") for r in results],
            "mode": self._combine if isinstance(self._combine, str) else "callable",
        }
        return merged


#: What the trail records about the decision itself, carried from the member
#: whose decision won: the reason and the thresholds it decided at. The
#: pipeline reads these off the merged result; the receipt states the
#: thresholds, and assumes an old operating point when they are absent.
_DECIDING_KEYS = ("reason", "threshold_allow", "threshold_deny")


def _member_decision(r: dict[str, Any]) -> str:
    decision = r.get("decision", r.get("action", "allow"))
    return decision if decision in _DECISION_RANK else "allow"


def _deciding_member(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The first member with the strongest decision: its verdict is the
    composite's, so its reason and operating point are what the trail
    records about the decision."""
    return max(results, key=lambda r: _DECISION_RANK[_member_decision(r)])


def _carry_decision(merged: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    deciding = _deciding_member(results)
    merged["decision"] = _member_decision(deciding)
    for key in _DECIDING_KEYS:
        if key in deciding:
            merged[key] = deciding[key]
        else:
            merged.pop(key, None)
    return merged


def _combine_max(results: list[dict[str, Any]]) -> dict[str, Any]:
    top_score = -1.0
    top_idx = 0
    for i, r in enumerate(results):
        try:
            s = float(r.get("point_estimate", 0.5))
        except (TypeError, ValueError):
            s = 0.5
        if s > top_score:
            top_score = s
            top_idx = i
    return _carry_decision(dict(results[top_idx]), results)


def _combine_mean(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    for r in results:
        try:
            scores.append(float(r.get("point_estimate", 0.5)))
        except (TypeError, ValueError):
            scores.append(0.5)
    avg = sum(scores) / len(scores)
    return _carry_decision({
        "point_estimate": avg,
        "raw_result": {
            "point_estimate": avg,
            "conformal_interval": [min(scores), max(scores)],
        },
    }, results)
