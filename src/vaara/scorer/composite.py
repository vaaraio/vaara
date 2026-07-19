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
        merged["composition"] = {
            "members": [r.get("backend", "?") for r in results],
            "mode": self._combine if isinstance(self._combine, str) else "callable",
        }
        return merged


def _combine_max(results: list[dict[str, Any]]) -> dict[str, Any]:
    top_score = -1.0
    top_idx = 0
    top_decision_rank = 0
    for i, r in enumerate(results):
        try:
            s = float(r.get("point_estimate", 0.5))
        except (TypeError, ValueError):
            s = 0.5
        if s > top_score:
            top_score = s
            top_idx = i
        dr = _DECISION_RANK.get(r.get("decision", "allow"), 0)
        if dr > top_decision_rank:
            top_decision_rank = dr
    base = dict(results[top_idx])
    base["decision"] = _RANK_DECISION.get(top_decision_rank, "allow")
    return base


def _combine_mean(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    for r in results:
        try:
            scores.append(float(r.get("point_estimate", 0.5)))
        except (TypeError, ValueError):
            scores.append(0.5)
    avg = sum(scores) / len(scores)
    top_decision_rank = max(
        _DECISION_RANK.get(r.get("decision", "allow"), 0) for r in results
    )
    return {
        "point_estimate": avg,
        "decision": _RANK_DECISION.get(top_decision_rank, "allow"),
        "raw_result": {
            "point_estimate": avg,
            "conformal_interval": [min(scores), max(scores)],
        },
    }
