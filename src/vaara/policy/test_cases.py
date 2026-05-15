"""Policy test framework — Conftest analog for Vaara YAML/JSON policies.

A test case is a synthetic action context plus an expected verdict.
`run_test_cases(policy, cases)` evaluates each case against the policy
and returns pass/fail results. `evaluate(policy, action_class,
risk_score, matched_sequences=())` is the underlying primitive: given a
risk score and any matched sequence patterns, what does the policy say
the verdict and (if escalating) the route should be.

YAML/JSON case-file loading lives in ``vaara.policy.test_cases_io`` —
this module stays a pure data + evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from vaara.policy.schema import Policy


_VALID_VERDICTS = frozenset({"allow", "escalate", "deny"})


@dataclass(frozen=True)
class EvaluationResult:
    verdict: str
    boosted_risk: float
    route: str | None


def evaluate(
    policy: Policy,
    action_class: str,
    risk_score: float,
    matched_sequences: Iterable[str] = (),
) -> EvaluationResult:
    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(f"risk_score must be in [0,1], got {risk_score}")
    if action_class not in policy.action_classes:
        raise ValueError(f"action_class {action_class!r} not declared in policy")

    matched = set(matched_sequences)
    known = {s.name for s in policy.sequences}
    unknown = matched - known
    if unknown:
        raise ValueError(
            f"matched_sequences references unknown pattern(s): {sorted(unknown)!r}"
        )

    matched_seq_articles: set[str] = set()
    boosted = float(risk_score)
    for seq in policy.sequences:
        if seq.name in matched:
            boosted = min(1.0, boosted + seq.risk_boost)
            matched_seq_articles.update(seq.regulatory)

    thr = policy.threshold_for(action_class)
    if boosted >= thr.deny:
        verdict = "deny"
    elif boosted >= thr.escalate:
        verdict = "escalate"
    else:
        verdict = "allow"

    route: str | None = None
    if verdict == "escalate":
        articles = set(policy.action_classes[action_class].regulatory)
        articles.update(matched_seq_articles)
        route = policy.escalation_route_for(articles)

    return EvaluationResult(verdict=verdict, boosted_risk=boosted, route=route)


@dataclass(frozen=True)
class PolicyTestCase:
    name: str
    action_class: str
    risk_score: float
    matched_sequences: tuple[str, ...] = field(default_factory=tuple)
    expected_verdict: str = "allow"
    expected_route: str | None = None

    def __post_init__(self) -> None:
        if self.expected_verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"case {self.name!r}: expected_verdict must be one of "
                f"{sorted(_VALID_VERDICTS)}, got {self.expected_verdict!r}"
            )
        if self.expected_route is not None and self.expected_verdict != "escalate":
            raise ValueError(
                f"case {self.name!r}: expected_route only meaningful for "
                f"expected_verdict='escalate'"
            )


@dataclass(frozen=True)
class PolicyTestResult:
    case: PolicyTestCase
    passed: bool
    actual: EvaluationResult | None
    diagnostic: str

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.case.name,
            "passed": self.passed,
            "diagnostic": self.diagnostic,
        }
        if self.actual is not None:
            d["actual"] = {
                "verdict": self.actual.verdict,
                "boosted_risk": self.actual.boosted_risk,
                "route": self.actual.route,
            }
        return d


def run_test_cases(
    policy: Policy, cases: Iterable[PolicyTestCase],
) -> list[PolicyTestResult]:
    results: list[PolicyTestResult] = []
    for case in cases:
        try:
            actual = evaluate(
                policy, case.action_class, case.risk_score, case.matched_sequences,
            )
        except (ValueError, KeyError) as e:
            results.append(PolicyTestResult(
                case=case, passed=False, actual=None,
                diagnostic=f"evaluation error: {e}",
            ))
            continue

        verdict_ok = actual.verdict == case.expected_verdict
        route_ok = case.expected_route is None or actual.route == case.expected_route
        passed = verdict_ok and route_ok

        if passed:
            diag = "ok"
        else:
            parts = []
            if not verdict_ok:
                parts.append(
                    f"verdict: expected {case.expected_verdict!r}, "
                    f"got {actual.verdict!r}"
                )
            if not route_ok:
                parts.append(
                    f"route: expected {case.expected_route!r}, "
                    f"got {actual.route!r}"
                )
            diag = "; ".join(parts)

        results.append(PolicyTestResult(
            case=case, passed=passed, actual=actual, diagnostic=diag,
        ))
    return results
