"""Structured policy validation.

`from_dict` / `from_json` / `from_yaml` raise PolicyError on the first
parse-time failure. That is right for the load path but coarse for a
CI / compliance workflow that wants every issue at once, surfaces
non-blocking warnings, and routes output to JSON. `validate(policy)`
runs semantic checks and returns a ValidationReport. `validate_source`
combines load and check so a single call yields (policy, report) or
(None, report-with-error).

Output rendering (text / JSON) lives in the CLI, not here, so the
core module stays a pure data layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union

from vaara.policy.loader import from_dict, from_json, from_yaml
from vaara.policy.schema import Policy, PolicyError


class IssueLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class PolicyIssue:
    level: IssueLevel
    code: str
    path: str
    message: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.value
        return d


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[PolicyIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[PolicyIssue, ...]:
        return tuple(i for i in self.issues if i.level is IssueLevel.ERROR)

    @property
    def warnings(self) -> tuple[PolicyIssue, ...]:
        return tuple(i for i in self.issues if i.level is IssueLevel.WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


_MIN_THRESHOLD_GAP = 0.05


def validate(policy: Policy) -> ValidationReport:
    issues: list[PolicyIssue] = []
    action_class_names = set(policy.action_classes)

    if not action_class_names:
        issues.append(PolicyIssue(
            IssueLevel.WARNING, "no_action_classes", "action_classes",
            "policy declares no action classes — pipeline cannot route any tool",
        ))

    default = policy.thresholds_default
    if default.deny - default.escalate < _MIN_THRESHOLD_GAP:
        issues.append(PolicyIssue(
            IssueLevel.WARNING, "narrow_threshold_band", "thresholds.default",
            f"escalate={default.escalate} and deny={default.deny} differ by less "
            f"than {_MIN_THRESHOLD_GAP} — operators get a narrow review band",
        ))

    for name in policy.thresholds_overrides:
        path = f"thresholds.{name}"
        if name not in action_class_names:
            issues.append(PolicyIssue(
                IssueLevel.WARNING, "threshold_override_dangling", path,
                f"threshold override targets action class {name!r} that is not "
                f"declared in action_classes — override will never fire",
            ))
        merged = policy.threshold_for(name)
        if merged.deny - merged.escalate < _MIN_THRESHOLD_GAP:
            issues.append(PolicyIssue(
                IssueLevel.WARNING, "narrow_threshold_band", path,
                f"merged escalate={merged.escalate} and deny={merged.deny} differ "
                f"by less than {_MIN_THRESHOLD_GAP} for {name!r}",
            ))

    for seq in policy.sequences:
        for i, step in enumerate(seq.pattern):
            if step not in action_class_names:
                issues.append(PolicyIssue(
                    IssueLevel.WARNING, "sequence_step_unknown_class",
                    f"sequences.{seq.name}.pattern[{i}]",
                    f"sequence step {step!r} does not name a declared action class "
                    f"— if this is a deployer-side tool name, ignore",
                ))

    emitted: set[str] = set()
    for ac in policy.action_classes.values():
        emitted.update(ac.regulatory)
    for seq in policy.sequences:
        emitted.update(seq.regulatory)

    has_default_route = False
    for i, route in enumerate(policy.escalation_routes):
        path = f"escalation.routes[{i}]"
        if not route.if_articles:
            has_default_route = True
            continue
        if not emitted.intersection(route.if_articles):
            issues.append(PolicyIssue(
                IssueLevel.WARNING, "escalation_route_unreachable", path,
                f"route to {route.operator_group!r} fires on "
                f"{list(route.if_articles)!r} but no action class or sequence "
                f"emits any of those articles",
            ))

    if policy.escalation_routes and not has_default_route:
        issues.append(PolicyIssue(
            IssueLevel.WARNING, "no_default_escalation_route", "escalation.routes",
            "no fallback route (route with empty `if`) declared — unmatched "
            "escalations fall through to the implicit 'on_call' group",
        ))

    return ValidationReport(issues=tuple(issues))


def validate_source(
    source: Union[str, Path, dict], *, fmt: str = "auto",
) -> tuple[Policy | None, ValidationReport]:
    try:
        policy = _load(source, fmt)
    except PolicyError as e:
        return None, ValidationReport(issues=(PolicyIssue(
            IssueLevel.ERROR, "parse_error", "", str(e),
        ),))
    except ImportError as e:
        return None, ValidationReport(issues=(PolicyIssue(
            IssueLevel.ERROR, "missing_extra", "", str(e),
        ),))
    return policy, validate(policy)


def _load(source: Union[str, Path, dict], fmt: str) -> Policy:
    if isinstance(source, dict):
        return from_dict(source)
    if fmt == "auto":
        fmt = _sniff_format(source)
    if fmt == "json":
        return from_json(source)
    if fmt == "yaml":
        return from_yaml(source)
    raise PolicyError(f"unknown format {fmt!r} (expected 'json' or 'yaml')")


def _sniff_format(source: Union[str, Path]) -> str:
    path: Path | None = None
    if isinstance(source, Path):
        path = source
    elif isinstance(source, str) and "\n" not in source:
        candidate = Path(source)
        if candidate.is_file():
            path = candidate
    if path is not None:
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return "yaml"
        if suffix == ".json":
            return "json"
    if isinstance(source, str) and source.lstrip().startswith("{"):
        return "json"
    return "yaml" if isinstance(source, str) else "json"
