"""Vaara policy schema — JSON-native, YAML via the optional `vaara[yaml]` extra.

The policy expresses, in declarative form, the deployer's choices for:
- Action-class taxonomy (category, reversibility, blast radius, urgency, regulatory tags)
- Risk thresholds (default plus per-action-class overrides)
- Sequence patterns that boost risk when matched in a sliding history window
- Escalation routes mapping regulatory-article sets to operator groups

Internal model stays zero-dependency. JSON loading uses stdlib. YAML loading is
gated on `vaara[yaml]` so the core library stays free of third-party imports.

The companion JSON Schema document at `docs/policy_schema.json` is the citable
spec for compliance-evidence purposes. Hand-rolled validation in `loader.py`
mirrors the schema, with clean error paths for human readers.

Beyond load and parse, two surfaces support reviewing the policy artifact
independently from agent code:

- ``validate`` / ``validate_source`` (in ``vaara.policy.validate``) returns a
  structured report with parse errors and semantic warnings — usable in CI.
- ``evaluate`` + ``run_test_cases`` (in ``vaara.policy.test_cases``) let a
  team write synthetic action contexts and assert expected verdicts against a
  policy, Conftest-style. YAML/JSON case files load via
  ``load_test_cases`` (in ``vaara.policy.test_cases_io``).
"""

from vaara.policy.schema import (
    SCHEMA_VERSION,
    ActionClassDef,
    EscalationRoute,
    Policy,
    PolicyError,
    SequencePattern,
    Thresholds,
)
from vaara.policy.loader import from_dict, from_json, from_yaml
from vaara.policy.controller import PolicyController, ReloadResult
from vaara.policy.validate import (
    IssueLevel,
    PolicyIssue,
    ValidationReport,
    validate,
    validate_source,
)
from vaara.policy.test_cases import (
    EvaluationResult,
    PolicyTestCase,
    PolicyTestResult,
    evaluate,
    run_test_cases,
)
from vaara.policy.test_cases_io import load_test_cases, parse_cases

__all__ = [
    "SCHEMA_VERSION",
    "ActionClassDef",
    "EscalationRoute",
    "EvaluationResult",
    "IssueLevel",
    "Policy",
    "PolicyController",
    "PolicyError",
    "PolicyIssue",
    "PolicyTestCase",
    "PolicyTestResult",
    "ReloadResult",
    "SequencePattern",
    "Thresholds",
    "ValidationReport",
    "evaluate",
    "from_dict",
    "from_json",
    "from_yaml",
    "load_test_cases",
    "parse_cases",
    "run_test_cases",
    "validate",
    "validate_source",
]
