"""YAML / JSON loaders for policy test-case files.

Cases document shape::

    cases:
      - name: baseline allow
        action_class: fs.write_file
        risk_score: 0.3
        matched_sequences: []       # optional
        expect:
          verdict: allow
          route: ai_oversight_team  # optional, only for verdict=escalate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from vaara.policy.schema import PolicyError
from vaara.policy.test_cases import PolicyTestCase


def parse_cases(data: object) -> list[PolicyTestCase]:
    if not isinstance(data, dict):
        raise PolicyError(
            f"cases document must be a mapping, got {type(data).__name__}"
        )
    raw = data.get("cases")
    if not isinstance(raw, list):
        raise PolicyError("cases document must have a 'cases:' list")

    out: list[PolicyTestCase] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise PolicyError(f"cases[{i}]: must be a mapping")
        name = entry.get("name") or f"case_{i}"
        try:
            action_class = entry["action_class"]
            risk_score = float(entry["risk_score"])
        except KeyError as e:
            raise PolicyError(
                f"cases[{i}] ({name}): missing required field {e.args[0]!r}"
            ) from None
        matched = tuple(entry.get("matched_sequences") or ())
        expect = entry.get("expect") or {}
        if not isinstance(expect, dict):
            raise PolicyError(f"cases[{i}] ({name}): 'expect' must be a mapping")
        try:
            out.append(PolicyTestCase(
                name=name,
                action_class=action_class,
                risk_score=risk_score,
                matched_sequences=matched,
                expected_verdict=expect.get("verdict", "allow"),
                expected_route=expect.get("route"),
            ))
        except ValueError as e:
            raise PolicyError(f"cases[{i}]: {e}") from None
    return out


def load_test_cases(source: Union[str, Path]) -> list[PolicyTestCase]:
    if isinstance(source, str) and "\n" in source:
        return parse_cases(_parse_text(source))
    path = Path(source) if not isinstance(source, Path) else source
    text = path.read_text(encoding="utf-8")
    prefer_yaml = path.suffix.lower() in {".yaml", ".yml"}
    return parse_cases(_parse_text(text, prefer_yaml=prefer_yaml))


def _parse_text(text: str, *, prefer_yaml: bool = False) -> object:
    if not prefer_yaml and text.lstrip().startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise PolicyError(f"invalid JSON in cases document: {e}") from None
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "loading a YAML cases document requires the [yaml] extra. "
            "Install with: pip install 'vaara[yaml]'"
        ) from e
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PolicyError(f"invalid YAML in cases document: {e}") from None
