"""Tests for vaara.policy.test_cases_io — YAML/JSON cases-file loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaara.policy.loader import from_yaml
from vaara.policy.schema import PolicyError
from vaara.policy.test_cases import run_test_cases
from vaara.policy.test_cases_io import load_test_cases, parse_cases


class TestParseCases:
    def test_minimal_case(self) -> None:
        cases = parse_cases({
            "cases": [{
                "name": "c1", "action_class": "tx.sign", "risk_score": 0.5,
                "expect": {"verdict": "escalate", "route": "ai_oversight_team"},
            }],
        })
        assert len(cases) == 1
        assert cases[0].expected_route == "ai_oversight_team"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(PolicyError, match="missing required field"):
            parse_cases({"cases": [{"name": "x", "action_class": "tx.sign"}]})

    def test_must_have_cases_list(self) -> None:
        with pytest.raises(PolicyError, match="cases:"):
            parse_cases({"version": "0.1"})

    def test_top_level_must_be_mapping(self) -> None:
        with pytest.raises(PolicyError, match="must be a mapping"):
            parse_cases(["not a mapping"])

    def test_each_entry_must_be_mapping(self) -> None:
        with pytest.raises(PolicyError, match="must be a mapping"):
            parse_cases({"cases": ["bad"]})

    def test_default_name_filled_in(self) -> None:
        cases = parse_cases({
            "cases": [{
                "action_class": "tx.sign", "risk_score": 0.3,
            }],
        })
        assert cases[0].name == "case_0"

    def test_invalid_expect_block_rejected(self) -> None:
        with pytest.raises(PolicyError, match="expect"):
            parse_cases({
                "cases": [{
                    "action_class": "tx.sign", "risk_score": 0.3,
                    "expect": "allow",
                }],
            })


class TestLoadFromFile:
    def test_load_json_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.json"
        path.write_text(json.dumps({
            "cases": [{
                "name": "c1", "action_class": "tx.sign", "risk_score": 0.3,
                "expect": {"verdict": "allow"},
            }],
        }), encoding="utf-8")
        cases = load_test_cases(path)
        assert cases[0].name == "c1"
        assert cases[0].expected_verdict == "allow"

    def test_load_yaml_file(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        path = tmp_path / "cases.yaml"
        path.write_text(
            "cases:\n"
            "  - name: c1\n"
            "    action_class: tx.sign\n"
            "    risk_score: 0.5\n"
            "    expect: {verdict: escalate, route: ai_oversight_team}\n",
            encoding="utf-8",
        )
        cases = load_test_cases(path)
        assert cases[0].expected_route == "ai_oversight_team"

    def test_load_yaml_invalid_raises_policy_error(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        path = tmp_path / "bad.yaml"
        path.write_text("cases: [: : invalid", encoding="utf-8")
        with pytest.raises(PolicyError, match="invalid YAML"):
            load_test_cases(path)


class TestExampleCasesFile:
    def test_full_example_passes_against_full_policy(self) -> None:
        pytest.importorskip("yaml")
        policy_path = Path("examples/policies/full.yaml")
        cases_path = Path("examples/policies/test_cases.yaml")
        if not policy_path.is_file() or not cases_path.is_file():
            pytest.skip("example files not present in working tree")
        policy = from_yaml(policy_path)
        cases = load_test_cases(cases_path)
        results = run_test_cases(policy, cases)
        failed = [r for r in results if not r.passed]
        assert not failed, [r.diagnostic for r in failed]
