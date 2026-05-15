"""Tests for vaara.policy.validate — structured semantic checks on a loaded Policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaara.policy import SCHEMA_VERSION, from_dict
from vaara.policy.validate import (
    IssueLevel,
    PolicyIssue,
    ValidationReport,
    validate,
    validate_source,
)


def _base_policy() -> dict:
    return {
        "version": SCHEMA_VERSION,
        "domains": ["eu_ai_act"],
        "action_classes": {
            "fs.write_file": {
                "category": "data", "reversibility": "partially_reversible",
                "blast_radius": "local", "urgency": "timely",
                "regulatory": ["aiact:9", "aiact:12"],
            },
        },
        "thresholds": {"default": {"escalate": 0.55, "deny": 0.85}},
        "sequences": {},
        "escalation": {"routes": [{"default": "on_call"}]},
    }


class TestReportShape:
    def test_empty_report_is_ok(self) -> None:
        report = ValidationReport()
        assert report.ok
        assert report.errors == ()
        assert report.warnings == ()

    def test_to_dict_has_stable_keys(self) -> None:
        issue = PolicyIssue(IssueLevel.WARNING, "x", "p", "m")
        report = ValidationReport(issues=(issue,))
        d = report.to_dict()
        assert set(d) == {"ok", "error_count", "warning_count", "issues"}
        assert d["ok"] is True
        assert d["warning_count"] == 1
        assert d["issues"][0] == {
            "level": "warning", "code": "x", "path": "p", "message": "m",
        }

    def test_error_makes_report_not_ok(self) -> None:
        report = ValidationReport(issues=(
            PolicyIssue(IssueLevel.ERROR, "e", "", "boom"),
        ))
        assert not report.ok
        assert len(report.errors) == 1
        assert len(report.warnings) == 0


class TestSemanticChecks:
    def test_clean_policy_passes(self) -> None:
        report = validate(from_dict(_base_policy()))
        assert report.ok
        assert report.warnings == ()

    def test_no_action_classes_warns(self) -> None:
        data = _base_policy()
        data["action_classes"] = {}
        report = validate(from_dict(data))
        assert report.ok
        codes = [i.code for i in report.issues]
        assert "no_action_classes" in codes

    def test_narrow_default_threshold_warns(self) -> None:
        data = _base_policy()
        data["thresholds"]["default"] = {"escalate": 0.60, "deny": 0.62}
        report = validate(from_dict(data))
        assert any(
            i.code == "narrow_threshold_band" and i.path == "thresholds.default"
            for i in report.issues
        )

    def test_threshold_override_dangling_warns(self) -> None:
        data = _base_policy()
        data["thresholds"]["nonexistent.tool"] = {"deny": 0.70}
        report = validate(from_dict(data))
        assert any(
            i.code == "threshold_override_dangling"
            and "nonexistent.tool" in i.path
            for i in report.issues
        )

    def test_narrow_override_band_warns(self) -> None:
        data = _base_policy()
        data["thresholds"]["fs.write_file"] = {"escalate": 0.60, "deny": 0.62}
        report = validate(from_dict(data))
        assert any(
            i.code == "narrow_threshold_band" and "fs.write_file" in i.path
            for i in report.issues
        )

    def test_sequence_step_unknown_class_warns(self) -> None:
        data = _base_policy()
        data["sequences"] = {
            "exfil": {
                "pattern": ["read_data", "fs.write_file"],
                "risk_boost": 0.2, "window_seconds": 60,
                "regulatory": [],
            },
        }
        report = validate(from_dict(data))
        unknown = [
            i for i in report.issues
            if i.code == "sequence_step_unknown_class"
        ]
        assert any("[0]" in i.path for i in unknown)
        assert not any("[1]" in i.path for i in unknown)

    def test_unreachable_escalation_route_warns(self) -> None:
        data = _base_policy()
        data["escalation"]["routes"] = [
            {"if": ["aiact:99"], "operator_group": "ghost_team"},
            {"default": "on_call"},
        ]
        report = validate(from_dict(data))
        assert any(
            i.code == "escalation_route_unreachable"
            for i in report.issues
        )

    def test_no_default_route_warns(self) -> None:
        data = _base_policy()
        data["escalation"]["routes"] = [
            {"if": ["aiact:9"], "operator_group": "data_team"},
        ]
        report = validate(from_dict(data))
        assert any(
            i.code == "no_default_escalation_route"
            for i in report.issues
        )


class TestValidateSource:
    def test_parse_error_yields_error_report(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"version": "9.9"}', encoding="utf-8")
        policy, report = validate_source(path)
        assert policy is None
        assert not report.ok
        assert report.errors[0].code == "parse_error"

    def test_clean_yaml_round_trip(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        path = tmp_path / "clean.yaml"
        path.write_text(
            "version: '0.1'\n"
            "domains: [eu_ai_act]\n"
            "action_classes:\n"
            "  fs.write_file:\n"
            "    category: data\n"
            "    reversibility: partially_reversible\n"
            "    blast_radius: local\n"
            "    urgency: timely\n"
            "    regulatory: [aiact:9]\n"
            "thresholds: {default: {escalate: 0.55, deny: 0.85}}\n"
            "sequences: {}\n"
            "escalation: {routes: [{default: on_call}]}\n",
            encoding="utf-8",
        )
        policy, report = validate_source(path)
        assert policy is not None
        assert report.ok

    def test_dict_input_is_accepted(self) -> None:
        policy, report = validate_source(_base_policy())
        assert policy is not None
        assert report.ok

    def test_explicit_format_overrides_sniff(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        path = tmp_path / "no_suffix"
        path.write_text(
            "version: '0.1'\n"
            "domains: [eu_ai_act]\n"
            "action_classes: {}\n"
            "thresholds: {default: {escalate: 0.55, deny: 0.85}}\n"
            "sequences: {}\n"
            "escalation: {routes: [{default: on_call}]}\n",
            encoding="utf-8",
        )
        policy, report = validate_source(path, fmt="yaml")
        assert policy is not None
        assert report.ok
