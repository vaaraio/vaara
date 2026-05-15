"""End-to-end CLI tests for `vaara policy validate` and `vaara policy test`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaara.cli import main


@pytest.fixture
def policy_file(tmp_path: Path) -> Path:
    pytest.importorskip("yaml")
    path = tmp_path / "policy.yaml"
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
    return path


@pytest.fixture
def cases_file(tmp_path: Path) -> Path:
    pytest.importorskip("yaml")
    path = tmp_path / "cases.yaml"
    path.write_text(
        "cases:\n"
        "  - name: allow_low\n"
        "    action_class: fs.write_file\n"
        "    risk_score: 0.3\n"
        "    expect: {verdict: allow}\n"
        "  - name: escalate_mid\n"
        "    action_class: fs.write_file\n"
        "    risk_score: 0.7\n"
        "    expect: {verdict: escalate, route: on_call}\n",
        encoding="utf-8",
    )
    return path


class TestValidateCLI:
    def test_clean_policy_exit_zero(
        self, policy_file: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["policy", "validate", str(policy_file)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "ok" in captured.out or "0 error" in captured.out

    def test_parse_error_exit_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"version": "9.9"}', encoding="utf-8")
        rc = main(["policy", "validate", str(bad)])
        captured = capsys.readouterr()
        assert rc == 1
        assert "error" in captured.out.lower()

    def test_json_output_shape(
        self, policy_file: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["policy", "validate", str(policy_file), "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        payload = json.loads(captured.out)
        assert payload["ok"] is True
        assert "error_count" in payload
        assert "warning_count" in payload
        assert isinstance(payload["issues"], list)


class TestTestCLI:
    def test_all_pass_exit_zero(
        self,
        policy_file: Path,
        cases_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([
            "policy", "test", str(policy_file), "--cases", str(cases_file),
        ])
        captured = capsys.readouterr()
        assert rc == 0
        assert "2 passed" in captured.out

    def test_failure_exit_one(
        self,
        policy_file: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pytest.importorskip("yaml")
        bad_cases = tmp_path / "bad_cases.yaml"
        bad_cases.write_text(
            "cases:\n"
            "  - name: wrong\n"
            "    action_class: fs.write_file\n"
            "    risk_score: 0.3\n"
            "    expect: {verdict: deny}\n",
            encoding="utf-8",
        )
        rc = main([
            "policy", "test", str(policy_file), "--cases", str(bad_cases),
        ])
        captured = capsys.readouterr()
        assert rc == 1
        assert "FAIL" in captured.out

    def test_policy_parse_error_exits_two(
        self,
        tmp_path: Path,
        cases_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"version": "9.9"}', encoding="utf-8")
        rc = main([
            "policy", "test", str(bad), "--cases", str(cases_file),
        ])
        assert rc == 2

    def test_json_output_shape(
        self,
        policy_file: Path,
        cases_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([
            "policy", "test", str(policy_file),
            "--cases", str(cases_file), "--json",
        ])
        captured = capsys.readouterr()
        assert rc == 0
        payload = json.loads(captured.out)
        assert payload["total"] == 2
        assert payload["passed"] == 2
        assert payload["failed"] == 0
        assert len(payload["results"]) == 2
