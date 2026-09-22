# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Nothing checked how the hooks were registered, so two faults ran unseen.

Both were found on the maintainer's machine on 2026-09-22, and neither is
visible from inside a tool call.

The first is a second registration. The universal install writes
``vaara hook pre-tool-use`` into ``~/.claude/settings.json``; the plugin
registers ``run.sh pre-tool-use`` through its own ``hooks.json``. Same
binary, same version, both fire. One logical call produced two action
ids, two full decision cycles and two escalations, seq 184 to 191. The
chain stays valid and every verifier passes, so the trail simply counts
one action as two.

``detect_stacked_governance`` in the MCP proxy does not reach this. It
runs in the proxy process, reads only the settings files, and matches the
literal string ``vaara hook pre-tool-use``. The plugin registration lives
in ``hooks.json`` and contains no such string, so the detector is blind to
this shape on all three counts. It answers hook-versus-proxy. This is
hook-versus-hook.

The second is an install left behind by its own package. ``HOOK_MATCHER``
widened to ``.*`` after an earlier drift, and ``write_claude_hooks``
strips and re-adds on every run, so ``vaara init-governance`` repairs a
stale file. Nothing ever asks anyone to re-run it. The machine that found
this was still dispatching PostToolUse on the fifteen-name list written by
an older version, so tools outside that list were scored and never
reported an outcome, and the conformal calibrator was fed from a subset
nobody chose.

Advisory throughout. This reports and never raises: a check that can take
a session down is worse than the drift it looks for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaara.integrations import hook_registration as reg

VAARA_PRE = "/usr/local/bin/vaara hook pre-tool-use"
VAARA_POST = "/usr/local/bin/vaara hook post-tool-use"


def _settings(pre_matcher: str = ".*", post_matcher: str = ".*",
              plugins: dict | None = None) -> dict:
    settings: dict = {
        "hooks": {
            "PreToolUse": [
                {"matcher": pre_matcher,
                 "hooks": [{"type": "command", "command": VAARA_PRE}]},
            ],
            "PostToolUse": [
                {"matcher": post_matcher,
                 "hooks": [{"type": "command", "command": VAARA_POST}]},
            ],
        }
    }
    if plugins is not None:
        settings["enabledPlugins"] = plugins
    return settings


def _write(tmp_path: Path, settings: dict) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return path


def _kinds(findings) -> list[str]:
    return [f.kind for f in findings]


class TestExpectedMatcherIsPinned:
    """The check is worthless if its idea of correct drifts from the installer."""

    def test_it_matches_the_installer(self):
        from vaara.integrations.init_governance import HOOK_MATCHER

        assert reg.EXPECTED_MATCHER == HOOK_MATCHER

    def test_it_matches_the_plugin_manifest(self):
        manifest = json.loads(
            (Path(__file__).resolve().parent.parent
             / "plugins" / "claude-code-vaara-governance"
             / "hooks" / "hooks.json").read_text()
        )
        for event in ("PreToolUse", "PostToolUse"):
            for entry in manifest["hooks"][event]:
                assert entry["matcher"] == reg.EXPECTED_MATCHER


class TestStackedRegistration:
    def test_plugin_enabled_alongside_the_universal_install(self, tmp_path):
        path = _write(tmp_path, _settings(plugins={"vaara-governance@vaara": True}))
        findings = reg.inspect_registration([path])
        assert "stacked" in _kinds(findings)
        text = " ".join(f.detail for f in findings)
        assert "vaara-governance@vaara" in text
        assert "twice" in text

    def test_a_disabled_plugin_is_not_stacked(self, tmp_path):
        path = _write(tmp_path, _settings(plugins={"vaara-governance@vaara": False}))
        assert "stacked" not in _kinds(reg.inspect_registration([path]))

    def test_the_plugin_alone_is_not_stacked(self, tmp_path):
        """One governance layer is the supported shape, whichever one it is."""
        settings = {"enabledPlugins": {"vaara-governance@vaara": True}}
        path = _write(tmp_path, settings)
        assert "stacked" not in _kinds(reg.inspect_registration([path]))

    def test_the_universal_install_alone_is_not_stacked(self, tmp_path):
        path = _write(tmp_path, _settings(plugins={"other@market": True}))
        assert "stacked" not in _kinds(reg.inspect_registration([path]))

    def test_any_marketplace_name_counts(self, tmp_path):
        """The plugin can be installed from a fork or a local marketplace."""
        path = _write(tmp_path, _settings(plugins={"vaara-governance@henri-local": True}))
        assert "stacked" in _kinds(reg.inspect_registration([path]))

    def test_two_settings_files_each_registering_the_hook(self, tmp_path):
        user = tmp_path / "user.json"
        user.write_text(json.dumps(_settings()))
        project = tmp_path / "project.json"
        project.write_text(json.dumps(_settings()))
        findings = reg.inspect_registration([user, project])
        assert "stacked" in _kinds(findings)


class TestStaleMatcher:
    NARROW = ("Bash|WebFetch|WebSearch|Write|Edit|NotebookEdit|Agent|Task|"
              "Workflow|CronCreate|ScheduleWakeup|RemoteTrigger|SendMessage|"
              "Skill|mcp__.*")

    def test_the_shape_found_on_the_maintainers_machine(self, tmp_path):
        """PreToolUse widened, PostToolUse left on the old enumerated list."""
        path = _write(tmp_path, _settings(post_matcher=self.NARROW))
        findings = reg.inspect_registration([path])
        assert _kinds(findings) == ["stale_matcher"]
        finding = findings[0]
        assert "PostToolUse" in finding.detail
        assert "init-governance" in finding.remedy

    def test_a_narrow_pre_tool_use_is_reported(self, tmp_path):
        path = _write(tmp_path, _settings(pre_matcher="Bash|mcp__.*"))
        assert _kinds(reg.inspect_registration([path])) == ["stale_matcher"]

    def test_both_narrow_reports_both_events(self, tmp_path):
        path = _write(tmp_path, _settings(pre_matcher="Bash", post_matcher="Bash"))
        findings = reg.inspect_registration([path])
        text = " ".join(f.detail for f in findings)
        assert "PreToolUse" in text and "PostToolUse" in text

    def test_a_current_install_is_quiet(self, tmp_path):
        path = _write(tmp_path, _settings())
        assert reg.inspect_registration([path]) == []

    def test_a_missing_matcher_is_not_reported(self, tmp_path):
        """SessionStart carries no matcher and is not a dispatch surface."""
        settings = {"hooks": {"SessionStart": [
            {"hooks": [{"type": "command", "command": "vaara hook session-start"}]},
        ]}}
        path = _write(tmp_path, settings)
        assert reg.inspect_registration([path]) == []

    def test_another_tools_narrow_matcher_is_ignored(self, tmp_path):
        settings = _settings()
        settings["hooks"]["PreToolUse"].append(
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "rtk hook claude"}]}
        )
        path = _write(tmp_path, settings)
        assert reg.inspect_registration([path]) == []


class TestItNeverRaises:
    @pytest.mark.parametrize("body", [
        "", "not json at all", "[]", "null", '{"hooks": "wrong type"}',
        '{"hooks": {"PreToolUse": "wrong"}}',
        '{"hooks": {"PreToolUse": [null, 3, {"hooks": null}]}}',
        '{"enabledPlugins": "wrong type"}',
        '{"enabledPlugins": {"vaara-governance@vaara": "yes"}}',
    ])
    def test_malformed_settings(self, tmp_path, body):
        path = tmp_path / "settings.json"
        path.write_text(body)
        assert reg.inspect_registration([path]) == []

    def test_a_missing_file(self, tmp_path):
        assert reg.inspect_registration([tmp_path / "nope.json"]) == []

    def test_a_directory_in_place_of_the_file(self, tmp_path):
        target = tmp_path / "settings.json"
        target.mkdir()
        assert reg.inspect_registration([target]) == []


class TestDefaultPaths:
    def test_it_reads_the_user_and_project_settings(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        project = tmp_path / "project"
        (project / ".claude").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
        paths = reg.settings_paths()
        assert home / ".claude" / "settings.json" in paths
        assert project / ".claude" / "settings.json" in paths
        assert project / ".claude" / "settings.local.json" in paths


class TestSessionStartReportsIt:
    """Session start is the only place the wiring can be seen."""

    @staticmethod
    def _run(tmp_path, settings: dict, monkeypatch, capsys):
        import io
        import sys

        from vaara.integrations import claude_code_hooks as hooks

        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        (home / ".claude" / "settings.json").write_text(json.dumps(settings))
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(tmp_path / "audit.db"))
        monkeypatch.setenv("VAARA_PLUGIN_NOTIFY", "0")
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"session_id": "s1"})))
        assert hooks.run_session_start() == 0
        return capsys.readouterr().err

    def test_a_stale_matcher_is_named(self, tmp_path, monkeypatch, capsys):
        err = self._run(
            tmp_path, _settings(post_matcher="Bash|mcp__.*"), monkeypatch, capsys
        )
        assert "PostToolUse" in err
        assert "init-governance" in err

    def test_a_second_layer_is_named(self, tmp_path, monkeypatch, capsys):
        err = self._run(
            tmp_path,
            _settings(plugins={"vaara-governance@vaara": True}),
            monkeypatch, capsys,
        )
        assert "twice" in err
        assert "vaara-governance@vaara" in err

    def test_a_healthy_install_says_nothing_about_registration(
        self, tmp_path, monkeypatch, capsys
    ):
        err = self._run(tmp_path, _settings(), monkeypatch, capsys)
        assert "twice" not in err
        assert "init-governance" not in err
