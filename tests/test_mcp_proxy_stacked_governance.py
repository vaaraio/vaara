# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stacked governance detection for the MCP proxy.

The gap this closes. Two Vaara layers can govern the same call without
either knowing about the other: the Claude Code hook (``vaara hook
pre-tool-use``, which sees every tool call the client makes) and this
proxy (which sees the MCP wire traffic). Run both and every MCP call is
decided twice and written to the trail twice, under two different names:
the client's namespaced ``mcp__serena__find_symbol`` from the hook, and
the bare ``find_symbol`` from the proxy.

Nothing was wrong in either layer, which is why a code-level audit does
not find it. The trail stays hash-linked and every verifier still passes;
it simply counts one action as two. The proxy has the evidence to notice
(``CLAUDECODE`` in its environment, and the hook registration readable
via ``CLAUDE_PROJECT_DIR``) and did not look.

Detection only. The proxy does not silently drop records when it finds a
second layer, because quietly changing what lands in someone's evidence
trail is the same class of mistake as quietly duplicating it. It says
which layer is doubling and lets the operator remove one.
"""
from __future__ import annotations

import json

from vaara.integrations.mcp_proxy import detect_stacked_governance


def _write_settings(path, commands, event="PreToolUse"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "hooks": {event: [{"hooks": [{"type": "command", "command": c} for c in commands]}]}
    }))


def test_no_detection_outside_claude_code(tmp_path):
    """Nothing to stack against when the client is not Claude Code."""
    _write_settings(tmp_path / ".claude" / "settings.json", ["vaara hook pre-tool-use"])
    assert detect_stacked_governance(
        env={"HOME": str(tmp_path), "CLAUDE_PROJECT_DIR": str(tmp_path)},
    ) is None


def test_no_detection_when_no_vaara_hook(tmp_path):
    """Claude Code alone is not a second governance layer."""
    _write_settings(tmp_path / ".claude" / "settings.json", ["rtk hook claude", "python3 other.py"])
    assert detect_stacked_governance(
        env={"CLAUDECODE": "1", "HOME": str(tmp_path), "CLAUDE_PROJECT_DIR": str(tmp_path)},
    ) is None


def test_detects_user_level_hook(tmp_path):
    _write_settings(
        tmp_path / ".claude" / "settings.json",
        ["rtk hook claude", "/home/claude/.local/bin/vaara hook pre-tool-use"],
    )
    found = detect_stacked_governance(
        env={"CLAUDECODE": "1", "HOME": str(tmp_path), "CLAUDE_PROJECT_DIR": str(tmp_path)},
    )
    assert found is not None
    assert "vaara hook pre-tool-use" in found
    assert "settings.json" in found


def test_detects_project_local_settings(tmp_path):
    """settings.local.json counts; it is where per-box wiring usually lands."""
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text("{}")
    _write_settings(proj / ".claude" / "settings.local.json", ["vaara hook pre-tool-use"])
    found = detect_stacked_governance(
        env={"CLAUDECODE": "1", "HOME": str(home), "CLAUDE_PROJECT_DIR": str(proj)},
    )
    assert found is not None
    assert "settings.local.json" in found


def test_post_tool_use_alone_does_not_count(tmp_path):
    """Only the deciding hook stacks. PostToolUse records an outcome, it does
    not decide the call, so it is not a second decision point."""
    _write_settings(
        tmp_path / ".claude" / "settings.json",
        ["vaara hook post-tool-use"],
        event="PostToolUse",
    )
    assert detect_stacked_governance(
        env={"CLAUDECODE": "1", "HOME": str(tmp_path), "CLAUDE_PROJECT_DIR": str(tmp_path)},
    ) is None


def test_unreadable_settings_never_raises(tmp_path):
    """Detection is advisory. A malformed or unreadable settings file must not
    take the proxy down, because the proxy owns the MCP server's process."""
    bad = tmp_path / ".claude" / "settings.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{ this is not json")
    assert detect_stacked_governance(
        env={"CLAUDECODE": "1", "HOME": str(tmp_path), "CLAUDE_PROJECT_DIR": str(tmp_path)},
    ) is None


def test_missing_project_dir_falls_back_to_home(tmp_path):
    _write_settings(tmp_path / ".claude" / "settings.json", ["vaara hook pre-tool-use"])
    found = detect_stacked_governance(env={"CLAUDECODE": "1", "HOME": str(tmp_path)})
    assert found is not None
