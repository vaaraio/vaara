"""A shell rule on the MCP content path fires on a command, not on prose about one.

After #790 fixed path rules, shell rules still ran over every string
argument. On 2026-09-24 a vaara-memory ``mem_save`` whose note described a
destructive command was refused as ``rm_rf_root``. A shell rule now reads an
argument only when it is command-shaped: under a command-like key, or a
one-line string that starts with the command (after launchers such as sudo,
or after a shell separator).

Commands are assembled at runtime: a governed session writing this file
would otherwise trip the rules under test.
"""
from __future__ import annotations

import io
import json
import sys

import pytest

from vaara.deny_rules import load_deny_rules, match_deny_rule_any_field
from vaara.integrations import claude_code_hooks as hooks

RM_ROOT = " ".join(["rm", "-rf", "/"])
DD_DISK = " ".join(["dd", "if=/dev/zero", "of=/dev/" + "sda"])

NOTE = (f"Lesson 24.9: the guard refused {RM_ROOT} in a Bash call, which is "
        f"right, and {DD_DISK} is in the same family.")


@pytest.fixture
def rules():
    return load_deny_rules()


def test_a_note_that_describes_a_destructive_command_passes(rules):
    assert match_deny_rule_any_field(
        rules, {"title": "guard lesson", "content": NOTE, "type": "discovery"}) is None


def test_a_multiline_note_with_the_command_on_its_own_line_passes(rules):
    assert match_deny_rule_any_field(
        rules, {"content": f"The refused call was:\n{RM_ROOT}\nand that was right."}) is None


@pytest.mark.parametrize("args", [
    {"command": RM_ROOT},
    {"cmd": f"echo hi; {RM_ROOT}"},
    {"script": f"set -e\n{RM_ROOT}\n"},
    {"args": ["bash", "-c", RM_ROOT]},
    {"input": RM_ROOT},
    {"nested": {"shell_command": RM_ROOT}},
])
def test_a_command_argument_is_still_refused(rules, args):
    assert match_deny_rule_any_field(rules, args)[0] == "rm_rf_root"


@pytest.mark.parametrize("text", [
    RM_ROOT,
    "  " + RM_ROOT,
    "sudo " + RM_ROOT,
    "FOO=1 sudo " + RM_ROOT,
    "cd /tmp && " + RM_ROOT,
    "true; " + RM_ROOT,
    "echo start | " + RM_ROOT,
])
def test_a_single_command_string_is_refused_under_any_key(rules, text):
    assert match_deny_rule_any_field(rules, {"q": text})[0] == "rm_rf_root"


def test_a_rule_matching_an_argument_fires_after_the_command_name(rules):
    shadow = "cat /etc/" + "shadow"
    assert match_deny_rule_any_field(rules, {"q": shadow})[0] == "etc_shadow_read"
    assert match_deny_rule_any_field(
        rules, {"q": f"Checked whether {shadow} was readable."}) is None


def test_the_hook_lets_the_memory_save_through(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_PLUGIN_AUDIT_DB", str(tmp_path / "audit.db"))
    monkeypatch.setenv("VAARA_PLUGIN_SHADOW", "0")
    monkeypatch.setenv("VAARA_PLUGIN_APPROVALS", "0")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "tool_name": "mcp__vaara-memory__mem_save",
        "tool_input": {"title": "guard lesson", "content": NOTE},
        "session_id": "s",
    })))
    assert hooks.run_pre_tool_use() == 0
