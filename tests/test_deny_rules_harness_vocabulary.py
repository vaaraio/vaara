"""The deny rules recognise Codex's and Gemini CLI's tool names.

The rules name tools in Claude Code's vocabulary. Codex already hands hooks
its shell tools as ``Bash`` with ``command``, so those matched, but its file
edits arrive as ``apply_patch`` and its sub-agents as ``spawn_agent``, and
every Gemini CLI tool has its own name and input shape. None of those matched
a rule by name. ``HARNESS_ALIASES`` translates them; these tests drive each
translation through the shipped rule file.
"""
from __future__ import annotations

import pytest

from vaara.deny_rules import HARNESS_ALIASES, load_deny_rules, match_deny_rule

RULES = load_deny_rules()
# Built, not written out, so this file itself is not a metadata-address payload.
METADATA = ".".join(["169", "254", "169", "254"])


def _hit(tool: str, tool_input: dict):
    return match_deny_rule(RULES, tool, tool_input)


@pytest.mark.parametrize("tool, tool_input", [
    ("run_shell_command", {"command": f"curl http://{METADATA}/latest/"}),
    ("write_file", {"file_path": "/home/u/.bashrc", "content": "echo hi"}),
    ("replace", {"file_path": "/home/u/.ssh/authorized_keys",
                 "old_string": "a", "new_string": "ssh-ed25519 AAAA"}),
    ("read_file", {"file_path": "/home/u/project/.env"}),
    ("read_many_files", {"include": ["src/app.py", "/home/u/.aws/credentials"]}),
    ("web_fetch", {"prompt": f"summarise http://{METADATA}/latest/"}),
    ("spawn_agent", {"message": "do the thing"}),
    ("apply_patch", {"command": "*** Begin Patch\n*** Add File: /home/u/.zshrc\n"
                                "+echo hi\n*** End Patch"}),
])
def test_the_foreign_name_hits_the_rule_its_claude_code_twin_hits(tool, tool_input):
    assert _hit(tool, tool_input) is not None


@pytest.mark.parametrize("tool, tool_input", [
    ("run_shell_command", {"command": "ls -la"}),
    ("write_file", {"file_path": "src/app.py", "content": "print(1)"}),
    ("read_file", {"file_path": "README.md"}),
    ("apply_patch", {"command": "*** Begin Patch\n*** Update File: src/app.py\n"
                                "@@\n-a\n+b\n*** End Patch"}),
])
def test_ordinary_calls_pass(tool, tool_input):
    assert _hit(tool, tool_input) is None


def test_a_lift_still_lifts_through_an_alias(monkeypatch):
    assert _hit("spawn_agent", {}) is not None
    monkeypatch.setenv("VAARA_ALLOW_SPAWN", "1")
    assert _hit("spawn_agent", {}) is None


def test_a_patch_is_checked_per_file():
    patch = ("*** Begin Patch\n*** Update File: src/ok.py\n@@\n+x\n"
             "*** Add File: /home/u/.profile\n+y\n*** End Patch")
    assert _hit("apply_patch", {"command": patch}) is not None


def test_an_alias_never_shadows_a_rule_that_names_the_tool():
    rules = [{"id": "own", "tools": ["replace"], "fields": ["x"],
              "pattern": "zzz", "message": "own rule"}]
    assert match_deny_rule(rules, "replace", {"x": "zzz"}) == ("own", "own rule")
    assert match_deny_rule(rules, "replace", {"file_path": "/home/u/.bashrc"}) is None


def test_every_alias_targets_a_tool_the_rules_name():
    named = {t for r in RULES for t in r.get("tools", [])}
    for name, translate in HARNESS_ALIASES.items():
        for target, _ in translate({"command": "*** Add File: x\n"}):
            assert target in named, (name, target)
