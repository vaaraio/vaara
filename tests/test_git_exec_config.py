# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Git configuration that runs a program on ordinary git commands.

A hook in .git/hooks was already refused. The same execution is reachable
through configuration: core.fsmonitor runs on `git status`, core.pager on
`git log`, a diff driver on `git diff`, core.hooksPath points the hooks
somewhere else. An agent that sets one in a repository leaves a program that
runs the next time anyone, or any tool, reads the repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "claude-code-vaara-governance"
sys.path.insert(0, str(PLUGIN / "hooks"))
from _deny_patterns import match_deny_rule  # noqa: E402

RULES = json.loads(
    (ROOT / "src" / "vaara" / "integrations" / "claude_code_deny.json").read_text(encoding="utf-8")
)["rules"]

REFUSED = [
    ("Bash", "git config core.fsmonitor ./x.sh"),
    ("Bash", "git config --local core.pager './x.sh'"),
    ("Bash", "git config diff.external ./x.sh"),
    ("Bash", "git config core.hooksPath .githooks"),
    ("Bash", "git config --global core.sshCommand 'ssh -o ProxyCommand=./x.sh'"),
    ("Bash", "git config set core.fsmonitor ./x.sh"),
    ("Bash", "git config diff.png.textconv ./x.sh"),
    ("Bash", "git config filter.lfs.smudge ./x.sh"),
    ("Bash", "git config merge.ours.driver ./x.sh"),
    ("Bash", "git config pager.log ./x.sh"),
    ("Bash", "git -C /repo config Core.FSMonitor ./x.sh"),
    ("Bash", "git -c core.fsmonitor=./x.sh status"),
    ("Bash", "git -c diff.external=./x.sh diff"),
    ("Bash", "cd /repo && git -c core.pager=./x.sh log"),
    ("Bash", "printf '[core]\\n\\tfsmonitor = ./x.sh\\n' >> .git/config"),
    ("Bash", "echo './x.sh' | tee -a .git/config"),
    ("Bash", "sed -i 's/pager = less/pager = .\\/x.sh/' .git/config"),
    ("Bash", "echo ./x.sh > .git/hooks/commit-msg"),
    ("Bash", "ln -sf ../../x.sh .git/hooks/post-checkout"),
]

REFUSED_FILES = [
    ("Write", "/repo/.git/config"),
    ("Edit", "/repo/.git/config"),
    ("Write", "/repo/.git/config.worktree"),
    ("Write", "/home/u/.gitconfig"),
    ("Write", "/home/u/.config/git/config"),
    ("Write", "/repo/.git/hooks/pre-commit"),
    ("Write", "/repo/.git/hooks/commit-msg"),
    ("Edit", "/repo/.git/hooks/post-rewrite"),
    ("Write", "/repo/.git/hooks/reference-transaction"),
    ("Write", "/repo/.git/hooks/fsmonitor-watchman"),
]

ORDINARY = [
    "git status",
    "git diff --stat",
    "git -c core.pager=cat log --oneline -5",
    "git -c core.pager=less log",
    "git -c core.editor=true rebase --continue",
    "git -c core.fsmonitor=false status",
    "git -c core.hooksPath=/dev/null commit -q -m 'notes'",
    "git config user.name 'A Person'",
    "git config --global user.email a@example.org",
    "git config --get core.pager",
    "git config core.pager",
    "git config --unset core.pager",
    "git config core.pager cat",
    "git config pull.rebase true",
    "git config --list",
    "cat .git/config",
    "grep fsmonitor .git/config",
    "ls .git/hooks",
    "cp .git/hooks/pre-commit.sample /tmp/look",
    "git log --format=%H -1 > /tmp/head.txt",
]

ORDINARY_FILES = [
    ("Write", "/repo/.gitignore"),
    ("Write", "/repo/.gitattributes"),
    ("Write", "/repo/docs/git/config.md"),
    ("Write", "/repo/.git/hooks/pre-commit.sample"),
    ("Write", "/repo/.githooks/pre-commit"),
]


@pytest.mark.parametrize("tool,command", REFUSED)
def test_exec_config_from_the_shell_is_refused(tool, command):
    assert match_deny_rule(RULES, tool, {"command": command}) is not None


@pytest.mark.parametrize("tool,path", REFUSED_FILES)
def test_git_config_and_hook_files_are_refused(tool, path):
    field = "content" if tool == "Write" else "new_string"
    assert match_deny_rule(RULES, tool, {"file_path": path, field: "x"}) is not None


@pytest.mark.parametrize("command", ORDINARY)
def test_ordinary_git_use_is_not_refused(command):
    assert match_deny_rule(RULES, "Bash", {"command": command}) is None


@pytest.mark.parametrize("tool,path", ORDINARY_FILES)
def test_ordinary_files_are_not_refused(tool, path):
    assert match_deny_rule(RULES, tool, {"file_path": path, "content": "x"}) is None


def test_the_lift_names_one_variable(monkeypatch):
    monkeypatch.setenv("VAARA_ALLOW_GIT_EXEC_CONFIG", "1")
    assert match_deny_rule(RULES, "Bash", {"command": "git config core.fsmonitor ./x.sh"}) is None
    assert match_deny_rule(RULES, "Write", {"file_path": "/repo/.git/config", "content": "x"}) is None


def test_codex_apply_patch_onto_git_config_is_refused():
    # Codex reports apply_patch as one file event per path; the path rules
    # see each one, so a patch that edits .git/config is refused like a Write.
    from vaara.integrations import codex

    patch = ("*** Begin Patch\n*** Update File: .git/config\n@@\n-x\n"
             "+[core]\n+\tfsmonitor = ./x.sh\n*** End Patch\n")
    payload = {"session_id": "s1", "turn_id": "t1", "transcript_path": "/x.jsonl",
               "cwd": "/w", "hook_event_name": "PreToolUse", "model": "m",
               "tool_name": "apply_patch", "tool_input": {"command": patch}}
    events = codex.to_hook_events(payload)
    assert any(match_deny_rule(RULES, e["tool_name"], e["tool_input"]) for e in events)
