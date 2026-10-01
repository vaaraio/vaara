"""Matching Claude Code deny rules: certain matches only, a miss asks."""
from __future__ import annotations

from vaara.integrations.claude_code_permissions import (
    deny_rules,
    matching_deny_rule,
    subcommands,
)

_DEL = "r" + "m"


def test_subcommands_split_on_operators_outside_quotes():
    assert subcommands(f"cd a && {_DEL} x; ls | wc -l") == [
        "cd a", f"{_DEL} x", "ls", "wc -l",
    ]
    assert subcommands(f"echo 'a; {_DEL} x'") == [f"echo a; {_DEL} x"]
    assert subcommands("echo 'unclosed") == []


def test_bash_patterns():
    rules = [f"Bash({_DEL} *)"]
    assert matching_deny_rule("Bash", {"command": f"{_DEL} -f a"}, rules)
    assert not matching_deny_rule("Bash", {"command": f"{_DEL}dir a"}, rules)
    assert not matching_deny_rule("Bash", {"command": "ls"}, rules)
    legacy = [f"Bash({_DEL}:*)"]
    assert matching_deny_rule("Bash", {"command": f"{_DEL} a"}, legacy)
    assert not matching_deny_rule("Bash", {"command": f"{_DEL}dir a"}, legacy)


def test_tool_and_mcp_rules():
    assert matching_deny_rule("WebFetch", {}, ["WebFetch"]) == "WebFetch"
    assert matching_deny_rule("mcp__s__t", {}, ["mcp__s"]) == "mcp__s"
    assert matching_deny_rule("mcp__s__t", {}, ["mcp__s__t"]) == "mcp__s__t"
    assert not matching_deny_rule("mcp__sx__t", {}, ["mcp__s"])
    # Path and domain specifiers are not interpreted: no certain match.
    assert not matching_deny_rule("Read", {"file_path": ".env"}, ["Read(./.env)"])


def test_deny_rules_reads_user_and_project_settings(tmp_path):
    import json

    home, project = tmp_path / "home", tmp_path / "proj"
    for base, rule in ((home, "WebFetch"), (project, "Bash(git push *)")):
        (base / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"deny": ["WebFetch"]}}))
    (project / ".claude" / "settings.local.json").write_text(
        json.dumps({"permissions": {"deny": ["Bash(git push *)", 3]}}))
    env = {"HOME": str(home), "CLAUDE_PROJECT_DIR": str(project)}
    assert deny_rules(env) == ["WebFetch", "Bash(git push *)"]
    # Without the project variable, the event's cwd names the project.
    assert deny_rules({"HOME": str(home)}, cwd=str(project)) == [
        "WebFetch", "Bash(git push *)",
    ]
