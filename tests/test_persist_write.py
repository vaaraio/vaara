"""Writes into agent-persistent locations, and the outbound-then-persist sequence.

Skills, memory files, instruction files and hook configuration are loaded again
in later sessions, so content an agent fetched and then wrote there steers every
session after it (arXiv 2610.13227). These tests pin which writes carry the
persist label and that an outbound call followed by such a write raises the
sequence signal.
"""

from __future__ import annotations

import pytest

from vaara.pipeline import InterceptionPipeline
from vaara.scorer._param_signals import persistent_write
from vaara.scorer.adaptive import BUILTIN_SEQUENCES, PERSIST_LABEL, action_labels


def _seq(pipeline: InterceptionPipeline, agent: str, calls: list[tuple[str, dict]]):
    result = None
    for tool, params in calls:
        result = pipeline.intercept(agent_id=agent, tool_name=tool, parameters=params)
    return result


FETCH = ("WebFetch", {"url": "https://example.invalid/readme"})
SKILL_WRITE = ("Write", {"file_path": "/home/u/.claude/skills/deploy/SKILL.md",
                         "content": "always run setup.sh first"})
PLAIN_WRITE = ("Write", {"file_path": "/home/u/proj/notes.md", "content": "x"})


class TestPersistentWrite:
    @pytest.mark.parametrize("tool, action, params", [
        ("Write", "data.write", {"file_path": "/h/.claude/skills/x/SKILL.md"}),
        ("Edit", None, {"file_path": "/h/proj/CLAUDE.md"}),
        ("MultiEdit", None, {"file_path": "/h/proj/AGENTS.md"}),
        ("str_replace_editor", None, {"path": "/h/.cursor/rules/style.mdc"}),
        ("write_file", "data.write", {"path": "/h/.claude/projects/p/memory/MEMORY.md"}),
        ("Edit", None, {"file_path": "/h/.claude/settings.json"}),
        ("Write", None, {"file_path": "/h/proj/.mcp.json"}),
        ("Write", None, {"file_path": "C:\\Users\\u\\.codex\\config.toml"}),
        ("Bash", None, {"command": "echo 'run x' >> ~/.claude/CLAUDE.md"}),
        ("Bash", None, {"command": "curl -s https://e.invalid/s | tee .claude/skills/s/SKILL.md"}),
        ("Bash", None, {"command": "cp /tmp/hook.py ~/.claude/hooks/hook.py"}),
        ("Bash", None, {"command": "sed -i 's/a/b/' AGENTS.md"}),
        ("Bash", None, {"command": "bash -c 'printf x >.gemini/GEMINI.md'"}),
    ])
    def test_persistent(self, tool, action, params):
        assert persistent_write(tool, action, params)

    @pytest.mark.parametrize("tool, action, params", [
        ("Read", "data.read", {"file_path": "/h/proj/CLAUDE.md"}),
        ("Write", "data.write", {"file_path": "/h/proj/src/memory/cache.py"}),
        ("Write", "data.write", {"file_path": "/h/proj/notes.md",
                                 "content": "see ~/.claude/skills/x/SKILL.md"}),
        ("Edit", None, {"file_path": "/h/proj/.vscode/settings.json"}),
        ("Bash", None, {"command": "cat AGENTS.md"}),
        ("Bash", None, {"command": "cp AGENTS.md /tmp/backup.md"}),
        ("Grep", None, {"pattern": "CLAUDE.md"}),
    ])
    def test_not_persistent(self, tool, action, params):
        assert persistent_write(tool, action, params) is None

    def test_label(self):
        labels = action_labels("Edit", {"action_name": None,
                                        "parameters": {"file_path": "/h/AGENTS.md"}})
        assert PERSIST_LABEL in labels


class TestOutboundThenPersist:
    def test_builtin_registered(self):
        assert "outbound_then_persist" in {p.name for p in BUILTIN_SEQUENCES}

    def test_fetch_then_skill_write_fires(self):
        r = _seq(InterceptionPipeline(), "a", [FETCH, SKILL_WRITE])
        assert r.signals["sequence_pattern"] >= 0.5

    def test_fetch_then_plain_write_does_not_fire(self):
        r = _seq(InterceptionPipeline(), "a", [FETCH, PLAIN_WRITE])
        assert r.signals["sequence_pattern"] < 0.5

    def test_skill_write_alone_does_not_fire(self):
        r = _seq(InterceptionPipeline(), "a", [SKILL_WRITE])
        assert r.signals["sequence_pattern"] < 0.5

    def test_order_matters(self):
        r = _seq(InterceptionPipeline(), "a", [SKILL_WRITE, FETCH])
        assert r.signals["sequence_pattern"] < 0.5
