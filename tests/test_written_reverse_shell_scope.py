# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""written_reverse_shell fires on a socket handed to a shell, not on any client.

The socket branch matched a ``socket.socket`` call followed by a connect, so
every file that opened a client socket was refused as a reverse shell,
Vaara's own OS guard client among them. What makes a reverse shell is the
connected socket wired to a process's stdio, so the branch now needs that
too. Samples come from files in the tree and from the rule's own pattern, so
no attack string lives here (the same choice test_every_deny_rule_fires.py
makes).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vaara.deny_rules import load_deny_rules, match_deny_rule

ROOT = Path(__file__).resolve().parents[1]
RULES = load_deny_rules()
RULE = next(r for r in RULES if r["id"] == "written_reverse_shell")


@pytest.fixture(autouse=True)
def _no_lifts(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VAARA_ALLOW_"):
            monkeypatch.delenv(key)


def _fires(content: str) -> bool:
    return match_deny_rule([RULE], "Write", {"file_path": "x.py", "content": content}) is not None


@pytest.mark.parametrize("path", [
    "src/vaara/oslayer/client.py",
    "src/vaara/oslayer/forward.py",
    "tests/test_oslayer_forward.py",
])
def test_a_plain_socket_client_is_not_a_reverse_shell(path):
    assert not _fires((ROOT / path).read_text())


def _socket_branch() -> str:
    pattern = RULE["pattern"]
    start = pattern.index(r"socket\.socket")
    assert pattern.endswith(")"), pattern
    return pattern[start:-1]


def test_the_socket_branch_needs_stdio_wiring():
    branch = _socket_branch()
    for wiring in (r"dup2\(", r"pty\.spawn\(", r"fileno\("):
        assert wiring in branch, (wiring, branch)


@settings(max_examples=50, database=None, deadline=None,
          suppress_health_check=list(HealthCheck))
@given(st.from_regex(re.compile(_socket_branch())))
def test_the_socket_branch_still_fires(sample):
    assert _fires(sample)
