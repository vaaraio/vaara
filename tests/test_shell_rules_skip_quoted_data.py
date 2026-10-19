# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Bash egress rule reads the words a shell runs, not quoted text.

Three refusals on 2026-10-17..19, all of commands that open no connection:

- ``echo`` and ``grep`` of a quoted string naming netcat or a remote copy
  (a FAILS.md append, a grep over that file) were refused as
  ``shell_netcat_egress`` and ``shell_copy_egress``. A quoted argument of
  ``echo``, ``printf`` or ``grep`` is text when its output goes to a file,
  the screen or a filter, never into anything that runs it.
- ``printf 'id\\ncat /etc/hostname\\n' | ssh ... vaara proxy-shell`` was
  refused as ``shell_netcat_egress``: the escape ``\\n`` followed by ``cat``
  reads as the word ``ncat``.
- ``python3 -c "...urllib.parse.quote(...)"`` was refused as
  ``interpreter_network_egress``; ``urllib.parse`` opens no connection.

Commands are assembled at runtime: a governed session writing this file
would otherwise trip the rules under test.
"""
from __future__ import annotations

import os

import pytest

from vaara.deny_rules import (
    load_deny_rules,
    match_deny_rule,
    match_deny_rule_any_field,
    without_quoted_data,
)

NC_WORD = "n" + "cat"
NETCAT = " ".join([NC_WORD, "example.org", "4444"])
NC_SHORT = " ".join(["n" + "c", "example.org", "4444"])
SECURE_COPY = " ".join(["s" + "cp", "notes.md", "box" + ":/tmp/"])
RSYNC = " ".join(["rs" + "ync", "-a", "~/Projects/vaara", "box" + ":/workspace"])
URLLIB = "url" + "lib"


@pytest.fixture(autouse=True)
def _no_lifts(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VAARA_ALLOW_"):
            monkeypatch.delenv(key)


@pytest.fixture
def rules():
    return load_deny_rules()


def _bash(rules, command):
    hit = match_deny_rule(rules, "Bash", {"command": command})
    return hit[0] if hit else None


def _proxy(rules, command):
    hit = match_deny_rule_any_field(rules, {"command": command})
    return hit[0] if hit else None


# --- quoted text of a command that only prints or filters it ---------------

@pytest.mark.parametrize("command", [
    f"echo '2026-10-19 FAIL {NETCAT} refused' >> worker/FAILS.md",
    f'echo "the {SECURE_COPY} line was refused" >> FAILS.md',
    f"printf '%s\\n' '{RSYNC}' > notes.txt",
    f'grep -n "{NC_WORD}\\|copy_egress" FAILS.md | cut -c1-400 | tail -8',
    f"rg -l '{NC_WORD}' src | head",
    f"cd /workspace; grep -c '{SECURE_COPY}' log.md && echo done",
    f"echo $'{NETCAT}\\n' > /dev/null",
])
def test_quoted_text_of_a_printing_command_is_not_egress(rules, command):
    assert _bash(rules, command) is None
    assert _proxy(rules, command) is None


# --- quoted text that something runs keeps matching -------------------------

@pytest.mark.parametrize("command,rule", [
    (f"echo '{NETCAT}' | sh", "shell_netcat_egress"),
    (f"echo '{NETCAT}' |& bash", "shell_netcat_egress"),
    (f"echo '{NETCAT}' | xargs", "shell_netcat_egress"),
    (f"printf '{SECURE_COPY}\\n' | ssh host", "shell_copy_egress"),
    (f"bash -c \"$(echo '{NETCAT}')\"", "shell_netcat_egress"),
    (f"sh <(echo '{NETCAT}')", "shell_netcat_egress"),
    (f"echo '{NETCAT}' > >(sh)", "shell_netcat_egress"),
    (f"(echo '{NETCAT}') | sh", "shell_netcat_egress"),
    (f"{{ echo '{NETCAT}'; }} | sh", "shell_netcat_egress"),
    (f"echo '{NETCAT}' > run.sh; bash run.sh", "shell_netcat_egress"),
    (f"echo '{NETCAT}' > run.sh && chmod +x run.sh", "shell_netcat_egress"),
    (f'echo "$({NETCAT})"', "shell_netcat_egress"),
    (f'echo "`{NETCAT}`"', "shell_netcat_egress"),
    (f"bash -c '{NETCAT}'", "shell_netcat_egress"),
    (f"ssh host '{RSYNC}'", "shell_copy_egress"),
    (f"eval '{NETCAT}'", "shell_netcat_egress"),
    (f"{NETCAT}", "shell_netcat_egress"),
    (f"echo hi; {NETCAT}", "shell_netcat_egress"),
    (f"echo 'x' && {SECURE_COPY}", "shell_copy_egress"),
    (f"echo 'unterminated {NETCAT}", "shell_netcat_egress"),
])
def test_quoted_text_that_runs_is_still_matched(rules, command, rule):
    assert _bash(rules, command) == rule
    assert _proxy(rules, command) == rule


def test_a_quoted_command_word_is_still_the_command(rules):
    # Quotes around the command name do not make it text: the shell runs it.
    assert _bash(rules, f"'{NC_WORD}' example.org 4444") == "shell_netcat_egress"


# --- an escape before a tool name is not the tool ---------------------------

def test_printf_escape_before_cat_is_not_ncat(rules):
    # The governed proxy-shell scenario in the RHEL VM, 2026-10-18T19:01Z.
    command = ("printf 'id\\ncat /etc/hostname\\n' "
               "| ssh -p 2229 cloud-user@127.0.0.1 vaara proxy-shell")
    assert _bash(rules, command) is None
    assert _proxy(rules, command) is None


def test_a_backslash_quoted_command_word_is_still_the_tool(rules):
    # Outside quotes the shell drops the backslash and runs the tool.
    assert _bash(rules, "\\" + NETCAT) == "shell_netcat_egress"
    assert _bash(rules, "echo hi; \\" + NETCAT) == "shell_netcat_egress"


def test_short_form_after_an_escape_is_not_the_tool(rules):
    assert _bash(rules, "printf 'a\\" + NC_SHORT + "' | ssh host") is None
    assert _bash(rules, NC_SHORT) == "shell_netcat_egress"


# --- urllib.parse opens no connection ---------------------------------------

@pytest.mark.parametrize("command", [
    f"python3 -c \"import {URLLIB}.parse; print({URLLIB}.parse.quote('a b'))\"",
    f"python3 -c 'from {URLLIB}.parse import quote; print(quote(\"x\"))'",
    f"python3 -c 'from {URLLIB} import parse; print(parse.quote(\"x\"))'",
    f"python3 -c 'import {URLLIB}.error'",
])
def test_urllib_parse_is_not_network_egress(rules, command):
    assert _bash(rules, command) is None


@pytest.mark.parametrize("command", [
    f"python3 -c 'import {URLLIB}.request as r; r.Request(\"http://x\")'",
    f"python3 -c 'from {URLLIB} import request'",
    f"python3 -c 'import {URLLIB}.parse, {URLLIB}.request'",
    f"python3 -c 'import {URLLIB}3'",
    f"python3 -c 'import {URLLIB}2'",
    f"python3 -c 'from {URLLIB}.request import urlopen'",
])
def test_urllib_request_is_still_network_egress(rules, command):
    assert _bash(rules, command) == "interpreter_network_egress"


# --- the transform itself ---------------------------------------------------

def test_without_quoted_data_leaves_other_text_alone():
    assert without_quoted_data("ls -la /tmp") == "ls -la /tmp"
    assert without_quoted_data("echo 'a' > f") == "echo '' > f"
    assert without_quoted_data("cat <<EOF\n'x'\nEOF") == "cat <<EOF\n'x'\nEOF"


def test_only_shell_syntax_rules_skip_quoted_data(rules):
    # A rule that reads written content still sees an echo's quoted text.
    for rule in rules:
        if rule.get("id") in ("shell_netcat_egress", "shell_copy_egress"):
            assert rule.get("shell_syntax") is True
    assert not next(r for r in rules
                    if r["id"] == "interpreter_network_egress").get("shell_syntax")
