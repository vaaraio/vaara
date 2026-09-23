# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""interpreter_config_write blocks writes to the harness config and lets reads through.

The rule used to count any ``>`` as a write, so a read-only interpreter call
with ``2>&1`` or ``2>/dev/null`` was blocked, and so was a command that read
one config file and later listed the hooks directory. A shell redirect into
these files is caught by harness_config_shell_write, which the redirect cases
below assert. On the write side the rule knew only open mode ``'w'``, so
appending, read-write and exclusive-create modes passed, as did os.rename and
os.replace.
"""
from __future__ import annotations

import os

import pytest

from vaara.deny_rules import load_deny_rules, match_deny_rule

RULES = load_deny_rules()
# Built, not written out, so this file's own text is not a rule payload.
CFG = "/home/u/" + ".".join(["", "claude"]) + "/"
SETTINGS = CFG + "settings.json"


@pytest.fixture(autouse=True)
def _no_lifts(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("VAARA_ALLOW_"):
            monkeypatch.delenv(key)


def _hit(command: str) -> str | None:
    hit = match_deny_rule(RULES, "Bash", {"command": command})
    return hit[0] if hit else None


@pytest.mark.parametrize("command", [
    f"python3 -c \"import json; print(json.load(open('{SETTINGS}')))\" 2>&1",
    f"python3 -c \"print(open('{SETTINGS}').read())\" 2>/dev/null",
    f"cd {CFG}; python3 -c \"import json; print(json.load(open('settings.json')))\" 2>&1; ls {CFG}hooks/",
    f"python3 -c \"print(open('{SETTINGS}', 'r').read())\"",
    f"node -e \"console.log(require('{SETTINGS}'))\"",
])
def test_reads_are_not_blocked(command: str) -> None:
    assert _hit(command) is None


@pytest.mark.parametrize("command", [
    f"python3 -c \"open('{SETTINGS}', 'w').write('x')\"",
    f"python3 -c \"open('{SETTINGS}', 'a').close()\"",
    f"python3 -c \"f = open('{SETTINGS}', 'r+'); f.seek(0)\"",
    f"python3 -c \"open('{SETTINGS}', 'wb').close()\"",
    f"python3 -c \"open('{SETTINGS}', 'x').close()\"",
    f"python3 -c \"open('{SETTINGS}', mode='a').close()\"",
    f"python3 -c \"import os; os.rename('/tmp/s', '{SETTINGS}')\"",
    f"python3 -c \"import os; os.replace('/tmp/s', '{SETTINGS}')\"",
    f"python3 -c \"import os; p = '{SETTINGS}'; os.remove(p)\"",
    f"python3 -c \"from pathlib import Path; Path('{SETTINGS}').write_text('{{}}')\"",
    f"python3 -c \"import shutil; shutil.rmtree('{CFG}hooks/')\"",
])
def test_interpreter_writes_are_blocked(command: str) -> None:
    assert _hit(command) == "interpreter_config_write"


@pytest.mark.parametrize("command", [
    f"python3 gen.py > {SETTINGS}",
    f"python3 gen.py >> {SETTINGS}",
])
def test_redirects_into_config_are_still_blocked_by_the_shell_rule(command: str) -> None:
    assert _hit(command) == "harness_config_shell_write"


def test_the_lift_lifts_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAARA_ALLOW_HARNESS_EDIT", "1")
    assert _hit(f"python3 -c \"open('{SETTINGS}', 'a').close()\"") is None
