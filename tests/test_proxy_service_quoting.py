# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The generated systemd unit has to survive values containing spaces.

The launchd path renders ProgramArguments as a plist array, so every argument
stays its own token no matter what is in it. The systemd path joined the same
argv with spaces into a single ExecStart line, which systemd re-splits on
whitespace. A home directory with a space in it — the default on macOS and
common enough on Linux — produced a unit that starts the proxy with the wrong
arguments, or fails to start at all.
"""

from __future__ import annotations

import plistlib
import shlex


from vaara.integrations.proxy_service import (
    render_launchd_plist,
    render_systemd_unit,
)

SPACED_DB = "/home/Henri Sirkkavaara/.vaara/trail/audit.db"


def _exec_start(unit: str) -> str:
    for line in unit.splitlines():
        if line.startswith("ExecStart="):
            return line[len("ExecStart="):]
    raise AssertionError("unit has no ExecStart")


def _tokens(unit: str) -> list[str]:
    """How systemd splits ExecStart: whitespace, honouring double quotes."""
    return shlex.split(_exec_start(unit))


def test_path_with_a_space_stays_one_argument():
    unit = render_systemd_unit(
        vaara_bin="/usr/bin/vaara", listen="127.0.0.1:8780",
        upstream="http://127.0.0.1:11434", trail_db=SPACED_DB,
    )
    tokens = _tokens(unit)
    assert SPACED_DB in tokens, tokens
    assert tokens[tokens.index("--trail") + 1] == SPACED_DB


def test_binary_path_with_a_space_stays_one_argument():
    unit = render_systemd_unit(
        vaara_bin="/opt/my tools/vaara", listen="127.0.0.1:8780",
        upstream="http://127.0.0.1:11434", trail_db="/tmp/a.db",
    )
    assert _tokens(unit)[0] == "/opt/my tools/vaara"


def test_allow_pattern_with_a_space_stays_one_argument():
    unit = render_systemd_unit(
        vaara_bin="vaara", listen="127.0.0.1:8780",
        upstream="http://127.0.0.1:11434", trail_db="/tmp/a.db",
        enforce=True, allow=["read file", "tx.*"],
    )
    tokens = _tokens(unit)
    assert "read file" in tokens
    assert "tx.*" in tokens


def test_a_value_cannot_inject_extra_arguments():
    """An unquoted value ending in a flag would become a real flag."""
    unit = render_systemd_unit(
        vaara_bin="vaara", listen="127.0.0.1:8780",
        upstream="http://127.0.0.1:11434",
        trail_db="/tmp/a.db --enforce",
    )
    tokens = _tokens(unit)
    assert "--enforce" not in tokens
    assert "/tmp/a.db --enforce" in tokens


def test_embedded_quotes_survive():
    unit = render_systemd_unit(
        vaara_bin="vaara", listen="127.0.0.1:8780",
        upstream="http://127.0.0.1:11434",
        trail_db='/tmp/we"ird.db',
    )
    assert '/tmp/we"ird.db' in _tokens(unit)


def test_ordinary_values_are_left_unquoted():
    """Quoting everything would work but makes the unit unreadable."""
    unit = render_systemd_unit(
        vaara_bin="vaara", listen="127.0.0.1:8780",
        upstream="http://127.0.0.1:11434", trail_db="/tmp/a.db",
    )
    assert '"' not in _exec_start(unit)


def test_launchd_was_always_safe_and_still_is():
    """The plist array is the shape the systemd path now matches."""
    text = render_launchd_plist(
        vaara_bin="vaara", listen="127.0.0.1:8780",
        upstream="http://127.0.0.1:11434", trail_db=SPACED_DB,
        log_dir="/tmp/logs",
    )
    argv = plistlib.loads(text.encode())["ProgramArguments"]
    assert SPACED_DB in argv


def test_both_renderers_agree_on_the_argument_list():
    kwargs = dict(
        vaara_bin="vaara", listen="127.0.0.1:8780",
        upstream="http://127.0.0.1:11434", trail_db=SPACED_DB,
        enforce=True, allow=["a b"], approvals_dir="/tmp/ap provals",
    )
    plist_argv = plistlib.loads(
        render_launchd_plist(log_dir="/tmp/logs", **kwargs).encode()
    )["ProgramArguments"]
    assert _tokens(render_systemd_unit(**kwargs)) == plist_argv
