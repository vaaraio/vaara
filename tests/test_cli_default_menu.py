# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bare ``vaara`` with no subcommand drops an interactive user into the menu,
while non-interactive callers (scripts, pipes, CI) keep the usage listing so
no automated caller changes behaviour.
"""
import vaara.cli as cli
import vaara.menu as menu


def test_bare_vaara_interactive_launches_menu(monkeypatch):
    monkeypatch.setattr(cli, "_is_interactive", lambda: True, raising=False)
    calls = {"menu": 0}

    def fake_menu(*a, **k):
        calls["menu"] += 1
        return 7

    monkeypatch.setattr(menu, "run_menu", fake_menu)

    rc = cli.main([])

    assert calls["menu"] == 1
    assert rc == 7


def test_bare_vaara_noninteractive_prints_usage(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_is_interactive", lambda: False, raising=False)
    calls = {"menu": 0}

    def fake_menu(*a, **k):
        calls["menu"] += 1
        return 0

    monkeypatch.setattr(menu, "run_menu", fake_menu)

    rc = cli.main([])

    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert calls["menu"] == 0
    assert rc == 0
    assert "COMMAND" in text or "usage" in text.lower()
