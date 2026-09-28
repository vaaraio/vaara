# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A dashboard bound off loopback is refused unless the operator asks for it.

The write token that guards policy thresholds, the hook config and OS-layer
decisions is served to the page by GET /api/config. On a loopback bind only
the operator's own browser can fetch it. On 0.0.0.0 anyone who can reach the
port fetches it too, and the old banner still said "not reachable from the
network". `vaara serve` already refuses such a bind without a credential;
the dashboard has no credential, so it refuses unless told otherwise.
"""
from __future__ import annotations

import pytest

from vaara import dashboard


class _NoServe:
    """Stands in for ThreadingHTTPServer so serve() returns after binding."""

    bound: list = []

    def __init__(self, address, handler):
        self.server_address = address
        _NoServe.bound.append(address)

    def serve_forever(self):
        return None

    def server_close(self):
        return None


@pytest.fixture(autouse=True)
def _no_real_server(monkeypatch):
    _NoServe.bound = []
    monkeypatch.setattr(dashboard, "ThreadingHTTPServer", _NoServe)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10", "example.test"])
def test_a_non_loopback_bind_is_refused_without_the_flag(host, capsys):
    rc = dashboard.serve(host=host, port=7517, open_browser=False)
    assert rc == 2
    assert _NoServe.bound == []
    out = capsys.readouterr().out
    assert "refusing to bind" in out
    assert "--allow-network" in out
    assert "write token" in out


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_a_loopback_bind_starts_and_says_it_is_local_only(host, capsys):
    rc = dashboard.serve(host=host, port=7517, open_browser=False)
    assert rc == 0
    assert _NoServe.bound == [(host, 7517)]
    out = capsys.readouterr().out
    assert "not reachable from the network" in out


def test_the_flag_binds_off_loopback_and_warns(capsys):
    rc = dashboard.serve(host="0.0.0.0", port=7517, open_browser=False,
                         allow_network=True)
    assert rc == 0
    assert _NoServe.bound == [("0.0.0.0", 7517)]
    out = capsys.readouterr().out
    assert "not reachable from the network" not in out
    assert "WARNING" in out
    assert "write token" in out


def test_the_cli_exposes_the_flag_and_defaults_it_off():
    from vaara.cli import build_parser

    args = build_parser().parse_args(["dashboard", "--db", "x.db", "--host", "0.0.0.0"])
    assert args.allow_network is False
    args = build_parser().parse_args(
        ["dashboard", "--db", "x.db", "--host", "0.0.0.0", "--allow-network"])
    assert args.allow_network is True
