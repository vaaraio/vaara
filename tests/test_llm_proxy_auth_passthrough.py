# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``--auth-passthrough``: forward the caller's credential, hold none.

The inject-only path could not govern an agent that authenticates with its own
subscription.  A subscriber has no API key to hand the proxy, and the proxy
stripped the token they did have, so the request arrived unauthenticated.
These tests pin both halves: the credential survives in pass-through mode, and
the operator-holds-the-key path still strips and replaces it.
"""

from __future__ import annotations

import pytest

from vaara.integrations._llm_proxy_shape import forward_request_headers
from vaara.integrations.llm_proxy import main


class TestForwardRequestHeaders:
    def test_auth_stripped_by_default(self):
        """The default must not change: operator key replaces caller's."""
        out = forward_request_headers(
            {"authorization": "Bearer caller-token", "accept": "application/json"}
        )
        assert "authorization" not in {k.lower() for k in out}
        assert out["accept"] == "application/json"

    def test_auth_kept_when_passthrough(self):
        out = forward_request_headers(
            {"authorization": "Bearer caller-token"}, keep_auth=True,
        )
        assert out["authorization"] == "Bearer caller-token"

    @pytest.mark.parametrize("header", ["authorization", "x-api-key", "api-key"])
    def test_every_known_auth_header_survives(self, header):
        """All three, not just the one Anthropic happens to use."""
        out = forward_request_headers({header: "secret"}, keep_auth=True)
        assert out[header] == "secret"

    def test_hop_by_hop_still_dropped_in_passthrough(self):
        """keep_auth must not turn into keep_everything."""
        out = forward_request_headers(
            {"host": "127.0.0.1", "content-length": "12",
             "authorization": "Bearer t"},
            keep_auth=True,
        )
        assert {k.lower() for k in out} == {"authorization"}


class TestCli:
    def test_passthrough_conflicts_with_api_key(self, capsys):
        """Two identities is a configuration error, not a precedence puzzle."""
        with pytest.raises(SystemExit):
            main(["--upstream", "https://x.test", "--api-key", "k",
                  "--auth-passthrough"])

    def test_one_of_the_three_is_required(self):
        with pytest.raises(SystemExit):
            main(["--upstream", "https://x.test"])

    def test_empty_key_file_still_fails_loudly(self, tmp_path, capsys):
        """Pass-through must not become the silent landing spot for a
        mistyped key file."""
        kf = tmp_path / "empty.key"
        kf.write_text("   \n")
        rc = main(["--upstream", "https://x.test", "--api-key-file", str(kf)])
        assert rc == 1
        assert "no API key" in capsys.readouterr().err


class TestBuildApp:
    def _app(self, api_key):
        from vaara.integrations._llm_proxy_app import build_app
        from vaara.integrations.llm_proxy import _build_pipeline
        return build_app(
            upstream="https://api.anthropic.com", api_key=api_key,
            api_key_header="authorization", pipeline=_build_pipeline(None),
        )

    def test_accepts_none_api_key(self):
        """None is the pass-through signal and must not raise."""
        assert self._app(None) is not None

    def test_still_accepts_a_key(self):
        assert self._app("sk-test") is not None
