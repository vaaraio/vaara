# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The startup line never prints credentials carried in the upstream URL (CodeQL 321)."""

from vaara.integrations.llm_proxy import _shown_upstream


def test_userinfo_is_dropped():
    shown = _shown_upstream("https://user:s3cret@gw.example.com:8443/v1?x=1")
    assert shown == "https://gw.example.com:8443/v1?x=1"
    assert "s3cret" not in shown


def test_password_with_at_sign_is_dropped():
    assert _shown_upstream("https://u:p@ss@gw.example.com/v1") == "https://gw.example.com/v1"


def test_plain_url_unchanged():
    assert _shown_upstream("https://api.anthropic.com") == "https://api.anthropic.com"
