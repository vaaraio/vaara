# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""The LLM proxy's default redaction covers every GitHub token prefix.

GitHub issues ``ghp_`` (classic personal), ``gho_`` (OAuth, what the GitHub
CLI holds), ``ghu_`` (app user-to-server), ``ghs_`` (app installation),
``ghr_`` (refresh) and ``github_pat_`` (fine-grained personal) tokens.
"""

from __future__ import annotations

import pytest

from vaara.integrations._llm_proxy_shape import redact_body

_TOKENS = {
    "installation": "ghs_" + "Ab1" * 12,
    "refresh": "ghr_" + "Zy9" * 25 + "x",
    "fine_grained": "github_pat_" + "11ABCDEFG0" + "a1B2c3D4e5F6" + "_" + "Qw3" * 19 + "rt",
}


@pytest.mark.parametrize("kind", sorted(_TOKENS))
def test_default_patterns_redact_github_token(kind: str) -> None:
    token = _TOKENS[kind]
    body = {"messages": [{"role": "user", "content": f"use {token} for the push"}]}
    out = redact_body(body)
    text = out["messages"][0]["content"]
    assert token not in text
    assert text == "use *** for the push"


@pytest.mark.parametrize("kind", sorted(_TOKENS))
def test_no_token_tail_survives(kind: str) -> None:
    # A match that stops short of the token's end would leave a usable tail.
    token = _TOKENS[kind]
    out = redact_body({"input": token})
    assert out["input"] == "***"
