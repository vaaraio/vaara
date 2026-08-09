# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-site guard on every local HTTP surface Vaara serves.

Each of these binds 127.0.0.1 by default and takes no inbound credential,
which is safe against other users on the box and not against the operator's
own browser: any page they visit can post to localhost. Without the guard a
cross-origin `fetch` with Content-Type text/plain is a CORS-simple request —
no preflight — and the side effect lands even though the page cannot read
the reply.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - server extra not installed
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )

from vaara.integrations._http_origin import (
    install_origin_guard,
    normalise_origins,
    origin_is_allowed,
    origins_from_env,
)

EVIL = "https://evil.example"


# ── the decision function ─────────────────────────────────────────────────

def test_absent_origin_is_allowed():
    assert origin_is_allowed(None, allowed=set()) is True


def test_unknown_origin_is_refused():
    assert origin_is_allowed(EVIL, allowed=set()) is False


def test_listed_origin_is_allowed():
    assert origin_is_allowed(EVIL, allowed={EVIL}) is True


def test_same_origin_is_allowed():
    assert origin_is_allowed(
        "http://127.0.0.1:8765", allowed=set(),
        self_origins={"http://127.0.0.1:8765"},
    ) is True


def test_null_origin_is_refused():
    """Sandboxed iframes, data: documents and file:// pages send "null"."""
    assert origin_is_allowed(
        "null", allowed=set(), self_origins={"http://127.0.0.1:8765"},
    ) is False


@pytest.mark.parametrize("hostile", [
    "https://app.example.evil.test",
    "https://evil-app.example",
    "http://app.example",
    "https://app.example:8443",
    "https://app.example/",
])
def test_matching_is_exact(hostile):
    assert origin_is_allowed(hostile, allowed={"https://app.example"}) is False


def test_normalise_origins_drops_blanks_and_whitespace():
    assert normalise_origins([" a ", "", None, "b"]) == {"a", "b"}


def test_origins_from_env_splits_on_commas():
    assert origins_from_env(" a , b ,, ") == {"a", "b"}
    assert origins_from_env(None) == set()


# ── the middleware ────────────────────────────────────────────────────────

def _guarded_app(**kwargs) -> TestClient:
    app = FastAPI()

    @app.post("/do")
    async def do() -> dict:
        return {"ran": True}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    install_origin_guard(app, **kwargs)
    return TestClient(app)


def test_middleware_refuses_a_cross_site_post():
    response = _guarded_app().post("/do", headers={"Origin": EVIL})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "origin_not_allowed"


def test_middleware_refuses_the_no_preflight_content_type():
    response = _guarded_app().post(
        "/do",
        content=json.dumps({"a": 1}),
        headers={"Origin": EVIL, "Content-Type": "text/plain;charset=UTF-8"},
    )
    assert response.status_code == 403


def test_middleware_allows_a_request_with_no_origin():
    assert _guarded_app().post("/do").json() == {"ran": True}


def test_middleware_allows_the_servers_own_page():
    """TestClient addresses the app as 'testserver'; its page posts back."""
    response = _guarded_app().post("/do", headers={"Origin": "http://testserver"})
    assert response.status_code == 200


def test_middleware_allows_a_listed_origin():
    client = _guarded_app(allowed_origins=[EVIL])
    assert client.post("/do", headers={"Origin": EVIL}).status_code == 200


def test_middleware_leaves_health_open():
    assert _guarded_app().get("/health", headers={"Origin": EVIL}).status_code == 200


def test_middleware_does_not_log_injected_newlines(caplog):
    """A hostile Origin ends up in the log; it must not forge a second line."""
    _guarded_app().post("/do", headers={"Origin": "https://a\r\nFAKE: line"})
    logged = caplog.text.split("refused cross-origin request from ")[-1].rstrip("\n")
    assert "\n" not in logged and "\r" not in logged
    assert "FAKE" in logged, "the value should be recorded, just neutered"


# ── the real surfaces ─────────────────────────────────────────────────────

def test_llm_proxy_refuses_cross_site():
    """This one injects the operator's upstream provider key."""
    from vaara.integrations._llm_proxy_app import build_app

    app = build_app(
        upstream="https://api.example/v1", api_key="k",
        api_key_header="x-api-key", pipeline=MagicMock(),
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        content=json.dumps({"model": "m", "messages": []}),
        headers={"Origin": EVIL, "Content-Type": "text/plain"},
    )
    assert response.status_code == 403


def test_infer_proxy_refuses_cross_site():
    """This one signs an attestation + receipt pair per chat call."""
    from vaara.integrations._infer_proxy_app import build_app

    app = build_app(emitter=None, upstream="http://127.0.0.1:11434",
                    client=MagicMock())
    response = TestClient(app).post(
        "/api/chat",
        content=json.dumps({"model": "m", "messages": []}),
        headers={"Origin": EVIL, "Content-Type": "text/plain"},
    )
    assert response.status_code == 403


def test_console_refuses_cross_site(tmp_path):
    from vaara.integrations._infer_console_app import build_app

    app = build_app(
        proxy_url="http://127.0.0.1:11435", receipts_dir=tmp_path,
        client=MagicMock(), judge_factory=lambda m: MagicMock(),
    )
    response = TestClient(app).post(
        "/api/chat",
        content=json.dumps({"model": "m", "messages": []}),
        headers={"Origin": EVIL, "Content-Type": "text/plain"},
    )
    assert response.status_code == 403


def test_console_own_page_still_works(tmp_path):
    """The console IS a browser app; refusing its own origin would break it."""
    from vaara.integrations._infer_console_app import build_app

    app = build_app(
        proxy_url="http://127.0.0.1:11435", receipts_dir=tmp_path,
        client=MagicMock(), judge_factory=lambda m: MagicMock(),
    )
    response = TestClient(app).get("/", headers={"Origin": "http://testserver"})
    assert response.status_code == 200
