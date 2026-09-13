"""End to end: a sealed secret does not reach the upstream, and comes back.

The unit tests in ``test_integrations_llm_seal`` cover the substitution.  This
covers the thing that actually matters in production and that unit tests
cannot see: a request travelling the real proxy handler reaches the provider
with the placeholder, and the caller still reads the original text.

The upstream is a stub inside the process, so running this sends nothing
anywhere.
"""

from __future__ import annotations

import json

import pytest

# The proxy extras are optional, and the signing-extras CI job installs only
# the signing ones. Importing these at module level errored collection for
# that whole job rather than skipping this file, which is what every other
# proxy test in the tree guards against.
httpx = pytest.importorskip("httpx", reason="proxy deps not installed: no httpx")
pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from vaara.integrations._llm_proxy_app import build_app  # noqa: E402
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402
from vaara.integrations.llm_seal import (  # noqa: E402
    SealRegistry,
    placeholder_for,
)

SECRET = "anti-note"
TOKEN = placeholder_for(SECRET)


@pytest.fixture
def seen() -> dict:
    """What the stub upstream actually received."""
    return {}


@pytest.fixture
def client(monkeypatch, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8")
        # Echo the prompt back the way a provider would quote it.
        return httpx.Response(
            200,
            json={"content": [{"type": "text",
                               "text": f"you said: {seen['body']}"}]},
            headers={"content-type": "application/json"},
        )

    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("http2", None)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)

    app = build_app(
        upstream="https://upstream.invalid",
        api_key="test-key",
        api_key_header="x-api-key",
        pipeline=_build_pipeline(None),
        seal_registry=SealRegistry({"concept": SECRET}),
    )
    return TestClient(app)


def test_upstream_never_sees_the_secret(client, seen):
    client.post("/v1/messages", json={
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": f"explain the {SECRET} idea"}],
    })
    assert SECRET not in seen["body"]
    assert TOKEN in seen["body"]


def test_caller_reads_the_secret_back(client, seen):
    resp = client.post("/v1/messages", json={
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": f"explain the {SECRET} idea"}],
    })
    assert resp.status_code == 200
    assert SECRET in resp.text
    assert TOKEN not in resp.text


def test_unsealed_proxy_forwards_the_body_byte_for_byte(monkeypatch, seen):
    """With sealing off the payload must be untouched, or prompt caching pays."""
    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, json={"ok": True})

    real = httpx.AsyncClient

    def fake(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("http2", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake)
    app = build_app(
        upstream="https://upstream.invalid", api_key="k",
        api_key_header="x-api-key", pipeline=_build_pipeline(None),
        seal_registry=SealRegistry(),
    )
    sent = json.dumps({"model": "m", "messages": [{"role": "user",
                                                   "content": "plain"}]})
    TestClient(app).post(
        "/v1/messages", content=sent,
        headers={"content-type": "application/json"},
    )
    assert seen["body"].decode("utf-8") == sent


class TestUpstreamPathIsConfined:
    """The proxy injects the operator's provider key into whatever it forwards.

    A caller-controlled path that escapes the configured upstream would spend
    that key against another host, so escapes are refused rather than
    normalised.
    """

    @pytest.mark.parametrize("path", [
        "//evil.example/v1/messages",   # protocol-relative, resolves off-host
        "/v1/messages",                 # absolute, replaces the base path
        "v1/../../v1/messages",         # traversal
        "v1\\messages",                 # backslash
        "v1/messages\nX-Injected: 1",   # header injection
    ])
    def test_escaping_paths_are_refused(self, path):
        from vaara.integrations._llm_proxy_app import _safe_upstream_path
        assert _safe_upstream_path(path) is None

    @pytest.mark.parametrize("path,expected", [
        ("v1/messages", "/v1/messages"),
        ("v1/chat/completions", "/v1/chat/completions"),
        ("", "/"),
    ])
    def test_ordinary_paths_pass(self, path, expected):
        from vaara.integrations._llm_proxy_app import _safe_upstream_path
        assert _safe_upstream_path(path) == expected
