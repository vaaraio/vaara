"""`--seal-known-secrets`: credentials in published formats never leave.

Until this flag the proxy sealed only strings an operator listed in advance,
so a key pasted into a prompt, or read from a file by the agent, reached the
provider as written. With the flag on, values matching a published credential
format are replaced with the same stable placeholders as named secrets and
restored in the reply. Each record counts what was sealed, by format, never
by value.

Personal data is out of scope and stays out: nothing here looks for emails or
names.
"""
from __future__ import annotations

import json

import pytest

from vaara.integrations.llm_seal import (
    KNOWN_SECRET_FORMATS,
    SealRegistry,
    StreamUnsealer,
    placeholder_for,
)

SAMPLES = {
    "anthropic_key": "sk-ant-api03-" + "A" * 40,
    "openai_key": "sk-proj-" + "b" * 40,
    "github_token": "ghp_" + "c" * 36,
    "aws_access_key_id": "AKIA" + "D" * 16,
    "google_api_key": "AIza" + "e" * 35,
    "slack_token": "xoxb-1234567890-abcdefghij",
    "stripe_key": "sk_live_" + "f" * 24,
    "huggingface_token": "hf_" + "g" * 34,
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf\\n-----END RSA PRIVATE KEY-----",
}


def _body(text: str) -> bytes:
    # Built by hand so the PEM sample keeps the \n escape it has on the wire.
    return ('{"messages":[{"role":"user","content":"' + text + '"}]}').encode()


def test_every_listed_format_has_a_sample():
    assert {k for k, _ in KNOWN_SECRET_FORMATS} == set(SAMPLES)


@pytest.mark.parametrize("kind", sorted(SAMPLES))
def test_each_format_is_sealed_and_restored(kind):
    reg = SealRegistry(known_formats=True)
    raw = _body(f"use {SAMPLES[kind]} please")
    out = reg.seal_bytes(raw)
    assert SAMPLES[kind].encode() not in out
    assert reg.last_kinds == {kind: 1}
    assert reg.unseal_bytes(out) == raw


@pytest.mark.parametrize("text", [
    "sk-short", "task-list-for-today", "AKIA123", "a git sha 3f2a9c1e0b7d",
    "550e8400-e29b-41d4-a716-446655440000", "henri@example.com",
    "risk-management-framework-overview-document",
])
def test_ordinary_text_is_left_alone(text):
    reg = SealRegistry(known_formats=True)
    raw = _body(text)
    assert reg.seal_bytes(raw) == raw
    assert reg.last_kinds == {}


def test_off_by_default():
    reg = SealRegistry()
    raw = _body(SAMPLES["github_token"])
    assert reg.seal_bytes(raw) == raw


def test_placeholder_is_stable_across_processes():
    a = SealRegistry(known_formats=True).seal_bytes(_body(SAMPLES["openai_key"]))
    b = SealRegistry(known_formats=True).seal_bytes(_body(SAMPLES["openai_key"]))
    assert a == b
    assert placeholder_for(SAMPLES["openai_key"]).encode() in a


def test_a_learned_value_survives_a_seal_file_reload(tmp_path):
    f = tmp_path / "seal.json"
    f.write_text(json.dumps({"x": "named-secret"}))
    reg = SealRegistry.from_file(f)
    reg.known_formats = True
    sealed = reg.seal_bytes(_body(SAMPLES["github_token"]))
    f.write_text(json.dumps({"y": "another-secret"}))
    import os
    os.utime(f, (1, 1))
    assert reg.refresh()
    assert SAMPLES["github_token"].encode() in reg.unseal_bytes(sealed)


def test_a_streamed_reply_is_restored():
    reg = SealRegistry(known_formats=True)
    reg.seal_bytes(_body(SAMPLES["stripe_key"]))
    token = placeholder_for(SAMPLES["stripe_key"]).encode()
    frame = b'data: {"delta":"your key is ' + token + b'"}\n\n'
    un = StreamUnsealer(reg)
    out = b"".join(un.feed(frame[i:i + 3]) for i in range(0, len(frame), 3))
    out += un.flush()
    assert SAMPLES["stripe_key"].encode() in out
    assert un.restored == 1


class TestThroughTheProxy:

    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        httpx = pytest.importorskip("httpx", reason="proxy deps not installed: no httpx")
        pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")
        from fastapi.testclient import TestClient

        from vaara.audit.trail import EventType
        from vaara.integrations._llm_proxy_app import build_app
        from vaara.integrations.llm_proxy import _build_pipeline

        seen: list[bytes] = []
        real = httpx.AsyncClient

        def handler(request):
            seen.append(request.content)
            echoed = request.content.decode()
            return httpx.Response(200, json={"content": [
                {"type": "text", "text": echoed}]})

        def fake(*a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            kw.pop("http2", None)
            return real(*a, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", fake)
        pipeline = _build_pipeline(tmp_path / "trail.db")
        client = TestClient(build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            seal_registry=SealRegistry(known_formats=True)))
        records = lambda: [r for r in pipeline.trail.get_records_by_type(  # noqa: E731
            EventType.ACTION_REQUESTED) if r.tool_name == "llm.prompt"]
        return client, seen, records

    def test_the_key_never_reaches_the_provider(self, run):
        client, seen, records = run
        key = SAMPLES["github_token"]
        resp = client.post("/v1/messages", json={
            "model": "m", "messages": [{"role": "user", "content": f"push with {key}"}]})
        assert key.encode() not in seen[0]
        assert key in resp.text
        (rec,) = records()
        params = rec.data["parameters"]
        assert params["seal_kinds"] == {"github_token": 1}
        assert key not in json.dumps(rec.data)
