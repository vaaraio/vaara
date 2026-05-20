"""OVERT 1.0 Base Envelope emission through the Vaara MCP proxy.

Smoke-level: with a real emitter wired into the proxy, exercise each
governed call site (allowed, blocked, perimeter-filtered) and confirm one
canonical-CBOR envelope per interaction lands in the receipts directory,
the monotonic counter advances strictly, and each envelope verifies under
the pinned public key.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

cbor2 = pytest.importorskip("cbor2")
pytest.importorskip("cryptography")


@dataclass
class _StubInterceptResult:
    allowed: bool
    action_id: str = "stub-action-id"
    reason: str = ""
    decision: str = "ALLOW"


@pytest.fixture
def signing_key_pem(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "signing.pem"
    path.write_bytes(pem)
    pub_raw = key.public_key().public_bytes_raw()
    return path, pub_raw


@pytest.fixture
def operator_key_path(tmp_path):
    p = tmp_path / "op.key"
    p.write_bytes(b"\x11" * 32)
    return p


@pytest.fixture
def emitter(signing_key_pem, operator_key_path, tmp_path):
    from vaara.integrations._mcp_overt import build_emitter

    receipts = tmp_path / "receipts"
    sk_path, _ = signing_key_pem
    return build_emitter(
        signing_key_path=sk_path,
        operator_key_path=operator_key_path,
        operator_key_hex=None,
        receipts_dir=receipts,
        policy_hash=b"\xab" * 32,
    )


def _make_proxy(monkeypatch, *, emitter, **kwargs):
    from vaara.integrations import mcp_proxy

    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    pipeline = MagicMock()
    p = mcp_proxy.VaaraMCPProxy(
        upstream_command=["echo"], pipeline=pipeline,
        overt_emitter=emitter, **kwargs,
    )
    p._upstream = MagicMock()
    return p, pipeline


def _envelopes(receipts_dir: Path) -> list[dict]:
    files = sorted(p for p in receipts_dir.iterdir() if p.suffix == ".cbor")
    return [cbor2.loads(f.read_bytes()) for f in files]


def test_emitter_disabled_by_default(monkeypatch, tmp_path):
    from vaara.integrations import mcp_proxy
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    pipeline = MagicMock()
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline)
    p._upstream = MagicMock()
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="a-1")
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "any", "arguments": {}},
    })


def test_allowed_tool_call_writes_one_envelope(monkeypatch, emitter):
    p, pipeline = _make_proxy(monkeypatch, emitter=emitter)
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="a-1")
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}

    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "sap.adt.read", "arguments": {"object": "ZCL"}},
    })

    envs = _envelopes(emitter.receipts_dir)
    assert len(envs) == 1
    meta = envs[0]["non_content_metadata"]
    assert meta["action_class"] == "mcp.tool.call"
    assert meta["tool_name"] == "sap.adt.read"
    assert meta["decision"] == "allow"
    assert meta["action_id"] == "a-1"


def test_blocked_tool_call_still_writes_envelope(monkeypatch, emitter):
    p, pipeline = _make_proxy(monkeypatch, emitter=emitter)
    pipeline.intercept.return_value = _StubInterceptResult(
        allowed=False, decision="DENY", reason="too risky", action_id="b-2",
    )
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "sap.abap.write", "arguments": {"path": "/etc/x"}},
    })
    envs = _envelopes(emitter.receipts_dir)
    assert len(envs) == 1
    meta = envs[0]["non_content_metadata"]
    assert meta["decision"] == "DENY"
    assert meta["reason"] == "too risky"


def test_perimeter_filtered_tool_writes_envelope(monkeypatch, emitter):
    p, _ = _make_proxy(
        monkeypatch, emitter=emitter, denylist={"delete_repository"},
    )
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "delete_repository", "arguments": {}},
    })
    envs = _envelopes(emitter.receipts_dir)
    assert len(envs) == 1
    assert envs[0]["non_content_metadata"]["decision"] == "FILTERED"


def test_resources_read_emits_envelopes_allowed_and_filtered(monkeypatch, emitter):
    p, _ = _make_proxy(
        monkeypatch, emitter=emitter,
        resource_denylist={"file:///etc/secret"},
    )
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
    p._handle_resources_read({
        "jsonrpc": "2.0", "id": 1, "method": "resources/read",
        "params": {"uri": "file:///etc/hosts"},
    })
    p._handle_resources_read({
        "jsonrpc": "2.0", "id": 2, "method": "resources/read",
        "params": {"uri": "file:///etc/secret"},
    })
    envs = _envelopes(emitter.receipts_dir)
    assert len(envs) == 2
    decisions = [e["non_content_metadata"]["decision"] for e in envs]
    assert decisions == ["allow", "FILTERED"]


def test_prompts_get_emits_envelopes_allowed_and_filtered(monkeypatch, emitter):
    p, _ = _make_proxy(
        monkeypatch, emitter=emitter,
        prompt_denylist={"jailbreak"},
    )
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
    p._handle_prompts_get({
        "jsonrpc": "2.0", "id": 1, "method": "prompts/get",
        "params": {"name": "summarize", "arguments": {}},
    })
    p._handle_prompts_get({
        "jsonrpc": "2.0", "id": 2, "method": "prompts/get",
        "params": {"name": "jailbreak", "arguments": {}},
    })
    envs = _envelopes(emitter.receipts_dir)
    assert len(envs) == 2
    assert envs[0]["non_content_metadata"]["prompt_name"] == "summarize"
    assert envs[1]["non_content_metadata"]["decision"] == "FILTERED"


def test_monotonic_counter_strictly_increases(monkeypatch, emitter):
    p, pipeline = _make_proxy(monkeypatch, emitter=emitter)
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="a")
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 0, "result": {}}
    for i in range(5):
        p._handle_tools_call({
            "jsonrpc": "2.0", "id": i, "method": "tools/call",
            "params": {"name": "x", "arguments": {}},
        })
    counters = [e["monotonic_counter"] for e in _envelopes(emitter.receipts_dir)]
    assert counters == [1, 2, 3, 4, 5]


def test_emitted_envelopes_verify_with_pinned_pubkey(monkeypatch, emitter, signing_key_pem):
    from vaara.attestation.overt import BaseEnvelope, verify_base_envelope
    p, pipeline = _make_proxy(monkeypatch, emitter=emitter)
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="v-1")
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "x", "arguments": {"k": 1}},
    })

    _, pub_raw = signing_key_pem
    pinned = (emitter.receipts_dir / "pubkey.bin").read_bytes()
    assert pinned == pub_raw

    envs = _envelopes(emitter.receipts_dir)
    assert len(envs) == 1
    e = envs[0]
    env = BaseEnvelope(
        blinded_identifier=e["blinded_identifier"],
        request_commitment=e["request_commitment"],
        encoder_binary_identity=e["encoder_binary_identity"],
        non_content_metadata=e["non_content_metadata"],
        monotonic_counter=int(e["monotonic_counter"]),
        nanosecond_timestamp=int(e["nanosecond_timestamp"]),
        key_identifier=e["key_identifier"],
        arbiter_instance_identifier=e["arbiter_instance_identifier"],
        signature=e["signature"],
    )
    assert verify_base_envelope(env, pub_raw) is True


def test_progress_notification_emits_envelope_with_parent_correlation(monkeypatch, emitter):
    p, pipeline = _make_proxy(monkeypatch, emitter=emitter)
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="parent-7")

    def upstream_request(req):
        if req["method"] == "tools/call":
            p._on_upstream_notification({
                "jsonrpc": "2.0", "method": "notifications/progress",
                "params": {"progressToken": "tok-z", "progress": 10, "total": 20},
            })
        return {"jsonrpc": "2.0", "id": req["id"], "result": {}}

    p._upstream.request.side_effect = upstream_request
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {
            "name": "slow_tool", "arguments": {},
            "_meta": {"progressToken": "tok-z"},
        },
    })
    envs = _envelopes(emitter.receipts_dir)
    # tools/call envelope + the in-flight progress envelope.
    assert len(envs) == 2
    surfaces = [e["non_content_metadata"]["action_class"] for e in envs]
    assert "mcp.notification.progress" in surfaces
    progress_env = next(
        e for e in envs
        if e["non_content_metadata"]["action_class"] == "mcp.notification.progress"
    )
    meta = progress_env["non_content_metadata"]
    assert meta["progress_token"] == "tok-z"
    assert meta["parent_action_id"] == "parent-7"
    assert meta["parent_tool"] == "slow_tool"
    assert meta["decision"] == "observed"


def test_message_notification_emits_envelope(monkeypatch, emitter):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._on_upstream_notification({
        "jsonrpc": "2.0", "method": "notifications/message",
        "params": {"level": "warning", "logger": "upstream", "data": "slow"},
    })
    envs = _envelopes(emitter.receipts_dir)
    assert len(envs) == 1
    meta = envs[0]["non_content_metadata"]
    assert meta["action_class"] == "mcp.notification.message"
    assert meta["level"] == "warning"
    assert meta["decision"] == "observed"


def test_orphan_progress_without_inflight_call_still_emits(monkeypatch, emitter):
    """Progress notifications with no matching tools/call still audit + emit.

    The MCP spec allows progressTokens that don't correlate cleanly (e.g.,
    upstream lagging past response, race between reader and handler). The
    envelope still lands, just with an empty parent_action_id — the auditor
    can spot the dangling event rather than having it disappear.
    """
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._on_upstream_notification({
        "jsonrpc": "2.0", "method": "notifications/progress",
        "params": {"progressToken": "orphan", "progress": 5},
    })
    envs = _envelopes(emitter.receipts_dir)
    assert len(envs) == 1
    meta = envs[0]["non_content_metadata"]
    assert meta["action_class"] == "mcp.notification.progress"
    assert meta["progress_token"] == "orphan"
    assert meta["parent_action_id"] == ""


def test_build_emitter_rejects_missing_operator_key(signing_key_pem, tmp_path):
    from vaara.integrations._mcp_overt import (
        OVERTConfigError, build_emitter,
    )
    sk_path, _ = signing_key_pem
    with pytest.raises(OVERTConfigError):
        build_emitter(
            signing_key_path=sk_path,
            operator_key_path=None,
            operator_key_hex=None,
            receipts_dir=tmp_path / "r",
            policy_hash=b"\x00" * 32,
        )


def test_build_emitter_rejects_short_operator_key(signing_key_pem, tmp_path):
    from vaara.integrations._mcp_overt import (
        OVERTConfigError, build_emitter,
    )
    sk_path, _ = signing_key_pem
    op = tmp_path / "short.key"
    op.write_bytes(b"\x01" * 8)
    with pytest.raises(OVERTConfigError):
        build_emitter(
            signing_key_path=sk_path,
            operator_key_path=op,
            operator_key_hex=None,
            receipts_dir=tmp_path / "r",
            policy_hash=b"\x00" * 32,
        )
