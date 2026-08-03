"""Tests for the MCP server approvals handshake (vaara_intercept).

A gated escalate on the MCP server path must block on the file-based
approvals handshake — the same protocol the Claude Code hook uses —
so every MCP-governed agent gets the human-in-the-loop surface (the
macOS notch approval panel). Shadow mode and VAARA_PLUGIN_APPROVALS=0
must skip it; deny and timeout must fail closed.
"""

import json

import pytest

import vaara.approvals
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.integrations.mcp_server import VaaraMCPServer
from vaara.pipeline import InterceptionPipeline


@pytest.fixture
def server():
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    pipeline = InterceptionPipeline(trail=trail)
    return VaaraMCPServer(pipeline=pipeline)


def _intercept(server, **overrides):
    args = {
        "tool_name": "tx.transfer",
        "agent_id": "mcp-agent",
        "parameters": {"amount": 5_000_000},
    }
    args.update(overrides)
    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "vaara_intercept", "arguments": args},
    })
    result = response["result"]
    payload = json.loads(result["content"][0]["text"])
    return payload, result


@pytest.fixture
def escalate_payload(server):
    # Precondition: this call must score into the escalate band on a
    # fresh pipeline; the handshake only engages there.
    import os
    os.environ["VAARA_PLUGIN_APPROVALS"] = "0"
    payload, _ = _intercept(server)
    os.environ.pop("VAARA_PLUGIN_APPROVALS")
    assert payload["decision"] == "escalate", (
        f"precondition failed: expected escalate, got {payload['decision']}"
    )
    return payload


class TestMCPServerApprovalHandshake:
    def test_approve_allows(self, server, escalate_payload, monkeypatch):
        monkeypatch.setattr(
            vaara.approvals, "request_approval",
            lambda *a, **kw: "approve",
        )
        payload, result = _intercept(server)
        assert payload["allowed"] is True
        assert "approved by human" in payload["reason"]
        assert result.get("isError") is not True
        # The trail carries the human resolution for the auditor.
        resolved = [
            r for r in server._pipeline.trail._records
            if r.event_type == EventType.ESCALATION_RESOLVED
            and r.action_id == payload["action_id"]
        ]
        assert len(resolved) == 1
        assert resolved[0].data["resolution"] == "allow"

    def test_deny_blocks(self, server, escalate_payload, monkeypatch):
        monkeypatch.setattr(
            vaara.approvals, "request_approval",
            lambda *a, **kw: "deny",
        )
        payload, result = _intercept(server)
        assert payload["allowed"] is False
        assert payload["decision"] == "escalate"
        resolved = [
            r for r in server._pipeline.trail._records
            if r.event_type == EventType.ESCALATION_RESOLVED
            and r.action_id == payload["action_id"]
        ]
        assert len(resolved) == 1
        assert resolved[0].data["resolution"] == "deny"

    def test_timeout_fails_closed(self, server, escalate_payload, monkeypatch):
        monkeypatch.setattr(
            vaara.approvals, "request_approval",
            lambda *a, **kw: "timeout",
        )
        payload, _ = _intercept(server)
        assert payload["allowed"] is False
        assert payload["decision"] == "escalate"

    def test_handshake_error_fails_closed(
        self, server, escalate_payload, monkeypatch
    ):
        def _boom(*a, **kw):
            raise RuntimeError("approvals dir unreadable")

        monkeypatch.setattr(
            vaara.approvals, "request_approval", _boom,
        )
        payload, _ = _intercept(server)
        assert payload["allowed"] is False
        assert payload["decision"] == "escalate"

    def test_env_kill_switch_skips_handshake(
        self, server, escalate_payload, monkeypatch
    ):
        monkeypatch.setenv("VAARA_PLUGIN_APPROVALS", "0")

        def _must_not_run(*a, **kw):
            raise AssertionError("handshake ran despite kill switch")

        monkeypatch.setattr(
            vaara.approvals, "request_approval", _must_not_run,
        )
        payload, _ = _intercept(server)
        assert payload["decision"] == "escalate"
        assert payload["allowed"] is False

    def test_allow_decision_never_handshakes(self, server, monkeypatch):
        def _must_not_run(*a, **kw):
            raise AssertionError("handshake ran on an allow")

        monkeypatch.setattr(
            vaara.approvals, "request_approval", _must_not_run,
        )
        payload, _ = _intercept(
            server, tool_name="data.read", parameters={"table": "users"},
        )
        assert payload["decision"] == "allow"
        assert payload["allowed"] is True
