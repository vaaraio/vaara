"""Tests for the MCP server integration."""

import json

import pytest

from vaara.integrations.mcp_server import (
    VAARA_CHECK_TOOL,
    VAARA_INTERCEPT_TOOL,
    VAARA_REPORT_TOOL,
    VaaraMCPServer,
)
from vaara.pipeline import InterceptionPipeline


@pytest.fixture
def server():
    pipeline = InterceptionPipeline()
    return VaaraMCPServer(pipeline=pipeline)


class TestMCPProtocol:
    def test_initialize(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        })
        assert response["id"] == 1
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "vaara-governance"
        assert "tools" in result["capabilities"]
        assert "resources" in result["capabilities"]

    def test_ping(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
            "params": {},
        })
        assert response["id"] == 2
        assert response["result"] == {}

    def test_unknown_method(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "nonexistent",
            "params": {},
        })
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_tools_list(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list",
            "params": {},
        })
        tools = response["result"]["tools"]
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"vaara_check", "vaara_intercept", "vaara_report_outcome"}

    def test_resources_list(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/list",
            "params": {},
        })
        resources = response["result"]["resources"]
        assert len(resources) == 2
        uris = {r["uri"] for r in resources}
        assert "vaara://status" in uris
        assert "vaara://compliance" in uris


class TestToolDefinitions:
    def test_check_tool_schema(self):
        schema = VAARA_CHECK_TOOL.inputSchema
        assert schema["type"] == "object"
        assert "tool_name" in schema["properties"]
        assert "tool_name" in schema["required"]

    def test_intercept_tool_schema(self):
        schema = VAARA_INTERCEPT_TOOL.inputSchema
        assert "tool_name" in schema["required"]
        assert "session_id" in schema["properties"]

    def test_report_tool_schema(self):
        schema = VAARA_REPORT_TOOL.inputSchema
        assert "action_id" in schema["required"]
        assert "outcome_severity" in schema["required"]


class TestVaaraCheck:
    def test_check_low_risk_action(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "vaara_check",
                "arguments": {"tool_name": "data.read", "agent_id": "test"},
            },
        })
        content = json.loads(response["result"]["content"][0]["text"])
        assert "risk_score" in content
        assert "decision" in content
        assert "risk_interval" in content
        assert content["action_type"] == "data.read"

    def test_check_high_risk_action(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "vaara_check",
                "arguments": {"tool_name": "phy.safety_override"},
            },
        })
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["risk_score"] > 0

    def test_check_unknown_tool(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "vaara_check",
                "arguments": {"tool_name": "completely.unknown"},
            },
        })
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["action_type"] == "unknown"

    def test_check_with_confidence(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "vaara_check",
                "arguments": {
                    "tool_name": "tx.transfer",
                    "confidence": 0.9,
                },
            },
        })
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["note"] == "Read-only check — use vaara_intercept for audited interception"


class TestVaaraIntercept:
    def test_intercept_creates_audit(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "vaara_intercept",
                "arguments": {
                    "tool_name": "data.read",
                    "agent_id": "test-agent",
                },
            },
        })
        content = json.loads(response["result"]["content"][0]["text"])
        assert "action_id" in content
        assert "allowed" in content
        assert "decision" in content
        assert "risk_score" in content

    def test_intercept_returns_action_id(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "vaara_intercept",
                "arguments": {"tool_name": "data.write"},
            },
        })
        content = json.loads(response["result"]["content"][0]["text"])
        assert len(content["action_id"]) > 0

    def test_intercept_with_parameters(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "vaara_intercept",
                "arguments": {
                    "tool_name": "tx.transfer",
                    "parameters": {"to": "0x1234", "amount": 1000},
                    "session_id": "session-001",
                },
            },
        })
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["action_type"] == "tx.transfer"


class TestVaaraReport:
    def test_report_outcome(self, server):
        # First intercept to get an action_id
        intercept_response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/call",
            "params": {
                "name": "vaara_intercept",
                "arguments": {"tool_name": "data.read", "agent_id": "test"},
            },
        })
        action_id = json.loads(
            intercept_response["result"]["content"][0]["text"]
        )["action_id"]

        # Report the outcome
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": {
                "name": "vaara_report_outcome",
                "arguments": {
                    "action_id": action_id,
                    "outcome_severity": 0.0,
                    "description": "Completed successfully",
                },
            },
        })
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["recorded"] is True
        assert content["calibration_size"] >= 1

    def test_report_builds_calibration(self, server):
        """Multiple reports should build calibration."""
        for i in range(5):
            intercept = server.handle_request({
                "jsonrpc": "2.0",
                "id": 40 + i * 2,
                "method": "tools/call",
                "params": {
                    "name": "vaara_intercept",
                    "arguments": {"tool_name": "data.read"},
                },
            })
            action_id = json.loads(
                intercept["result"]["content"][0]["text"]
            )["action_id"]

            server.handle_request({
                "jsonrpc": "2.0",
                "id": 41 + i * 2,
                "method": "tools/call",
                "params": {
                    "name": "vaara_report_outcome",
                    "arguments": {
                        "action_id": action_id,
                        "outcome_severity": 0.1 * i,
                    },
                },
            })

        # Check status resource
        status = server.handle_request({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "resources/read",
            "params": {"uri": "vaara://status"},
        })
        status_data = json.loads(
            status["result"]["contents"][0]["text"]
        )
        assert status_data["scorer"]["calibration_size"] >= 5


class TestResources:
    def test_status_resource(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 50,
            "method": "resources/read",
            "params": {"uri": "vaara://status"},
        })
        contents = response["result"]["contents"]
        assert len(contents) == 1
        assert contents[0]["mimeType"] == "application/json"

        data = json.loads(contents[0]["text"])
        assert "scorer" in data
        assert "trail_size" in data

    def test_compliance_resource(self, server):
        # Need some audit data first
        server.handle_request({
            "jsonrpc": "2.0",
            "id": 60,
            "method": "tools/call",
            "params": {
                "name": "vaara_intercept",
                "arguments": {"tool_name": "tx.transfer"},
            },
        })

        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 61,
            "method": "resources/read",
            "params": {"uri": "vaara://compliance"},
        })
        assert "result" in response, f"Expected result, got error: {response.get('error')}"
        data = json.loads(response["result"]["contents"][0]["text"])
        assert "overall_status" in data
        assert "articles" in data

    def test_unknown_resource(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 70,
            "method": "resources/read",
            "params": {"uri": "vaara://nonexistent"},
        })
        assert "error" in response


class TestUnknownTool:
    def test_call_unknown_tool(self, server):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 80,
            "method": "tools/call",
            "params": {"name": "unknown_tool", "arguments": {}},
        })
        assert "error" in response


class TestReportOutcomeValidation:
    """Loop 45: malformed report_outcome args must map to -32602 Invalid params,
    and null/non-numeric severity must be rejected at the boundary (otherwise
    it silently becomes 0.0 downstream, biasing the scorer toward ALLOW)."""

    @pytest.mark.parametrize("bad_action_id", [{"x": 1}, ["a"], 42])
    def test_non_string_action_id_is_invalid_params(self, server, bad_action_id):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 90,
            "method": "tools/call",
            "params": {
                "name": "vaara_report_outcome",
                "arguments": {"action_id": bad_action_id, "outcome_severity": 0.1},
            },
        })
        assert response["error"]["code"] == -32602

    @pytest.mark.parametrize("bad_severity", [None, "high", float("nan"), float("inf"), 1.5, -0.1, True])
    def test_non_numeric_or_out_of_range_severity_is_invalid_params(self, server, bad_severity):
        response = server.handle_request({
            "jsonrpc": "2.0",
            "id": 91,
            "method": "tools/call",
            "params": {
                "name": "vaara_report_outcome",
                "arguments": {"action_id": "abc", "outcome_severity": bad_severity},
            },
        })
        assert response["error"]["code"] == -32602
