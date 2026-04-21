"""Model Context Protocol (MCP) server for agent governance.

Exposes Vaara's interception pipeline as an MCP server so that AI agents
(Claude, GPT, Cursor, etc.) can discover and use Vaara governance as a
standard tool via the MCP protocol.

The MCP server provides three tools:
1. **vaara_check** — Score an action before execution (read-only assessment)
2. **vaara_intercept** — Full interception with audit trail (blocks/allows)
3. **vaara_report_outcome** — Report what happened after execution

And two resources:
1. **vaara://status** — Current scorer status, calibration state
2. **vaara://compliance** — Latest compliance assessment

Exposes Vaara's adaptive scorer over MCP so any MCP-capable agent host
can consult the gate before executing a tool call.

Architecture::

    AI Agent (Claude, GPT, etc.)
        │
        ▼
    MCP Client (in agent framework)
        │  JSON-RPC over stdio/SSE
        ▼
    VaaraMCPServer
        │
        ▼
    InterceptionPipeline
        ├── ActionRegistry (classify)
        ├── AdaptiveScorer (risk score + conformal interval)
        ├── AuditTrail (hash-chained log)
        └── ComplianceEngine (EU AI Act, DORA)

Protocol: JSON-RPC 2.0 over stdio (default) or SSE.

Usage::

    # As a standalone MCP server
    python -m vaara.integrations.mcp_server

    # In Claude Code's .claude/settings.json:
    {
        "mcpServers": {
            "vaara": {
                "command": "python",
                "args": ["-m", "vaara.integrations.mcp_server"],
                "env": {"VAARA_DB": "/path/to/audit.db"}
            }
        }
    }

    # Programmatic usage
    from vaara.integrations.mcp_server import VaaraMCPServer
    server = VaaraMCPServer()
    server.run()

Reference: Model Context Protocol specification (modelcontextprotocol.io)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from vaara import __version__ as _VAARA_VERSION
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline

logger = logging.getLogger(__name__)


def _scrub_nonfinite(value: Any) -> Any:
    """Replace NaN/+Inf/-Inf floats with None, recursively.

    RFC 8259 does not permit NaN/Infinity as JSON number tokens. Python's
    json.dumps defaults to allow_nan=True and emits JavaScript-style
    `NaN`/`Infinity`/`-Infinity` literals — strict parsers (Go, Rust,
    browsers, and many MCP clients) reject those responses, which would
    silently break governance at the wire boundary. Scrub before dumping.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _scrub_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_nonfinite(v) for v in value]
    return value


def _strict_json_dumps(obj: Any, **kwargs: Any) -> str:
    """json.dumps with strict (RFC 8259) output.

    Runs the payload through _scrub_nonfinite first, then passes
    allow_nan=False so any NaN/Inf that escapes scrubbing fails loudly
    in tests instead of producing invalid wire format in production.
    """
    return json.dumps(_scrub_nonfinite(obj), allow_nan=False, **kwargs)


class _InvalidParams(Exception):
    """Marker raised by tool handlers when params don't match the contract.
    handle_request maps this to JSON-RPC -32602 (Invalid params)."""
    pass


# ── MCP Protocol Types ──────────────────────────────────────────────────
# These mirror the MCP spec without depending on any MCP SDK.

@dataclass
class MCPToolDefinition:
    """MCP tool definition — describes a callable tool to the client."""
    name: str
    description: str
    inputSchema: dict  # JSON Schema for parameters

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


@dataclass
class MCPResourceDefinition:
    """MCP resource definition — describes a readable resource."""
    uri: str
    name: str
    description: str
    mimeType: str = "application/json"

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mimeType,
        }


# ── Tool Definitions ────────────────────────────────────────────────────

VAARA_CHECK_TOOL = MCPToolDefinition(
    name="vaara_check",
    description=(
        "Check the risk level of an action BEFORE executing it. "
        "Returns a risk assessment with point estimate, conformal interval, "
        "and recommended decision (allow/deny/escalate). Does NOT record "
        "to the audit trail — use vaara_intercept for that."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "The tool/action to check (e.g., 'tx.transfer', 'data.delete')",
            },
            "agent_id": {
                "type": "string",
                "description": "Identifier for the agent requesting the action",
                "default": "mcp-agent",
            },
            "parameters": {
                "type": "object",
                "description": "Action parameters (optional, for context)",
                "default": {},
            },
            "confidence": {
                "type": "number",
                "description": "Agent's self-assessed confidence in the action (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["tool_name"],
    },
)

VAARA_INTERCEPT_TOOL = MCPToolDefinition(
    name="vaara_intercept",
    description=(
        "Intercept an action with full risk scoring and audit trail. "
        "Classifies the action, scores risk with conformal prediction, "
        "decides allow/deny/escalate, and records everything in a "
        "tamper-evident audit trail. Returns the decision and action_id "
        "for later outcome reporting."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "The tool/action to intercept",
            },
            "agent_id": {
                "type": "string",
                "description": "Identifier for the agent",
                "default": "mcp-agent",
            },
            "parameters": {
                "type": "object",
                "description": "Action parameters",
                "default": {},
            },
            "confidence": {
                "type": "number",
                "description": "Agent's self-assessed confidence (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "session_id": {
                "type": "string",
                "description": "Session identifier for grouping related actions",
                "default": "",
            },
        },
        "required": ["tool_name"],
    },
)

VAARA_REPORT_TOOL = MCPToolDefinition(
    name="vaara_report_outcome",
    description=(
        "Report what happened after an action was executed. "
        "This closes the feedback loop — the scorer learns from outcomes "
        "to improve future risk assessments. Provide the action_id from "
        "vaara_intercept and the observed severity."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "action_id": {
                "type": "string",
                "description": "The action_id returned by vaara_intercept",
            },
            "outcome_severity": {
                "type": "number",
                "description": "How severe was the outcome? 0.0 = completely safe, 1.0 = catastrophic",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of what happened",
                "default": "",
            },
        },
        "required": ["action_id", "outcome_severity"],
    },
)


# ── Resource Definitions ────────────────────────────────────────────────

VAARA_STATUS_RESOURCE = MCPResourceDefinition(
    uri="vaara://status",
    name="Vaara Status",
    description="Current scorer status: calibration state, MWU weights, thresholds, tracked agents",
)

VAARA_COMPLIANCE_RESOURCE = MCPResourceDefinition(
    uri="vaara://compliance",
    name="Vaara Compliance Report",
    description="Latest EU AI Act and DORA compliance assessment against the audit trail",
)


# ── MCP Server ──────────────────────────────────────────────────────────

class VaaraMCPServer:
    """MCP server that exposes Vaara governance to AI agents.

    Handles JSON-RPC 2.0 messages over stdio.
    """

    SERVER_NAME = "vaara-governance"
    # Pulled from the installed vaara package so a version bump doesn't
    # leave stale strings in MCP serverInfo or compliance reports.
    SERVER_VERSION = _VAARA_VERSION

    def __init__(
        self,
        pipeline: Optional[InterceptionPipeline] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        # Set up pipeline with optional SQLite persistence
        if pipeline is not None:
            self._pipeline = pipeline
            self._backend = None
        else:
            db = db_path or Path(os.environ.get("VAARA_DB", "vaara_audit.db"))
            self._backend = SQLiteAuditBackend(db)
            from vaara.audit.trail import AuditTrail
            trail = AuditTrail(on_record=self._backend.write_record)
            self._pipeline = InterceptionPipeline(trail=trail)

        self._tools = {
            "vaara_check": VAARA_CHECK_TOOL,
            "vaara_intercept": VAARA_INTERCEPT_TOOL,
            "vaara_report_outcome": VAARA_REPORT_TOOL,
        }
        self._resources = {
            "vaara://status": VAARA_STATUS_RESOURCE,
            "vaara://compliance": VAARA_COMPLIANCE_RESOURCE,
        }
        # Optional API key auth. Set VAARA_API_KEY env var to require all
        # tool calls to pass _api_key in arguments. Leave unset for
        # single-tenant stdio deployments where process isolation is enough.
        self._required_api_key: Optional[str] = os.environ.get("VAARA_API_KEY") or None
        if self._required_api_key:
            logger.info("VaaraMCPServer: API key authentication enabled")

    def handle_request(self, request: Any) -> dict:
        """Handle a single JSON-RPC 2.0 request."""
        # JSON-RPC 2.0 §5: a non-object request is an Invalid Request.
        # Lists (batches), strings, numbers, and null must not crash the
        # server — return -32600 so the client sees a well-formed error.
        if not isinstance(request, dict):
            return self._error_response(None, -32600, "Invalid Request: not a JSON object")

        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        # params must be a structured value (object or array) per §4.2.
        # A non-dict params coerces to empty for downstream dict access.
        if not isinstance(params, dict):
            params = {}

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                result = self._handle_tools_list()
            elif method == "tools/call":
                result = self._handle_tools_call(params)
            elif method == "resources/list":
                result = self._handle_resources_list()
            elif method == "resources/read":
                result = self._handle_resources_read(params)
            elif method == "ping":
                result = {}
            else:
                return self._error_response(req_id, -32601, f"Method not found: {method}")
        except _InvalidParams as e:
            return self._error_response(req_id, -32602, str(e))
        except Exception:
            logger.exception("Error handling %s", method)
            # Don't leak internal exception messages (Python tracebacks,
            # attribute errors) to the client — emit a generic message.
            # The full exception is in the server's logs for diagnostics.
            return self._error_response(req_id, -32603, "Internal server error")

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    # ── Protocol handlers ────────────────────────────────────────

    # Protocol versions the server speaks. Newest first.
    SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05",)

    def _handle_initialize(self, params: dict) -> dict:
        # Per MCP spec: if the server supports the requested version it
        # MUST respond with the same version, otherwise it MUST respond
        # with a version it does support. Hardcoding our own version
        # regardless of the client request breaks negotiation for
        # clients that speak an older version we also support.
        requested = params.get("protocolVersion")
        if isinstance(requested, str) and requested in self.SUPPORTED_PROTOCOL_VERSIONS:
            negotiated = requested
        else:
            negotiated = self.SUPPORTED_PROTOCOL_VERSIONS[0]
            if requested:
                logger.info(
                    "Client requested protocolVersion=%r; responding with %r",
                    requested, negotiated,
                )
        return {
            "protocolVersion": negotiated,
            "serverInfo": {
                "name": self.SERVER_NAME,
                "version": self.SERVER_VERSION,
            },
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
        }

    def _handle_tools_list(self) -> dict:
        return {
            "tools": [t.to_dict() for t in self._tools.values()],
        }

    def _handle_tools_call(self, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # A misbehaving client can send arguments as a string/list/number;
        # tool handlers then AttributeError on .get and we leak a Python
        # traceback via the generic -32603 handler. Coerce to empty dict
        # so handlers always see a dict — they already validate fields.
        if not isinstance(arguments, dict):
            arguments = {}

        if self._required_api_key:
            provided = arguments.pop("_api_key", None)
            if not isinstance(provided, str) or provided != self._required_api_key:
                raise _InvalidParams("Authentication required: missing or invalid _api_key")

        if tool_name == "vaara_check":
            return self._call_check(arguments)
        elif tool_name == "vaara_intercept":
            return self._call_intercept(arguments)
        elif tool_name == "vaara_report_outcome":
            return self._call_report(arguments)
        else:
            # Unknown tool → Invalid params (-32602), not Internal (-32603).
            # Raise a marker that handle_request translates properly.
            raise _InvalidParams(f"Unknown tool: {tool_name!r}")

    def _handle_resources_list(self) -> dict:
        return {
            "resources": [r.to_dict() for r in self._resources.values()],
        }

    def _handle_resources_read(self, params: dict) -> dict:
        uri = params.get("uri", "")
        if uri == "vaara://status":
            content = _strict_json_dumps(self._pipeline.status(), indent=2)
        elif uri == "vaara://compliance":
            report = self._pipeline.run_compliance_assessment(
                system_version=self.SERVER_VERSION,
            )
            content = _strict_json_dumps({
                "overall_status": report.overall_status.value,
                "system_name": report.system_name,
                "system_version": report.system_version,
                "generated_at": report.generated_at,
                "articles": [
                    {
                        "article": ev.requirement.article,
                        "title": ev.requirement.title,
                        "status": ev.status.value,
                        "evidence_count": ev.evidence_count,
                        "gaps": ev.gaps,
                        "recommendations": ev.recommendations,
                    }
                    for ev in report.articles
                ],
            }, indent=2)
        else:
            # Unknown URI is an invalid-params condition (-32602), not an
            # internal server error. Use the same marker exception as
            # _handle_tools_call so handle_request maps it correctly.
            raise _InvalidParams(f"Unknown resource: {uri!r}")

        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": content,
            }],
        }

    # ── Tool implementations ─────────────────────────────────────

    def _call_check(self, args: dict) -> dict:
        """Read-only risk assessment — no audit trail."""
        tool_name = args.get("tool_name", "unknown")
        agent_id = args.get("agent_id", "mcp-agent")
        confidence = args.get("confidence")
        # A non-string tool_name (list/int/None/dict) crashes registry.classify
        # via str.startswith and escapes as a generic -32603 Internal error.
        # Per JSON-RPC 2.0 §5.1, malformed client input is -32602 Invalid params;
        # -32603 is reserved for server-side faults.
        if not isinstance(tool_name, str):
            raise _InvalidParams("tool_name must be a string")
        if not isinstance(agent_id, str):
            raise _InvalidParams("agent_id must be a string")

        action_type = self._pipeline.registry.classify(tool_name)
        context = {
            "tool_name": tool_name,
            "agent_id": agent_id,
            "base_risk_score": action_type.base_risk_score,
            "agent_confidence": confidence,
            "reversibility": action_type.reversibility.value,
            "blast_radius": action_type.blast_radius.value,
        }
        # vaara_check is advertised as read-only. scorer.evaluate mutates
        # agent profiles and trips the sequence detector, so use the
        # dry-run path which preserves state.
        if hasattr(self._pipeline.scorer, "dry_run_evaluate"):
            scorer_result = self._pipeline.scorer.dry_run_evaluate(context)
        else:
            logger.warning(
                "vaara_check: scorer %r has no dry_run_evaluate — "
                "falling back to stateful evaluate(), which mutates "
                "agent profiles and sequence-detector state. "
                "Implement dry_run_evaluate to preserve read-only contract.",
                type(self._pipeline.scorer).__name__,
            )
            scorer_result = self._pipeline.scorer.evaluate(context)

        raw = scorer_result.get("raw_result", {})
        interval = raw.get("conformal_interval", [0.2, 0.8])

        return {
            "content": [{
                "type": "text",
                "text": _strict_json_dumps({
                    "tool_name": tool_name,
                    "action_type": action_type.name,
                    "category": action_type.category.value,
                    "decision": scorer_result.get("action", "escalate"),
                    "risk_score": raw.get("point_estimate", 0.5),
                    "risk_interval": interval,
                    "base_risk": action_type.base_risk_score,
                    "calibrated": self._pipeline.scorer.is_calibrated,
                    "note": "Read-only check — use vaara_intercept for audited interception",
                }, indent=2),
            }],
        }

    def _call_intercept(self, args: dict) -> dict:
        """Full interception with audit trail."""
        tool_name = args.get("tool_name", "unknown")
        agent_id = args.get("agent_id", "mcp-agent")
        if not isinstance(tool_name, str):
            raise _InvalidParams("tool_name must be a string")
        if not isinstance(agent_id, str):
            raise _InvalidParams("agent_id must be a string")
        parameters = args.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        session_id = args.get("session_id", "")
        if not isinstance(session_id, str):
            session_id = ""
        result = self._pipeline.intercept(
            agent_id=agent_id,
            tool_name=tool_name,
            parameters=parameters,
            agent_confidence=args.get("confidence"),
            session_id=session_id,
        )

        return {
            "content": [{
                "type": "text",
                "text": _strict_json_dumps({
                    "allowed": result.allowed,
                    "decision": result.decision,
                    "action_id": result.action_id,
                    "risk_score": result.risk_score,
                    "risk_interval": list(result.risk_interval),
                    "reason": result.reason,
                    "action_type": result.action_type.name,
                    "evaluation_ms": round(result.evaluation_ms, 2),
                }, indent=2),
            }],
            "isError": not result.allowed and result.decision == "deny",
        }

    def _call_report(self, args: dict) -> dict:
        """Report outcome for learning."""
        # inputSchema marks both fields as required. Silently defaulting
        # a missing outcome_severity to 0.0 would feed a fake "completely
        # safe" signal into the conformal calibrator and MWU weights,
        # permanently biasing the scorer toward ALLOW. Reject instead.
        if "action_id" not in args or not args.get("action_id"):
            raise _InvalidParams("vaara_report_outcome requires action_id")
        if "outcome_severity" not in args:
            raise _InvalidParams("vaara_report_outcome requires outcome_severity")

        action_id = args["action_id"]
        # Non-string action_id (dict/list/int) hits _pending_outcomes.get
        # with an unhashable key and escapes as -32603. Per JSON-RPC 2.0
        # §5.1 this is client-side malformed input → -32602.
        if not isinstance(action_id, str):
            raise _InvalidParams("action_id must be a string")

        severity = args["outcome_severity"]
        # Null/non-numeric severity passes the presence check but is
        # silently coerced to 0.0 downstream ("completely safe") — the
        # exact false-allow-bias the presence check was written to
        # prevent. Reject at the boundary so missing severity is
        # indistinguishable from malformed severity to the client.
        if severity is None or isinstance(severity, bool):
            raise _InvalidParams("outcome_severity must be a number in [0, 1]")
        try:
            severity_f = float(severity)
        except (TypeError, ValueError):
            raise _InvalidParams("outcome_severity must be a number in [0, 1]")
        if not math.isfinite(severity_f) or not (0.0 <= severity_f <= 1.0):
            raise _InvalidParams("outcome_severity must be a finite number in [0, 1]")
        severity = severity_f

        description = args.get("description", "")
        if not isinstance(description, str):
            description = str(description)

        self._pipeline.report_outcome(
            action_id=action_id,
            outcome_severity=severity,
            description=description,
        )

        return {
            "content": [{
                "type": "text",
                "text": _strict_json_dumps({
                    "recorded": True,
                    "action_id": action_id,
                    "outcome_severity": severity,
                    "calibration_size": self._pipeline.scorer.calibration_size,
                }, indent=2),
            }],
        }

    # ── JSON-RPC helpers ─────────────────────────────────────────

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    # ── stdio transport ──────────────────────────────────────────

    def run(self) -> None:
        """Run the MCP server on stdio (JSON-RPC over stdin/stdout)."""
        logger.info("Vaara MCP server starting on stdio")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                response = self._error_response(None, -32700, "Parse error")
                sys.stdout.write(_strict_json_dumps(response) + "\n")
                sys.stdout.flush()
                continue

            # JSON-RPC 2.0 §4.1: notifications (no "id") MUST NOT get a
            # response. MCP sends notifications/initialized, notifications/
            # cancelled, etc. — answering them confuses the client.
            # Non-dict requests (lists, strings, null) get an Invalid
            # Request error from handle_request and must be written back.
            is_notification = isinstance(request, dict) and "id" not in request
            response = self.handle_request(request)
            if not is_notification:
                sys.stdout.write(_strict_json_dumps(response) + "\n")
                sys.stdout.flush()

    def close(self) -> None:
        """Clean up resources."""
        if self._backend is not None:
            self._backend.close()


# ── CLI entry point ─────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,  # Logs go to stderr, JSON-RPC goes to stdout
    )
    server = VaaraMCPServer()
    try:
        server.run()
    finally:
        server.close()


if __name__ == "__main__":
    main()
