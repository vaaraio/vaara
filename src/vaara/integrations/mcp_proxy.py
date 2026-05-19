"""MCP proxy: Vaara as a transparent runtime governance layer for MCP tool calls.

Sits between an MCP client (Claude Code, Cursor, any MCP-capable host) and an
upstream MCP server (SAP ADT MCP, SAP Graph API MCP, SAP Cloud ALM MCP, any
community-built MCP server). Forwards every request to the upstream, but
routes ``tools/call`` through Vaara's interception pipeline first. Allowed
calls flow through transparently. Blocked calls return an MCP tool error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from vaara import __version__ as _VAARA_VERSION
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail
from vaara.integrations._mcp_upstream import (
    ProxyError, UpstreamMCPClient, strict_json_dumps,
)
from vaara.pipeline import InterceptionPipeline

logger = logging.getLogger(__name__)


class VaaraMCPProxy:
    """Transparent MCP proxy with Vaara interception on tool calls."""

    PROXY_NAME = f"vaara-mcp-proxy/{_VAARA_VERSION}"

    def __init__(
        self,
        upstream_command: list[str],
        pipeline: Optional[InterceptionPipeline] = None,
        db_path: Optional[Path] = None,
        agent_id_default: str = "mcp-proxy-client",
    ) -> None:
        if pipeline is not None:
            self._pipeline = pipeline
            self._backend = None
        else:
            db = db_path or Path(os.environ.get("VAARA_DB", "vaara_audit.db"))
            self._backend = SQLiteAuditBackend(db)
            trail = AuditTrail(on_record=self._backend.write_record)
            self._pipeline = InterceptionPipeline(trail=trail)
        self._agent_id_default = agent_id_default
        self._stdout_lock = threading.Lock()
        self._upstream = UpstreamMCPClient(
            command=upstream_command,
            on_notification=self._forward_notification_to_client,
        )

    def run(self) -> None:
        """Read JSON-RPC from stdin, write to stdout, route through upstream."""
        logger.info("Vaara MCP proxy starting on stdio (%s)", self.PROXY_NAME)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._write_to_client(self._error_response(None, -32700, "Parse error"))
                continue
            # Notifications (no id) forward silently per JSON-RPC 2.0 §4.1.
            if isinstance(request, dict) and "id" not in request:
                try:
                    self._upstream.notify(request)
                except ProxyError:
                    logger.exception("Failed to forward notification")
                continue
            self._write_to_client(self._handle_request(request))

    def _handle_request(self, request: Any) -> dict:
        if not isinstance(request, dict):
            return self._error_response(None, -32600, "Invalid Request: not a JSON object")
        method = request.get("method", "")
        req_id = request.get("id")
        if method == "tools/call":
            try:
                return self._handle_tools_call(request)
            except ProxyError as e:
                return self._error_response(req_id, -32603, str(e))
            except Exception:
                logger.exception("Error in tools/call interception")
                return self._error_response(req_id, -32603, "Internal proxy error")
        try:
            return self._upstream.request(request)
        except ProxyError as e:
            return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")

    def _handle_tools_call(self, request: dict) -> dict:
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        # _vaara_agent_id is a proxy-side override for audit attribution;
        # strip before forwarding so the upstream never sees Vaara metadata.
        agent_id = arguments.pop("_vaara_agent_id", self._agent_id_default)
        if not isinstance(agent_id, str):
            agent_id = self._agent_id_default
        # Unknown upstream tool names classify as generic high-risk in the
        # registry (fail-closed). Correct default for runtime governance.
        result = self._pipeline.intercept(
            agent_id=agent_id, tool_name=tool_name, parameters=arguments,
        )
        if not result.allowed:
            block_payload = {
                "vaara_blocked": True,
                "reason": getattr(result, "reason", None) or "Blocked by Vaara policy",
                "decision": getattr(result, "decision", None),
                "action_id": getattr(result, "action_id", None),
            }
            return {
                "jsonrpc": "2.0", "id": request.get("id"),
                "result": {
                    "content": [{"type": "text", "text": strict_json_dumps(block_payload, indent=2)}],
                    "isError": True,
                },
            }
        upstream_response = self._upstream.request(request)
        outcome_severity = self._severity_from_response(upstream_response)
        try:
            self._pipeline.report_outcome(
                action_id=result.action_id, outcome_severity=outcome_severity,
            )
        except Exception:
            logger.exception("report_outcome failed for action_id=%s", result.action_id)
        return upstream_response

    @staticmethod
    def _severity_from_response(response: dict) -> float:
        # Protocol/tool errors → 1.0 (failure signal). Clean success → 0.0.
        if not isinstance(response, dict) or "error" in response:
            return 1.0
        result = response.get("result")
        if isinstance(result, dict) and result.get("isError"):
            return 1.0
        return 0.0

    def _forward_notification_to_client(self, message: dict) -> None:
        self._write_to_client(message)

    def _write_to_client(self, payload: dict) -> None:
        # Serialize stdout writes between main loop and upstream reader thread.
        with self._stdout_lock:
            sys.stdout.write(strict_json_dumps(payload) + "\n")
            sys.stdout.flush()

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def close(self) -> None:
        self._upstream.close()
        if self._backend is not None:
            self._backend.close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vaara-mcp-proxy",
        description="Vaara runtime governance proxy in front of an upstream MCP server.",
    )
    parser.add_argument("--upstream", required=True, help="Upstream MCP server command")
    parser.add_argument("--upstream-arg", action="append", default=[], dest="upstream_args",
                        help="Argument to pass to the upstream command (repeatable)")
    parser.add_argument("--db", type=Path, default=None,
                        help="Audit database path (default: $VAARA_DB or ./vaara_audit.db)")
    parser.add_argument("--agent-id", default="mcp-proxy-client",
                        help="Default agent_id for the audit trail")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    proxy = VaaraMCPProxy(
        upstream_command=[args.upstream, *args.upstream_args],
        db_path=args.db, agent_id_default=args.agent_id,
    )
    try:
        proxy.run()
    finally:
        proxy.close()


if __name__ == "__main__":
    main()
