"""MCP proxy: Vaara as a transparent runtime governance layer for MCP.

Sits between an MCP client (Claude Code, Cursor, any MCP-capable host) and an
upstream MCP server. Forwards every request to the upstream, but routes the
governed MCP surfaces through Vaara first:

* ``tools/call`` runs through the full interception pipeline (classify, score,
  decide, audit) and either flows through or returns an MCP tool error.
* ``resources/read`` and ``prompts/get`` write a request+decision audit pair
  for every access so a regulator can reconstruct what the agent read or
  retrieved. These are read-oriented MCP surfaces. The perimeter
  allow/deny lists gate exposure, but they do not run through the risk
  scorer.
* ``tools/list``, ``resources/list``, and ``prompts/list`` are filtered
  symmetrically against the operator-supplied allow/deny lists before the
  client sees them.

Operator-side filtering (``--allow-tool``/``--deny-tool``,
``--allow-resource``/``--deny-resource``, ``--allow-prompt``/``--deny-prompt``):
when set, the proxy filters the upstream's discovery response before the
client sees it, and rejects matching access at the perimeter with a
``FILTERED`` block payload, without contacting the upstream.
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
from vaara.taxonomy.actions import ActionRequest

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
        allowlist: Optional[set[str]] = None,
        denylist: Optional[set[str]] = None,
        resource_allowlist: Optional[set[str]] = None,
        resource_denylist: Optional[set[str]] = None,
        prompt_allowlist: Optional[set[str]] = None,
        prompt_denylist: Optional[set[str]] = None,
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
        # An empty allowlist means "no restriction" (None and empty set are equivalent
        # here); a non-empty allowlist restricts to exactly those names. Denylist
        # always subtracts. When both are set, denylist wins on overlap.
        self._allowlist: Optional[set[str]] = set(allowlist) if allowlist else None
        self._denylist: set[str] = set(denylist) if denylist else set()
        self._resource_allowlist: Optional[set[str]] = (
            set(resource_allowlist) if resource_allowlist else None
        )
        self._resource_denylist: set[str] = (
            set(resource_denylist) if resource_denylist else set()
        )
        self._prompt_allowlist: Optional[set[str]] = (
            set(prompt_allowlist) if prompt_allowlist else None
        )
        self._prompt_denylist: set[str] = (
            set(prompt_denylist) if prompt_denylist else set()
        )
        self._stdout_lock = threading.Lock()
        self._upstream = UpstreamMCPClient(
            command=upstream_command,
            on_notification=self._forward_notification_to_client,
        )

    @staticmethod
    def _is_filtered(name: object, allowlist: Optional[set[str]], denylist: set[str]) -> bool:
        if not isinstance(name, str):
            return True
        if name in denylist:
            return True
        if allowlist is not None and name not in allowlist:
            return True
        return False

    def _tool_filtered(self, name: str) -> bool:
        return self._is_filtered(name, self._allowlist, self._denylist)

    def _resource_filtered(self, uri: str) -> bool:
        return self._is_filtered(uri, self._resource_allowlist, self._resource_denylist)

    def _prompt_filtered(self, name: str) -> bool:
        return self._is_filtered(name, self._prompt_allowlist, self._prompt_denylist)

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
        if method == "tools/list":
            try:
                return self._handle_tools_list(request)
            except ProxyError as e:
                return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")
        if method == "resources/list":
            try:
                return self._handle_list(
                    request, "resources", "uri",
                    self._resource_allowlist, self._resource_denylist,
                )
            except ProxyError as e:
                return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")
        if method == "resources/read":
            try:
                return self._handle_resources_read(request)
            except ProxyError as e:
                return self._error_response(req_id, -32603, str(e))
            except Exception:
                logger.exception("Error in resources/read interception")
                return self._error_response(req_id, -32603, "Internal proxy error")
        if method == "prompts/list":
            try:
                return self._handle_list(
                    request, "prompts", "name",
                    self._prompt_allowlist, self._prompt_denylist,
                )
            except ProxyError as e:
                return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")
        if method == "prompts/get":
            try:
                return self._handle_prompts_get(request)
            except ProxyError as e:
                return self._error_response(req_id, -32603, str(e))
            except Exception:
                logger.exception("Error in prompts/get interception")
                return self._error_response(req_id, -32603, "Internal proxy error")
        try:
            return self._upstream.request(request)
        except ProxyError as e:
            return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")

    def _handle_tools_list(self, request: dict) -> dict:
        return self._handle_list(
            request, "tools", "name", self._allowlist, self._denylist,
        )

    def _handle_list(
        self,
        request: dict,
        item_key: str,
        name_field: str,
        allowlist: Optional[set[str]],
        denylist: set[str],
    ) -> dict:
        """Filter a discovery-list response by operator-supplied perimeter lists.

        Shared between ``tools/list``, ``resources/list``, and ``prompts/list``.
        ``item_key`` is the field on ``result`` holding the array. ``name_field``
        is the identifier on each item to match against the lists ("name" for
        tools and prompts, "uri" for resources).
        """
        response = self._upstream.request(request)
        if not (denylist or allowlist is not None):
            return response
        if not isinstance(response, dict) or "result" not in response:
            return response
        result = response.get("result")
        if not isinstance(result, dict):
            return response
        items = result.get(item_key)
        if not isinstance(items, list):
            return response
        filtered = [
            item for item in items
            if isinstance(item, dict)
            and not self._is_filtered(item.get(name_field, ""), allowlist, denylist)
        ]
        # Mutate a shallow copy so the upstream response object the reader
        # parked is not aliased into the client-facing payload.
        new_result = dict(result)
        new_result[item_key] = filtered
        return {**response, "result": new_result}

    def _handle_tools_call(self, request: dict) -> dict:
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if self._tool_filtered(tool_name):
            logger.warning(
                "tools/call rejected at perimeter (operator filter): %s", tool_name,
            )
            block_payload = {
                "vaara_blocked": True,
                "reason": "Tool filtered by operator policy",
                "decision": "FILTERED",
                "tool": tool_name,
            }
            return {
                "jsonrpc": "2.0", "id": request.get("id"),
                "result": {
                    "content": [{"type": "text", "text": strict_json_dumps(block_payload, indent=2)}],
                    "isError": True,
                },
            }
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

    def _handle_resources_read(self, request: dict) -> dict:
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        uri = params.get("uri", "")
        if not isinstance(uri, str):
            uri = ""
        if self._resource_filtered(uri):
            logger.warning(
                "resources/read rejected at perimeter (operator filter): %s", uri,
            )
            return self._perimeter_block(
                request, code=-32000,
                message="Resource filtered by Vaara operator policy",
                data={
                    "vaara_blocked": True,
                    "reason": "Resource filtered by operator policy",
                    "decision": "FILTERED",
                    "uri": uri,
                },
            )
        self._record_perimeter_audit(
            agent_id=self._agent_id_default,
            tool_name="mcp.resource.read",
            parameters={"uri": uri},
            decision="allow",
            reason="resource within operator perimeter",
        )
        return self._upstream.request(request)

    def _handle_prompts_get(self, request: dict) -> dict:
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        name = params.get("name", "")
        if not isinstance(name, str):
            name = ""
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if self._prompt_filtered(name):
            logger.warning(
                "prompts/get rejected at perimeter (operator filter): %s", name,
            )
            return self._perimeter_block(
                request, code=-32000,
                message="Prompt filtered by Vaara operator policy",
                data={
                    "vaara_blocked": True,
                    "reason": "Prompt filtered by operator policy",
                    "decision": "FILTERED",
                    "prompt": name,
                },
            )
        self._record_perimeter_audit(
            agent_id=self._agent_id_default,
            tool_name="mcp.prompt.get",
            parameters={"name": name, "arguments": arguments},
            decision="allow",
            reason="prompt within operator perimeter",
        )
        return self._upstream.request(request)

    @staticmethod
    def _perimeter_block(request: dict, code: int, message: str, data: dict) -> dict:
        """Return a JSON-RPC error for resources/prompts blocked at the perimeter.

        ``resources/read`` and ``prompts/get`` carry no ``isError`` envelope
        (unlike ``tools/call``'s CallToolResult), so a perimeter block must
        surface as a JSON-RPC error. Code -32000 is the server-defined
        application-error range. The ``data`` field carries the same
        ``vaara_blocked``/``decision``/``reason`` payload tool-call
        blocks emit, so downstream readers can use one schema.
        """
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": code, "message": message, "data": data},
        }

    def _record_perimeter_audit(
        self,
        agent_id: str,
        tool_name: str,
        parameters: dict,
        decision: str,
        reason: str,
    ) -> None:
        """Write a request+decision audit pair for a read-oriented MCP access.

        Resource reads and prompt gets are read-oriented MCP surfaces,
        not actions. They need audit coverage so regulators can
        reconstruct what was accessed, but they do not run through the
        risk scorer. Writing directly to the trail bypasses scorer and
        policy while still producing the two records that anchor every
        access to the hash chain. Failures here are logged and
        swallowed: a perimeter audit failure must not block legitimate
        upstream traffic.
        """
        import time as _time
        try:
            registry = self._pipeline.registry
            action_type = registry.classify(tool_name, parameters)
            req = ActionRequest(
                agent_id=agent_id,
                tool_name=tool_name,
                action_type=action_type,
                parameters=parameters or {},
                timestamp_utc=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            )
            action_id = self._pipeline.trail.record_action_requested(req)
            self._pipeline.trail.record_decision(
                action_id=action_id,
                agent_id=agent_id,
                tool_name=tool_name,
                decision=decision,
                reason=reason,
                risk_score=0.0,
                regulatory_domains=action_type.regulatory_domains,
            )
        except Exception:
            logger.exception("Failed to record perimeter audit for %s", tool_name)

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
    parser.add_argument("--allow-tool", action="append", default=[], dest="allow_tools",
                        help="Only expose this tool name (repeatable). If any --allow-tool "
                             "is given, all other upstream tools are filtered from tools/list "
                             "and rejected at tools/call.")
    parser.add_argument("--deny-tool", action="append", default=[], dest="deny_tools",
                        help="Filter this tool name from tools/list and reject any tools/call "
                             "to it (repeatable). Denylist wins on overlap with --allow-tool.")
    parser.add_argument("--allow-resource", action="append", default=[], dest="allow_resources",
                        help="Only expose this resource URI (repeatable). If any "
                             "--allow-resource is given, all other upstream resources are "
                             "filtered from resources/list and rejected at resources/read.")
    parser.add_argument("--deny-resource", action="append", default=[], dest="deny_resources",
                        help="Filter this resource URI from resources/list and reject any "
                             "resources/read to it (repeatable). Denylist wins on overlap.")
    parser.add_argument("--allow-prompt", action="append", default=[], dest="allow_prompts",
                        help="Only expose this prompt name (repeatable). If any --allow-prompt "
                             "is given, all other upstream prompts are filtered from "
                             "prompts/list and rejected at prompts/get.")
    parser.add_argument("--deny-prompt", action="append", default=[], dest="deny_prompts",
                        help="Filter this prompt name from prompts/list and reject any "
                             "prompts/get to it (repeatable). Denylist wins on overlap.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    proxy = VaaraMCPProxy(
        upstream_command=[args.upstream, *args.upstream_args],
        db_path=args.db, agent_id_default=args.agent_id,
        allowlist=set(args.allow_tools) if args.allow_tools else None,
        denylist=set(args.deny_tools) if args.deny_tools else None,
        resource_allowlist=set(args.allow_resources) if args.allow_resources else None,
        resource_denylist=set(args.deny_resources) if args.deny_resources else None,
        prompt_allowlist=set(args.allow_prompts) if args.allow_prompts else None,
        prompt_denylist=set(args.deny_prompts) if args.deny_prompts else None,
    )
    try:
        proxy.run()
    finally:
        proxy.close()


if __name__ == "__main__":
    main()
