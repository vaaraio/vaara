# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Record the tool calls a model requests through the interception pipeline.

This is the observe half of the model-proxy governance layer: whatever
framework drove the chat call, the tool requests inside the response are
what the agent is about to do, so they belong in the same audit trail the
MCP proxy and the hooks write. Phase 1 records only (callers pass a
pipeline in shadow/enforce=False mode); the gate that rewrites a denied
tool call into a policy error arrives with Phase 2.

Recording must never break passthrough: any failure is logged and the
response still reaches the caller byte-for-byte.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("vaara.infer_proxy")

__all__ = ["record_tool_calls"]


def _parse_one(tool_call: Any) -> "Optional[tuple[str, dict]]":
    """(tool_name, parameters) from an OpenAI- or ollama-shaped tool call."""
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):  # OpenAI: JSON-encoded string
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"_raw_arguments": arguments}
    if not isinstance(arguments, dict):  # ollama passes a dict already
        arguments = {} if arguments is None else {"_raw_arguments": arguments}
    return name, arguments


def record_tool_calls(
    pipeline: Any,
    tool_calls: Any,
    *,
    model_name: str,
    agent_id: str = "infer-proxy",
    session_id: str = "",
) -> None:
    """Run each requested tool call through ``pipeline.intercept``.

    ``pipeline`` is an ``InterceptionPipeline``; with ``enforce=False`` the
    intercepts classify, score, and audit without gating anything.
    """
    if pipeline is None or not isinstance(tool_calls, list):
        return
    for tool_call in tool_calls:
        parsed = _parse_one(tool_call)
        if parsed is None:
            continue
        name, parameters = parsed
        try:
            pipeline.intercept(
                agent_id=agent_id,
                tool_name=name,
                parameters=parameters,
                context={
                    "vaara_interception_layer": "model_proxy",
                    "model": model_name,
                },
                session_id=session_id,
            )
        except Exception:
            logger.exception(
                "recording requested tool call %r failed; passthrough continues",
                name,
            )
