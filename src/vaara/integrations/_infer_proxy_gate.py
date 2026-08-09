# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""The gate: block denied tool calls at the model layer (proxy Phase 2).

With an enforcing pipeline, every tool call the model requests is decided
before the agent ever sees it. A deny rewrites the response so the tool
call is gone and a policy-error text explains why; an escalate blocks on
the file-based approvals handshake (``vaara.approvals``) and fails closed
on timeout or when no approvals directory is configured. One denied call
strips the whole batch: a partially executed tool batch is ambiguous for
the agent, and the explanation text names what was blocked.

Streams cannot be un-sent, so gated streaming requests are buffered
upstream and either replayed byte-for-byte (all allowed) or replaced with
a synthesized, shape-correct stream carrying the policy text.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
from pathlib import Path
from typing import Any, Optional

from vaara.integrations._infer_proxy_govern import _parse_one

logger = logging.getLogger("vaara.infer_proxy")

__all__ = [
    "UNREADABLE_RESPONSE_DENIAL",
    "gate_tool_calls",
    "refusal_buffered",
    "rewrite_buffered",
    "synthesize_stream",
]

#: Denial used when the upstream reply could not be parsed at all. The gate
#: can only decide tool calls it can read, so an unreadable reply is not
#: evidence that there were none in it.
UNREADABLE_RESPONSE_DENIAL = (
    "upstream response: could not be read, so any tool call it contained "
    "could not be decided"
)


async def gate_tool_calls(
    pipeline: Any,
    tool_calls: Any,
    *,
    model_name: str,
    agent_id: str = "infer-proxy",
    approvals_dir: Optional[Path] = None,
    approvals_timeout: float = 60.0,
    allow_patterns: "Optional[list[str]]" = None,
) -> "list[str]":
    """Decide every requested tool call; return the denial messages.

    Empty list means everything is allowed. Each intercept lands in the
    audit trail; approved escalations are resolved on the trail too.
    """
    denials: list[str] = []
    if pipeline is None or not isinstance(tool_calls, list):
        return denials
    for tool_call in tool_calls:
        parsed = _parse_one(tool_call)
        if parsed is None:
            # The observe path skips what it cannot name, because there is
            # nothing to record. The gate must not: a call this does not
            # recognise is still a call the downstream agent framework may
            # well understand, and letting it through undecided is the one
            # outcome the gate exists to prevent. Deny it.
            logger.warning(
                "gate: unrecognised tool-call shape, denying (fail closed)",
            )
            denials.append(
                "unrecognised tool call: could not be read, so it could not "
                "be decided",
            )
            continue
        name, parameters = parsed
        if allow_patterns and any(
            fnmatch.fnmatchcase(name, pat) for pat in allow_patterns
        ):
            # Operator-allow-listed: passes without gating. Recording still
            # happens on the observe path, so the trail keeps the call.
            continue
        try:
            result = pipeline.intercept(
                agent_id=agent_id,
                tool_name=name,
                parameters=parameters,
                context={
                    "vaara_interception_layer": "model_proxy",
                    "model": model_name,
                },
            )
        except Exception:
            logger.exception("gate intercept failed for %r; failing closed", name)
            denials.append(f"{name}: policy evaluation failed")
            continue
        if result.allowed:
            continue
        if result.decision == "escalate" and approvals_dir is not None:
            from vaara.approvals import request_approval

            human = await asyncio.to_thread(
                request_approval,
                result.action_id, name,
                f"risk {result.risk_score:.2f}: {result.reason}",
                approvals_dir=approvals_dir, timeout=approvals_timeout,
            )
            if human in ("approve", "deny"):
                try:
                    pipeline.resolve_escalation(
                        result.action_id,
                        "allow" if human == "approve" else "deny",
                        reviewer="approvals-handshake",
                        justification="human decision via approvals directory",
                    )
                except Exception:
                    logger.exception("could not record escalation resolution")
            if human == "approve":
                continue
        denials.append(f"{name}: {result.reason}")
    return denials


def _policy_text(denials: "list[str]") -> str:
    return "blocked by Vaara policy: " + "; ".join(denials)


def rewrite_buffered(shape: str, parsed: dict, denials: "list[str]") -> dict:
    """Strip every tool call from a buffered response and explain why."""
    doc = json.loads(json.dumps(parsed))  # deep copy, JSON-safe by definition
    text = _policy_text(denials)
    if shape == "anthropic":
        content = [b for b in doc.get("content") or []
                   if not (isinstance(b, dict) and b.get("type") == "tool_use")]
        content.append({"type": "text", "text": text})
        doc["content"] = content
        doc["stop_reason"] = "end_turn"
    elif shape == "ollama":
        message = doc.setdefault("message", {})
        message.pop("tool_calls", None)
        message["content"] = ((message.get("content") or "") + "\n" + text).strip()
        doc["done"] = True
    else:  # openai
        for choice in doc.get("choices") or []:
            message = choice.get("message") or {}
            message.pop("tool_calls", None)
            message["content"] = ((message.get("content") or "") + "\n" + text).strip()
            choice["message"] = message
            choice["finish_reason"] = "stop"
    return doc


def refusal_buffered(shape: str, denials: "list[str]", model_name: str) -> dict:
    """A minimal, shape-correct buffered reply carrying the policy text.

    ``rewrite_buffered`` edits the upstream document in place, which needs a
    document to edit. When the upstream reply could not be parsed there is
    none, so build one from scratch rather than fall back to forwarding the
    bytes the gate could not read.
    """
    text = _policy_text(denials)
    if shape == "anthropic":
        return {
            "id": "vaara_policy_block",
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    if shape == "ollama":
        return {
            "model": model_name,
            "message": {"role": "assistant", "content": text},
            "done": True,
        }
    return {
        "object": "chat.completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


def synthesize_stream(shape: str, denials: "list[str]", model_name: str) -> bytes:
    """A minimal, well-formed stream for the shape, carrying the policy text."""
    text = _policy_text(denials)
    if shape == "anthropic":
        events = [
            ("message_start", {"type": "message_start", "message": {
                "type": "message", "role": "assistant", "model": model_name,
                "content": [], "usage": {"input_tokens": 0, "output_tokens": 0}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn"}, "usage": {}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        return b"".join(
            f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
            for name, data in events
        )
    if shape == "ollama":
        line = {"model": model_name, "message": {"role": "assistant",
                                                 "content": text}, "done": True}
        return (json.dumps(line) + "\n").encode()
    # openai
    delta = {"choices": [{"index": 0, "delta": {"role": "assistant",
                                                "content": text},
                          "finish_reason": None}]}
    stop = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    return (f"data: {json.dumps(delta)}\n\n"
            f"data: {json.dumps(stop)}\n\n"
            "data: [DONE]\n\n").encode()
