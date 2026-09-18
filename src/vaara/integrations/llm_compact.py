"""History compaction: old tool payloads leave the machine once.

A coding agent resends its whole conversation with every call. The bulk of
that conversation is tool traffic: the files the agent read, the files it
wrote, the command output it saw. Each of those payloads left the machine
when it was fresh. Resending it on every later call multiplies the surface
without giving the model anything it did not already have, because a model
that needs an old file again can read it again.

:func:`compact_messages` keeps the last ``keep_turns`` assistant turns
verbatim and, in everything older, replaces tool results and tool inputs
with a stub naming the byte count and a digest prefix. Text from the user or
the assistant is never touched: the ask and the answers are the conversation,
the payloads are its baggage. The rewrite is deterministic, so a given
history always compacts to the same bytes.

Both provider shapes are handled: Anthropic ``tool_use`` and ``tool_result``
content blocks, and OpenAI ``tool_calls`` on assistant messages with
``role: tool`` replies.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

STUB_KEY = "vaara_compacted"


def _payload_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _stub_text(raw: bytes) -> str:
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"[{STUB_KEY}: {len(raw)} bytes, sha256 {digest}]"


def _stub_obj(raw: bytes) -> dict[str, Any]:
    return {STUB_KEY: {"bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()[:16]}}


def compact_messages(messages: list[dict[str, Any]], keep_turns: int
                     ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return a compacted copy of ``messages`` and what the compaction did.

    ``keep_turns`` is the number of most recent assistant turns whose tool
    payloads stay verbatim. ``0`` disables compaction and returns the input
    unchanged.
    """
    before = len(_payload_bytes(messages))
    if keep_turns <= 0 or not messages:
        return messages, {"bytes_before": before, "bytes_after": before,
                          "compacted_blocks": 0}

    # Find the index of the first message that belongs to the kept window:
    # walk back from the end counting assistant turns.
    seen = 0
    cutoff = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            seen += 1
            if seen == keep_turns:
                cutoff = i
                break
    else:
        # fewer assistant turns than keep_turns: nothing is old enough
        return messages, {"bytes_before": before, "bytes_after": before,
                          "compacted_blocks": 0}

    out = copy.deepcopy(messages)
    count = 0
    for msg in out[:cutoff]:
        role = msg.get("role")
        content = msg.get("content")

        # OpenAI: a tool reply is a whole message
        if role == "tool" and isinstance(content, str):
            raw = content.encode("utf-8")
            msg["content"] = _stub_text(raw)
            count += 1
            continue

        # OpenAI: tool_calls carry the arguments string
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and "arguments" in fn:
                raw = _payload_bytes(fn["arguments"])
                fn["arguments"] = json.dumps(_stub_obj(raw))
                count += 1

        # Anthropic: content blocks
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "tool_result" and "content" in block:
                    raw = _payload_bytes(block["content"])
                    block["content"] = _stub_text(raw)
                    count += 1
                elif kind == "tool_use" and "input" in block:
                    raw = _payload_bytes(block["input"])
                    block["input"] = _stub_obj(raw)
                    count += 1

    after = len(_payload_bytes(out))
    return out, {"bytes_before": before, "bytes_after": after,
                 "compacted_blocks": count}
