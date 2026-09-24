# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Tool calls in model replies, decided by the deny rules.

The proxy read only the request. A tool call the model asked for went back
to the agent untouched, so an agent with no Vaara hook of its own (any agent
whose base URL points here) ran whatever the model named. This module reads
the tool calls out of the reply, in the three shapes the proxy serves
(Anthropic messages, OpenAI chat completions, OpenAI Responses), streamed or
not, and runs each through the same deny rules as the hooks: by tool name
where the name is one the rules know, by content otherwise, as for MCP.

Every tool call is recorded. A rule hit is refused only when the proxy
enforces: the call is taken out of the reply and a text block in its place
tells the agent, and the model's user, which rule refused it and why. With
enforcement off the hit is recorded as a rule match that was not enforced.

The gate sees what the model asked for, not what the agent then runs. An
agent can still run something no reply named, which is what the per-agent
hooks are for.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from vaara.deny_rules import (
    load_deny_rules,
    match_deny_rule,
    match_deny_rule_any_field,
)

logger = logging.getLogger("vaara.llm_proxy")


@dataclass
class Verdict:
    allowed: bool
    rule_id: str = ""
    message: str = ""


def tool_args(raw: Any) -> dict:
    """A tool call's arguments as a dict the rules can read.

    Chat and Responses carry them as a JSON string, Anthropic as an object.
    A freeform tool (Codex's ``apply_patch``) carries plain text, which is
    kept under both ``input`` and ``command``, the field its alias reads.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {"input": raw, "command": raw}
        return value if isinstance(value, dict) else {"input": raw}
    return {}


def refusal_text(name: str, verdict: Verdict) -> str:
    return (f"[Vaara refused the tool call {name} (rule {verdict.rule_id}): "
            f"{verdict.message}]")


class ToolGate:
    """Decides and records the tool calls in one reply."""

    def __init__(self, pipeline: Any, agent_id: str, enforce: bool,
                 rules: Optional[list] = None):
        self.pipeline = pipeline
        self.agent_id = agent_id
        self.enforce = enforce
        self.rules = load_deny_rules() if rules is None else rules
        self.refused = 0
        self.passed = 0

    def match(self, name: str, args: dict) -> Optional[tuple[str, str]]:
        hit = match_deny_rule(self.rules, name, args)
        if hit is None:
            try:
                hit = match_deny_rule_any_field(self.rules, args)
            except Exception as exc:  # a broken rule must not take the proxy down
                logger.warning("content deny rules failed: %r", exc)
        return hit

    def decide(self, name: str, args: dict) -> Verdict:
        hit = self.match(name, args)
        digest = hashlib.sha256(
            json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()
        params: dict[str, Any] = {"tool": name, "args_sha256": digest,
                                  "enforce": self.enforce}
        kwargs: dict[str, Any] = {}
        if hit is not None:
            params.update(rule_id=hit[0], rule_message=hit[1],
                          enforced=self.enforce)
            if self.enforce:
                kwargs = {"policy_decision": "deny",
                          "policy_reason": f"deny rule {hit[0]}: {hit[1]}"}
        try:
            self.pipeline.intercept(agent_id=self.agent_id,
                                    tool_name="llm.tool_call",
                                    parameters=params, **kwargs)
        except Exception as exc:  # recording must not decide the call
            logger.error("could not record tool call %s: %r", name, exc)
        if hit is not None and self.enforce:
            self.refused += 1
            return Verdict(False, hit[0], hit[1])
        self.passed += 1
        return Verdict(True)


# ---------------------------------------------------------------------------
# Whole replies


def gate_reply(body: Any, gate: ToolGate) -> Any:
    """The reply with every refused tool call replaced by a text block."""
    if not isinstance(body, dict):
        return body
    if isinstance(body.get("content"), list) and body.get("type") == "message":
        return _gate_anthropic(body, gate)
    if isinstance(body.get("choices"), list):
        return _gate_chat(body, gate)
    if isinstance(body.get("output"), list):
        return _gate_responses(body, gate)
    return body


def _gate_anthropic(body: dict, gate: ToolGate) -> dict:
    blocks, kept_tools, changed = [], 0, False
    for block in body["content"]:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = str(block.get("name", ""))
            verdict = gate.decide(name, tool_args(block.get("input")))
            if not verdict.allowed:
                blocks.append({"type": "text", "text": refusal_text(name, verdict)})
                changed = True
                continue
            kept_tools += 1
        blocks.append(block)
    if not changed:
        return body
    out = {**body, "content": blocks}
    if kept_tools == 0 and out.get("stop_reason") == "tool_use":
        out["stop_reason"] = "end_turn"
    return out


def _gate_chat(body: dict, gate: ToolGate) -> dict:
    choices, changed = [], False
    for choice in body["choices"]:
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list) or not calls:
            choices.append(choice)
            continue
        kept, notes = [], []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            name = str((fn or {}).get("name", ""))
            verdict = gate.decide(name, tool_args((fn or {}).get("arguments")))
            if verdict.allowed:
                kept.append(call)
            else:
                notes.append(refusal_text(name, verdict))
        if not notes:
            choices.append(choice)
            continue
        changed = True
        msg = dict(message)
        text = "\n".join(filter(None, [msg.get("content") or "", *notes]))
        msg["content"] = text
        if kept:
            msg["tool_calls"] = kept
        else:
            msg.pop("tool_calls", None)
        new_choice = {**choice, "message": msg}
        if not kept and new_choice.get("finish_reason") == "tool_calls":
            new_choice["finish_reason"] = "stop"
        choices.append(new_choice)
    return {**body, "choices": choices} if changed else body


_RESPONSES_TOOLS = ("function_call", "custom_tool_call", "local_shell_call")


def _responses_call(item: dict) -> tuple[str, dict]:
    kind = item.get("type")
    if kind == "custom_tool_call":
        return str(item.get("name", "")), tool_args(str(item.get("input", "")))
    if kind == "local_shell_call":
        command = (item.get("action") or {}).get("command")
        text = " ".join(command) if isinstance(command, list) else str(command or "")
        return "local_shell", {"command": text}
    return str(item.get("name", "")), tool_args(item.get("arguments"))


def _refusal_item(item: dict, text: str) -> dict:
    return {"type": "message", "id": item.get("id", ""), "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _gate_responses(body: dict, gate: ToolGate,
                    decided: Optional[dict] = None) -> dict:
    """``decided`` maps item ids already decided in a stream to their
    replacement (or None when allowed), so the final event is not decided,
    or recorded, a second time."""
    out, changed = [], False
    for item in body["output"]:
        if isinstance(item, dict) and item.get("type") in _RESPONSES_TOOLS:
            key = item.get("id") or item.get("call_id")
            if decided is not None and key in decided:
                replacement = decided[key]
            else:
                name, args = _responses_call(item)
                verdict = gate.decide(name, args)
                replacement = None if verdict.allowed else _refusal_item(
                    item, refusal_text(name, verdict))
            if replacement is not None:
                out.append(replacement)
                changed = True
                continue
        out.append(item)
    return {**body, "output": out} if changed else body


# ---------------------------------------------------------------------------
# Streams


def _event(name: Optional[str], data: dict) -> bytes:
    head = f"event: {name}\n" if name else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@dataclass
class _Held:
    raws: list = field(default_factory=list)
    name: str = ""
    args: str = ""
    template: dict = field(default_factory=dict)
    call_id: str = ""


class SseToolGate:
    """Holds each streamed tool call until it is complete, then decides it.

    Text streams through as it arrives. A tool call is held from its first
    event to its last, which costs the agent nothing it could use: a tool
    call is not runnable until its arguments are complete. An allowed call
    is then released as received; a refused one is replaced by a text block
    in the same position.
    """

    def __init__(self, gate: ToolGate):
        self.gate = gate
        self._buf = b""
        # Anthropic: content block index -> held tool_use block.
        self._blocks: dict[int, _Held] = {}
        self._anthropic_kept = 0
        self._anthropic_refused = 0
        # Chat: choice index -> tool call index -> held call.
        self._chat: dict[int, dict[int, _Held]] = {}
        self._chat_raws: dict[int, list] = {}
        # Responses: output index -> held item; item id -> replacement.
        self._items: dict[int, _Held] = {}
        self._decided: dict[str, Optional[dict]] = {}

    def feed(self, chunk: bytes) -> bytes:
        self._buf += chunk
        out = b""
        while True:
            cut = self._split()
            if cut is None:
                return out
            raw, self._buf = self._buf[:cut[0]], self._buf[cut[1]:]
            out += self._on_event(raw + b"\n\n")

    def flush(self) -> bytes:
        out, self._buf = self._buf, b""
        for held in self._blocks.values():
            out += b"".join(held.raws)
        for raws in self._chat_raws.values():
            out += b"".join(raws)
        for held in self._items.values():
            out += b"".join(held.raws)
        self._blocks, self._chat_raws, self._items = {}, {}, {}
        return out

    def _split(self) -> Optional[tuple[int, int]]:
        for sep in (b"\r\n\r\n", b"\n\n"):
            i = self._buf.find(sep)
            if i >= 0:
                return i, i + len(sep)
        return None

    @staticmethod
    def _parse(raw: bytes) -> tuple[Optional[str], Any]:
        name, data = None, []
        for line in raw.decode("utf-8", "replace").splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        text = "\n".join(data)
        if not text or text == "[DONE]":
            return name, None
        try:
            return name, json.loads(text)
        except ValueError:
            return name, None

    def _on_event(self, raw: bytes) -> bytes:
        name, data = self._parse(raw)
        if not isinstance(data, dict):
            return raw
        kind = data.get("type")
        if isinstance(kind, str) and kind.startswith(("content_block", "message_")):
            return self._anthropic(raw, name, data)
        if isinstance(data.get("choices"), list):
            return self._chat_event(raw, data)
        if isinstance(kind, str) and kind.startswith("response."):
            return self._responses(raw, name, data)
        return raw

    # Anthropic ---------------------------------------------------------

    def _anthropic(self, raw: bytes, name: Optional[str], data: dict) -> bytes:
        kind, index = data.get("type"), data.get("index")
        if kind == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._blocks[index] = _Held([raw], str(block.get("name", "")))
                return b""
        if index in self._blocks:
            held = self._blocks[index]
            held.raws.append(raw)
            delta = data.get("delta") or {}
            if kind == "content_block_delta" and delta.get("type") == "input_json_delta":
                held.args += str(delta.get("partial_json", ""))
            if kind != "content_block_stop":
                return b""
            del self._blocks[index]
            verdict = self.gate.decide(held.name, tool_args(held.args or "{}"))
            if verdict.allowed:
                self._anthropic_kept += 1
                return b"".join(held.raws)
            self._anthropic_refused += 1
            text = refusal_text(held.name, verdict)
            return (_event("content_block_start", {
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "text", "text": ""}})
                    + _event("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": text}})
                    + _event("content_block_stop", {
                        "type": "content_block_stop", "index": index}))
        if (kind == "message_delta" and self._anthropic_refused
                and not self._anthropic_kept
                and (data.get("delta") or {}).get("stop_reason") == "tool_use"):
            data = {**data, "delta": {**data["delta"], "stop_reason": "end_turn"}}
            return _event(name, data)
        return raw

    # OpenAI chat -------------------------------------------------------

    def _chat_event(self, raw: bytes, data: dict) -> bytes:
        out = b""
        holding = False
        for choice in data["choices"]:
            if not isinstance(choice, dict):
                continue
            ci = choice.get("index", 0)
            delta = choice.get("delta") or {}
            for call in delta.get("tool_calls") or []:
                held = self._chat.setdefault(ci, {}).setdefault(
                    call.get("index", 0), _Held(template=data))
                fn = call.get("function") or {}
                held.name += str(fn.get("name") or "")
                held.args += str(fn.get("arguments") or "")
                held.call_id = held.call_id or str(call.get("id") or "")
                holding = True
            if ci in self._chat and choice.get("finish_reason"):
                self._chat_raws.setdefault(ci, []).append(raw)
                out += self._release_chat(ci, data, choice)
                return out
        if holding:
            self._chat_raws.setdefault(
                data["choices"][0].get("index", 0) if data["choices"] else 0,
                []).append(raw)
            return out
        return raw

    def _release_chat(self, ci: int, final: dict, choice: dict) -> bytes:
        calls = self._chat.pop(ci)
        raws = self._chat_raws.pop(ci, [])
        kept, notes = [], []
        for ti in sorted(calls):
            held = calls[ti]
            verdict = self.gate.decide(held.name, tool_args(held.args))
            if verdict.allowed:
                kept.append((ti, held))
            else:
                notes.append(refusal_text(held.name, verdict))
        if not notes:
            return b"".join(raws)
        base = {k: v for k, v in final.items() if k != "choices"}
        out = b""
        if kept:
            out += _event(None, {**base, "choices": [{"index": ci, "delta": {
                "tool_calls": [{"index": ti, "id": h.call_id, "type": "function",
                                "function": {"name": h.name, "arguments": h.args}}
                               for ti, h in kept]}, "finish_reason": None}]})
        out += _event(None, {**base, "choices": [{"index": ci, "delta": {
            "content": "\n".join(notes)}, "finish_reason": None}]})
        finish = choice.get("finish_reason")
        if not kept and finish == "tool_calls":
            finish = "stop"
        closing = {**choice, "delta": {}, "finish_reason": finish}
        return out + _event(None, {**final, "choices": [closing]})

    # OpenAI Responses --------------------------------------------------

    def _responses(self, raw: bytes, name: Optional[str], data: dict) -> bytes:
        kind = data.get("type")
        index = data.get("output_index")
        item = data.get("item") if isinstance(data.get("item"), dict) else None
        if kind == "response.output_item.added" and item \
                and item.get("type") in _RESPONSES_TOOLS:
            self._items[index] = _Held([raw])
            return b""
        if kind == "response.output_item.done" and item \
                and item.get("type") in _RESPONSES_TOOLS:
            held = self._items.pop(index, _Held())
            held.raws.append(raw)
            call_name, args = _responses_call(item)
            verdict = self.gate.decide(call_name, args)
            key = item.get("id") or item.get("call_id")
            if verdict.allowed:
                self._decided[key] = None
                return b"".join(held.raws)
            replacement = _refusal_item(item, refusal_text(call_name, verdict))
            self._decided[key] = replacement
            return (_event("response.output_item.added", {
                        **data, "type": "response.output_item.added",
                        "item": {**replacement, "status": "in_progress"}})
                    + _event("response.output_item.done", {**data, "item": replacement}))
        if index is not None and index in self._items:
            self._items[index].raws.append(raw)
            return b""
        if kind in ("response.completed", "response.incomplete") \
                and isinstance(data.get("response"), dict) \
                and isinstance(data["response"].get("output"), list):
            gated = _gate_responses(data["response"], self.gate, self._decided)
            if gated is not data["response"]:
                return _event(name, {**data, "response": gated})
        return raw


def gate_json_bytes(content: bytes, gate: ToolGate) -> bytes:
    """A non-streamed JSON reply, gated; anything else as received."""
    try:
        body = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return content
    gated = gate_reply(body, gate)
    if gated is body:
        return content
    return json.dumps(gated, ensure_ascii=False).encode("utf-8")

