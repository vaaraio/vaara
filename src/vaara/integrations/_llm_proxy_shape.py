# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Prompt inspection and redaction for the LLM proxy.

Extracts messages, model names, and system prompts from provider-shaped request
bodies.  Redacts sensitive patterns (API keys, secrets) before they reach the
audit trail.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any, Optional

logger = logging.getLogger("vaara.llm_proxy")

_DEFAULT_REDACT_PATTERNS: list[re.Pattern] = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),           # OpenAI keys
    re.compile(r"sk-ant-[a-zA-Z0-9]{20,}"),        # Anthropic keys
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),            # GitHub tokens
    re.compile(r"gho_[a-zA-Z0-9]{36}"),            # GitHub oauth
    re.compile(r"ghu_[a-zA-Z0-9]{36}"),            # GitHub user tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),               # AWS access keys
    re.compile(r"fw_[a-zA-Z0-9]+"),                # Fireworks keys
    re.compile(r"sk-mel-[a-zA-Z0-9]+"),            # Melious keys
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),  # emails
]

KNOWN_AUTH_HEADERS = frozenset({
    "authorization", "x-api-key", "api-key",
})

_DROP_REQUEST_HEADERS = frozenset({
    "host", "content-length", "accept-encoding",
})

_AUDIT_MAX_CHARS = 4096


def extract_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract messages list from any provider-shaped request body.

    Handles OpenAI ``/v1/chat/completions`` and Anthropic ``/v1/messages``
    shapes.  Returns an empty list if the body doesn't look like either.
    """
    if "messages" in body:
        return body["messages"]
    msgs: list[dict[str, Any]] = []
    for block in (body.get("system") or []):
        if isinstance(block, dict) and block.get("type") == "text":
            msgs.append({"role": "system", "content": block.get("text", "")})
    for block in (body.get("messages") or []):
        msgs.append(block)
    for block in (body.get("content") or []):
        if isinstance(block, dict):
            msgs.append({"role": "assistant" if body.get("role") == "assistant" else "user",
                         "content": block.get("text", "")})
    return msgs


def extract_model_name(body: dict[str, Any]) -> str:
    """Extract model identifier from the request body."""
    return body.get("model", "")


def extract_system_prompt(body: dict[str, Any]) -> str:
    """Extract the system prompt, provider-agnostic."""
    if "system" in body:
        if isinstance(body["system"], str):
            return body["system"]
        if isinstance(body["system"], list):
            parts: list[str] = []
            for b in body["system"]:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
            return "".join(parts)
    return ""


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Concatenate message content for scanning."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def redact_string(value: str, patterns: list[re.Pattern]) -> str:
    """Replace all pattern matches with ``***``."""
    result = value
    for pat in patterns:
        result = pat.sub("***", result)
    return result


def redact_body(body: dict[str, Any],
                patterns: Optional[list[re.Pattern]] = None) -> dict[str, Any]:
    """Deep-copy the body and redact matching patterns in all string values.

    Also strips any top-level field whose key hints at a secret
    (e.g. ``api_key``).
    """
    pats = patterns if patterns is not None else _DEFAULT_REDACT_PATTERNS
    copied = deepcopy(body)

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return redact_string(obj, pats)
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                if k.lower() in ("api_key", "apikey", "api-key", "secret",
                                 "password", "token", "passphrase"):
                    out[k] = "***"
                else:
                    out[k] = _walk(v)
            return out
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        return obj

    return _walk(copied)


def truncate_for_audit(messages: list[dict[str, Any]],
                       max_chars: int = _AUDIT_MAX_CHARS) -> list[dict[str, Any]]:
    """Truncate message content so the audit trail doesn't balloon.

    Keeps the first and last messages intact (for context), caps content
    length of messages in between.
    """
    if not messages:
        return messages
    total = len(messages)
    if total <= 2:
        return messages
    out: list[dict[str, Any]] = []
    chars = 0
    for i, msg in enumerate(messages):
        content = msg.get("content") or ""
        if isinstance(content, str):
            remaining = max_chars - chars
            if remaining <= 0 and i > 0 and i < total - 1:
                out.append({**msg, "content": "(truncated)"})
                continue
            if chars + len(content) > max_chars and i > 0 and i < total - 1:
                out.append({**msg, "content": content[:max(0, remaining)] + "…(truncated)"})
                chars = max_chars
            else:
                out.append(msg)
                chars += len(content)
        else:
            out.append(msg)
    return out


def forward_request_headers(headers: Any) -> dict[str, str]:
    """Return headers suitable for forwarding, dropping hop-by-hop and auth."""
    result: dict[str, str] = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in _DROP_REQUEST_HEADERS:
            continue
        if kl in KNOWN_AUTH_HEADERS:
            continue
        result[k] = v
    return result


def forward_response_headers(headers: Any) -> dict[str, str]:
    """Return response headers suitable for forwarding."""
    drop = {"content-length", "content-encoding", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in drop}
