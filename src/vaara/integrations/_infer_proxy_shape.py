# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Request/response shaping for the inference proxy.

Pulls the sampling params into the request commitment,
reconstructs the output + integer eval counters from a buffered or streamed
chat response, and filters hop-by-hop headers. Public surface is
``infer_proxy``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("vaara.infer_proxy")

# Sampling parameters that, with the messages, define the request. Pulled into
# the request commitment; floats among them are normalized to shortest
# round-trip decimal strings by ``make_request_commitment`` before hashing.
SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "typical_p", "seed",
    "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    "max_tokens", "num_predict", "num_ctx", "stop", "mirostat",
    "mirostat_tau", "mirostat_eta", "tfs_z", "penalize_newline",
})

# Hop-by-hop / length headers not to forward; httpx and Starlette recompute
# framing for the body we actually send.
_DROP_REQUEST_HEADERS = frozenset({"host", "content-length", "accept-encoding"})
_DROP_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection"}
)


def extract_sampling(data: dict[str, Any], is_ollama: bool) -> dict[str, Any]:
    """Pull the sampling params that define the request, OpenAI or ollama."""
    if is_ollama:
        opts = data.get("options") or {}
        return {k: v for k, v in opts.items() if k in SAMPLING_KEYS}
    return {k: data[k] for k in data if k in SAMPLING_KEYS}


def _coerce_eval_stats(raw: dict[str, Any]) -> dict[str, int]:
    """Keep only integer counters (the signed schema rejects floats/bools)."""
    out: dict[str, int] = {}
    for k, v in raw.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out[k] = v
    return out


def parse_ollama_response(obj: dict[str, Any]) -> "tuple[Any, dict[str, int]]":
    message = obj.get("message") or {}
    output: dict[str, Any] = {"content": message.get("content", "")}
    if message.get("tool_calls"):
        output["toolCalls"] = message["tool_calls"]
    stats = _coerce_eval_stats({
        "promptEvalCount": obj.get("prompt_eval_count"),
        "evalCount": obj.get("eval_count"),
        "promptEvalDurationNs": obj.get("prompt_eval_duration"),
        "evalDurationNs": obj.get("eval_duration"),
        "totalDurationNs": obj.get("total_duration"),
        "loadDurationNs": obj.get("load_duration"),
    })
    return output, stats


def parse_openai_response(obj: dict[str, Any]) -> "tuple[Any, dict[str, int]]":
    choices = obj.get("choices") or []
    message = (choices[0].get("message") if choices else None) or {}
    output: dict[str, Any] = {"content": message.get("content", "")}
    if message.get("tool_calls"):
        output["toolCalls"] = message["tool_calls"]
    usage = obj.get("usage") or {}
    stats = _coerce_eval_stats({
        "promptTokens": usage.get("prompt_tokens"),
        "completionTokens": usage.get("completion_tokens"),
        "totalTokens": usage.get("total_tokens"),
    })
    return output, stats


class StreamAccumulator:
    """Buffers a streamed chat response to reconstruct output + eval stats.

    Handles OpenAI SSE (``data: {json}`` ... ``data: [DONE]``) and ollama
    NDJSON (one JSON object per line). Best-effort: a parse failure yields no
    output commitment but never breaks the streamed bytes.
    """

    def __init__(self, is_ollama: bool) -> None:
        self._is_ollama = is_ollama
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._buf.extend(chunk)

    def finalize(self) -> "tuple[Any, Optional[dict[str, int]]]":
        try:
            text = self._buf.decode("utf-8", errors="replace")
            return (
                self._finalize_ollama(text)
                if self._is_ollama
                else self._finalize_openai(text)
            )
        except Exception:
            logger.debug("Stream accumulation parse failed", exc_info=True)
            return None, None

    def _finalize_ollama(self, text: str) -> "tuple[Any, Optional[dict[str, int]]]":
        parts: list[str] = []
        tool_calls: list[Any] = []
        stats: dict[str, int] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message") or {}
            if msg.get("content"):
                parts.append(msg["content"])
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])
            if obj.get("done"):
                stats = parse_ollama_response(obj)[1]
        output: dict[str, Any] = {"content": "".join(parts)}
        if tool_calls:
            output["toolCalls"] = tool_calls
        return output, (stats or None)

    def _finalize_openai(self, text: str) -> "tuple[Any, Optional[dict[str, int]]]":
        parts: list[str] = []
        tool_calls: list[Any] = []
        stats: dict[str, int] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    parts.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
            if obj.get("usage"):
                stats = parse_openai_response(obj)[1]
        output: dict[str, Any] = {"content": "".join(parts)}
        if tool_calls:
            output["toolCalls"] = tool_calls
        return output, (stats or None)


def forward_request_headers(headers: Any) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}


def forward_response_headers(headers: Any) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}
