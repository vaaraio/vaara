# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""What the provider said it counted. Numbers only, never content.

The envelope says how many bytes left. It cannot say what the provider
charged for them, and the two are not the same number: a request whose
prefix is unchanged is read from the provider's cache at a fraction of the
price, and a request that rewrites one early byte is not. The reply carries
that split and the proxy used to drop it.

``extract_usage`` normalises the Anthropic and the OpenAI shapes into one
set of keys, so a record can be read the same way whichever provider served
it. ``StreamUsage`` does the same for a server-sent-event stream, where the
input counts arrive in the first frame and the output count in the last.

Nothing here reads a token of the prompt or the completion. Every value is
an integer the provider put in its own reply.
"""
from __future__ import annotations

import json
from typing import Any, Optional

#: The normalised keys. Absent rather than zero when the provider said
#: nothing, so a reader can tell "not reported" from "reported as none".
_KEYS = ("input_tokens", "output_tokens",
         "cache_read_input_tokens", "cache_creation_input_tokens")


def _normalise(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}

    def put(key: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            return
        out[key] = value

    # Anthropic names them outright.
    put("input_tokens", usage.get("input_tokens"))
    put("output_tokens", usage.get("output_tokens"))
    put("cache_read_input_tokens", usage.get("cache_read_input_tokens"))
    put("cache_creation_input_tokens",
        usage.get("cache_creation_input_tokens"))

    # The Responses API borrows Anthropic's names with OpenAI's meaning: its
    # input_tokens INCLUDES the cached share, reported in
    # input_tokens_details. Anthropic never sends that key, so its presence
    # says which meaning applies.
    details = usage.get("input_tokens_details")
    if isinstance(details, dict) and isinstance(
            details.get("cached_tokens"), int) and "input_tokens" in out:
        cached = details["cached_tokens"]
        out["cache_read_input_tokens"] = cached
        out["input_tokens"] = max(0, out["input_tokens"] - cached)

    # OpenAI uses prompt/completion, and hides the cached share one level
    # down. Its prompt_tokens INCLUDES the cached part, where Anthropic's
    # input_tokens excludes it, so the cached count is subtracted back out
    # to make the two shapes mean the same thing.
    if "input_tokens" not in out and isinstance(
            usage.get("prompt_tokens"), int):
        cached = 0
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(
                details.get("cached_tokens"), int):
            cached = details["cached_tokens"]
            out["cache_read_input_tokens"] = cached
        out["input_tokens"] = max(0, usage["prompt_tokens"] - cached)
    if "output_tokens" not in out:
        put("output_tokens", usage.get("completion_tokens"))
    return out


def summarise(usage: dict[str, int]) -> dict[str, Any]:
    """Add the one derived number that matters: the cached share of input.

    Billed-input is what the provider read fresh plus what it wrote to cache;
    cache reads are the rest, and they are the cheap part. The percentage is
    the regression alarm: a prefix that stopped being stable shows up here as
    a drop, on the first call, without anyone estimating anything.
    """
    if not usage:
        return {}
    out: dict[str, Any] = dict(usage)
    fresh = usage.get("input_tokens", 0)
    created = usage.get("cache_creation_input_tokens", 0)
    read = usage.get("cache_read_input_tokens", 0)
    total = fresh + created + read
    if total > 0 and ("cache_read_input_tokens" in usage
                      or "cache_creation_input_tokens" in usage):
        out["cache_hit_pct"] = round(100.0 * read / total, 1)
    return out


def extract_usage(raw: bytes) -> dict[str, Any]:
    """Usage out of a non-streamed reply body. ``{}`` when there is none."""
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(body, dict):
        return {}
    return summarise(_normalise(body.get("usage")))


class StreamUsage:
    """Usage accumulated across a server-sent-event stream.

    Anthropic reports the input counts in ``message_start`` and the final
    output count in ``message_delta``; OpenAI, when asked for it, sends one
    trailing chunk carrying the whole object. Both land in the same keys.

    Bytes are fed as they pass through, so this never buffers the stream: a
    partial trailing line is held until the rest of it arrives, and only
    complete ``data:`` lines are parsed. A frame that does not parse is
    skipped, because a usage number is not worth breaking a reply over.
    """

    def __init__(self) -> None:
        self._usage: dict[str, int] = {}
        self._pending = b""

    def feed(self, chunk: bytes) -> None:
        self._pending += chunk
        # Keep whatever follows the last newline: it may be half a line.
        head, sep, tail = self._pending.rpartition(b"\n")
        if not sep:
            return
        self._pending = tail
        for line in head.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                event = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            self._absorb(event.get("usage"))
            # Anthropic nests it in message_start's "message", the
            # Responses API in response.completed's "response".
            for key in ("message", "response"):
                inner = event.get(key)
                if isinstance(inner, dict):
                    self._absorb(inner.get("usage"))

    def _absorb(self, usage: Any) -> None:
        for key, value in _normalise(usage).items():
            # Later frames supersede earlier ones: the output count grows as
            # the stream runs, and the input counts are sent once. Taking the
            # larger value is right for both and is immune to a frame that
            # repeats an input count as zero.
            if value >= self._usage.get(key, 0):
                self._usage[key] = value

    @property
    def usage(self) -> dict[str, Any]:
        return summarise(self._usage)


def usage_fields(usage: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The outcome-description fields for a usage dict. Empty when unknown."""
    if not usage:
        return {}
    return {"usage": usage}
