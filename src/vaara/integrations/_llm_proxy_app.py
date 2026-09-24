# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""FastAPI application for the Vaara LLM proxy.

Two modes:
- ``relay`` — blind route, no prompt inspection. Governance on metadata only.
- ``govern`` — inspect prompts, scan for secrets, redact for audit.

Three audit levels:
- ``meta`` — model, provider, agent, timestamp, token count.
- ``hash`` — meta + ``sha256(prompt)``
- ``full`` — meta + redacted prompt messages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from vaara.pipeline import InterceptionPipeline
from ._http_origin import install_origin_guard
from ._llm_proxy_shape import (
    extract_messages,
    extract_model_name,
    flatten_messages,
    forward_request_headers,
    forward_response_headers,
    redact_body,
    truncate_for_audit,
)
from ._llm_proxy_toolgate import SseToolGate, ToolGate, gate_json_bytes
from .llm_compact import compact_messages
from .llm_envelope import measure_envelope
from .llm_usage import StreamUsage, extract_usage, usage_fields

logger = logging.getLogger("vaara.llm_proxy")

#: Ordered so a matched path can be recovered *by index* from the constant
#: rather than passed through from the request. The chat handler only ever runs
#: for a path already known to be in this set, so forwarding the constant is
#: both what we mean and the form that leaves no caller-controlled string in
#: the outgoing URL at all.
_CHAT_PATH_LIST = ["/v1/chat/completions", "/v1/messages", "/v1/responses"]
_CHAT_PATHS = frozenset(_CHAT_PATH_LIST)


def _canonical_chat_path(path: str) -> Optional[str]:
    """The constant this request matched, or None.

    Validating a caller-controlled string and forwarding it still forwards a
    caller-controlled string. Returning the element of the constant list is a
    different thing: whatever the caller sent, the value that reaches the
    upstream URL provably originated here.
    """
    try:
        return _CHAT_PATH_LIST[_CHAT_PATH_LIST.index(f"/{path}")]
    except ValueError:
        return None


def _safe_upstream_path(path: str) -> Optional[str]:
    """Return a path that cannot leave the configured upstream, or None.

    Used for the general pass-through, where the path is not drawn from a fixed
    set and so cannot be replaced by a constant.

    The caller controls this segment, and the proxy holds the operator's
    provider key and injects it into whatever it forwards. A path beginning
    ``//`` is protocol-relative and resolves to a different host, so a request
    to ``//evil.example/x`` would spend that key somewhere else entirely.
    Traversal is refused for the same reason rather than normalised, because a
    normalised path that still escapes is worse than a rejection.
    """
    if not path:
        return "/"
    if path.startswith("/") or "//" in path:
        return None
    if ".." in path.split("/"):
        return None
    if "\\" in path or "\n" in path or "\r" in path:
        return None
    return f"/{path}"


def _detect_provider(upstream: str) -> str:
    u = upstream.lower()
    if "anthropic" in u:
        return "anthropic"
    if "fireworks" in u:
        return "fireworks"
    if "melious" in u:
        return "melious"
    if "openai" in u:
        return "openai"
    return "custom"


def build_app(*, upstream: str, api_key: Optional[str], api_key_header: str,
              pipeline: InterceptionPipeline, mode: str = "relay",
              audit_level: str = "meta", enforce: bool = False,
              model_allow: Optional[list[str]] = None,
              model_deny: Optional[list[str]] = None,
              rate_limit_rpm: int = 0,
              redact_patterns: Optional[list[str]] = None,
              agent_id_header: str = "x-agent-id",
              agent_id_default: str = "llm-agent",
              seal_registry: Optional[Any] = None,
              allowed_origins: Optional[list[str]] = None,
              marker_watch: Optional[Any] = None,
              compact_keep_turns: int = 0,
              fail_open: bool = False,
              gate_tool_calls: bool = True) -> FastAPI:
    app = FastAPI(title="Vaara LLM Proxy")
    # This proxy holds the operator's upstream provider key and injects it
    # into every forwarded call, and it binds loopback with no inbound
    # credential. Without the guard, a page the operator visits can spend
    # that key and land turns in the trail under whatever agent id it likes.
    install_origin_guard(
        app, allowed_origins=allowed_origins, surface="vaara llm-proxy",
    )
    provider = _detect_provider(upstream)
    # Pass-through auth. With no key of its own the proxy has nothing to
    # inject, so the caller's own credential is the only one there is and it
    # travels untouched. This is what lets Vaara govern an agent that
    # authenticates with its own subscription (Claude Code, Cursor) instead of
    # an operator-held API key, which the inject-only path could never do:
    # a subscriber has no API key to hand over, and stripping the token they
    # do have left the request unauthenticated.
    #
    # The credential is forwarded and never recorded. Auditing reads the body,
    # never the headers, so nothing here can put a bearer token in the trail.
    passthrough_auth = api_key is None

    def _upstream_headers(request_headers: Any) -> dict[str, str]:
        headers = forward_request_headers(
            request_headers, keep_auth=passthrough_auth)
        if not passthrough_auth:
            if api_key_header.lower() == "authorization":
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                headers[api_key_header] = api_key
        return headers
    # HTTP/2 needs the optional `h2` package. The llm-proxy extra now pulls
    # it in, but httpx can also arrive from somewhere else without it, and
    # AsyncClient(http2=True) raises ImportError at construction — which
    # took the whole proxy down at startup rather than costing it
    # multiplexing. Fall back to HTTP/1.1, which every provider speaks.
    # Per-phase timeout. httpx defaults to 5s on every phase, including each
    # read of a buffered response, so an upstream that pauses longer than
    # that mid-generation raised ReadTimeout and the caller saw a 502. Long
    # generations pause for longer than 5s routinely, so read is unbounded.
    # Connect and write are not: Timeout(None) everywhere meant a dead
    # upstream hung the request forever with nothing recorded.
    _timeout = httpx.Timeout(connect=10.0, write=30.0, read=None, pool=None)
    try:
        client = httpx.AsyncClient(
            base_url=upstream, http2=True, timeout=_timeout)
    except ImportError:
        logger.info(
            "h2 is not installed; llm-proxy is using HTTP/1.1. Install "
            "'vaara[llm-proxy]' (or httpx[http2]) for HTTP/2 multiplexing.",
        )
        client = httpx.AsyncClient(base_url=upstream, timeout=_timeout)

    _rate_buckets: dict[str, list[float]] = {}
    _model_allow_pats = _compile_glob_patterns(model_allow or [])
    _model_deny_pats = _compile_glob_patterns(model_deny or [])
    _redact_pats = _compile_redact_patterns(redact_patterns) \
        if mode == "govern" else []
    # Sealing is independent of `mode`. Redaction protects the trail; sealing
    # protects the provider request, and an operator may want either alone.
    _seal = seal_registry
    # The marker watch reports which private markers were in the bytes that
    # left, by id. Like the seal it is refreshed per request and its strings
    # never reach a log line or the trail.
    _markers = marker_watch

    # A request whose record did not reach the store is not forwarded. On
    # 2026-09-22 this proxy forwarded 169 prompts across 68 minutes while the
    # trail logged THE TRAIL IS NOT RECORDING on every one: the write failure
    # was counted, and the request went out anyway. Before refusing, the
    # proxy repairs the store and tries once more, because every agent that
    # talks through it, including the one that would fix the trail, stops
    # while it refuses. Repair reads the whole file, so it runs at most once
    # per interval; between attempts a failing store is refused straight away.
    _repair_state = {"last": 0.0}
    _REPAIR_INTERVAL = 30.0

    def _trail_failures() -> int:
        return getattr(pipeline.trail, "persistence_failures", 0)

    def _intercept_recorded(**kwargs: Any) -> tuple[Any, Optional[str]]:
        """``(result, None)`` when the records landed, else ``(result, why)``."""
        before = _trail_failures()
        try:
            result = pipeline.intercept(**kwargs)
        except Exception as exc:
            return None, f"the trail raised {type(exc).__name__}: {exc}"
        if _trail_failures() == before:
            return result, None

        why = "the prompt record did not reach the audit store"
        now = time.time()
        repair = getattr(pipeline.trail, "repair_store", None)
        if repair is None or now - _repair_state["last"] < _REPAIR_INTERVAL:
            return result, why
        _repair_state["last"] = now
        try:
            report = repair()
        except Exception as exc:  # repair must never take the proxy down
            logger.error("trail repair raised %s: %s", type(exc).__name__, exc)
            return result, f"{why}, and repair raised {type(exc).__name__}"
        if report is None or not report.ok:
            detail = getattr(report, "error", None) or "no repairable store"
            logger.error("trail repair failed: %s", detail)
            return result, f"{why}, and repair failed ({detail})"
        logger.warning(
            "trail repaired by %s after a failed write; %d record(s) lost%s",
            report.method, len(report.lost_seqs),
            f", damaged file kept at {report.damaged_copy}"
            if report.damaged_copy else "",
        )
        before = _trail_failures()
        try:
            result = pipeline.intercept(**kwargs)
        except Exception as exc:
            return None, f"{why}, repaired, and the retry raised {type(exc).__name__}"
        if _trail_failures() == before:
            return result, None
        return result, f"{why}, and still after repair by {report.method}"

    def _unrecorded_response(why: str) -> JSONResponse:
        trail_path = None
        backend = getattr(pipeline.trail, "_chain_backend", lambda: None)()
        if backend is not None:
            trail_path = getattr(backend, "db_path", None) or getattr(backend, "_db_path", None)
        where = f" at {trail_path}" if trail_path else ""
        return JSONResponse(
            {
                "error": (
                    f"vaara llm-proxy: this request was not forwarded because "
                    f"{why}. The audit trail{where} is not recording, and this "
                    "proxy does not forward what it cannot record. It retries "
                    "the repair on a later request by itself. By hand: `vaara "
                    f"trail repair --db {trail_path or '<trail>'}`. Do not delete "
                    "the file, it is the evidence. To forward unrecorded on "
                    "purpose, restart the proxy with --fail-open."
                ),
                "type": "vaara_trail_not_recording",
            },
            status_code=503,
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def handle_request(path: str, request: Request):
        if request.method == "POST" and f"/{path}" in _CHAT_PATHS:
            return await _handle_chat(path, request)
        return await _proxy_pass_through(path, request)

    async def _handle_chat(path: str, request: Request) -> Response:
        agent_id = request.headers.get(agent_id_header, agent_id_default)
        body_bytes = await request.body()
        if not body_bytes:
            return JSONResponse({"error": "empty request body"}, status_code=400)

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        model_name = extract_model_name(body)

        if _model_deny_pats and _matches_any(model_name, _model_deny_pats):
            result = pipeline.intercept(
                agent_id=agent_id, tool_name="llm.prompt",
                parameters={"model": model_name, "reason": "model denied"},
            )
            pipeline.report_outcome(result.action_id, outcome_severity=1.0)
            return JSONResponse(
                {"error": f"Model '{model_name}' is denied"}, status_code=403)

        if _model_allow_pats and not _matches_any(model_name, _model_allow_pats):
            result = pipeline.intercept(
                agent_id=agent_id, tool_name="llm.prompt",
                parameters={"model": model_name, "reason": "model not allowed"},
            )
            pipeline.report_outcome(result.action_id, outcome_severity=1.0)
            return JSONResponse(
                {"error": f"Model '{model_name}' not allowed"}, status_code=403)

        if rate_limit_rpm > 0:
            now = time.time()
            window = _rate_buckets.setdefault(agent_id, [])
            cutoff = now - 60.0
            window[:] = [t for t in window if t > cutoff]
            if len(window) >= rate_limit_rpm:
                result = pipeline.intercept(
                    agent_id=agent_id, tool_name="llm.prompt",
                    parameters={"model": model_name, "reason": "rate_limited"},
                )
                pipeline.report_outcome(result.action_id, outcome_severity=0.5)
                return JSONResponse({"error": "rate limited"}, status_code=429)
            window.append(now)

        # Seal named secrets on the way out, BEFORE the trail record is
        # written, so the record can say what actually left. Fails open: a
        # sealing fault costs the sealing, never the request, but the fault
        # is recorded rather than hidden. Forwarding the raw bytes rather than
        # re-serialising the parsed body also keeps the payload byte-identical
        # when sealing is off, which matters because any rewrite of the early
        # message content invalidates the provider's prompt-cache prefix.
        outbound = body_bytes
        # History compaction runs first, so that sealing and the envelope
        # both see what will actually leave. Old tool payloads become
        # size-and-digest stubs; the last N turns and every line of user or
        # assistant text stay as sent. Off by default, and when it changes
        # nothing the raw bytes are forwarded untouched.
        compact_stats = {"bytes_before": len(body_bytes),
                         "bytes_after": len(body_bytes),
                         "compacted_blocks": 0}
        if compact_keep_turns > 0 and isinstance(body.get("messages"), list):
            new_msgs, compact_stats = compact_messages(
                body["messages"], compact_keep_turns)
            if compact_stats["compacted_blocks"] > 0:
                body = dict(body)
                body["messages"] = new_msgs
                outbound = json.dumps(
                    body, ensure_ascii=False).encode("utf-8")
                compact_stats["bytes_before"] = len(body_bytes)
                compact_stats["bytes_after"] = len(outbound)
        seal_state: dict[str, Any] = {
            "seal_active": False, "seal_count": 0, "seal_fault": None,
            "seal_kinds": {}}
        if _seal is not None:
            try:
                _seal.refresh()
            except Exception as exc:  # pragma: no cover - guard, not a path
                logger.warning("seal refresh failed: %s", exc)
            if _seal.active:
                seal_state["seal_active"] = True
                try:
                    unsealed = outbound
                    outbound = _seal.seal_bytes(unsealed)
                    seal_state["seal_count"] = _seal.count_sealed(outbound)
                    seal_state["seal_kinds"] = dict(_seal.last_kinds)
                except Exception as exc:
                    logger.warning(
                        "sealing failed, forwarding unsealed: %s", exc)
                    outbound = unsealed
                    seal_state["seal_fault"] = \
                        f"{type(exc).__name__}: {exc}"[:200]

        audit_params = _build_audit_params(
            body, body_bytes, model_name, provider,
            agent_id, mode, audit_level, _redact_pats)
        audit_params.update(seal_state)
        # Sizes, not content, so they are recorded at every audit level. The
        # envelope is measured on the bytes that leave, after sealing, and so
        # is the marker check: a marker the seal replaced did not leave.
        audit_params["envelope"] = measure_envelope(body, outbound)
        audit_params["envelope"]["bytes_before_compaction"] = \
            compact_stats["bytes_before"]
        audit_params["envelope"]["compacted_blocks"] = \
            compact_stats["compacted_blocks"]
        present: list[str] = []
        if _markers is not None:
            try:
                _markers.refresh()
                present = _markers.present(outbound)
            except Exception as exc:  # pragma: no cover - guard, not a path
                logger.warning("marker watch failed: %s", type(exc).__name__)
        audit_params["markers_present"] = present
        # What governed this request. A reader of the record can then tell
        # a proxy that would have blocked from one that only watched.
        audit_params["enforce"] = enforce
        audit_params["audit_level"] = audit_level

        result, unrecorded = _intercept_recorded(
            agent_id=agent_id, tool_name="llm.prompt",
            parameters=audit_params,
        )
        if unrecorded is not None:
            if not fail_open or result is None:
                logger.error("refusing an unrecorded request: %s", unrecorded)
                return _unrecorded_response(unrecorded)
            logger.error("forwarding UNRECORDED (--fail-open): %s", unrecorded)

        if not result.allowed and enforce:
            pipeline.report_outcome(
                result.action_id, outcome_severity=1.0,
                description=_outcome("denied"))
            return JSONResponse(
                {"error": result.reason or "denied"}, status_code=403)

        headers = _upstream_headers(request.headers)

        is_stream = body.get("stream", False)

        # `path` reached here only because handle_request matched it against
        # _CHAT_PATHS, so the constant is recoverable and is what we forward.
        chat_path = _canonical_chat_path(path)
        if chat_path is None:  # pragma: no cover - dispatcher guarantees it
            return JSONResponse(
                {"error": "invalid upstream path"}, status_code=400)

        try:
            upstream_response = await client.post(
                chat_path, content=outbound, headers=headers,
            )
        except httpx.RequestError as exc:
            # httpx.RequestError often stringifies empty (ReadError,
            # RemoteProtocolError), which logs a failure with no reason.
            # The class name is always there, so lead with it.
            logger.error("Upstream request failed: %s: %s",
                         type(exc).__name__, exc)
            pipeline.report_outcome(
                result.action_id, outcome_severity=0.5,
                description=_outcome(
                    "upstream_error", error=type(exc).__name__))
            return JSONResponse({"error": f"upstream: {exc}"}, status_code=502)

        # Every tool call the model asks for goes through the deny rules
        # before the agent sees it. See _llm_proxy_toolgate.
        gate = ToolGate(pipeline, agent_id, enforce) if gate_tool_calls else None

        if is_stream:
            return StreamingResponse(
                _forward_stream(upstream_response, pipeline,
                                result.action_id, _seal, gate),
                status_code=upstream_response.status_code,
                headers=forward_response_headers(upstream_response.headers),
                media_type=upstream_response.headers.get("content-type"),
            )

        raw_response = upstream_response.content
        unseal_count = 0
        unmapped: list[str] = []
        if _seal is not None and _seal.active:
            try:
                text, unseal_count = _seal.unseal_text_counted(
                    raw_response.decode("utf-8"))
                raw_response = text.encode("utf-8")
                unmapped = _seal.unmapped_placeholders(raw_response)
            except Exception as exc:
                logger.warning("unsealing failed, passing through: %s", exc)
                raw_response = upstream_response.content
        if gate is not None and upstream_response.is_success:
            raw_response = gate_json_bytes(raw_response, gate)

        # What the provider says it counted, including the cached share of
        # the input. Read off the reply the proxy already holds; the envelope
        # alone cannot tell a cheap resend from an expensive one.
        pipeline.report_outcome(
            result.action_id,
            outcome_severity=0.0 if upstream_response.is_success else 0.5,
            description=_outcome(
                "ok" if upstream_response.is_success else "upstream_status",
                http_status=upstream_response.status_code,
                unseal_count=unseal_count, unmapped_placeholders=unmapped,
                **usage_fields(extract_usage(raw_response))))
        return Response(
            content=raw_response,
            status_code=upstream_response.status_code,
            headers=forward_response_headers(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    async def _proxy_pass_through(path: str, request: Request) -> Response:
        # No fixed set to draw from here, so this one is validated rather than
        # replaced. Same reason as the chat path: the proxy injects the
        # operator's key into whatever it forwards.
        safe_path = _safe_upstream_path(path)
        if safe_path is None:
            return JSONResponse(
                {"error": "invalid upstream path"}, status_code=400)
        agent_id = request.headers.get(agent_id_header, agent_id_default)
        body_bytes = await request.body()

        # Until 1.95.0 every path outside _CHAT_PATHS left with no record and
        # no sealing, so a --seal-file secret sent to /v1/responses reached
        # the provider as written. The body here has no shape the proxy
        # knows, so it is sealed as bytes and recorded by size and digest,
        # never by content.
        outbound = body_bytes
        seal_state: dict[str, Any] = {
            "seal_active": False, "seal_count": 0, "seal_fault": None,
            "seal_kinds": {}}
        if _seal is not None and body_bytes:
            try:
                _seal.refresh()
            except Exception as exc:  # pragma: no cover - guard, not a path
                logger.warning("seal refresh failed: %s", exc)
            if _seal.active:
                seal_state["seal_active"] = True
                try:
                    outbound = _seal.seal_bytes(body_bytes)
                    seal_state["seal_count"] = _seal.count_sealed(outbound)
                    seal_state["seal_kinds"] = dict(_seal.last_kinds)
                except Exception as exc:
                    logger.warning(
                        "sealing failed, forwarding unsealed: %s", exc)
                    outbound = body_bytes
                    seal_state["seal_fault"] = \
                        f"{type(exc).__name__}: {exc}"[:200]
        present: list[str] = []
        if _markers is not None and outbound:
            try:
                _markers.refresh()
                present = _markers.present(outbound)
            except Exception as exc:  # pragma: no cover - guard, not a path
                logger.warning("marker watch failed: %s", type(exc).__name__)

        audit_params: dict[str, Any] = {
            "method": request.method,
            "path": safe_path,
            "provider": provider,
            "agent_id": agent_id,
            "bytes_out": len(outbound),
            "sha256_out": hashlib.sha256(outbound).hexdigest(),
            "markers_present": present,
            "enforce": enforce,
        }
        audit_params.update(seal_state)
        result, unrecorded = _intercept_recorded(
            agent_id=agent_id, tool_name="llm.passthrough",
            parameters=audit_params,
        )
        if unrecorded is not None:
            if not fail_open or result is None:
                logger.error("refusing an unrecorded request: %s", unrecorded)
                return _unrecorded_response(unrecorded)
            logger.error("forwarding UNRECORDED (--fail-open): %s", unrecorded)

        if not result.allowed and enforce:
            pipeline.report_outcome(
                result.action_id, outcome_severity=1.0,
                description=_outcome("denied"))
            return JSONResponse(
                {"error": result.reason or "denied"}, status_code=403)

        headers = _upstream_headers(request.headers)
        try:
            resp = await client.request(
                method=request.method, url=safe_path,
                content=outbound, headers=headers,
            )
        except httpx.RequestError as exc:
            logger.error("Upstream request failed: %s: %s",
                         type(exc).__name__, exc)
            pipeline.report_outcome(
                result.action_id, outcome_severity=0.5,
                description=_outcome(
                    "upstream_error", error=type(exc).__name__))
            return JSONResponse({"error": str(exc)}, status_code=502)

        # A reply that is not UTF-8 (a file, an image) cannot carry a
        # placeholder the caller needs back, so it passes as received.
        content = resp.content
        unseal_count = 0
        unmapped: list[str] = []
        if _seal is not None and _seal.active and content:
            try:
                text, unseal_count = _seal.unseal_text_counted(
                    content.decode("utf-8"))
                content = text.encode("utf-8")
                unmapped = _seal.unmapped_placeholders(content)
            except UnicodeDecodeError:
                pass
            except Exception as exc:
                logger.warning("unsealing failed, passing through: %s", exc)
                content = resp.content
        pipeline.report_outcome(
            result.action_id,
            outcome_severity=0.0 if resp.is_success else 0.5,
            description=_outcome(
                "ok" if resp.is_success else "upstream_status",
                http_status=resp.status_code,
                unseal_count=unseal_count, unmapped_placeholders=unmapped))
        return Response(
            content=content, status_code=resp.status_code,
            headers=forward_response_headers(resp.headers),
        )

    return app


def _build_audit_params(body: dict, body_bytes: bytes,
                         model_name: str, provider: str,
                         agent_id: str, mode: str,
                         audit_level: str,
                         redact_pats: list) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model_name,
        "provider": provider,
        "agent_id": agent_id,
        "mode": mode,
    }

    if audit_level == "meta":
        return params

    if audit_level == "hash":
        params["prompt_hash"] = hashlib.sha256(body_bytes).hexdigest()
        params["prompt_bytes"] = len(body_bytes)
        return params

    if mode == "relay":
        params["prompt_hash"] = hashlib.sha256(body_bytes).hexdigest()
        params["prompt_bytes"] = len(body_bytes)
        return params

    msgs = extract_messages(body)
    flat = flatten_messages(msgs)
    params["classification"] = "sensitive" \
        if _contains_sensitive(flat, redact_pats) else "normal"
    redacted = redact_body(body, redact_pats)
    params["messages"] = truncate_for_audit(extract_messages(redacted))
    return params


def _outcome(status: str, **fields: Any) -> str:
    """The outcome description: one JSON object, so a reader can parse it."""
    return json.dumps({"status": status, **fields}, sort_keys=True)


async def _forward_stream(upstream_response: Any,
                           pipeline: InterceptionPipeline,
                           action_id: str,
                           seal: Optional[Any] = None,
                           gate: Optional[ToolGate] = None):
    unsealer = None
    sse_gate = SseToolGate(gate) if gate is not None else None
    if seal is not None and seal.active:
        # Frame-aware, not byte-aware: a placeholder that straddles two SSE
        # events has event framing between its halves and no byte regex can
        # see it. See SseUnsealer for the measurement that found this.
        from .llm_seal import SseUnsealer
        unsealer = SseUnsealer(seal)
    done = False
    # Counted on the chunks as they arrive, before any unsealing: the usage
    # frames carry integers no placeholder appears in, and feeding the raw
    # bytes means a sealing fault cannot also cost the numbers.
    usage = StreamUsage()
    try:
        async for chunk in upstream_response.aiter_bytes():
            usage.feed(chunk)
            out = chunk
            if unsealer is not None:
                try:
                    out = unsealer.feed(chunk)
                except Exception as exc:  # pragma: no cover - guard, not a path
                    logger.warning("stream unseal failed, passing through: %s", exc)
                    unsealer = None
                    out = chunk
            # Gated after unsealing, so a rule reads the argument the agent
            # will get, not a placeholder.
            if sse_gate is not None and out:
                out = sse_gate.feed(out)
            if out:
                yield out
        tail = unsealer.flush() if unsealer is not None else b""
        if sse_gate is not None:
            tail = sse_gate.feed(tail) + sse_gate.flush() if tail else sse_gate.flush()
        if tail:
            yield tail
        done = True
        pipeline.report_outcome(
            action_id, outcome_severity=0.0,
            description=_outcome(
                "ok",
                unseal_count=unsealer.restored if unsealer else 0,
                unmapped_placeholders=unsealer.unmapped if unsealer else [],
                **usage_fields(usage.usage)))
    except BaseException as exc:
        # BaseException, not Exception. A client that goes away mid-stream
        # arrives here as GeneratorExit (Starlette closing the body
        # generator) or CancelledError, and neither is an Exception. With the
        # narrower clause nothing was recorded and the action stayed pending
        # for good: 178 of them at the 2026-09-16 audit.
        if not done:
            aborted = isinstance(exc, (GeneratorExit,
                                       __import__("asyncio").CancelledError))
            logger.warning("Stream %s: %s: %s",
                           "aborted" if aborted else "error",
                           type(exc).__name__, exc)
            pipeline.report_outcome(
                action_id,
                outcome_severity=0.5 if aborted else 0.8,
                description=_outcome(
                    "aborted" if aborted else "stream_error",
                    error=type(exc).__name__,
                    unseal_count=unsealer.restored if unsealer else 0,
                    # A stream that died still spent its input. Whatever the
                    # provider had reported by then is the honest number.
                    **usage_fields(usage.usage)))
        raise


def sweep_orphaned_outcomes(pipeline: InterceptionPipeline,
                            process_started: float) -> int:
    """Close ``llm.prompt`` outcomes left pending by an earlier process.

    A pending outcome older than this process belongs to a request no one
    will ever finish reporting: the process that made it is gone. Closing it
    as ``orphaned`` says so in the trail instead of leaving a request that
    looks in flight forever. Other tools' pending rows are left alone; the
    cross-process ``vaara check`` / ``vaara outcome`` pair owns those.
    """
    backend = getattr(pipeline.trail, "_backend", None)
    lister = getattr(backend, "list_pending_outcomes", None)
    if lister is None:
        return 0
    try:
        rows = lister(tool_name="llm.prompt", created_before=process_started)
    except Exception:
        logger.exception("could not list pending outcomes")
        return 0
    closed = 0
    for row in rows:
        try:
            pipeline.report_outcome(
                row["action_id"], outcome_severity=0.5,
                description=_outcome("orphaned"))
            closed += 1
        except Exception:
            logger.exception("could not close orphaned outcome %s",
                             row.get("action_id"))
    if closed:
        logger.info("closed %d orphaned llm.prompt outcome(s)", closed)
    return closed


def _compile_redact_patterns(extra: Optional[list[str]] = None) -> list[Any]:
    from ._llm_proxy_shape import _DEFAULT_REDACT_PATTERNS
    if not extra:
        return _DEFAULT_REDACT_PATTERNS
    import re
    return _DEFAULT_REDACT_PATTERNS + [re.compile(p) for p in extra]


def _compile_glob_patterns(patterns: list[str]) -> list[Any]:
    import fnmatch
    if not patterns:
        return []
    return [(p, fnmatch.translate(p)) for p in patterns]


def _matches_any(value: str, patterns: list[Any]) -> bool:
    import re
    for raw, regex in patterns:
        if re.fullmatch(regex, value):
            return True
    return False


def _contains_sensitive(text: str, patterns: list[Any]) -> bool:
    for pat in patterns:
        if pat.search(text):
            return True
    return False
