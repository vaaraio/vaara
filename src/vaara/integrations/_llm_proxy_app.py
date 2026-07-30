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
from ._llm_proxy_shape import (
    extract_messages,
    extract_model_name,
    flatten_messages,
    forward_request_headers,
    forward_response_headers,
    redact_body,
    truncate_for_audit,
)

logger = logging.getLogger("vaara.llm_proxy")

_CHAT_PATHS = frozenset({"/v1/chat/completions", "/v1/messages"})


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


def build_app(*, upstream: str, api_key: str, api_key_header: str,
              pipeline: InterceptionPipeline, mode: str = "relay",
              audit_level: str = "meta", enforce: bool = False,
              model_allow: Optional[list[str]] = None,
              model_deny: Optional[list[str]] = None,
              rate_limit_rpm: int = 0,
              redact_patterns: Optional[list[str]] = None,
              agent_id_header: str = "x-agent-id") -> FastAPI:
    app = FastAPI(title="Vaara LLM Proxy")
    provider = _detect_provider(upstream)
    client = httpx.AsyncClient(base_url=upstream, http2=True)

    _rate_buckets: dict[str, list[float]] = {}
    _model_allow_pats = _compile_glob_patterns(model_allow or [])
    _model_deny_pats = _compile_glob_patterns(model_deny or [])
    _redact_pats = _compile_redact_patterns(redact_patterns) \
        if mode == "govern" else []

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def handle_request(path: str, request: Request):
        if request.method == "POST" and f"/{path}" in _CHAT_PATHS:
            return await _handle_chat(path, request)
        return await _proxy_pass_through(path, request)

    async def _handle_chat(path: str, request: Request) -> Response:
        agent_id = request.headers.get(agent_id_header, "llm-agent")
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

        audit_params = _build_audit_params(
            body, body_bytes, model_name, provider,
            agent_id, mode, audit_level, _redact_pats)

        result = pipeline.intercept(
            agent_id=agent_id, tool_name="llm.prompt",
            parameters=audit_params,
        )

        if not result.allowed and enforce:
            pipeline.report_outcome(result.action_id, outcome_severity=1.0)
            return JSONResponse(
                {"error": result.reason or "denied"}, status_code=403)

        headers = forward_request_headers(request.headers)
        if api_key_header.lower() == "authorization":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers[api_key_header] = api_key

        is_stream = body.get("stream", False)
        try:
            upstream_response = await client.post(
                f"/{path}", json=body, headers=headers,
            )
        except httpx.RequestError as exc:
            logger.error("Upstream request failed: %s", exc)
            return JSONResponse({"error": f"upstream: {exc}"}, status_code=502)

        if is_stream:
            return StreamingResponse(
                _forward_stream(upstream_response, pipeline, result.action_id),
                status_code=upstream_response.status_code,
                headers=forward_response_headers(upstream_response.headers),
                media_type=upstream_response.headers.get("content-type"),
            )

        try:
            response_body = upstream_response.json()
        except json.JSONDecodeError:
            pipeline.report_outcome(result.action_id, outcome_severity=0.8)
            return Response(
                content=upstream_response.content,
                status_code=upstream_response.status_code,
                headers=forward_response_headers(upstream_response.headers),
            )

        pipeline.report_outcome(result.action_id, outcome_severity=0.0
                                if upstream_response.is_success else 0.5)
        return JSONResponse(
            content=response_body,
            status_code=upstream_response.status_code,
            headers=forward_response_headers(upstream_response.headers),
        )

    async def _proxy_pass_through(path: str, request: Request) -> Response:
        body_bytes = await request.body()
        headers = forward_request_headers(request.headers)
        if api_key_header.lower() == "authorization":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers[api_key_header] = api_key
        try:
            resp = await client.request(
                method=request.method, url=f"/{path}",
                content=body_bytes, headers=headers,
            )
        except httpx.RequestError as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)
        return Response(
            content=resp.content, status_code=resp.status_code,
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


async def _forward_stream(upstream_response: httpx.Response,
                           pipeline: InterceptionPipeline,
                           action_id: str):
    try:
        async for chunk in upstream_response.aiter_bytes():
            yield chunk
        pipeline.report_outcome(action_id, outcome_severity=0.0)
    except Exception as exc:
        logger.warning("Stream error: %s", exc)
        pipeline.report_outcome(action_id, outcome_severity=0.8)
        raise


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
