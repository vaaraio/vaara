# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""FastAPI app factory for the inference proxy.

Builds the transparent reverse proxy that signs chat calls and
passes everything else through. Public surface is ``infer_proxy``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from vaara.integrations._infer_proxy_emit import InferenceAttestEmitter
from vaara.integrations._infer_proxy_model import CHAT_PATHS, ModelResolver
from vaara.integrations._infer_proxy_shape import (
    StreamAccumulator,
    extract_sampling,
    forward_request_headers,
    forward_response_headers,
    parse_ollama_response,
    parse_openai_response,
)

logger = logging.getLogger("vaara.infer_proxy")


def build_app(
    *, emitter: InferenceAttestEmitter, upstream: str, client: Any = None
) -> Any:
    """Build the FastAPI app fronting ``upstream`` and signing chat calls.

    ``client`` injects the upstream ``httpx.AsyncClient`` (tests pass one with
    a ``MockTransport``); in production it is created here.
    """
    upstream = upstream.rstrip("/")
    if client is None:
        import httpx

        client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    resolver = ModelResolver(client, upstream)
    app = FastAPI(title="vaara-infer-proxy")

    async def _handle_buffered(url, body, fwd_headers, is_ollama, emitted):
        try:
            upstream_resp = await client.post(url, content=body, headers=fwd_headers)
        except Exception as exc:
            # Upstream unreachable: complete the chain with an honest errored
            # receipt rather than leaving a dangling attestation, then surface
            # the failure to the caller as a gateway error.
            logger.warning("Upstream request failed: %s", exc)
            if emitted is not None:
                attestation, counter = emitted
                emitter.emit_receipt(
                    attestation=attestation, counter=counter, status="errored",
                    output=None, eval_stats=None,
                )
            return Response(
                content=json.dumps({"error": "upstream request failed"}),
                status_code=502, media_type="application/json",
            )
        status = "completed" if upstream_resp.is_success else "errored"
        output: Any = None
        eval_stats: Optional[dict[str, int]] = None
        if upstream_resp.is_success:
            try:
                parsed = upstream_resp.json()
                output, eval_stats = (
                    parse_ollama_response(parsed)
                    if is_ollama
                    else parse_openai_response(parsed)
                )
                eval_stats = eval_stats or None
            except Exception:
                logger.debug("Buffered response parse failed", exc_info=True)
        if emitted is not None:
            attestation, counter = emitted
            emitter.emit_receipt(
                attestation=attestation, counter=counter, status=status,
                output=output, eval_stats=eval_stats,
            )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=forward_response_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type"),
        )

    async def _handle_stream(url, body, fwd_headers, is_ollama, emitted):
        # Tee the bytes to the client while accumulating, then sign the receipt
        # once the upstream stream completes.
        stream_cm = client.stream("POST", url, content=body, headers=fwd_headers)
        try:
            upstream_resp = await stream_cm.__aenter__()
        except Exception as exc:
            logger.warning("Upstream stream failed to open: %s", exc)
            if emitted is not None:
                attestation, counter = emitted
                emitter.emit_receipt(
                    attestation=attestation, counter=counter, status="errored",
                    output=None, eval_stats=None,
                )
            return Response(
                content=json.dumps({"error": "upstream stream failed"}),
                status_code=502, media_type="application/json",
            )
        status_code = upstream_resp.status_code
        media_type = upstream_resp.headers.get("content-type")
        resp_headers = forward_response_headers(upstream_resp.headers)

        async def _tee() -> Any:
            acc = StreamAccumulator(is_ollama)
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    acc.feed(chunk)
                    yield chunk
            finally:
                await stream_cm.__aexit__(None, None, None)
                if emitted is not None:
                    attestation, counter = emitted
                    status = "completed" if 200 <= status_code < 300 else "errored"
                    output, eval_stats = acc.finalize()
                    emitter.emit_receipt(
                        attestation=attestation, counter=counter, status=status,
                        output=output, eval_stats=eval_stats,
                    )

        return StreamingResponse(
            _tee(), status_code=status_code, media_type=media_type,
            headers=resp_headers,
        )

    async def _handle_chat(full_path: str, body: bytes, request: "Request") -> Any:
        is_ollama = full_path == "/api/chat"
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        model_name = data.get("model") or "unknown"
        messages = data.get("messages")
        sampling = extract_sampling(data, is_ollama)
        stream = bool(data.get("stream"))

        model_derived = await resolver.resolve(model_name)
        emitted = emitter.emit_attestation(
            model_ref=model_name, model_derived=model_derived,
            messages=messages, sampling=sampling,
        )

        url = f"{upstream}{full_path}"
        fwd_headers = forward_request_headers(request.headers)
        fwd_headers["accept-encoding"] = "identity"  # keep the body parseable

        if not stream:
            return await _handle_buffered(url, body, fwd_headers, is_ollama, emitted)
        return await _handle_stream(url, body, fwd_headers, is_ollama, emitted)

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    )
    async def proxy(full_path: str, request: "Request") -> Any:
        full = "/" + full_path
        body = await request.body()
        if request.method == "POST" and full in CHAT_PATHS:
            return await _handle_chat(full, body, request)
        upstream_resp = await client.request(
            request.method, f"{upstream}{full}",
            content=body if body else None,
            headers=forward_request_headers(request.headers),
            params=request.query_params,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=forward_response_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type"),
        )

    @app.on_event("shutdown")
    async def _close() -> None:
        await client.aclose()

    return app
