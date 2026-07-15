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
    parse_anthropic_response,
    parse_ollama_response,
    parse_openai_response,
)

logger = logging.getLogger("vaara.infer_proxy")


def build_app(
    *, emitter: Optional[InferenceAttestEmitter], upstream: str,
    client: Any = None, pipeline: Any = None,
    approvals_dir: Any = None, approvals_timeout: float = 60.0,
    allow_patterns: Any = None,
) -> Any:
    """Build the FastAPI app fronting ``upstream`` and signing chat calls.

    ``client`` injects the upstream ``httpx.AsyncClient`` (tests pass one with
    a ``MockTransport``); in production it is created here.

    ``pipeline`` is an optional ``InterceptionPipeline``: every tool call the
    model requests in a chat response (buffered or streamed) is run through
    ``pipeline.intercept`` so it lands in the audit trail. Pass a pipeline
    with ``enforce=False`` to observe without gating; recording failures
    never break passthrough. With an ENFORCING pipeline the proxy gates:
    denied tool calls are rewritten out of the response, escalations block
    on the approvals handshake in ``approvals_dir`` (fail closed on timeout
    or when unset), and gated streams are buffered then replayed or
    synthesized.
    """
    upstream = upstream.rstrip("/")
    gating = pipeline is not None and getattr(pipeline, "_enforce", False)
    if client is None:
        import httpx

        client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    resolver = ModelResolver(client, upstream)
    app = FastAPI(title="vaara-infer-proxy")

    def _record_requested_tools(output: Any, model_name: str) -> None:
        if pipeline is None or not isinstance(output, dict):
            return
        from vaara.integrations._infer_proxy_govern import record_tool_calls

        record_tool_calls(pipeline, output.get("toolCalls"),
                          model_name=model_name)

    async def _gate(output: Any, model_name: str) -> "list[str]":
        """Decide the requested tool calls; returns denial messages."""
        if not isinstance(output, dict):
            return []
        from vaara.integrations._infer_proxy_gate import gate_tool_calls

        return await gate_tool_calls(
            pipeline, output.get("toolCalls"), model_name=model_name,
            approvals_dir=approvals_dir, approvals_timeout=approvals_timeout,
            allow_patterns=allow_patterns,
        )

    async def _handle_buffered(url, body, fwd_headers, shape, emitted,
                               model_name=""):
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
        parsed: Any = None
        if upstream_resp.is_success:
            try:
                parsed = upstream_resp.json()
                if shape == "ollama":
                    output, eval_stats = parse_ollama_response(parsed)
                elif shape == "anthropic":
                    output, eval_stats = parse_anthropic_response(parsed)
                else:
                    output, eval_stats = parse_openai_response(parsed)
                eval_stats = eval_stats or None
            except Exception:
                logger.debug("Buffered response parse failed", exc_info=True)
        if emitted is not None:
            attestation, counter = emitted
            emitter.emit_receipt(
                attestation=attestation, counter=counter, status=status,
                output=output, eval_stats=eval_stats,
            )
        if gating:
            denials = await _gate(output, model_name)
            if denials and isinstance(parsed, dict):
                from vaara.integrations._infer_proxy_gate import rewrite_buffered

                return Response(
                    content=json.dumps(rewrite_buffered(shape, parsed, denials)),
                    status_code=200, media_type="application/json",
                )
        else:
            _record_requested_tools(output, model_name)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=forward_response_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type"),
        )

    async def _handle_stream(url, body, fwd_headers, shape, emitted,
                             model_name=""):
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

        if gating:
            # A stream cannot be un-sent, so a gated stream is buffered
            # first: replayed byte-for-byte when everything is allowed,
            # replaced by a synthesized policy stream when anything is
            # denied. Latency trades for the pre-execution block.
            chunks: list[bytes] = []
            acc = StreamAccumulator(shape=shape)
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    acc.feed(chunk)
                    chunks.append(chunk)
            finally:
                await stream_cm.__aexit__(None, None, None)
            output, eval_stats = acc.finalize()
            if emitted is not None:
                attestation, counter = emitted
                status = "completed" if 200 <= status_code < 300 else "errored"
                emitter.emit_receipt(
                    attestation=attestation, counter=counter, status=status,
                    output=output, eval_stats=eval_stats,
                )
            denials = await _gate(output, model_name)
            if denials:
                from vaara.integrations._infer_proxy_gate import synthesize_stream

                return Response(
                    content=synthesize_stream(shape, denials, model_name),
                    status_code=200, media_type=media_type,
                )
            return Response(
                content=b"".join(chunks), status_code=status_code,
                headers=resp_headers, media_type=media_type,
            )

        async def _tee() -> Any:
            acc = StreamAccumulator(shape=shape)
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    acc.feed(chunk)
                    yield chunk
            finally:
                await stream_cm.__aexit__(None, None, None)
                output, eval_stats = acc.finalize()
                if emitted is not None:
                    attestation, counter = emitted
                    status = "completed" if 200 <= status_code < 300 else "errored"
                    emitter.emit_receipt(
                        attestation=attestation, counter=counter, status=status,
                        output=output, eval_stats=eval_stats,
                    )
                _record_requested_tools(output, model_name)

        return StreamingResponse(
            _tee(), status_code=status_code, media_type=media_type,
            headers=resp_headers,
        )

    async def _handle_chat(full_path: str, body: bytes, request: "Request") -> Any:
        shape = ("ollama" if full_path == "/api/chat"
                 else "anthropic" if full_path == "/v1/messages"
                 else "openai")
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        model_name = data.get("model") or "unknown"
        messages = data.get("messages")
        sampling = extract_sampling(data, shape == "ollama")
        stream = bool(data.get("stream"))

        emitted = None
        if emitter is not None:
            model_derived = await resolver.resolve(model_name)
            emitted = emitter.emit_attestation(
                model_ref=model_name, model_derived=model_derived,
                messages=messages, sampling=sampling,
            )

        url = f"{upstream}{full_path}"
        fwd_headers = forward_request_headers(request.headers)
        fwd_headers["accept-encoding"] = "identity"  # keep the body parseable

        if not stream:
            return await _handle_buffered(url, body, fwd_headers, shape,
                                          emitted, model_name=model_name)
        return await _handle_stream(url, body, fwd_headers, shape,
                                    emitted, model_name=model_name)

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
