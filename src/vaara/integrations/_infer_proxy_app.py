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

from vaara.integrations._http_origin import install_origin_guard
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
    allow_patterns: Any = None, allowed_origins: Any = None,
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
    # Loopback bind, no inbound credential, and every chat call it signs
    # becomes an attestation + receipt pair. A page that can post here mints
    # signed evidence for inference the operator never asked for.
    install_origin_guard(
        app, allowed_origins=allowed_origins, surface="vaara-infer-proxy",
    )

    # Registered before the catch-all below, because Starlette matches routes
    # in registration order and every path on this app otherwise forwards
    # upstream. A probe that forwards reports the MODEL's health, not the
    # proxy's, so an orchestrator would restart Vaara whenever the model was
    # slow to load. "/healthz" rather than "/health": vLLM serves /health and
    # shadowing it would hide the upstream's own check from anyone routing
    # through the proxy. The path is in the origin guard's exempt set, so a
    # probe arriving with a foreign Origin is not refused.
    @app.get("/healthz")
    async def healthz() -> dict:
        from vaara import __version__

        return {
            "status": "ok",
            "version": __version__,
            "mode": "enforce" if gating else "observe",
            # No upstream URL: this answers to anything that can reach the
            # port, and naming an internal model host is free reconnaissance.
            "upstream_configured": bool(upstream),
            "signing": emitter is not None,
        }

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
        parse_failed = False
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
                parse_failed = True
                # Debug level is right when only recording is at stake. When
                # the pipeline is enforcing, this is the gate going blind, so
                # it is a warning and, below, a refusal.
                logger.log(
                    logging.WARNING if gating else logging.DEBUG,
                    "Buffered response parse failed", exc_info=True,
                )
        if emitted is not None:
            attestation, counter = emitted
            emitter.emit_receipt(
                attestation=attestation, counter=counter, status=status,
                output=output, eval_stats=eval_stats,
            )
        if gating:
            from vaara.integrations._infer_proxy_gate import (
                UNREADABLE_RESPONSE_DENIAL,
                refusal_buffered,
                rewrite_buffered,
            )

            # A reply the proxy could not read is not a reply with no tool
            # calls in it. Forwarding it because the gate found nothing to
            # decide is a silent bypass of the whole enforcement path.
            denials = (
                [UNREADABLE_RESPONSE_DENIAL] if parse_failed
                else await _gate(output, model_name)
            )
            if denials:
                doc = (
                    rewrite_buffered(shape, parsed, denials)
                    if isinstance(parsed, dict)
                    else refusal_buffered(shape, denials, model_name)
                )
                return Response(
                    content=json.dumps(doc),
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
            from vaara.integrations._infer_proxy_gate import (
                UNREADABLE_RESPONSE_DENIAL,
                synthesize_stream,
            )

            # Same rule as the buffered path, in the two ways a stream can be
            # unreadable: finalize raised (output is None), or it completed
            # without recognising a single message of the expected shape,
            # which looks exactly like a clean stream carrying no tool calls.
            # Replaying either would ship whatever the bytes contain with no
            # decision behind it.
            unreadable = (
                200 <= status_code < 300
                and (output is None or not (acc.recognised or acc.empty))
            )
            if unreadable:
                logger.warning(
                    "Streamed response could not be reconstructed; refusing "
                    "rather than replaying it past the gate",
                )
            denials = (
                [UNREADABLE_RESPONSE_DENIAL] if unreadable
                else await _gate(output, model_name)
            )
            if denials:
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
