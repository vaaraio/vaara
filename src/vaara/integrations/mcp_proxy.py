"""MCP proxy: Vaara as a transparent runtime governance layer for MCP.

Sits between an MCP client (Claude Code, Cursor, any MCP-capable host) and an
upstream MCP server. Forwards every request to the upstream, but routes the
governed MCP surfaces through Vaara first:

* ``tools/call`` runs through the full interception pipeline (classify, score,
  decide, audit) and either flows through or returns an MCP tool error.
* ``resources/read`` and ``prompts/get`` write a request+decision audit pair
  for every access so a regulator can reconstruct what the agent read or
  retrieved. These are read-oriented MCP surfaces. The perimeter
  allow/deny lists gate exposure, but they do not run through the risk
  scorer.
* ``tools/list``, ``resources/list``, and ``prompts/list`` are filtered
  symmetrically against the operator-supplied allow/deny lists before the
  client sees them.

Operator-side filtering (``--allow-tool``/``--deny-tool``,
``--allow-resource``/``--deny-resource``, ``--allow-prompt``/``--deny-prompt``):
when set, the proxy filters the upstream's discovery response before the
client sees it, and rejects matching access at the perimeter with a
``FILTERED`` block payload, without contacting the upstream.
"""

from __future__ import annotations

import argparse
import contextvars
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from vaara import __version__ as _VAARA_VERSION
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail
from vaara.integrations._mcp_notify import (
    HttpRouter,
    NotificationRouter,
    StdioRouter,
)
from vaara.integrations._mcp_attest import (
    AttestConfigError,
    AttestPairEmitter,
    build_attest_emitter,
)
from vaara.integrations._mcp_overt import (
    OVERTConfigError,
    OVERTReceiptEmitter,
    build_emitter,
    policy_hash_from_perimeter,
)
from vaara.integrations._mcp_upstream import (
    ProxyError, UpstreamClient, UpstreamMCPClient, strict_json_dumps,
)
from vaara.integrations._mcp_upstream_http import HttpUpstreamClient
from vaara.pipeline import InterceptionPipeline
from vaara.taxonomy.actions import ActionRequest

# Optional dependency: only the streamable-HTTP transport needs FastAPI /
# Starlette. Keep the import lazy so the stdio path stays installable with
# the base extras only.
try:  # pragma: no cover - import guard
    from starlette.requests import Request as _StarletteRequest
except ImportError:  # pragma: no cover
    _StarletteRequest = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# v0.40 per-request request scope. HTTP transport sets these per inbound
# request so _handle_request and friends can look up the right upstream and
# tag the audit/interception trail with the request's tenant_id without
# threading the values through every helper signature.
_REQUEST_UPSTREAM: contextvars.ContextVar[str] = contextvars.ContextVar(
    "vaara_mcp_upstream", default="default",
)
_REQUEST_TENANT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "vaara_mcp_tenant", default="",
)
# v0.41 GET-SSE: HTTP transport sets this per inbound POST /mcp call so the
# tools/call handler can capture the originating session in the inflight
# map. The upstream reader thread later uses that captured session_id to
# route notifications/progress through the HTTP router to the right SSE
# subscriber. Empty default keeps stdio transport unaffected.
_REQUEST_SESSION: contextvars.ContextVar[str] = contextvars.ContextVar(
    "vaara_mcp_session", default="",
)
# v0.43 proxy pairing: HTTP transport sets this from X-Vaara-Intent per request.
# Empty default means intent falls back to "tools/call/{tool_name}" in the emitter.
_REQUEST_INTENT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "vaara_mcp_intent", default="",
)

def _safe_log(value: Any, max_len: int = 200) -> str:
    """Sanitise a user-supplied string for safe logging.

    Strips CR/LF and other control characters so an attacker who controls
    a tool / resource / prompt name can't inject fake log lines, and caps
    length so a multi-megabyte name doesn't blow the log up.
    """
    if not isinstance(value, str):
        value = str(value)
    cleaned = "".join(c if c.isprintable() and c not in ("\r", "\n") else "?" for c in value)
    return cleaned[:max_len]


# Largest single MCP JSON-RPC message accepted on the /mcp HTTP endpoint.
# Real tool calls and responses fit comfortably; the cap stops a malicious
# client from exhausting memory at parse time. v0.41 can promote this to
# a CLI flag if a real workload needs more headroom.
_MCP_HTTP_MAX_BODY_BYTES = 1 * 1024 * 1024

# Maximum Mcp-Session-Id length accepted on the /mcp HTTP endpoint. The id
# keys the inflight-progress and HttpRouter session maps, so the cap bounds
# those keys against a malicious client submitting an absurdly long header.
# 128 is comfortably wider than any realistic cryptographically-random id.
_MCP_SESSION_ID_MAX_LEN = 128

# Streamable HTTP transport revisions the /mcp endpoint speaks. The
# MCP-Protocol-Version header arrived in 2025-06-18; per spec a request that
# omits it is assumed to be 2025-03-26, and an unrecognised value is a 400.
_SUPPORTED_HTTP_PROTOCOL_VERSIONS = frozenset({"2025-03-26", "2025-06-18"})


def _session_id_is_visible_ascii(value: str) -> bool:
    """True iff every character is visible ASCII (0x21-0x7E).

    The Streamable HTTP transport requires session ids to be visible
    ASCII. The empty string passes vacuously; callers cap length and
    presence separately.
    """
    return all("\x21" <= ch <= "\x7e" for ch in value)


def _protocol_version_supported(version: Optional[str]) -> bool:
    """True iff the MCP-Protocol-Version header is absent, blank, or known.

    An absent or blank header is allowed: per spec the server assumes
    2025-03-26. A present, non-blank value must be one the transport speaks.
    """
    if version is None:
        return True
    stripped = version.strip()
    return stripped == "" or stripped in _SUPPORTED_HTTP_PROTOCOL_VERSIONS


def _accept_satisfies(accept: Optional[str], media_type: str) -> bool:
    """True iff an Accept header value can receive ``media_type``.

    A missing or blank header states no preference and is accepted. A
    present header satisfies ``media_type`` when it lists ``*/*``, the
    matching type wildcard (e.g. ``application/*``), or the exact type.
    Wildcard-aware where a literal substring check would not be.
    """
    if not accept or not accept.strip():
        return True
    main_type = media_type.split("/", 1)[0]
    tokens = {
        token.strip().split(";", 1)[0].strip().lower()
        for token in accept.split(",")
    }
    return (
        "*/*" in tokens
        or f"{main_type}/*" in tokens
        or media_type.lower() in tokens
    )


class VaaraMCPProxy:
    """Transparent MCP proxy with Vaara interception on tool calls."""

    PROXY_NAME = f"vaara-mcp-proxy/{_VAARA_VERSION}"

    def __init__(
        self,
        upstream_command: Optional[list[str]] = None,
        pipeline: Optional[InterceptionPipeline] = None,
        db_path: Optional[Path] = None,
        agent_id_default: str = "mcp-proxy-client",
        allowlist: Optional[set[str]] = None,
        denylist: Optional[set[str]] = None,
        resource_allowlist: Optional[set[str]] = None,
        resource_denylist: Optional[set[str]] = None,
        prompt_allowlist: Optional[set[str]] = None,
        prompt_denylist: Optional[set[str]] = None,
        overt_emitter: Optional[OVERTReceiptEmitter] = None,
        attest_emitter: Optional[AttestPairEmitter] = None,
        upstreams: Optional[dict[str, list[str]]] = None,
        upstream_urls: Optional[dict[str, str]] = None,
        upstream_headers: Optional[dict[str, dict[str, str]]] = None,
        allow_private_upstream_hosts: Optional[bool] = None,
        router: Optional[NotificationRouter] = None,
    ) -> None:
        if pipeline is not None:
            self._pipeline = pipeline
            self._backend = None
        else:
            db = db_path or Path(os.environ.get("VAARA_DB", "vaara_audit.db"))
            self._backend = SQLiteAuditBackend(db)
            trail = AuditTrail(on_record=self._backend.write_record)
            self._pipeline = InterceptionPipeline(trail=trail)
        self._agent_id_default = agent_id_default
        # An empty allowlist means "no restriction" (None and empty set are equivalent
        # here); a non-empty allowlist restricts to exactly those names. Denylist
        # always subtracts. When both are set, denylist wins on overlap.
        self._allowlist: Optional[set[str]] = set(allowlist) if allowlist else None
        self._denylist: set[str] = set(denylist) if denylist else set()
        self._resource_allowlist: Optional[set[str]] = (
            set(resource_allowlist) if resource_allowlist else None
        )
        self._resource_denylist: set[str] = (
            set(resource_denylist) if resource_denylist else set()
        )
        self._prompt_allowlist: Optional[set[str]] = (
            set(prompt_allowlist) if prompt_allowlist else None
        )
        self._prompt_denylist: set[str] = (
            set(prompt_denylist) if prompt_denylist else set()
        )
        self._stdout_lock = threading.Lock()
        self._overt = overt_emitter
        self._attest = attest_emitter
        # Notification router. stdio default writes through the shared stdout
        # lock; HTTP transport swaps in HttpRouter in run_http(). The router is
        # the only surface allowed to deliver upstream-initiated notifications
        # to clients; tools/call response replies still go through
        # _write_to_client / JSONResponse on their own paths.
        self._router: NotificationRouter = router or StdioRouter(self._stdout_lock)
        # progressToken -> (action_id, agent_id, tool_name, tenant_id, session_id).
        # Populated when a tools/call enters interception with
        # params._meta.progressToken set, consulted by the upstream-notification
        # handler so progress events arriving mid-call carry the originating
        # action_id into the audit record and into the OVERT envelope's
        # non_content_metadata. The tenant and session_id are captured at
        # tools/call time because the upstream reader thread that delivers
        # later notifications does not inherit the request ContextVars.
        self._inflight_progress: dict[Any, tuple[str, str, str, str, str]] = {}
        # request_id -> upstream_name for every tools/call still running.
        # Used to route notifications/cancelled to the upstream actually
        # serving the matching request, regardless of which header the
        # cancellation POST carries. Without this, cancellations under
        # fan-out would land on the upstream named by X-Vaara-Upstream
        # on the cancel POST, which the client has no reason to set
        # correctly. Request-id ownership is the only stable identifier.
        self._inflight_requests: dict[Any, str] = {}
        self._inflight_lock = threading.Lock()
        # v0.40 fan-out: hold N upstream MCP servers in a name -> client map.
        # The single-upstream legacy entry point (positional ``upstream_command``)
        # lands under the "default" name. ``--upstream NAME=CMD`` via CLI or
        # ``upstreams={"NAME": [cmd, ...]}`` populates the map directly. The
        # HTTP transport reads X-Vaara-Upstream per inbound request to pick;
        # stdio transport stays on "default".
        if upstreams and upstream_command is not None:
            raise ValueError(
                "Pass either upstream_command (single-upstream legacy) or "
                "upstreams (multi-upstream fan-out), not both.",
            )
        upstream_map: dict[str, list[str]] = {}
        if upstreams:
            upstream_map = {name: list(cmd) for name, cmd in upstreams.items()}
        elif upstream_command is not None:
            upstream_map = {"default": list(upstream_command)}
        # v0.45: remote upstreams reached over the MCP Streamable HTTP transport.
        # A slot is either a stdio command or a URL, never both. Optional static
        # headers (auth) attach per URL slot.
        url_map: dict[str, str] = dict(upstream_urls) if upstream_urls else {}
        header_map: dict[str, dict[str, str]] = (
            {n: dict(h) for n, h in upstream_headers.items()} if upstream_headers else {}
        )
        collisions = set(upstream_map) & set(url_map)
        if collisions:
            raise ValueError(
                f"Upstream slot(s) {sorted(collisions)!r} given as both a stdio "
                "command and a URL; each slot is exactly one transport.",
            )
        stray_headers = set(header_map) - set(url_map)
        if stray_headers:
            raise ValueError(
                f"Upstream header(s) for {sorted(stray_headers)!r} match no "
                "--upstream-url slot; headers apply to URL upstreams only.",
            )
        if not upstream_map and not url_map:
            raise ValueError(
                "VaaraMCPProxy requires at least one upstream "
                "(upstream_command, upstreams, or upstream_urls).",
            )
        all_names = set(upstream_map) | set(url_map)
        default_alias_target: Optional[str] = None
        if "default" not in all_names:
            # Pick a stable fallback so requests without X-Vaara-Upstream
            # still resolve. Lexicographic first keeps multi-tenant fleets
            # deterministic across restarts. Alias the slot rather than
            # cloning the transport so we never open a duplicate connection.
            default_alias_target = sorted(all_names)[0]
        # Wrap on_notification per upstream so the reader/listener callback
        # carries the upstream's name. Default-arg ``n=name`` binds the loop
        # variable at definition, avoiding the late-binding bug that would
        # otherwise pin every upstream to the last name iterated.
        self._upstreams: dict[str, UpstreamClient] = {}
        for name, command in upstream_map.items():
            self._upstreams[name] = UpstreamMCPClient(
                command=command,
                on_notification=(
                    lambda msg, n=name: self._on_upstream_notification(n, msg)
                ),
            )
        for name, url in url_map.items():
            # SSRF egress floor defaults SAFE; allow_private_upstream_hosts (or
            # the VAARA_MCP_ALLOW_PRIVATE_UPSTREAM env flag) opts a trusted
            # internal host in. Refused targets raise at construction here.
            self._upstreams[name] = HttpUpstreamClient(
                url=url,
                headers=header_map.get(name),
                on_notification=(
                    lambda msg, n=name: self._on_upstream_notification(n, msg)
                ),
                allow_private_hosts=allow_private_upstream_hosts,
            )
        if default_alias_target is not None:
            self._upstreams["default"] = self._upstreams[default_alias_target]
        # ``self._upstream`` resolves to the per-request upstream via the
        # ``_REQUEST_UPSTREAM`` ContextVar. stdio transport leaves the
        # default ("default") in place, so existing single-upstream callers
        # see exactly one client. HTTP transport sets the ctxvar per
        # request to dispatch into the right fleet member.

    @property
    def _upstream(self) -> UpstreamClient:
        """Resolve the upstream MCP client for the current request scope.

        HTTP transport sets ``_REQUEST_UPSTREAM`` per inbound request. stdio
        transport never sets it; the default value "default" routes to the
        legacy single-upstream slot. Unknown names raise ``ProxyError`` so
        a client asking for a fleet member that does not exist gets a
        loud failure instead of a silent reroute.
        """
        name = _REQUEST_UPSTREAM.get()
        client = self._upstreams.get(name)
        if client is None:
            raise ProxyError(
                f"No upstream named {name!r}; configured names: "
                f"{sorted(self._upstreams)!r}",
            )
        return client

    @_upstream.setter
    def _upstream(self, client: UpstreamClient) -> None:
        """Replace the default-slot upstream client.

        Test fixtures and embedders that previously assigned
        ``proxy._upstream = MagicMock()`` keep working under the v0.40
        fan-out shape: the assignment lands in the "default" slot and the
        property reads it back.
        """
        self._upstreams["default"] = client

    @staticmethod
    def _is_filtered(name: object, allowlist: Optional[set[str]], denylist: set[str]) -> bool:
        if not isinstance(name, str):
            return True
        if name in denylist:
            return True
        if allowlist is not None and name not in allowlist:
            return True
        return False

    def _tool_filtered(self, name: str) -> bool:
        return self._is_filtered(name, self._allowlist, self._denylist)

    def _resource_filtered(self, uri: str) -> bool:
        return self._is_filtered(uri, self._resource_allowlist, self._resource_denylist)

    def _prompt_filtered(self, name: str) -> bool:
        return self._is_filtered(name, self._prompt_allowlist, self._prompt_denylist)

    def run(self) -> None:
        """Read JSON-RPC from stdin, write to stdout, route through upstream."""
        logger.info("Vaara MCP proxy starting on stdio (%s)", self.PROXY_NAME)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._write_to_client(self._error_response(None, -32700, "Parse error"))
                continue
            # Notifications (no id) forward silently per JSON-RPC 2.0 §4.1.
            if isinstance(request, dict) and "id" not in request:
                try:
                    self._handle_client_notification(request)
                except ProxyError:
                    logger.exception("Failed to forward notification")
                continue
            self._write_to_client(self._handle_request(request))

    def _build_http_app(self):
        """Construct the FastAPI app that backs the Streamable HTTP transport.

        Split out from ``run_http`` so tests can drive the endpoints via
        ``fastapi.testclient.TestClient`` without standing up uvicorn. As a
        side effect, swaps the proxy's notification router to ``HttpRouter``
        so upstream-initiated notifications fan out to SSE subscribers
        instead of stdout.
        """
        try:
            import asyncio
            from fastapi import FastAPI, Header, HTTPException, Response
            from fastapi.responses import JSONResponse, StreamingResponse
        except ImportError as exc:
            raise RuntimeError(
                "vaara-mcp-proxy --transport http requires the 'server' "
                "extra. Install with: pip install 'vaara[server]'"
            ) from exc
        if _StarletteRequest is None:
            raise RuntimeError(
                "starlette is required for the streamable-HTTP transport. "
                "Install with: pip install 'vaara[server]'"
            )

        proxy = self
        # Replace the default StdioRouter with an HTTP-aware fan-out router.
        # Mutation here is safe: it happens once at HTTP startup, before any
        # request arrives, so the upstream reader threads consistently see
        # the HTTP router for the lifetime of the serve loop. The StdioRouter
        # instance is dropped.
        http_router = HttpRouter()
        proxy._router = http_router

        app = FastAPI(
            title="Vaara MCP Proxy",
            version=_VAARA_VERSION,
            description=(
                "Streamable HTTP transport in front of one or more upstream "
                "MCP servers, with Vaara runtime governance applied to every "
                "tools/call."
            ),
        )

        @app.get("/health")
        async def health() -> dict:
            return {
                "status": "ok",
                "proxy": proxy.PROXY_NAME,
                "upstreams": sorted(proxy._upstreams.keys()),
            }

        @app.post("/mcp")
        async def mcp_endpoint(
            request: _StarletteRequest,
            x_vaara_tenant: Optional[str] = Header(default=None, alias="X-Vaara-Tenant"),
            x_vaara_upstream: Optional[str] = Header(default=None, alias="X-Vaara-Upstream"),
            mcp_session_id: Optional[str] = Header(default=None, alias="Mcp-Session-Id"),
            x_vaara_intent: Optional[str] = Header(default=None, alias="X-Vaara-Intent"),
        ) -> Response:
            # Streamable HTTP transport header validation (MCP 2025-03-26 /
            # 2025-06-18): the client must be able to accept both
            # application/json and text/event-stream, and any
            # MCP-Protocol-Version it sends must be one we speak. Reject
            # transport violations before reading or parsing the body.
            accept = request.headers.get("accept")
            if not (
                _accept_satisfies(accept, "application/json")
                and _accept_satisfies(accept, "text/event-stream")
            ):
                raise HTTPException(
                    status_code=406,
                    detail={"error": {
                        "code": "not_acceptable",
                        "message": (
                            "Accept header must allow both application/json "
                            "and text/event-stream"
                        ),
                    }},
                )
            if not _protocol_version_supported(
                request.headers.get("mcp-protocol-version")
            ):
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "code": "unsupported_protocol_version",
                        "message": (
                            "MCP-Protocol-Version is not supported; supported: "
                            f"{sorted(_SUPPORTED_HTTP_PROTOCOL_VERSIONS)!r}"
                        ),
                    }},
                )
            # 1 MiB cap on a single MCP JSON-RPC message. Real tool calls and
            # responses fit comfortably; anything larger is either a misuse or
            # a DoS attempt against the proxy's JSON parser. The cap is the
            # same order as the MCP reference servers' limit. Hard-cap here
            # before json.loads runs so a malicious payload cannot exhaust
            # memory at parse time.
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > _MCP_HTTP_MAX_BODY_BYTES:
                        return JSONResponse(
                            status_code=413,
                            content={"error": {
                                "code": "payload_too_large",
                                "message": (
                                    f"MCP message exceeds "
                                    f"{_MCP_HTTP_MAX_BODY_BYTES} bytes"
                                ),
                            }},
                        )
                except ValueError:
                    pass  # bogus content-length, fall through to actual read
            raw = await request.body()
            if len(raw) > _MCP_HTTP_MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": {
                        "code": "payload_too_large",
                        "message": (
                            f"MCP message exceeds "
                            f"{_MCP_HTTP_MAX_BODY_BYTES} bytes"
                        ),
                    }},
                )
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content=proxy._error_response(None, -32700, "Parse error"),
                )

            header_name = (x_vaara_upstream or "").strip()
            # Real upstream slots are everything except the "default" alias.
            # When the operator configured exactly one real slot (single-
            # upstream deployment), silent fallback preserves the v0.39
            # contract. When the operator configured a fleet, ambiguity is
            # an error: missing X-Vaara-Upstream returns 400 with the list
            # so the client knows which slots are available, instead of
            # silently routing to whichever slot won the sort.
            real_slots = sorted(n for n in proxy._upstreams if n != "default")
            ambiguous_fanout = len(real_slots) > 1
            if not header_name:
                if ambiguous_fanout:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "code": "upstream_required",
                                "message": (
                                    "X-Vaara-Upstream header is required "
                                    "when the proxy serves more than one "
                                    "upstream. Available upstreams: "
                                    f"{real_slots!r}"
                                ),
                            }
                        },
                    )
                upstream_name = "default"
            else:
                upstream_name = header_name
            if upstream_name not in proxy._upstreams:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": {
                            "code": "unknown_upstream",
                            "message": (
                                f"No upstream named {upstream_name!r}; "
                                f"configured: {sorted(proxy._upstreams)!r}"
                            ),
                        }
                    },
                )

            session_value = (mcp_session_id or "").strip()
            if len(session_value) > _MCP_SESSION_ID_MAX_LEN:
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "code": "session_id_too_long",
                        "message": (
                            f"Mcp-Session-Id must be {_MCP_SESSION_ID_MAX_LEN} "
                            "characters or fewer"
                        ),
                    }},
                )
            if not _session_id_is_visible_ascii(session_value):
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "code": "session_id_invalid",
                        "message": (
                            "Mcp-Session-Id must contain only visible ASCII "
                            "characters (0x21-0x7E)"
                        ),
                    }},
                )
            upstream_token = _REQUEST_UPSTREAM.set(upstream_name)
            tenant_token = _REQUEST_TENANT.set((x_vaara_tenant or "").strip())
            session_token = _REQUEST_SESSION.set(session_value)
            intent_token = _REQUEST_INTENT.set((x_vaara_intent or "").strip())
            try:
                if isinstance(payload, dict) and "id" not in payload:
                    # _handle_client_notification forwards to the upstream via a
                    # blocking sync notify(); offload it on the same copied
                    # context as the request path so a slow upstream notify does
                    # not park the event loop and serialise other HTTP traffic.
                    nctx = contextvars.copy_context()
                    try:
                        await asyncio.to_thread(
                            nctx.run, proxy._handle_client_notification, payload,
                        )
                    except ProxyError:
                        logger.exception("Failed to forward HTTP notification")
                    return Response(status_code=202)
                # _handle_request is a blocking sync call that waits on the
                # upstream (up to its request timeout). Running it inline would
                # park the event loop for the whole call, serialising every
                # other POST /mcp, GET /mcp drain, and /health to concurrency 1.
                # Offload to a worker thread. The per-request ContextVars set
                # just above live on this task's context, which a bare
                # to_thread target would not inherit, so copy the current
                # context and run the handler inside it on the worker thread.
                ctx = contextvars.copy_context()
                response = await asyncio.to_thread(
                    ctx.run, proxy._handle_request, payload,
                )
                return JSONResponse(content=response)
            finally:
                _REQUEST_UPSTREAM.reset(upstream_token)
                _REQUEST_TENANT.reset(tenant_token)
                _REQUEST_SESSION.reset(session_token)
                _REQUEST_INTENT.reset(intent_token)

        @app.get("/mcp")
        async def mcp_sse_endpoint(
            request: _StarletteRequest,
            x_vaara_tenant: Optional[str] = Header(default=None, alias="X-Vaara-Tenant"),
            x_vaara_upstream: Optional[str] = Header(default=None, alias="X-Vaara-Upstream"),
            mcp_session_id: Optional[str] = Header(default=None, alias="Mcp-Session-Id"),
            last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
        ) -> StreamingResponse:
            if not _protocol_version_supported(
                request.headers.get("mcp-protocol-version")
            ):
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "code": "unsupported_protocol_version",
                        "message": (
                            "MCP-Protocol-Version is not supported; supported: "
                            f"{sorted(_SUPPORTED_HTTP_PROTOCOL_VERSIONS)!r}"
                        ),
                    }},
                )
            session_value = (mcp_session_id or "").strip()
            if not session_value:
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "code": "session_id_required",
                        "message": "Mcp-Session-Id header is required for GET /mcp",
                    }},
                )
            if len(session_value) > _MCP_SESSION_ID_MAX_LEN:
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "code": "session_id_too_long",
                        "message": (
                            f"Mcp-Session-Id must be {_MCP_SESSION_ID_MAX_LEN} "
                            "characters or fewer"
                        ),
                    }},
                )
            if not _session_id_is_visible_ascii(session_value):
                raise HTTPException(
                    status_code=400,
                    detail={"error": {
                        "code": "session_id_invalid",
                        "message": (
                            "Mcp-Session-Id must contain only visible ASCII "
                            "characters (0x21-0x7E)"
                        ),
                    }},
                )
            header_name = (x_vaara_upstream or "").strip()
            real_slots = sorted(n for n in proxy._upstreams if n != "default")
            ambiguous_fanout = len(real_slots) > 1
            if not header_name:
                if ambiguous_fanout:
                    raise HTTPException(
                        status_code=400,
                        detail={"error": {
                            "code": "upstream_required",
                            "message": (
                                "X-Vaara-Upstream header is required when "
                                "the proxy serves more than one upstream. "
                                f"Available upstreams: {real_slots!r}"
                            ),
                        }},
                    )
                upstream_name = "default"
            else:
                upstream_name = header_name
            if upstream_name not in proxy._upstreams:
                raise HTTPException(
                    status_code=404,
                    detail={"error": {
                        "code": "unknown_upstream",
                        "message": (
                            f"No upstream named {upstream_name!r}; "
                            f"configured: {sorted(proxy._upstreams)!r}"
                        ),
                    }},
                )
            # Last-Event-ID is the SSE resumption cursor. Non-integer values
            # are tolerated and treated as 0 (full replay window) rather than
            # rejected, matching the EventSource spec's lenient parsing.
            resume_after = 0
            if last_event_id is not None:
                try:
                    resume_after = max(int(last_event_id.strip()), 0)
                except ValueError:
                    resume_after = 0
            loop = asyncio.get_running_loop()
            # my_state is THIS stream's session state. On a reconnect with the
            # same Mcp-Session-Id, register_session installs a fresh state and
            # closes this one; the unregister in the finally below is then
            # identity-checked against my_state so the tearing-down old stream
            # never pops the NEW state out from under the live reconnection.
            my_state = http_router.register_session(
                session_id=session_value,
                upstream=upstream_name,
                tenant=(x_vaara_tenant or "").strip(),
                loop=loop,
            )
            state = my_state

            async def event_stream():
                # enqueue populates both the buffer (for replay) and the queue
                # (for live drain), so an event that lands between
                # register_session and the first replay_since call would arrive
                # on both paths. Tracking the highest yielded id and filtering
                # the drain stream by it keeps each event on the wire exactly
                # once.
                last_yielded = resume_after
                try:
                    # Initial retry hint (5s) so EventSource clients reconnect
                    # at a predictable cadence on transient disconnect.
                    yield b"retry: 5000\n\n"
                    for event_id, payload in state.replay_since(resume_after):
                        yield (
                            f"id: {event_id}\n"
                            f"data: {strict_json_dumps(payload)}\n\n"
                        ).encode("utf-8")
                        last_yielded = max(last_yielded, event_id)
                    while True:
                        try:
                            entry = await asyncio.wait_for(state.drain(), timeout=15.0)
                        except asyncio.TimeoutError:
                            # Heartbeat. A failing yield here surfaces a dead
                            # socket and lets the finally block tear the
                            # session down.
                            yield b": keepalive\n\n"
                            continue
                        if entry is None:
                            break
                        event_id, payload = entry
                        if event_id <= last_yielded:
                            continue
                        yield (
                            f"id: {event_id}\n"
                            f"data: {strict_json_dumps(payload)}\n\n"
                        ).encode("utf-8")
                        last_yielded = event_id
                finally:
                    # Identity-checked: only tear down the map entry if it is
                    # still this stream's state. A reconnect that already
                    # replaced it leaves the live session registered.
                    http_router.unregister_session(session_value, expected=my_state)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return app

    def run_http(self, host: str, port: int, log_level: str = "info") -> None:
        """Run the proxy on Streamable HTTP (MCP 2026 transport).

        POST /mcp accepts one JSON-RPC message and returns one JSON response.
        GET /mcp opens a server-sent-events stream for upstream-initiated
        notifications scoped to the ``Mcp-Session-Id`` header. Notifications
        (no ``id``) return 202 Accepted and are forwarded to the upstream
        without a reply, with ``notifications/cancelled`` routed to the
        upstream that owns the named request id. Multi-tenant / fan-out scope
        is read per request from the ``X-Vaara-Tenant`` and
        ``X-Vaara-Upstream`` headers and pushed into the per-request
        ContextVars before dispatch.
        """
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "vaara-mcp-proxy --transport http requires the 'server' "
                "extra. Install with: pip install 'vaara[server]'"
            ) from exc
        app = self._build_http_app()
        logger.info(
            "Vaara MCP proxy starting on http://%s:%d (%s, upstreams=%s)",
            host, port, self.PROXY_NAME, sorted(self._upstreams.keys()),
        )
        uvicorn.run(app, host=host, port=port, log_level=log_level)

    def _handle_client_notification(self, payload: dict) -> None:
        """Route a client-originated JSON-RPC notification to the right upstream.

        Most notifications follow the per-request ContextVar set by the HTTP
        transport from ``X-Vaara-Upstream`` (or stdio's default of "default").
        ``notifications/cancelled`` overrides that with the upstream actually
        serving the named ``requestId``, so under fan-out the cancel reaches
        the upstream subprocess that owns the long-running tools/call rather
        than whatever slot the cancel POST's header guessed at.
        """
        target_upstream: Optional[str] = None
        if (
            isinstance(payload, dict)
            and payload.get("method") == "notifications/cancelled"
        ):
            params = payload.get("params")
            if isinstance(params, dict):
                req_id = params.get("requestId")
                if req_id is not None:
                    with self._inflight_lock:
                        target_upstream = self._inflight_requests.get(req_id)
        if target_upstream is not None and target_upstream in self._upstreams:
            token = _REQUEST_UPSTREAM.set(target_upstream)
            try:
                self._upstream.notify(payload)
            finally:
                _REQUEST_UPSTREAM.reset(token)
        else:
            self._upstream.notify(payload)

    def _handle_request(self, request: Any) -> dict:
        if not isinstance(request, dict):
            return self._error_response(None, -32600, "Invalid Request: not a JSON object")
        method = request.get("method", "")
        req_id = request.get("id")
        if method == "tools/call":
            try:
                return self._handle_tools_call(request)
            except ProxyError as e:
                return self._error_response(req_id, -32603, str(e))
            except Exception:
                logger.exception("Error in tools/call interception")
                return self._error_response(req_id, -32603, "Internal proxy error")
        if method == "tools/list":
            try:
                return self._handle_tools_list(request)
            except ProxyError as e:
                return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")
        if method == "resources/list":
            try:
                return self._handle_list(
                    request, "resources", "uri",
                    self._resource_allowlist, self._resource_denylist,
                )
            except ProxyError as e:
                return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")
        if method == "resources/read":
            try:
                return self._handle_resources_read(request)
            except ProxyError as e:
                return self._error_response(req_id, -32603, str(e))
            except Exception:
                logger.exception("Error in resources/read interception")
                return self._error_response(req_id, -32603, "Internal proxy error")
        if method == "prompts/list":
            try:
                return self._handle_list(
                    request, "prompts", "name",
                    self._prompt_allowlist, self._prompt_denylist,
                )
            except ProxyError as e:
                return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")
        if method == "prompts/get":
            try:
                return self._handle_prompts_get(request)
            except ProxyError as e:
                return self._error_response(req_id, -32603, str(e))
            except Exception:
                logger.exception("Error in prompts/get interception")
                return self._error_response(req_id, -32603, "Internal proxy error")
        try:
            return self._upstream.request(request)
        except ProxyError as e:
            return self._error_response(req_id, -32603, f"Upstream unavailable: {e}")

    def _handle_tools_list(self, request: dict) -> dict:
        response = self._handle_list(
            request, "tools", "name", self._allowlist, self._denylist,
        )
        if self._attest is not None:
            self._attest.update_manifest_fingerprint(
                _REQUEST_UPSTREAM.get(), response
            )
        return response

    def _handle_list(
        self,
        request: dict,
        item_key: str,
        name_field: str,
        allowlist: Optional[set[str]],
        denylist: set[str],
    ) -> dict:
        """Filter a discovery-list response by operator-supplied perimeter lists.

        Shared between ``tools/list``, ``resources/list``, and ``prompts/list``.
        ``item_key`` is the field on ``result`` holding the array. ``name_field``
        is the identifier on each item to match against the lists ("name" for
        tools and prompts, "uri" for resources).
        """
        response = self._upstream.request(request)
        if not (denylist or allowlist is not None):
            return response
        if not isinstance(response, dict) or "result" not in response:
            return response
        result = response.get("result")
        if not isinstance(result, dict):
            return response
        items = result.get(item_key)
        if not isinstance(items, list):
            return response
        filtered = [
            item for item in items
            if isinstance(item, dict)
            and not self._is_filtered(item.get(name_field, ""), allowlist, denylist)
        ]
        # Mutate a shallow copy so the upstream response object the reader
        # parked is not aliased into the client-facing payload.
        new_result = dict(result)
        new_result[item_key] = filtered
        return {**response, "result": new_result}

    def _handle_tools_call(self, request: dict) -> dict:
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if self._tool_filtered(tool_name):
            logger.warning(
                "tools/call rejected at perimeter (operator filter): %s",
                _safe_log(tool_name),
            )
            block_payload = {
                "vaara_blocked": True,
                "reason": "Tool filtered by operator policy",
                "decision": "FILTERED",
                "tool": tool_name,
            }
            self._overt_emit(
                surface="mcp.tool.call",
                identifier=tool_name,
                identifier_field="tool_name",
                request_obj={"tool": tool_name, "arguments": arguments},
                decision="FILTERED",
                reason="Tool filtered by operator policy",
                extra={"agent_id": self._agent_id_default},
            )
            return {
                "jsonrpc": "2.0", "id": request.get("id"),
                "result": {
                    "content": [{"type": "text", "text": strict_json_dumps(block_payload, indent=2)}],
                    "isError": True,
                },
            }
        # _vaara_agent_id is a proxy-side override for audit attribution;
        # strip before forwarding so the upstream never sees Vaara metadata.
        agent_id = arguments.pop("_vaara_agent_id", self._agent_id_default)
        if not isinstance(agent_id, str):
            agent_id = self._agent_id_default
        # Unknown upstream tool names classify as generic high-risk in the
        # registry (fail-closed). Correct default for runtime governance.
        result = self._pipeline.intercept(
            agent_id=agent_id, tool_name=tool_name, parameters=arguments,
            tenant_id=_REQUEST_TENANT.get(),
        )
        progress_token = self._progress_token(params)
        if not result.allowed:
            decision = getattr(result, "decision", None) or "DENY"
            reason = getattr(result, "reason", None) or "Blocked by Vaara policy"
            block_payload = {
                "vaara_blocked": True,
                "reason": reason,
                "decision": decision,
                "action_id": getattr(result, "action_id", None),
            }
            self._overt_emit(
                surface="mcp.tool.call",
                identifier=tool_name,
                identifier_field="tool_name",
                request_obj={"tool": tool_name, "arguments": arguments},
                decision=str(decision),
                reason=reason,
                extra={
                    "agent_id": agent_id,
                    "action_id": getattr(result, "action_id", None) or "",
                },
            )
            return {
                "jsonrpc": "2.0", "id": request.get("id"),
                "result": {
                    "content": [{"type": "text", "text": strict_json_dumps(block_payload, indent=2)}],
                    "isError": True,
                },
            }
        request_id = request.get("id")
        upstream_name = _REQUEST_UPSTREAM.get()
        attest_pair = None
        if self._attest is not None:
            attest_pair = self._attest.emit_attestation(
                tool_name=tool_name,
                arguments=arguments,
                upstream_name=upstream_name,
                tenant_id=_REQUEST_TENANT.get(),
                intent_override=_REQUEST_INTENT.get(),
            )
        with self._inflight_lock:
            if progress_token is not None:
                self._inflight_progress[progress_token] = (
                    str(getattr(result, "action_id", None) or ""),
                    agent_id,
                    tool_name,
                    _REQUEST_TENANT.get(),
                    _REQUEST_SESSION.get(),
                )
            if request_id is not None:
                self._inflight_requests[request_id] = upstream_name
        # Default to failure severity so a paired receipt is still emitted
        # (errored) if the upstream raises before returning a response. The
        # attestation was already written above; pairing it with a receipt on
        # every path keeps the evidence trail complete with no orphans.
        outcome_severity = 1.0
        try:
            upstream_response = self._upstream.request(request)
            outcome_severity = self._severity_from_response(upstream_response)
        finally:
            with self._inflight_lock:
                if progress_token is not None:
                    self._inflight_progress.pop(progress_token, None)
                if request_id is not None:
                    self._inflight_requests.pop(request_id, None)
            if self._attest is not None and attest_pair is not None:
                _attestation, _counter = attest_pair
                self._attest.emit_receipt(
                    attestation=_attestation,
                    counter=_counter,
                    outcome_severity=outcome_severity,
                    upstream_name=upstream_name,
                    tenant_id=_REQUEST_TENANT.get(),
                )
        try:
            self._pipeline.report_outcome(
                action_id=result.action_id, outcome_severity=outcome_severity,
            )
        except Exception:
            logger.exception("report_outcome failed for action_id=%s", result.action_id)
        self._overt_emit(
            surface="mcp.tool.call",
            identifier=tool_name,
            identifier_field="tool_name",
            request_obj={"tool": tool_name, "arguments": arguments},
            decision="allow",
            reason="allowed by Vaara policy",
            extra={
                "agent_id": agent_id,
                "action_id": getattr(result, "action_id", None) or "",
                "outcome_severity": int(outcome_severity),
            },
        )
        return upstream_response

    def _handle_resources_read(self, request: dict) -> dict:
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        uri = params.get("uri", "")
        if not isinstance(uri, str):
            uri = ""
        if self._resource_filtered(uri):
            logger.warning(
                "resources/read rejected at perimeter (operator filter): %s",
                _safe_log(uri),
            )
            self._overt_emit(
                surface="mcp.resource.read",
                identifier=uri,
                identifier_field="uri",
                request_obj={"uri": uri},
                decision="FILTERED",
                reason="Resource filtered by operator policy",
                extra={"agent_id": self._agent_id_default},
            )
            return self._perimeter_block(
                request, code=-32000,
                message="Resource filtered by Vaara operator policy",
                data={
                    "vaara_blocked": True,
                    "reason": "Resource filtered by operator policy",
                    "decision": "FILTERED",
                    "uri": uri,
                },
            )
        self._record_perimeter_audit(
            agent_id=self._agent_id_default,
            tool_name="mcp.resource.read",
            parameters={"uri": uri},
            decision="allow",
            reason="resource within operator perimeter",
        )
        self._overt_emit(
            surface="mcp.resource.read",
            identifier=uri,
            identifier_field="uri",
            request_obj={"uri": uri},
            decision="allow",
            reason="resource within operator perimeter",
            extra={"agent_id": self._agent_id_default},
        )
        return self._upstream.request(request)

    def _handle_prompts_get(self, request: dict) -> dict:
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        name = params.get("name", "")
        if not isinstance(name, str):
            name = ""
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if self._prompt_filtered(name):
            logger.warning(
                "prompts/get rejected at perimeter (operator filter): %s",
                _safe_log(name),
            )
            self._overt_emit(
                surface="mcp.prompt.get",
                identifier=name,
                identifier_field="prompt_name",
                request_obj={"name": name, "arguments": arguments},
                decision="FILTERED",
                reason="Prompt filtered by operator policy",
                extra={"agent_id": self._agent_id_default},
            )
            return self._perimeter_block(
                request, code=-32000,
                message="Prompt filtered by Vaara operator policy",
                data={
                    "vaara_blocked": True,
                    "reason": "Prompt filtered by operator policy",
                    "decision": "FILTERED",
                    "prompt": name,
                },
            )
        self._record_perimeter_audit(
            agent_id=self._agent_id_default,
            tool_name="mcp.prompt.get",
            parameters={"name": name, "arguments": arguments},
            decision="allow",
            reason="prompt within operator perimeter",
        )
        self._overt_emit(
            surface="mcp.prompt.get",
            identifier=name,
            identifier_field="prompt_name",
            request_obj={"name": name, "arguments": arguments},
            decision="allow",
            reason="prompt within operator perimeter",
            extra={"agent_id": self._agent_id_default},
        )
        return self._upstream.request(request)

    @staticmethod
    def _perimeter_block(request: dict, code: int, message: str, data: dict) -> dict:
        """Return a JSON-RPC error for resources/prompts blocked at the perimeter.

        ``resources/read`` and ``prompts/get`` carry no ``isError`` envelope
        (unlike ``tools/call``'s CallToolResult), so a perimeter block must
        surface as a JSON-RPC error. Code -32000 is the server-defined
        application-error range. The ``data`` field carries the same
        ``vaara_blocked``/``decision``/``reason`` payload tool-call
        blocks emit, so downstream readers can use one schema.
        """
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": code, "message": message, "data": data},
        }

    def _record_perimeter_audit(
        self,
        agent_id: str,
        tool_name: str,
        parameters: dict,
        decision: str,
        reason: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Write a request+decision audit pair for a read-oriented MCP access.

        Resource reads and prompt gets are read-oriented MCP surfaces,
        not actions. They need audit coverage so regulators can
        reconstruct what was accessed, but they do not run through the
        risk scorer. Writing directly to the trail bypasses scorer and
        policy while still producing the two records that anchor every
        access to the hash chain. Failures here are logged and
        swallowed: a perimeter audit failure must not block legitimate
        upstream traffic. ``tenant_id`` is taken from the request
        ContextVar by default; async callers that run outside the
        originating request (upstream notification reader thread) pass
        the captured tenant explicitly.
        """
        import time as _time
        if tenant_id is None:
            tenant_id = _REQUEST_TENANT.get()
        try:
            registry = self._pipeline.registry
            action_type = registry.classify(tool_name, parameters)
            req = ActionRequest(
                agent_id=agent_id,
                tool_name=tool_name,
                action_type=action_type,
                parameters=parameters or {},
                timestamp_utc=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                tenant_id=tenant_id,
            )
            action_id = self._pipeline.trail.record_action_requested(req)
            self._pipeline.trail.record_decision(
                action_id=action_id,
                agent_id=agent_id,
                tool_name=tool_name,
                decision=decision,
                reason=reason,
                risk_score=0.0,
                regulatory_domains=action_type.regulatory_domains,
            )
        except Exception:
            logger.exception("Failed to record perimeter audit for %s", tool_name)

    def _overt_emit(
        self,
        *,
        surface: str,
        identifier: str,
        identifier_field: str,
        request_obj: dict,
        decision: str,
        reason: str,
        extra: Optional[dict] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Emit one OVERT Base Envelope for an MCP interaction.

        No-op when no emitter is configured. Failures are logged and
        swallowed: an attestation-side failure must not block legitimate
        upstream traffic, mirroring the perimeter-audit rule. ``tenant_id``
        is taken from the request ContextVar by default; async callers
        that run outside the originating request pass the captured tenant
        explicitly so the OVERT claim attributes to the right tenant.
        """
        if self._overt is None:
            return
        try:
            non_content_metadata = {
                "action_class": surface,
                identifier_field: identifier,
                "decision": decision,
                "reason": reason,
            }
            if tenant_id is None:
                tenant_id = _REQUEST_TENANT.get()
            if tenant_id:
                non_content_metadata["tenant_id"] = tenant_id
            if extra:
                non_content_metadata.update(extra)
            request_payload = strict_json_dumps(
                request_obj, sort_keys=True,
            ).encode("utf-8")
            self._overt.emit(
                request_payload=request_payload,
                non_content_metadata=non_content_metadata,
            )
        except Exception:
            logger.exception("OVERT envelope emission failed for %s", surface)

    @staticmethod
    def _severity_from_response(response: dict) -> float:
        # Protocol/tool errors to 1.0 (failure signal). Clean success to 0.0.
        if not isinstance(response, dict) or "error" in response:
            return 1.0
        result = response.get("result")
        if isinstance(result, dict) and result.get("isError"):
            return 1.0
        return 0.0

    def _on_upstream_notification(self, upstream_name: str, message: dict) -> None:
        """Audit + OVERT-emit upstream-originated notifications, then forward.

        Streaming surfaces (notifications/progress, notifications/message) flow
        upstream-to-client during a long-running tools/call. v0.25.0 brings
        them inside the audit boundary: each notification is recorded as a
        perimeter-style audit event (no scorer) and, when OVERT is configured,
        emits its own Base Envelope. Progress events correlate to the
        originating tools/call via params.progressToken when present. Routing
        to the destination client goes through the transport-aware
        ``NotificationRouter``: stdio writes to the proxy stdout, HTTP fans
        out to the SSE session registered for the originating tools/call.
        """
        method = message.get("method") if isinstance(message, dict) else None
        try:
            if method == "notifications/progress":
                self._audit_progress_notification(message)
            elif method == "notifications/message":
                self._audit_message_notification(message)
        except Exception:
            logger.exception("Streaming-notification audit failed for %s", method)
        # Resolve the session that originated this notification, if any. Only
        # progress notifications carry a progressToken; log notifications and
        # any other server-pushed message broadcast across the sessions
        # subscribed to this upstream (HttpRouter handles broadcast when
        # session_id is None; StdioRouter ignores both args).
        session_id: Optional[str] = None
        if method == "notifications/progress":
            params = message.get("params") if isinstance(message, dict) else None
            if isinstance(params, dict):
                token = params.get("progressToken")
                if isinstance(token, (str, int)):
                    with self._inflight_lock:
                        entry = self._inflight_progress.get(token)
                    if entry is not None:
                        captured_session = entry[4]
                        if captured_session:
                            session_id = captured_session
        self._router.deliver(message, session_id=session_id, upstream=upstream_name)

    @staticmethod
    def _progress_token(params: dict) -> Any:
        meta = params.get("_meta") if isinstance(params, dict) else None
        if not isinstance(meta, dict):
            return None
        token = meta.get("progressToken")
        if isinstance(token, (str, int)):
            return token
        return None

    def _audit_progress_notification(self, message: dict) -> None:
        params = message.get("params") if isinstance(message, dict) else None
        if not isinstance(params, dict):
            params = {}
        token = params.get("progressToken")
        if not isinstance(token, (str, int)):
            token = None
        parent_action_id = ""
        agent_id = self._agent_id_default
        parent_tool = ""
        # Notifications arrive on the upstream reader thread, which does
        # not inherit the request ContextVars. Pull the tenant captured
        # at tools/call time out of the inflight map instead of reading
        # _REQUEST_TENANT here, otherwise the audit + OVERT claim land
        # under empty tenant scope.
        captured_tenant = ""
        if token is not None:
            with self._inflight_lock:
                entry = self._inflight_progress.get(token)
            if entry is not None:
                # Session id is captured for routing in _on_upstream_notification;
                # the audit/OVERT path discards it intentionally so the perimeter
                # record schema stays unchanged.
                parent_action_id, agent_id, parent_tool, captured_tenant, _ = entry
        self._record_perimeter_audit(
            agent_id=agent_id,
            tool_name="mcp.notification.progress",
            parameters={
                "progressToken": token,
                "parent_action_id": parent_action_id,
                "parent_tool": parent_tool,
            },
            decision="observed",
            reason="upstream progress notification observed",
            tenant_id=captured_tenant,
        )
        self._overt_emit(
            surface="mcp.notification.progress",
            identifier=str(token) if token is not None else "",
            identifier_field="progress_token",
            request_obj={k: v for k, v in params.items() if k != "_meta"},
            decision="observed",
            reason="upstream progress notification observed",
            extra={
                "agent_id": agent_id,
                "parent_action_id": parent_action_id,
                "parent_tool": parent_tool,
            },
            tenant_id=captured_tenant,
        )

    def _audit_message_notification(self, message: dict) -> None:
        params = message.get("params") if isinstance(message, dict) else None
        if not isinstance(params, dict):
            params = {}
        level = params.get("level", "")
        if not isinstance(level, str):
            level = ""
        log_logger = params.get("logger", "")
        if not isinstance(log_logger, str):
            log_logger = ""
        # Log notifications carry no progressToken, so there is no way to
        # recover the originating request's tenant from the reader thread.
        # Pass tenant_id="" explicitly to make the fail-soft scope visible
        # rather than reading _REQUEST_TENANT in a thread that never set it.
        self._record_perimeter_audit(
            agent_id=self._agent_id_default,
            tool_name="mcp.notification.message",
            parameters={"level": level, "logger": log_logger},
            decision="observed",
            reason="upstream log notification observed",
            tenant_id="",
        )
        self._overt_emit(
            surface="mcp.notification.message",
            identifier=level,
            identifier_field="level",
            request_obj={"level": level, "logger": log_logger},
            decision="observed",
            reason="upstream log notification observed",
            extra={"agent_id": self._agent_id_default},
            tenant_id="",
        )

    def _write_to_client(self, payload: dict) -> None:
        # Serialize stdout writes between main loop and upstream reader thread.
        with self._stdout_lock:
            sys.stdout.write(strict_json_dumps(payload) + "\n")
            sys.stdout.flush()

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def close(self) -> None:
        for client in self._upstreams.values():
            try:
                client.close()
            except Exception:
                logger.exception("Failed to close upstream MCP client")
        if self._backend is not None:
            self._backend.close()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vaara-mcp-proxy",
        description="Vaara runtime governance proxy in front of one or more upstream MCP servers.",
    )
    parser.add_argument(
        "--upstream", action="append", default=[], dest="upstreams",
        help=(
            "Upstream MCP server command. Repeatable for v0.40 fan-out: "
            "`--upstream NAME=CMD` registers under a named slot; bare "
            "`--upstream CMD` lands under 'default'. The first slot (or "
            "'default' when supplied) is the stdio fallback."
        ),
    )
    parser.add_argument("--upstream-arg", action="append", default=[], dest="upstream_args",
                        help="Argument to pass to the (first) upstream command (repeatable)")
    parser.add_argument(
        "--upstream-url", action="append", default=[], dest="upstream_urls",
        help=(
            "Remote upstream MCP server reached over the Streamable HTTP "
            "transport. Repeatable: `--upstream-url NAME=URL` registers a named "
            "slot; bare `--upstream-url URL` lands under 'default'. A slot is "
            "either --upstream (stdio) or --upstream-url (remote), never both."
        ),
    )
    parser.add_argument(
        "--upstream-header", action="append", default=[], dest="upstream_headers",
        help=(
            "Static header sent on every request to a --upstream-url slot, e.g. "
            "`--upstream-header NAME=Authorization: Bearer TOKEN`. Repeatable for "
            "multiple headers or slots. The slot NAME must match an --upstream-url."
        ),
    )
    parser.add_argument(
        "--allow-private-upstream-hosts",
        action="store_true",
        # default None (not False) so an absent flag leaves the env opt-in
        # VAARA_MCP_ALLOW_PRIVATE_UPSTREAM live; passing False here would
        # shadow it and silently break the documented process-wide opt-in.
        default=None,
        help=(
            "Permit --upstream-url targets that resolve to loopback, "
            "link-local, RFC1918, or ULA addresses. OFF by default: such "
            "targets are refused to block SSRF. The cloud-metadata address "
            "stays refused even with this flag. Only set it for a trusted "
            "internal upstream you control."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help=(
            "stdio (default) reads JSON-RPC from stdin/stdout, suitable for "
            "in-process MCP clients. http exposes Streamable HTTP "
            "(POST /mcp) for fleet / multi-tenant deployments and "
            "requires the [server] extra."
        ),
    )
    parser.add_argument("--http-host", default="127.0.0.1",
                        help="Bind host when --transport http (default 127.0.0.1)")
    parser.add_argument("--http-port", type=int, default=8765,
                        help="Bind port when --transport http (default 8765)")
    parser.add_argument(
        "--http-log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    parser.add_argument("--db", type=Path, default=None,
                        help="Audit database path (default: $VAARA_DB or ./vaara_audit.db)")
    parser.add_argument("--agent-id", default="mcp-proxy-client",
                        help="Default agent_id for the audit trail")
    parser.add_argument("--allow-tool", action="append", default=[], dest="allow_tools",
                        help="Only expose this tool name (repeatable). If any --allow-tool "
                             "is given, all other upstream tools are filtered from tools/list "
                             "and rejected at tools/call.")
    parser.add_argument("--deny-tool", action="append", default=[], dest="deny_tools",
                        help="Filter this tool name from tools/list and reject any tools/call "
                             "to it (repeatable). Denylist wins on overlap with --allow-tool.")
    parser.add_argument("--allow-resource", action="append", default=[], dest="allow_resources",
                        help="Only expose this resource URI (repeatable). If any "
                             "--allow-resource is given, all other upstream resources are "
                             "filtered from resources/list and rejected at resources/read.")
    parser.add_argument("--deny-resource", action="append", default=[], dest="deny_resources",
                        help="Filter this resource URI from resources/list and reject any "
                             "resources/read to it (repeatable). Denylist wins on overlap.")
    parser.add_argument("--allow-prompt", action="append", default=[], dest="allow_prompts",
                        help="Only expose this prompt name (repeatable). If any --allow-prompt "
                             "is given, all other upstream prompts are filtered from "
                             "prompts/list and rejected at prompts/get.")
    parser.add_argument("--deny-prompt", action="append", default=[], dest="deny_prompts",
                        help="Filter this prompt name from prompts/list and reject any "
                             "prompts/get to it (repeatable). Denylist wins on overlap.")
    parser.add_argument("--attest-signing-key", type=Path, default=None,
                        help="PEM private key (EC P-256 = ES256, RSA = RS256) or raw "
                             "bytes file (HS256) for SEP-2787 attestation + receipt "
                             "pairing. Off when absent. Generate EC key: openssl ecparam "
                             "-genkey -name prime256v1 | openssl pkcs8 -topk8 -nocrypt "
                             "-out attest_key.pem")
    parser.add_argument("--attest-receipts-dir", type=Path, default=None,
                        help="Directory to write paired attestation + receipt JSON files "
                             "({n}-attest.json / {n}-receipt.json). Required when "
                             "--attest-signing-key is set.")
    parser.add_argument("--overt-signing-key", type=Path, default=None,
                        help="Ed25519 PEM private key used to sign OVERT 1.0 Base "
                             "Envelopes for every governed MCP interaction. Off when "
                             "absent. Generate one with: vaara keygen --dev --out PATH.")
    parser.add_argument("--overt-operator-key", type=Path, default=None,
                        help="Raw bytes file holding the operator HMAC key for OVERT "
                             "request_commitment (min 16 bytes). Required when "
                             "--overt-signing-key is set, unless "
                             "VAARA_OVERT_OPERATOR_KEY_HEX is set in the environment.")
    parser.add_argument("--overt-receipts-dir", type=Path, default=None,
                        help="Directory to write OVERT Base Envelopes into "
                             "(one canonical-CBOR file per envelope). Required when "
                             "--overt-signing-key is set.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    tool_allow = set(args.allow_tools) if args.allow_tools else None
    tool_deny = set(args.deny_tools) if args.deny_tools else set()
    resource_allow = set(args.allow_resources) if args.allow_resources else None
    resource_deny = set(args.deny_resources) if args.deny_resources else set()
    prompt_allow = set(args.allow_prompts) if args.allow_prompts else None
    prompt_deny = set(args.deny_prompts) if args.deny_prompts else set()

    overt_emitter = _build_overt_emitter_from_args(
        args,
        policy_hash=policy_hash_from_perimeter(
            tool_allow=tool_allow, tool_deny=tool_deny,
            resource_allow=resource_allow, resource_deny=resource_deny,
            prompt_allow=prompt_allow, prompt_deny=prompt_deny,
        ),
    )

    upstreams = _parse_upstream_specs(args.upstreams, args.upstream_args)
    upstream_urls = _parse_upstream_url_specs(args.upstream_urls)
    upstream_headers = _parse_upstream_header_specs(args.upstream_headers)
    if not upstreams and not upstream_urls:
        parser.error(
            "at least one --upstream or --upstream-url is required (e.g. "
            "`--upstream github=github-mcp-server` or "
            "`--upstream-url remote=https://host/mcp`).",
        )

    attest_emitter = _build_attest_emitter_from_args(
        args, upstreams=_attest_upstreams_for_slots(upstreams),
    )

    # The legacy single-upstream entry point only applies to a lone stdio
    # upstream with no remote slots in play.
    legacy_single = (
        list(next(iter(upstreams.values())))
        if (len(upstreams) == 1 and not upstream_urls) else None
    )
    try:
        proxy = VaaraMCPProxy(
            upstream_command=legacy_single,
            upstreams=upstreams if (legacy_single is None and upstreams) else None,
            upstream_urls=upstream_urls or None,
            upstream_headers=upstream_headers or None,
            allow_private_upstream_hosts=args.allow_private_upstream_hosts,
            db_path=args.db, agent_id_default=args.agent_id,
            allowlist=tool_allow,
            denylist=tool_deny if tool_deny else None,
            resource_allowlist=resource_allow,
            resource_denylist=resource_deny if resource_deny else None,
            prompt_allowlist=prompt_allow,
            prompt_denylist=prompt_deny if prompt_deny else None,
            overt_emitter=overt_emitter,
            attest_emitter=attest_emitter,
        )
    except (ValueError, ProxyError) as e:
        # ProxyError here means a --upstream-url target was refused by the SSRF
        # egress floor at client construction; surface it as a clean CLI error.
        parser.error(str(e))
    try:
        if args.transport == "http":
            proxy.run_http(
                host=args.http_host,
                port=args.http_port,
                log_level=args.http_log_level,
            )
        else:
            proxy.run()
    finally:
        proxy.close()


# A fan-out slot name is a short alphanumeric slug. The narrow pattern
# stops _parse_upstream_specs from confusing a command that itself
# contains '=' (e.g. ``python -m foo --bar=baz``) with a NAME=CMD prefix.
_UPSTREAM_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _parse_upstream_specs(
    upstream_specs: list[str], legacy_args: list[str],
) -> dict[str, list[str]]:
    """Turn ``--upstream`` / ``--upstream-arg`` CLI input into a fan-out map.

    Each ``--upstream`` is either ``NAME=CMD`` (named — NAME is a short
    alphanumeric slug) or ``CMD`` (lands under "default"). Commands that
    contain ``=`` (e.g. ``python -m foo --bar=baz``) stay intact because
    the NAME-prefix check rejects anything whose left-of-``=`` half isn't
    a valid slug. Legacy ``--upstream-arg`` values append to the first
    named slot for back-compat with single-upstream callers.
    """
    upstreams: dict[str, list[str]] = {}
    first_name: Optional[str] = None
    for spec in upstream_specs:
        if "=" in spec:
            candidate_name, _, candidate_cmd = spec.partition("=")
            candidate_name = candidate_name.strip()
            candidate_cmd = candidate_cmd.strip()
            if _UPSTREAM_NAME_RE.match(candidate_name) and candidate_cmd:
                name, command = candidate_name, candidate_cmd
            else:
                name, command = "default", spec
        else:
            name, command = "default", spec
        if not name or not command:
            raise SystemExit(
                f"invalid --upstream value {spec!r}; expected NAME=CMD or CMD",
            )
        upstreams[name] = [command]
        if first_name is None:
            first_name = name
    if legacy_args and first_name is not None:
        upstreams[first_name].extend(legacy_args)
    return upstreams


def _parse_upstream_url_specs(url_specs: list[str]) -> dict[str, str]:
    """Turn ``--upstream-url`` CLI input into a name -> URL map.

    Each value is ``NAME=URL`` (NAME a short slug) or a bare ``URL`` (lands
    under "default"). The URL must be http(s); the scheme check also keeps a
    bare URL containing ``=`` in its query string from being misread as
    ``NAME=URL``.
    """
    urls: dict[str, str] = {}
    for spec in url_specs:
        spec = spec.strip()
        if spec.lower().startswith(("http://", "https://")):
            name, url = "default", spec
        else:
            candidate_name, sep, candidate_url = spec.partition("=")
            candidate_name = candidate_name.strip()
            candidate_url = candidate_url.strip()
            if (
                sep
                and _UPSTREAM_NAME_RE.match(candidate_name)
                and candidate_url.lower().startswith(("http://", "https://"))
            ):
                name, url = candidate_name, candidate_url
            else:
                raise SystemExit(
                    f"invalid --upstream-url value {spec!r}; expected "
                    "NAME=URL or URL (http/https)",
                )
        if name in urls:
            raise SystemExit(f"duplicate --upstream-url slot {name!r}")
        urls[name] = url
    return urls


def _parse_upstream_header_specs(header_specs: list[str]) -> dict[str, dict[str, str]]:
    """Turn ``--upstream-header`` CLI input into a name -> {header: value} map.

    Each value is ``NAME=Header-Name: header value``. Splitting NAME off the
    first ``=`` is unambiguous because the slug pattern never matches a header
    line, so a base64 token carrying ``=`` in the value stays intact.
    """
    headers: dict[str, dict[str, str]] = {}
    for spec in header_specs:
        name, sep, header_line = spec.partition("=")
        name = name.strip()
        if not sep or not _UPSTREAM_NAME_RE.match(name):
            raise SystemExit(
                f"invalid --upstream-header value {spec!r}; expected "
                "NAME=Header-Name: value",
            )
        field, colon, value = header_line.partition(":")
        field = field.strip()
        if not colon or not field:
            raise SystemExit(
                f"invalid --upstream-header value {spec!r}; the header must be "
                "'Header-Name: value'",
            )
        headers.setdefault(name, {})[field] = value.strip()
    return headers


def _attest_upstreams_for_slots(
    upstreams: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Key the attestation fingerprint table the way the proxy slots upstreams.

    A single upstream (named ``NAME=CMD`` or bare ``CMD``) collapses into the
    ``"default"`` slot inside ``VaaraMCPProxy``, and ``_REQUEST_UPSTREAM``
    resolves to ``"default"`` at runtime. The emitter must be keyed the same
    way, or ``fingerprint_for("default")`` misses the precomputed cmd-hash and
    emits a ``cmd:sha256:unknown-default`` placeholder. Multi-upstream fan-out
    keeps the operator-supplied slot names.
    """
    if len(upstreams) == 1:
        return {"default": list(next(iter(upstreams.values())))}
    return dict(upstreams)


def _build_attest_emitter_from_args(
    args: argparse.Namespace,
    *,
    upstreams: dict[str, list[str]],
) -> Optional[AttestPairEmitter]:
    """Construct the attestation pair emitter from CLI args, or None if not configured.

    Off when --attest-signing-key is absent. If signing-key is set,
    --attest-receipts-dir must also be present, or the proxy refuses to start.
    """
    if args.attest_signing_key is None:
        if args.attest_receipts_dir is not None:
            print(
                "vaara-mcp-proxy: --attest-receipts-dir has no effect without "
                "--attest-signing-key. Exiting.",
                file=sys.stderr,
            )
            sys.exit(2)
        return None
    if args.attest_receipts_dir is None:
        print(
            "vaara-mcp-proxy: --attest-signing-key requires --attest-receipts-dir.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        return build_attest_emitter(
            signing_key_path=args.attest_signing_key,
            receipts_dir=args.attest_receipts_dir,
            upstream_commands=upstreams,
        )
    except AttestConfigError as exc:
        print(f"vaara-mcp-proxy: {exc}", file=sys.stderr)
        sys.exit(2)


def _build_overt_emitter_from_args(
    args: argparse.Namespace, *, policy_hash: bytes,
) -> Optional[OVERTReceiptEmitter]:
    """Construct the OVERT emitter from CLI args, or None if not configured.

    Off when --overt-signing-key is absent. If signing-key is set, both
    --overt-receipts-dir and an operator HMAC key (file or env var) must
    also be present, or the proxy refuses to start.
    """
    if args.overt_signing_key is None:
        if args.overt_receipts_dir is not None or args.overt_operator_key is not None:
            print(
                "vaara-mcp-proxy: --overt-receipts-dir and --overt-operator-key "
                "have no effect without --overt-signing-key. Exiting.",
                file=sys.stderr,
            )
            sys.exit(2)
        return None
    if args.overt_receipts_dir is None:
        print(
            "vaara-mcp-proxy: --overt-signing-key requires --overt-receipts-dir.",
            file=sys.stderr,
        )
        sys.exit(2)
    operator_key_hex = os.environ.get("VAARA_OVERT_OPERATOR_KEY_HEX")
    try:
        return build_emitter(
            signing_key_path=args.overt_signing_key,
            operator_key_path=args.overt_operator_key,
            operator_key_hex=operator_key_hex,
            receipts_dir=args.overt_receipts_dir,
            policy_hash=policy_hash,
        )
    except OVERTConfigError as exc:
        print(f"vaara-mcp-proxy: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
