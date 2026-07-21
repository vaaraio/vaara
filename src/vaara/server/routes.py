# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Route handlers for the Vaara HTTP API reference server."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from vaara import __version__ as _vaara_version
from vaara.audit.timeanchor import TimeAnchorError
from vaara.audit.trail import AuditRecord, EventType
from vaara.server import schemas as S
from vaara.server.state import ServerState


_SERVER_NAME = "vaara-reference-server"
_SERVER_VERSION = "1.0.2"


def _error(code: str, message: str, http_status: int, **details) -> HTTPException:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return HTTPException(status_code=http_status, detail=body)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _resolve_tenant(body_value: str, header_value: Optional[str]) -> str:
    body = (body_value or "").strip()
    header = (header_value or "").strip()
    return body or header


def register(app: FastAPI, state: ServerState) -> None:

    @app.exception_handler(HTTPException)
    async def _http_exc_handler(_request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": str(exc.detail)}},
        )

    @app.get("/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/server", response_model=S.ServerInfo)
    async def server_info():
        return S.ServerInfo(
            name=_SERVER_NAME,
            version=_SERVER_VERSION,
            vaara_version=_vaara_version,
            capabilities=S.Capabilities(
                score=True, audit=True, outcome_feedback=True,
            ),
            scorer=S.ScorerInfo(
                type=type(state.scorer).__name__,
                calibration_size=state.scorer._conformal.calibration_size,
                threshold_allow=state.scorer._threshold_allow,
                threshold_deny=state.scorer._threshold_deny,
                alpha=state.scorer._conformal._alpha,
            ),
        )

    @app.post("/v1/score", response_model=S.ScoreResponse)
    async def score(
        req: S.ScoreRequest,
        x_vaara_tenant: Optional[str] = Header(default=None, alias="X-Vaara-Tenant"),
    ):
        tenant_id = _resolve_tenant(req.tenant_id, x_vaara_tenant)
        ctx = req.model_dump(exclude_none=True)
        ctx["tenant_id"] = tenant_id
        try:
            decision_dict = state.scorer.evaluate(ctx)
        except Exception as exc:
            raise _error(
                "scorer_error", str(exc), status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        raw = decision_dict.get("raw_result", {}) or {}
        lower, upper = (raw.get("conformal_interval") or [0.0, 1.0])
        action_id = str(uuid.uuid4())
        signals = {k: float(v) for k, v in (raw.get("signals") or {}).items()}
        state.remember_action(
            action_id=action_id,
            agent_id=req.agent_id,
            tool_name=req.tool_name,
            predicted_risk=float(raw.get("point_estimate", 0.5) or 0.5),
            signals=signals,
            tenant_id=tenant_id,
        )

        return S.ScoreResponse(
            action_id=action_id,
            decision=decision_dict.get("action", "escalate"),
            risk=S.RiskBlock(
                point=raw.get("point_estimate", 0.5),
                lower=lower,
                upper=upper,
                alpha=raw.get("effective_alpha", 0.10),
                bucket=raw.get("bucket_category"),
            ),
            signals=signals,
            mwu_weights={k: float(v) for k, v in state.scorer._mwu.weights.items()},
            thresholds=S.Thresholds(
                allow=float(
                    decision_dict.get("threshold_allow", state.scorer._threshold_allow)
                ),
                deny=float(
                    decision_dict.get("threshold_deny", state.scorer._threshold_deny)
                ),
            ),
            sequence_risk=float(raw.get("sequence_risk", 0.0) or 0.0),
            calibration_size=int(raw.get("calibration_size", 0) or 0),
            evaluation_ms=float(decision_dict.get("evaluation_ms", 0.0) or 0.0),
            explanation=decision_dict.get("reason", ""),
        )

    @app.post("/v1/score/outcome", status_code=204)
    async def score_outcome(req: S.OutcomeRequest):
        info = state.lookup_action(req.action_id)
        if info is None:
            raise _error(
                "unknown_action", f"action_id {req.action_id!r} not found",
                status.HTTP_404_NOT_FOUND,
            )
        state.scorer.record_outcome(
            agent_id=info.agent_id,
            tool_name=info.tool_name,
            predicted_risk=info.predicted_risk,
            actual_outcome=req.outcome_severity,
            signals=info.signals,
        )
        return None

    @app.post(
        "/v1/audit/events",
        response_model=S.AuditEventResponse,
        status_code=201,
    )
    async def append_audit_event(
        req: S.AuditEventRequest,
        x_vaara_tenant: Optional[str] = Header(default=None, alias="X-Vaara-Tenant"),
    ):
        try:
            event_type = EventType(req.event_type)
        except ValueError:
            raise _error(
                "bad_event_type", f"unknown event_type {req.event_type!r}",
                status.HTTP_400_BAD_REQUEST,
            )

        tenant_id = _resolve_tenant(req.tenant_id, x_vaara_tenant)
        if not tenant_id:
            info = state.lookup_action(req.action_id)
            if info is not None:
                tenant_id = info.tenant_id

        record = AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=req.action_id,
            event_type=event_type,
            timestamp=time.time(),
            agent_id=req.agent_id or "",
            tool_name=req.tool_name or "",
            data=req.payload or {},
            regulatory_articles=[],
            tenant_id=tenant_id,
        )
        state.audit._append(record)
        return S.AuditEventResponse(
            event_id=record.record_id,
            chain_position=state.audit.size - 1,
            event_hash=record.record_hash,
            previous_hash=record.previous_hash,
            timestamp=_iso(record.timestamp),
        )

    @app.get(
        "/v1/audit/actions/{action_id}/chain",
        response_model=S.AuditChain,
    )
    async def read_action_chain(
        action_id: str,
        x_vaara_tenant: Optional[str] = Header(default=None, alias="X-Vaara-Tenant"),
    ):
        tenant_id = (x_vaara_tenant or "").strip()
        scoped = state.audit.get_action_chain_scoped(action_id, tenant_id)
        if not scoped:
            # Unknown action and cross-tenant action both 404 with the same
            # message: a caller scoped to one tenant gets no signal that an
            # action_id exists for another tenant.
            raise _error(
                "unknown_action", f"no audit records for {action_id!r}",
                status.HTTP_404_NOT_FOUND,
            )
        return S.AuditChain(
            action_id=action_id,
            events=[
                S.AuditChainEvent(
                    event_id=r.record_id,
                    event_type=r.event_type.value,
                    chain_position=pos,
                    event_hash=r.record_hash,
                    previous_hash=r.previous_hash,
                    timestamp=_iso(r.timestamp),
                    payload=r.data or {},
                )
                for pos, r in scoped
            ],
        )

    @app.post("/v1/audit/verify", response_model=S.VerifyResponse)
    async def verify_audit_chain(_req: Optional[S.VerifyRequest] = None):
        # v1: full-chain verify only. Ranged verify is in the spec but
        # not yet implemented server-side.
        problem = state.audit.verify_chain()
        if problem is None:
            return S.VerifyResponse(
                valid=True, events_checked=state.audit.size,
            )
        return S.VerifyResponse(
            valid=False,
            events_checked=state.audit.size,
            first_break=None,
        )

    @app.post(
        "/v1/detect/injection", response_model=S.DetectInjectionResponse,
    )
    async def detect_injection_endpoint(req: S.DetectInjectionRequest):
        from vaara.detect import detect_injection

        result = detect_injection(req.text, threshold=req.threshold)
        return S.DetectInjectionResponse(**result.to_dict())

    @app.post("/v1/detect/pii", response_model=S.DetectPIIResponse)
    async def detect_pii_endpoint(req: S.DetectPIIRequest):
        from vaara.detect import detect_pii

        result = detect_pii(req.text)
        return S.DetectPIIResponse(**result.to_dict())

    @app.post("/v1/policy/reload", response_model=S.PolicyReloadResponse)
    async def reload_policy(
        req: S.PolicyReloadRequest,
        x_vaara_tenant: Optional[str] = Header(default=None, alias="X-Vaara-Tenant"),
    ):
        from vaara.policy.schema import PolicyError

        tenant_id = _resolve_tenant(req.tenant_id, x_vaara_tenant)
        registry = state.policy_registry
        controller = (
            registry.get_exact(tenant_id) if registry is not None else None
        )
        if controller is None and not tenant_id:
            controller = state.policy_controller
        if registry is None and controller is None:
            raise _error(
                code="policy_not_configured",
                message=(
                    "Server has no policy plane; start with "
                    "`vaara serve --policy PATH` or `--policy-dir DIR` to "
                    "enable reload."
                ),
                http_status=status.HTTP_409_CONFLICT,
            )

        if (req.path is None) == (req.body is None):
            raise _error(
                code="invalid_request",
                message="Exactly one of `path` or `body` must be supplied.",
                http_status=status.HTTP_400_BAD_REQUEST,
            )

        source = req.body if req.body is not None else req.path
        try:
            if registry is not None:
                result = registry.reload(tenant_id, source, format=req.format)
            else:
                result = controller.reload(source, format=req.format)
        except PolicyError as exc:
            raise _error(
                code="policy_invalid",
                message=str(exc),
                http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        return S.PolicyReloadResponse(
            version=result.version,
            thresholds_default={
                "escalate": result.thresholds_default_escalate,
                "deny": result.thresholds_default_deny,
            },
            sequence_count=result.sequence_count,
            action_class_count=result.action_class_count,
            escalation_route_count=result.escalation_route_count,
            tenant_id=tenant_id,
        )

    @app.post("/v1/anchor")
    async def anchor_receipt_endpoint(req: S.AnchorRequest, request: Request):
        # Paid coin slot: x402-gated in front of the qualified eIDAS anchor.
        # Free (no-op gate) until an operator sets VAARA_X402_ENABLED; a
        # non-None challenge is the 402 payment-required response to return.
        challenge = state.x402.check(
            request,
            resource="/v1/anchor",
            description="Qualified eIDAS receipt anchor (rfc3161-eidas-qualified)",
        )
        if challenge is not None:
            return challenge
        try:
            anchor = state.anchorer.anchor(req.receipt)
            attested = state.anchorer.attested_time(req.receipt, anchor)
        except TimeAnchorError as exc:
            # Upstream QTSP condition (refusal, timeout, pin mismatch) — the
            # anchor is a dependency the server does not control.
            raise _error(
                "anchor_failed", str(exc), status.HTTP_502_BAD_GATEWAY,
            )
        return S.AnchorResponse(
            anchor=S.TimestampAnchor(**anchor),
            attested=attested,
        )
