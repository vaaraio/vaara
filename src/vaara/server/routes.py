"""Route handlers for the Vaara HTTP API reference server."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from vaara import __version__ as _vaara_version
from vaara.audit.trail import AuditRecord, EventType
from vaara.server import schemas as S
from vaara.server.state import ServerState


_SERVER_NAME = "vaara-reference-server"
_SERVER_VERSION = "1.0.0"


def _error(code: str, message: str, http_status: int, **details) -> HTTPException:
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return HTTPException(status_code=http_status, detail=body)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


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
    async def score(req: S.ScoreRequest):
        ctx = req.model_dump(exclude_none=True)
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
                allow=state.scorer._threshold_allow,
                deny=state.scorer._threshold_deny,
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
    async def append_audit_event(req: S.AuditEventRequest):
        try:
            event_type = EventType(req.event_type)
        except ValueError:
            raise _error(
                "bad_event_type", f"unknown event_type {req.event_type!r}",
                status.HTTP_400_BAD_REQUEST,
            )

        record = AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=req.action_id,
            event_type=event_type,
            timestamp=time.time(),
            agent_id=req.agent_id or "",
            tool_name=req.tool_name or "",
            data=req.payload or {},
            regulatory_articles=[],
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
    async def read_action_chain(action_id: str):
        records = state.audit._by_action.get(action_id, [])
        if not records:
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
                    chain_position=state.audit._records.index(r),
                    event_hash=r.record_hash,
                    previous_hash=r.previous_hash,
                    timestamp=_iso(r.timestamp),
                    payload=r.data or {},
                )
                for r in records
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
