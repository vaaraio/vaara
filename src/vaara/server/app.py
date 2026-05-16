"""FastAPI application factory for the Vaara HTTP API reference server.

The server holds a single in-process `AdaptiveScorer` and `AuditTrail` for
the lifetime of the process. Both are stateful: the scorer maintains
conformal calibration and MWU weights across requests, and the audit trail
is a single hash chain.

State persistence is out of scope for v1. State is in-memory unless the
embedder wires the audit trail to a persistent backend.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from vaara.audit.trail import AuditTrail
from vaara.policy.controller import PolicyController
from vaara.scorer.adaptive import AdaptiveScorer
from vaara.server.routes import register
from vaara.server.state import ServerState


def create_app(
    scorer: Optional[AdaptiveScorer] = None,
    audit: Optional[AuditTrail] = None,
    policy_controller: Optional[PolicyController] = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        scorer: Pre-configured scorer, or None for default `AdaptiveScorer()`.
        audit: Pre-configured audit trail, or None for default in-memory.
        policy_controller: Pre-loaded ``PolicyController``. When supplied,
            the scorer is registered as a listener and ``POST
            /v1/policy/reload`` becomes available. When omitted, the
            reload endpoint returns ``409 policy_not_configured``.
    """
    state = ServerState(
        scorer=scorer, audit=audit, policy_controller=policy_controller
    )
    app = FastAPI(
        title="Vaara HTTP API",
        version="1.0.0",
        description=(
            "Conformal-scoring risk evaluation and hash-chained audit "
            "emission. Authoritative spec: docs/openapi.yaml in the vaara "
            "repository."
        ),
    )
    app.state.vaara = state
    register(app, state)
    return app
