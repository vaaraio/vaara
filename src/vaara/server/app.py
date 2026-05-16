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
from vaara.scorer.adaptive import AdaptiveScorer
from vaara.server.routes import register
from vaara.server.state import ServerState


def create_app(
    scorer: Optional[AdaptiveScorer] = None,
    audit: Optional[AuditTrail] = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        scorer: Pre-configured scorer, or None for default `AdaptiveScorer()`.
        audit: Pre-configured audit trail, or None for default in-memory.
    """
    state = ServerState(scorer=scorer, audit=audit)
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
