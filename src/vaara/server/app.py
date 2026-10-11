# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
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
from fastapi.openapi.utils import get_openapi

from vaara.audit.trail import AuditTrail
from vaara.policy.controller import PolicyController
from vaara.policy.registry import PolicyRegistry
from vaara.scorer.adaptive import AdaptiveScorer
from vaara.server.routes import register
from vaara.server.state import ServerState


def create_app(
    scorer: Optional[AdaptiveScorer] = None,
    audit: Optional[AuditTrail] = None,
    policy_controller: Optional[PolicyController] = None,
    policy_registry: Optional[PolicyRegistry] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        scorer: Pre-configured scorer, or None for default `AdaptiveScorer()`.
        audit: Pre-configured audit trail, or None for default in-memory.
        policy_controller: Pre-loaded ``PolicyController``. When supplied,
            the scorer is registered as a listener and ``POST
            /v1/policy/reload`` becomes available. When omitted (and no
            ``policy_registry``), the reload endpoint returns
            ``409 policy_not_configured``.
        policy_registry: Pre-loaded ``PolicyRegistry`` for multi-tenant
            deployments. Mutually exclusive with ``policy_controller`` —
            single-controller callers are wrapped into a registry's ""
            slot automatically by ``ServerState``.
        api_key: Bearer key required on every endpoint except
            ``GET /v1/health`` when set. ``None`` means unauthenticated —
            acceptable only on a loopback bind; the CLI refuses to start
            a non-loopback server without one.
    """
    state = ServerState(
        scorer=scorer,
        audit=audit,
        policy_controller=policy_controller,
        policy_registry=policy_registry,
        api_key=api_key,
    )
    app = FastAPI(
        title="Vaara HTTP API",
        version="1.0.2",
        description=(
            "Conformal-scoring risk evaluation and hash-chained audit "
            "emission. Authoritative spec: docs/openapi.yaml in the vaara "
            "repository."
        ),
    )
    app.state.vaara = state
    register(app, state)
    _publish_openapi(app)
    return app


def _publish_openapi(app: FastAPI) -> None:
    """Make the generated schema say what the server answers.

    FastAPI adds a ``422`` with its own ``HTTPValidationError`` body to every
    route that takes a body. This server answers a malformed request with
    the ``400`` ``Error`` body the routes declare (see ``routes.register``),
    so that automatic entry would promise a body nobody receives. Only the
    automatic entries go: a route that declares its own ``422`` keeps it.
    """

    def openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title, version=app.version, description=app.description,
            routes=app.routes,
        )
        for methods in schema.get("paths", {}).values():
            for operation in methods.values():
                response = operation.get("responses", {}).get("422")
                ref = ((response or {}).get("content", {}).get("application/json", {})
                       .get("schema", {}).get("$ref", ""))
                if ref.endswith("/HTTPValidationError"):
                    del operation["responses"]["422"]
        components = schema.get("components", {}).get("schemas", {})
        components.pop("HTTPValidationError", None)
        components.pop("ValidationError", None)
        app.openapi_schema = schema
        return schema

    app.openapi = openapi  # type: ignore[method-assign]
