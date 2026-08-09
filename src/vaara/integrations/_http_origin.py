# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-site request guard for Vaara's local HTTP surfaces.

Every proxy and console Vaara ships binds loopback by default and takes no
inbound credential, on the reasoning that only the operator can reach the
port. A browser breaks that reasoning: any page the operator visits can issue
requests to ``127.0.0.1`` from inside their machine.

A cross-origin ``fetch`` with ``Content-Type: text/plain`` is a CORS-"simple"
request, so it is sent without a preflight the server could refuse, and none
of these surfaces inspect Content-Type before parsing JSON. The attacker page
cannot *read* the reply — the same-origin policy still holds — but the request
already ran: a tool call executed, an LLM call spent the operator's upstream
key, a turn landed in the audit trail attributed to someone else. For a
governance layer, an action it cannot attribute is the whole loss.

The MCP Streamable HTTP transport spec states the requirement directly:
servers MUST validate the Origin header on incoming connections. This module
is that check, in one place, for every app Vaara serves.

The rule:

* **No Origin header** — allowed. Native clients (Claude Code, Cursor, curl,
  the Python SDKs) send none, and a browser cannot suppress it.
* **Origin matching the server's own host** — allowed. This is the page the
  server itself served talking back to it, which is how the console works.
* **Origin in the operator's allow-list** — allowed, compared exactly, so
  ``https://app.example.evil.test`` never passes for ``https://app.example``.
* **Anything else** — 403.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: Paths that stay reachable from any origin. Health checks expose no operator
#: data and load balancers poll them from wherever they happen to run.
DEFAULT_EXEMPT_PATHS = frozenset({"/health", "/healthz"})


def normalise_origins(origins: Optional[Iterable[str]]) -> set[str]:
    """Drop blanks and surrounding whitespace from an operator-supplied list."""
    if not origins:
        return set()
    return {origin.strip() for origin in origins if origin and origin.strip()}


def origins_from_env(raw: Optional[str]) -> set[str]:
    """Parse the comma-separated env-var form of an allow-list."""
    if not raw:
        return set()
    return normalise_origins(raw.split(","))


def origin_is_allowed(
    origin: Optional[str],
    *,
    allowed: set[str],
    self_origins: Iterable[str] = (),
) -> bool:
    """Decide one Origin header value. See the module docstring for the rule."""
    if origin is None:
        return True
    origin = origin.strip()
    if origin in allowed:
        return True
    # "null" is what a sandboxed iframe, a data: document, or a file:// page
    # sends. It names no host, so it can never be same-origin with us, and
    # treating it as anonymous would readmit exactly the pages this refuses.
    if origin == "null":
        return False
    return origin in set(self_origins)


def install_origin_guard(
    app,
    *,
    allowed_origins: Optional[Iterable[str]] = None,
    exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS,
    surface: str = "vaara",
) -> None:
    """Attach the cross-site guard to a FastAPI/Starlette app.

    Registered as HTTP middleware, so it runs before any route handler and
    before any other middleware registered earlier — Starlette applies them
    in reverse registration order, and a cross-site request should be refused
    without the rest of the stack looking at it.
    """
    from fastapi.responses import JSONResponse

    allowed = normalise_origins(allowed_origins)
    exempt = frozenset(exempt_paths)

    @app.middleware("http")
    async def _origin_guard(request, call_next):
        origin = request.headers.get("origin")
        if origin is not None and request.url.path not in exempt:
            # The origin the server is being addressed as. Host carries the
            # port, which Origin includes and request.url.hostname does not.
            host = request.headers.get("host", "")
            self_origins = (
                {f"{request.url.scheme}://{host}", f"http://{host}", f"https://{host}"}
                if host else set()
            )
            if not origin_is_allowed(
                origin, allowed=allowed, self_origins=self_origins,
            ):
                cleaned = "".join(
                    c if c.isprintable() and c not in "\r\n" else "?"
                    for c in origin
                )[:200]
                logger.warning(
                    "%s: refused cross-origin request from %s", surface, cleaned,
                )
                return JSONResponse(
                    status_code=403,
                    content={"error": {
                        "code": "origin_not_allowed",
                        "message": (
                            "This endpoint refuses requests carrying an Origin "
                            "header from another site. That is what stops a web "
                            "page you visit from driving a loopback-bound Vaara "
                            "service. Allow the origin explicitly if a browser "
                            "client needs it."
                        ),
                    }},
                )
        return await call_next(request)
