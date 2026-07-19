# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ExternalScorer — call a remote /v1/score endpoint as a scorer backend.

The v0.10.0 HTTP API defines a stable wire contract for scorers; this
module reuses the same contract on the inbound side. The remote can be
another Vaara instance, a NeMo Guardrails service, or any peer that
implements the /v1/score request/response shape.

Fail-closed deny on any transport error or malformed response so a
``CompositeScorer`` cannot silently downgrade a decision because one
member was unreachable.

Pair with ``vaara.scorer.composite.CompositeScorer`` to run Vaara's
``AdaptiveScorer`` alongside external scorers and merge the results.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional


class ExternalScorerError(RuntimeError):
    """Raised when an external scorer returns an unusable response."""


class ExternalScorer:
    """Vaara-compatible scorer that POSTs to a remote /v1/score endpoint."""

    def __init__(
        self,
        url: str,
        *,
        name: Optional[str] = None,
        timeout: float = 5.0,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        if not url:
            raise ValueError("ExternalScorer requires a non-empty url")
        self._url = url
        self._name = name or url
        self._timeout = float(timeout)
        self._headers = {
            "content-type": "application/json",
            **(headers or {}),
        }

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(context).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, headers=self._headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            return _fail_closed(
                self._name,
                f"external scorer transport failure: {exc.__class__.__name__}",
            )
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return _fail_closed(self._name, "external scorer returned non-JSON body")
        return _coerce_external_result(self._name, data)


def _fail_closed(name: str, message: str) -> dict[str, Any]:
    return {
        "point_estimate": 1.0,
        "decision": "deny",
        "backend": name,
        "raw_result": {
            "point_estimate": 1.0,
            "conformal_interval": [1.0, 1.0],
            "fail_closed_reason": message,
        },
    }


def _coerce_external_result(name: str, data: Any) -> dict[str, Any]:
    """Map a remote /v1/score response into Vaara's internal scorer dict."""
    if not isinstance(data, dict):
        return _fail_closed(name, "external scorer returned non-object body")
    risk = data.get("risk") or data.get("raw_result") or {}
    if isinstance(risk, dict):
        point = risk.get("point", risk.get("point_estimate"))
        lower = risk.get("lower", point)
        upper = risk.get("upper", point)
        interval = risk.get("conformal_interval", [lower, upper])
    else:
        point, interval = data.get("point_estimate"), None
    try:
        point = float(point) if point is not None else 0.5
    except (TypeError, ValueError):
        point = 0.5
    point = max(0.0, min(1.0, point))
    decision = data.get("decision") or _decision_from_score(point)
    return {
        "point_estimate": point,
        "decision": decision,
        "backend": name,
        "raw_result": {
            "point_estimate": point,
            "conformal_interval": interval or [point, point],
        },
    }


def _decision_from_score(score: float) -> str:
    if score >= 0.7:
        return "deny"
    if score >= 0.4:
        return "escalate"
    return "allow"
