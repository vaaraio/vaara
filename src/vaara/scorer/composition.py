# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ExternalScorer — call a remote /v1/score endpoint as a scorer backend.

The v0.10.0 HTTP API defines a stable wire contract for scorers; this
module reuses the same contract on the inbound side. The remote can be
another Vaara instance, a NeMo Guardrails service, or any peer that
implements the /v1/score request/response shape.

Everything the contract carries reaches the trail: the remote's
``explanation`` is the recorded reason, its ``thresholds`` are the operating
point the receipt states, its ``signals``, ``risk.alpha``, ``risk.bucket``
and ``calibration_size`` land on the risk record. A remote that answers a
score with no decision is decided here, at a stated point. A remote that
answers a decision without the thresholds it decided at is refused: the
receipt would otherwise state an operating point nobody was at.

Fail-closed deny on any transport error or malformed response so a
``CompositeScorer`` cannot silently downgrade a decision because one
member was unreachable.

Pair with ``vaara.scorer.composite.CompositeScorer`` to run Vaara's
``AdaptiveScorer`` alongside external scorers and merge the results.
"""

from __future__ import annotations

import http.client
import json
import math
import urllib.parse
import urllib.request
from typing import Any, Optional

#: The point this module decides at when the remote sent a score and no
#: decision. Stated on the record, so the receipt reports what ran.
LOCAL_THRESHOLD_ALLOW = 0.4
LOCAL_THRESHOLD_DENY = 0.7

#: A remote answer longer than this is not a /v1/score response.
_MAX_BODY_BYTES = 1 << 20


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
            # Operator-configured scoring backend.
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # nosec B310
                payload = resp.read(_MAX_BODY_BYTES + 1)
                # A sized read returns what arrived before the peer went
                # away instead of raising; the bytes still owed say it did.
                owed = getattr(resp, "length", None)
        except (OSError, http.client.HTTPException) as exc:
            # URLError, HTTPError and TimeoutError are OSErrors; so are a
            # connection reset or an SSL failure while reading. IncompleteRead
            # and RemoteDisconnected are HTTPExceptions. All of them mean the
            # scorer never answered.
            return _fail_closed(
                self._name,
                f"external scorer transport failure: {exc.__class__.__name__}",
            )
        if len(payload) > _MAX_BODY_BYTES:
            return _fail_closed(self._name, "external scorer returned an oversized body")
        if isinstance(owed, int) and owed > 0:
            return _fail_closed(
                self._name, "external scorer transport failure: incomplete body",
            )
        try:
            data = json.loads(payload)
        except ValueError:
            return _fail_closed(self._name, "external scorer returned non-JSON body")
        return _coerce_external_result(self._name, data)


def _safe_name(name: str) -> str:
    """The scorer's name as a record may carry it: a URL is cut to its host.

    The name defaults to the endpoint URL, which can hold credentials or a
    token in its query, and a receipt is built to leave the machine.
    """
    parts = urllib.parse.urlsplit(name)
    if parts.scheme and parts.hostname:
        return parts.hostname + (f":{parts.port}" if parts.port else "")
    return name


def _fail_closed(name: str, message: str, missing: str = "") -> dict[str, Any]:
    # The scorer never answered, or answered something that is not a
    # judgement. The ground and the missing input say so; without them the
    # pipeline records the deny as a risk threshold the action crossed.
    unavailable = f"scorer:{_safe_name(name)}" + (f".{missing}" if missing else "")
    return {
        "point_estimate": 1.0,
        "decision": "deny",
        "action": "deny",
        "reason": message,
        "violation_type": "scorer_unavailable",
        "unavailable_input": unavailable,
        "backend": name,
        "raw_result": {
            "point_estimate": 1.0,
            "conformal_interval": [1.0, 1.0],
            "fail_closed_reason": message,
        },
    }


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _coerce_external_result(name: str, data: Any) -> dict[str, Any]:
    """Map a remote /v1/score response into Vaara's internal scorer dict.

    Reads the published response shape (``risk{point,lower,upper,alpha,
    bucket}``, ``thresholds{allow,deny}``, ``signals``, ``explanation``) and
    the internal scorer shape (``raw_result``, ``threshold_allow``,
    ``reason``) a Vaara peer may hand back instead.
    """
    if not isinstance(data, dict):
        return _fail_closed(name, "external scorer returned non-object body")
    risk = data.get("risk") or data.get("raw_result") or {}
    if not isinstance(risk, dict):
        risk = {}
    point = risk.get("point", risk.get("point_estimate", data.get("point_estimate")))
    lower = risk.get("lower", point)
    upper = risk.get("upper", point)
    interval = risk.get("conformal_interval", [lower, upper])
    try:
        point = float(point) if point is not None else 0.5
    except (TypeError, ValueError):
        point = 0.5
    point = max(0.0, min(1.0, point))

    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        thresholds = {}
    allow_at = _number(thresholds.get("allow", data.get("threshold_allow",
                                                        risk.get("threshold_allow"))))
    deny_at = _number(thresholds.get("deny", data.get("threshold_deny",
                                                      risk.get("threshold_deny"))))

    reason = data.get("explanation", data.get("reason", ""))
    reason = reason if isinstance(reason, str) else str(reason)

    decision = data.get("decision", data.get("action"))
    if not isinstance(decision, str) or not decision:
        # A score and nothing else: the decision is made here, at a point the
        # record states, so the receipt reports the operating point that ran.
        decision = _decision_from_score(point)
        allow_at, deny_at = LOCAL_THRESHOLD_ALLOW, LOCAL_THRESHOLD_DENY
        reason = (f"decided here from the remote score {point:.3f} at "
                  f"allow<{allow_at}, deny>={deny_at}" + (f"; {reason}" if reason else ""))
    elif allow_at is None or deny_at is None:
        # The contract requires thresholds. Carrying a verdict without the
        # point it was made at would leave the receipt to assume one.
        return _fail_closed(
            name,
            "external scorer stated a decision without the thresholds it "
            "decided at",
            missing="thresholds",
        )

    raw: dict[str, Any] = {
        "point_estimate": point,
        "conformal_interval": interval or [point, point],
    }
    signals = data.get("signals", risk.get("signals"))
    if isinstance(signals, dict):
        raw["signals"] = signals
    alpha = _number(risk.get("alpha", risk.get("effective_alpha")))
    if alpha is not None:
        raw["effective_alpha"] = alpha
    bucket = risk.get("bucket", risk.get("bucket_category"))
    if isinstance(bucket, str) and bucket:
        raw["bucket_category"] = bucket
    size = data.get("calibration_size", risk.get("calibration_size"))
    if isinstance(size, int) and not isinstance(size, bool):
        raw["calibration_size"] = size

    return {
        "point_estimate": point,
        "decision": decision,
        "action": decision,
        "reason": reason,
        "threshold_allow": allow_at,
        "threshold_deny": deny_at,
        "backend": name,
        "raw_result": raw,
    }


def _decision_from_score(score: float) -> str:
    if score >= LOCAL_THRESHOLD_DENY:
        return "deny"
    if score >= LOCAL_THRESHOLD_ALLOW:
        return "escalate"
    return "allow"
