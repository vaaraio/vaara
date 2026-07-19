# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rebuff adapter. Upstream prompt-injection signal.

Rebuff layers four checks: heuristic, LLM-based, vector-DB
similarity, and canary-word leak detection. The adapter wraps a
pre-configured ``Rebuff`` (hosted) or ``RebuffSdk`` (self-hosted)
instance and parses the ``DetectResponse`` into a
``ContentSafetyFinding`` the deployer routes through
``InterceptionPipeline.intercept(context=...)``.

Two entry points:

* ``parse_detect_response(response)`` for callers who already have a
  ``DetectResponse`` from ``Rebuff.detect_injection``.
* ``RebuffAdapter`` holds the Rebuff client and exposes
  ``scan_prompt`` plus ``scan_response`` (the latter checks for
  canary-word leakage when a canary word is supplied).

Optional dependency: ``pip install vaara[rebuff]``.
"""

from __future__ import annotations

from typing import Any

from vaara.integrations._content_safety_base import (
    ContentSafetyFinding,
    FindingCategory,
    build_finding,
    mapping_for,
)


_PROVIDER = "rebuff"


def _sev_str(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.4f}"


def _layer_category(
    key: str,
    *,
    score: float,
    threshold: float,
    ran: bool,
) -> FindingCategory:
    triggered = ran and (score >= threshold > 0.0)
    return FindingCategory(
        provider_category=key,
        severity_label=("BLOCKED" if triggered else ("PASS" if ran else "SKIPPED")),
        normalized_severity=_sev_str(score if ran else 0.0),
        action=("BLOCKED" if triggered else "NONE"),
        mapping=mapping_for(_PROVIDER, key),
        evidence={
            "score": f"{score:.4f}",
            "threshold": f"{threshold:.4f}",
            "ran": bool(ran),
        },
    )


def parse_detect_response(
    response: Any,
    *,
    scanned_role: str = "prompt",
) -> ContentSafetyFinding:
    """Parse a Rebuff ``DetectResponse`` into a finding.

    Accepts the SDK object or a dict shape. Three layers are recorded
    regardless of trigger so the audit trail reflects which checks
    ran.
    """
    get = (lambda k, d=None: response.get(k, d)) if isinstance(response, dict) else (
        lambda k, d=None: getattr(response, k, d)
    )
    cats: list[FindingCategory] = [
        _layer_category(
            "heuristic_injection",
            score=float(get("heuristicScore", 0.0) or 0.0),
            threshold=float(get("maxHeuristicScore", 0.75) or 0.75),
            ran=bool(get("runHeuristicCheck", True)),
        ),
        _layer_category(
            "model_injection",
            score=float(get("modelScore", 0.0) or 0.0),
            threshold=float(get("maxModelScore", 0.9) or 0.9),
            ran=bool(get("runLanguageModelCheck", True)),
        ),
        _layer_category(
            "vector_injection",
            score=float(get("vectorScore", {}).get("topScore", 0.0)
                        if isinstance(get("vectorScore"), dict)
                        else (get("vectorScore", 0.0) or 0.0)),
            threshold=float(get("maxVectorScore", 0.9) or 0.9),
            ran=bool(get("runVectorCheck", True)),
        ),
    ]

    raw = response if isinstance(response, dict) else {
        "injectionDetected": bool(get("injectionDetected", False)),
    }
    return build_finding(
        provider=_PROVIDER,
        categories=cats,
        raw=raw,
        scanned_role=scanned_role,
    )


def parse_canary_leak(
    leaked: bool,
    *,
    canary_word: str,
    scanned_role: str = "response",
) -> ContentSafetyFinding:
    cats = [FindingCategory(
        provider_category="canary_leak",
        severity_label=("BLOCKED" if leaked else "PASS"),
        normalized_severity=_sev_str(0.95 if leaked else 0.0),
        action=("BLOCKED" if leaked else "NONE"),
        mapping=mapping_for(_PROVIDER, "canary_leak"),
        evidence={"canary_word": canary_word, "leaked": bool(leaked)},
    )]
    return build_finding(
        provider=_PROVIDER,
        categories=cats,
        raw={"canary_word": canary_word, "leaked": bool(leaked)},
        scanned_role=scanned_role,
    )


class RebuffAdapter:
    """Wraps a configured ``Rebuff`` or ``RebuffSdk`` instance."""

    provider = _PROVIDER

    def __init__(self, client: Any) -> None:
        if not hasattr(client, "detect_injection"):
            raise TypeError(
                "RebuffAdapter: client must expose detect_injection "
                "(pass a Rebuff or RebuffSdk instance)."
            )
        self._client = client

    def scan_prompt(self, text: str, **kwargs: Any) -> ContentSafetyFinding:
        result = self._client.detect_injection(text, **kwargs)
        # SDK returns either DetectResponse alone or (metrics, is_injection).
        if isinstance(result, tuple) and len(result) == 2:
            result = result[0]
        return parse_detect_response(result, scanned_role="prompt")

    def scan_response(
        self,
        text: str,
        *,
        prompt: str = "",
        canary_word: str = "",
        **_: Any,
    ) -> ContentSafetyFinding:
        if not canary_word:
            raise ValueError(
                "RebuffAdapter.scan_response: canary_word is required for "
                "Rebuff response-side checks. Use Rebuff.add_canary_word "
                "before generation and pass the canary back in."
            )
        leaked = bool(self._client.is_canary_word_leaked(prompt, text, canary_word))
        return parse_canary_leak(
            leaked, canary_word=canary_word, scanned_role="response",
        )


__all__ = [
    "RebuffAdapter",
    "parse_canary_leak",
    "parse_detect_response",
]
