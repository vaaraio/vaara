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


_MISSING = object()


def _field(response: Any, *names: str, default: Any = None) -> Any:
    """First present attribute or key among ``names``.

    Rebuff ships TWO response shapes with different spellings, and they
    disagree on more than case:

    * ``rebuff.rebuff.DetectApiSuccessResponse`` (hosted ``Rebuff``) —
      camelCase, and ``vectorScore`` is a ``Dict[str, float]``.
    * ``rebuff.sdk.RebuffDetectionResponse`` (self-hosted ``RebuffSdk``)
      — snake_case, ``vector_score`` is a plain float, and the model
      layer is called ``openai_score``, not ``model_score``.

    Reading only the camelCase names made every self-hosted field fall
    back to its default, so all three layers scored 0.0 and a detected
    injection came back as verdict "allow".
    """
    for name in names:
        if isinstance(response, dict):
            if name in response:
                return response[name]
            continue
        value = getattr(response, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _vector_score(response: Any) -> float:
    raw = _field(response, "vectorScore", "vector_score", default=0.0)
    if isinstance(raw, dict):
        # Hosted API returns per-match scores; topScore is the max.
        raw = raw.get("topScore", 0.0)
    return float(raw or 0.0)


def parse_detect_response(
    response: Any,
    *,
    scanned_role: str = "prompt",
) -> ContentSafetyFinding:
    """Parse a Rebuff detection response into a finding.

    Accepts either SDK response object or a dict shape, in either
    naming convention. Three layers are recorded regardless of trigger
    so the audit trail reflects which checks ran.
    """
    cats: list[FindingCategory] = [
        _layer_category(
            "heuristic_injection",
            score=float(_field(response, "heuristicScore", "heuristic_score",
                               default=0.0) or 0.0),
            threshold=float(_field(response, "maxHeuristicScore", "max_heuristic_score",
                                   default=0.75) or 0.75),
            ran=bool(_field(response, "runHeuristicCheck", "run_heuristic_check",
                            default=True)),
        ),
        _layer_category(
            "model_injection",
            score=float(_field(response, "modelScore", "openai_score", "model_score",
                               default=0.0) or 0.0),
            threshold=float(_field(response, "maxModelScore", "max_model_score",
                                   default=0.9) or 0.9),
            ran=bool(_field(response, "runLanguageModelCheck", "run_language_model_check",
                            default=True)),
        ),
        _layer_category(
            "vector_injection",
            score=_vector_score(response),
            threshold=float(_field(response, "maxVectorScore", "max_vector_score",
                                   default=0.9) or 0.9),
            ran=bool(_field(response, "runVectorCheck", "run_vector_check",
                            default=True)),
        ),
    ]

    # Rebuff's own aggregate verdict. If it says injection and no layer
    # threshold tripped, trust the provider rather than reporting clean:
    # that disagreement is exactly what field drift looks like.
    detected = bool(_field(response, "injectionDetected", "injection_detected",
                           default=False))
    if detected and all(c.action == "NONE" for c in cats):
        cats.append(FindingCategory(
            provider_category="injection_detected",
            severity_label="BLOCKED",
            normalized_severity=_sev_str(0.9),
            action="BLOCKED",
            mapping=mapping_for(_PROVIDER, "heuristic_injection"),
            evidence={"reason": "provider reported injectionDetected "
                                "with no layer above threshold"},
        ))

    raw = response if isinstance(response, dict) else {"injectionDetected": detected}
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
