# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AWS Bedrock Guardrails adapter — upstream content-safety signal.

Wraps the Bedrock Runtime ``ApplyGuardrail`` API. Caller supplies a
pre-configured boto3 client and a guardrail identifier; the adapter
returns a ``ContentSafetyFinding`` for the deployer to route into
``InterceptionPipeline.intercept(context=...)``.

Optional dependency: ``pip install vaara[bedrock]``.
"""

from __future__ import annotations

from typing import Any

from vaara.integrations._content_safety_base import (
    ContentSafetyFinding,
    FindingCategory,
    build_finding,
    mapping_for,
)


_PROVIDER = "aws-bedrock-guardrails"

# Bedrock content-filter confidence -> Vaara severity in [0.0, 1.0].
_CONFIDENCE_SEVERITY = {"NONE": 0.0, "LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.9}


def _sev_str(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.4f}"


def _topic_cats(assessment: dict[str, Any]) -> list[FindingCategory]:
    out: list[FindingCategory] = []
    for t in (assessment.get("topicPolicy") or {}).get("topics", []) or []:
        if not t.get("detected"):
            continue
        out.append(FindingCategory(
            provider_category="topicPolicy",
            severity_label=t.get("type") or "DENY",
            normalized_severity=_sev_str(0.9),
            action=t.get("action") or "NONE",
            mapping=mapping_for(_PROVIDER, "topicPolicy"),
            evidence={"topic_name": t.get("name"), "type": t.get("type")},
        ))
    return out


def _content_cats(assessment: dict[str, Any]) -> list[FindingCategory]:
    out: list[FindingCategory] = []
    for f in (assessment.get("contentPolicy") or {}).get("filters", []) or []:
        if not f.get("detected"):
            continue
        ftype = (f.get("type") or "").upper()
        confidence = (f.get("confidence") or "NONE").upper()
        key = f"contentPolicy.{ftype}"
        out.append(FindingCategory(
            provider_category=key,
            severity_label=confidence,
            normalized_severity=_sev_str(_CONFIDENCE_SEVERITY.get(confidence, 0.5)),
            action=f.get("action") or "NONE",
            mapping=mapping_for(_PROVIDER, key),
            evidence={"filter_strength": f.get("filterStrength")},
        ))
    return out


def _word_cats(assessment: dict[str, Any]) -> list[FindingCategory]:
    out: list[FindingCategory] = []
    wp = assessment.get("wordPolicy") or {}
    for w in (wp.get("customWords") or []) + (wp.get("managedWordLists") or []):
        if not w.get("detected"):
            continue
        out.append(FindingCategory(
            provider_category="wordPolicy",
            severity_label=w.get("type") or "CUSTOM",
            normalized_severity=_sev_str(0.7),
            action=w.get("action") or "NONE",
            mapping=mapping_for(_PROVIDER, "wordPolicy"),
            evidence={"match": w.get("match")},
        ))
    return out


def _pii_cats(assessment: dict[str, Any]) -> list[FindingCategory]:
    out: list[FindingCategory] = []
    sip = assessment.get("sensitiveInformationPolicy") or {}
    for p in (sip.get("piiEntities") or []) + (sip.get("regexes") or []):
        if not p.get("detected"):
            continue
        # Bedrock returns ANONYMIZED for soft-redact; normalise to the
        # common REDACTED action so the verdict aggregator catches it.
        raw_action = p.get("action") or "NONE"
        action = "REDACTED" if raw_action == "ANONYMIZED" else raw_action
        out.append(FindingCategory(
            provider_category="sensitiveInformationPolicy",
            severity_label=p.get("type") or p.get("name") or "PII",
            normalized_severity=_sev_str(0.7),
            action=action,
            mapping=mapping_for(_PROVIDER, "sensitiveInformationPolicy"),
            evidence={"match": p.get("match"), "type": p.get("type"), "name": p.get("name"),
                      "raw_action": raw_action},
        ))
    return out


def _grounding_cats(assessment: dict[str, Any]) -> list[FindingCategory]:
    out: list[FindingCategory] = []
    for g in (assessment.get("contextualGroundingPolicy") or {}).get("filters", []) or []:
        if not g.get("detected"):
            continue
        score = float(g.get("score") or 0.0)
        threshold = float(g.get("threshold") or 0.0)
        # Severity is the gap below threshold, anchored at 0.5 when at threshold.
        severity = threshold - score + 0.5
        out.append(FindingCategory(
            provider_category="contextualGroundingPolicy",
            severity_label=g.get("type") or "GROUNDING",
            normalized_severity=_sev_str(severity),
            action=g.get("action") or "NONE",
            mapping=mapping_for(_PROVIDER, "contextualGroundingPolicy"),
            evidence={"score": str(score), "threshold": str(threshold)},
        ))
    return out


def parse_apply_guardrail_response(
    response: dict[str, Any],
    *,
    scanned_role: str = "",
) -> ContentSafetyFinding:
    """Parse an already-issued ``ApplyGuardrail`` response.

    For callers running Bedrock Guardrails inline (InvokeModel with
    ``GuardrailIdentifier`` set) — feed the response in without a
    second API call.
    """
    cats: list[FindingCategory] = []
    for assessment in response.get("assessments", []) or []:
        cats.extend(_topic_cats(assessment))
        cats.extend(_content_cats(assessment))
        cats.extend(_word_cats(assessment))
        cats.extend(_pii_cats(assessment))
        cats.extend(_grounding_cats(assessment))
    return build_finding(
        provider=_PROVIDER,
        categories=cats,
        raw=response,
        scanned_role=scanned_role,
    )


class BedrockGuardrailsAdapter:
    """Wraps a boto3 ``bedrock-runtime`` client."""

    provider = _PROVIDER

    def __init__(
        self,
        client: Any,
        guardrail_id: str,
        guardrail_version: str = "DRAFT",
    ) -> None:
        if not hasattr(client, "apply_guardrail"):
            raise TypeError(
                "BedrockGuardrailsAdapter: client must expose apply_guardrail "
                "(pass a boto3 bedrock-runtime client)."
            )
        if not guardrail_id:
            raise ValueError("BedrockGuardrailsAdapter: guardrail_id is required.")
        self._client = client
        self._guardrail_id = guardrail_id
        self._guardrail_version = guardrail_version

    def _apply(self, text: str, source: str) -> dict[str, Any]:
        return self._client.apply_guardrail(
            guardrailIdentifier=self._guardrail_id,
            guardrailVersion=self._guardrail_version,
            source=source,
            content=[{"text": {"text": text}}],
        )

    def scan_prompt(self, text: str, **_: Any) -> ContentSafetyFinding:
        return parse_apply_guardrail_response(
            self._apply(text, source="INPUT"), scanned_role="prompt",
        )

    def scan_response(self, text: str, **_: Any) -> ContentSafetyFinding:
        return parse_apply_guardrail_response(
            self._apply(text, source="OUTPUT"), scanned_role="response",
        )


__all__ = ["BedrockGuardrailsAdapter", "parse_apply_guardrail_response"]
