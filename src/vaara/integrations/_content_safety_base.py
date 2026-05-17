"""Shared shape for cloud-guardrail content-safety adapters.

Three adapters live alongside this module — Bedrock, Azure, GCP — and
each scans prompts and responses through its respective cloud filter.
The cloud filter is an upstream signal, not Vaara's replacement: every
adapter returns a ``ContentSafetyFinding`` that the deployer routes
into ``InterceptionPipeline.intercept(context=...)`` so the finding
lands in Vaara's hash-chained audit trail and, when the deployer is
emitting OVERT envelopes, in ``non_content_metadata``.

Design choices (per the v0.19.0 plan):

* Adapters accept a pre-configured cloud client. They do not handle
  auth, region selection, retries, or pagination.
* Cloud SDKs are lazy-imported at adapter construction. Installing
  Vaara does not pull boto3, azure-ai-contentsafety, or
  google-cloud-modelarmor.
* The Finding shape is the contract. SDK glue may shift; the Finding
  fields, ``vaara_category`` vocabulary, and the published article
  mapping in ``_content_safety_articles`` are stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol

from vaara.integrations._content_safety_articles import (
    CategoryMapping,
    lookup as _lookup_mapping,
)


# OVERT Protocol Profile 1.0 prohibits IEEE-754 floats in
# non_content_metadata; rates and probabilities are decimal strings.
# Adapters compute float severity internally and stringify here.
_SEVERITY_DECIMALS = 4


def _decimal_str(value: float) -> str:
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    return f"{value:.{_SEVERITY_DECIMALS}f}"


@dataclass(frozen=True)
class FindingCategory:
    """One category triggered inside a single provider response."""

    provider_category: str
    severity_label: str
    normalized_severity: str
    action: str
    mapping: Optional[CategoryMapping]
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def vaara_category(self) -> str:
        return self.mapping.vaara_category if self.mapping else "unmapped"

    @property
    def ai_act_articles(self) -> tuple[str, ...]:
        return self.mapping.ai_act_articles if self.mapping else ()


@dataclass(frozen=True)
class ContentSafetyFinding:
    """Adapter output. Shared shape across Bedrock, Azure, GCP."""

    provider: str
    verdict: str
    severity: str
    categories: tuple[FindingCategory, ...]
    raw: dict[str, Any] = field(default_factory=dict)
    scanned_role: str = ""

    def triggered_categories(self) -> tuple[FindingCategory, ...]:
        return tuple(c for c in self.categories if c.action != "NONE")

    def ai_act_articles(self) -> tuple[str, ...]:
        seen: list[str] = []
        for cat in self.categories:
            for art in cat.ai_act_articles:
                if art not in seen:
                    seen.append(art)
        return tuple(seen)

    def to_audit_context(self) -> dict[str, Any]:
        """Lands in the audit trail's per-action context.

        Pipeline ingress sanitises and size-caps, so this adapter does
        not need to truncate spans or normalise types.
        """
        return {
            "upstream_guardrail": {
                "provider": self.provider,
                "verdict": self.verdict,
                "severity": self.severity,
                "scanned_role": self.scanned_role,
                "ai_act_articles": list(self.ai_act_articles()),
                "categories": [
                    {
                        "provider_category": c.provider_category,
                        "vaara_category": c.vaara_category,
                        "severity_label": c.severity_label,
                        "normalized_severity": c.normalized_severity,
                        "action": c.action,
                        "ai_act_articles": list(c.ai_act_articles),
                        "evidence": c.evidence,
                    }
                    for c in self.categories
                ],
            }
        }

    def to_overt_metadata(self) -> dict[str, Any]:
        """Shape for OVERT envelope ``non_content_metadata``.

        Decimal-string severities only. Raw provider responses excluded —
        OVERT envelopes carry only structural metadata.
        """
        return {
            "upstream_guardrail_provider": self.provider,
            "upstream_guardrail_verdict": self.verdict,
            "upstream_guardrail_severity": self.severity,
            "upstream_guardrail_ai_act_articles": list(self.ai_act_articles()),
            "upstream_guardrail_vaara_categories": [
                c.vaara_category for c in self.triggered_categories()
            ],
        }


class ContentSafetyScorer(Protocol):
    """Protocol every cloud-guardrail adapter satisfies."""

    provider: str

    def scan_prompt(self, text: str, **kwargs: Any) -> ContentSafetyFinding: ...

    def scan_response(self, text: str, **kwargs: Any) -> ContentSafetyFinding: ...


def aggregate_verdict(categories: Iterable[FindingCategory]) -> str:
    """Roll triggered actions into a single verdict.

    Any BLOCKED -> ``"block"``. Else any FLAGGED or REDACTED -> ``"flag"``.
    Else ``"allow"``.
    """
    actions = {c.action for c in categories}
    if "BLOCKED" in actions:
        return "block"
    if actions & {"FLAGGED", "REDACTED"}:
        return "flag"
    return "allow"


def aggregate_severity(categories: Iterable[FindingCategory]) -> str:
    severities = [c.normalized_severity for c in categories]
    if not severities:
        return _decimal_str(0.0)
    return max(severities)


def build_finding(
    *,
    provider: str,
    categories: Iterable[FindingCategory],
    raw: dict[str, Any],
    scanned_role: str,
) -> ContentSafetyFinding:
    cats = tuple(categories)
    return ContentSafetyFinding(
        provider=provider,
        verdict=aggregate_verdict(cats),
        severity=aggregate_severity(cats),
        categories=cats,
        raw=raw,
        scanned_role=scanned_role,
    )


def mapping_for(provider: str, provider_category: str) -> Optional[CategoryMapping]:
    return _lookup_mapping(provider, provider_category)


__all__ = [
    "ContentSafetyFinding",
    "ContentSafetyScorer",
    "FindingCategory",
    "aggregate_severity",
    "aggregate_verdict",
    "build_finding",
    "mapping_for",
]
