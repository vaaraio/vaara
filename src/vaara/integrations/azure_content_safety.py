"""Azure AI Content Safety adapter — upstream content-safety signal.

Wraps Azure's ContentSafetyClient. The Azure surface is broader than
Bedrock or GCP: analyze_text, Prompt Shields, Protected Material, and
Groundedness Detection are separate endpoints. The adapter exposes a
single ``scan_prompt`` / ``scan_response`` pair and routes internally
based on which endpoints the caller asks for via ``include``.

Azure severity ladder 0/2/4/6 projects onto [0.0, 1.0]. Block
threshold defaults to 4. Optional dep: ``pip install vaara[azure-content-safety]``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from vaara.integrations._content_safety_base import (
    ContentSafetyFinding, FindingCategory, build_finding, mapping_for,
)

_PROVIDER = "azure-content-safety"
_SEVERITY_BY_LABEL = {0: 0.0, 2: 0.25, 4: 0.5, 6: 1.0}


def _sev_str(v: float) -> str:
    return f"{max(0.0, min(1.0, v)):.4f}"


def _harm_action(severity: int, block_threshold: int) -> str:
    if severity >= block_threshold:
        return "BLOCKED"
    return "FLAGGED" if severity > 0 else "NONE"


def _harm_cats(analysis: Iterable[dict[str, Any]], block_threshold: int) -> list[FindingCategory]:
    out: list[FindingCategory] = []
    for entry in analysis or []:
        category = entry.get("category") or ""
        if not category:
            continue
        severity = int(entry.get("severity") or 0)
        out.append(FindingCategory(
            provider_category=category,
            severity_label=str(severity),
            normalized_severity=_sev_str(_SEVERITY_BY_LABEL.get(severity, severity / 6.0)),
            action=_harm_action(severity, block_threshold),
            mapping=mapping_for(_PROVIDER, category),
            evidence={"severity": severity},
        ))
    return out


def _shield_cats(shield: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not shield:
        return []
    out: list[FindingCategory] = []
    if (shield.get("userPromptAnalysis") or {}).get("attackDetected"):
        out.append(FindingCategory(
            provider_category="PromptShield.UserPrompt",
            severity_label="ATTACK_DETECTED", normalized_severity=_sev_str(1.0),
            action="BLOCKED",
            mapping=mapping_for(_PROVIDER, "PromptShield.UserPrompt"), evidence={},
        ))
    for idx, doc in enumerate(shield.get("documentsAnalysis") or []):
        if doc.get("attackDetected"):
            out.append(FindingCategory(
                provider_category="PromptShield.Documents",
                severity_label="ATTACK_DETECTED", normalized_severity=_sev_str(1.0),
                action="BLOCKED",
                mapping=mapping_for(_PROVIDER, "PromptShield.Documents"),
                evidence={"document_index": idx},
            ))
    return out


def _protected_cats(protected: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not protected:
        return []
    out: list[FindingCategory] = []
    if (protected.get("protectedMaterialAnalysis") or {}).get("detected"):
        out.append(FindingCategory(
            provider_category="ProtectedMaterial.Text",
            severity_label="DETECTED", normalized_severity=_sev_str(0.7),
            action="FLAGGED",
            mapping=mapping_for(_PROVIDER, "ProtectedMaterial.Text"), evidence={},
        ))
    code = protected.get("protectedMaterialCodeAnalysis") or {}
    if code.get("detected"):
        out.append(FindingCategory(
            provider_category="ProtectedMaterial.Code",
            severity_label="DETECTED", normalized_severity=_sev_str(0.7),
            action="FLAGGED",
            mapping=mapping_for(_PROVIDER, "ProtectedMaterial.Code"),
            evidence={"matched_citation": code.get("citation")},
        ))
    return out


def _groundedness_cats(grounded: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not grounded or not grounded.get("ungroundedDetected"):
        return []
    pct = grounded.get("ungroundedPercentage")
    severity = float(pct) if isinstance(pct, (int, float)) else 0.7
    return [FindingCategory(
        provider_category="Groundedness",
        severity_label=str(pct) if pct is not None else "DETECTED",
        normalized_severity=_sev_str(severity), action="FLAGGED",
        mapping=mapping_for(_PROVIDER, "Groundedness"),
        evidence={"ungrounded_percentage": pct},
    )]


def parse_responses(
    *,
    analyze_text: Optional[dict[str, Any]] = None,
    shield: Optional[dict[str, Any]] = None,
    protected: Optional[dict[str, Any]] = None,
    grounded: Optional[dict[str, Any]] = None,
    scanned_role: str = "",
    block_threshold: int = 4,
) -> ContentSafetyFinding:
    """Parse one or more Azure responses into a single Finding."""
    cats: list[FindingCategory] = []
    if analyze_text:
        cats.extend(_harm_cats(analyze_text.get("categoriesAnalysis") or [], block_threshold))
    cats.extend(_shield_cats(shield))
    cats.extend(_protected_cats(protected))
    cats.extend(_groundedness_cats(grounded))
    return build_finding(
        provider=_PROVIDER, categories=cats,
        raw={"analyze_text": analyze_text, "shield": shield,
             "protected": protected, "grounded": grounded},
        scanned_role=scanned_role,
    )


class AzureContentSafetyAdapter:
    """Wraps an azure-ai-contentsafety client and optional siblings."""

    provider = _PROVIDER

    def __init__(self, client: Any, *, shield_client: Any = None,
                 protected_client: Any = None, groundedness_client: Any = None,
                 block_threshold: int = 4) -> None:
        if not hasattr(client, "analyze_text"):
            raise TypeError(
                "AzureContentSafetyAdapter: client must expose analyze_text "
                "(pass an azure-ai-contentsafety ContentSafetyClient).")
        self._client = client
        self._shield_client = shield_client or client
        self._protected_client = protected_client or client
        self._groundedness_client = groundedness_client or client
        self._block_threshold = block_threshold

    @staticmethod
    def _call(fn_name: str, client: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        fn = getattr(client, fn_name, None)
        if fn is None:
            return None
        result = fn(**kwargs)
        if hasattr(result, "as_dict"):
            return result.as_dict()
        return result if isinstance(result, dict) else None

    def scan_prompt(self, text: str, *, include: Optional[set[str]] = None) -> ContentSafetyFinding:
        include = include if include is not None else {"analyze_text", "shield"}
        analyze = self._call("analyze_text", self._client, text=text) if "analyze_text" in include else None
        shield = self._call("shield_prompt", self._shield_client, user_prompt=text, documents=None) if "shield" in include else None
        return parse_responses(analyze_text=analyze, shield=shield,
                               scanned_role="prompt", block_threshold=self._block_threshold)

    def scan_response(self, text: str, *, include: Optional[set[str]] = None) -> ContentSafetyFinding:
        include = include if include is not None else {"analyze_text", "protected"}
        analyze = self._call("analyze_text", self._client, text=text) if "analyze_text" in include else None
        protected = self._call("detect_text_protected_material", self._protected_client, text=text) if "protected" in include else None
        grounded = self._call("detect_groundedness", self._groundedness_client, text=text) if "grounded" in include else None
        return parse_responses(analyze_text=analyze, protected=protected, grounded=grounded,
                               scanned_role="response", block_threshold=self._block_threshold)


__all__ = ["AzureContentSafetyAdapter", "parse_responses"]
