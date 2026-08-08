# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Azure AI Content Safety adapter — upstream content-safety signal.

Wraps Azure's ContentSafetyClient. The Azure surface is broader than
Bedrock or GCP: analyze_text, Prompt Shields, Protected Material, and
Groundedness Detection are separate endpoints. The adapter exposes a
single ``scan_prompt`` / ``scan_response`` pair and routes internally
based on which endpoints the caller asks for via ``include``.

Calling convention: every endpoint takes ONE positional mapping whose
keys are the REST request body's keys. That is what the GA SDK does —
``ContentSafetyClient.analyze_text`` takes a required positional
``options`` argument, not ``text=`` — and injected sibling clients are
expected to match it.

Which endpoints the GA SDK actually has, verified against
azure-ai-contentsafety 1.0.0 (the only non-preview release on PyPI):

* ``analyze_text`` — present. This is the default for both scans.
* ``shield_prompt``, ``detect_text_protected_material``,
  ``detect_groundedness`` — NOT present on ``ContentSafetyClient`` at
  any released version. They exist in the REST API only. To use them,
  inject a client that exposes the method (``shield_client=``,
  ``protected_client=``, ``groundedness_client=``).

Asking for an endpoint whose client cannot serve it raises. It used to
return ``None`` silently, which produced a clean-looking finding with
no categories — a guardrail that reports "allow" because it never ran.

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
    def _call(
        fn_name: str, client: Any, options: dict[str, Any], *, include_key: str,
    ) -> dict[str, Any]:
        """Invoke one endpoint with a single positional options mapping.

        Raises when the client cannot serve the endpoint. A missing
        method used to yield ``None`` and therefore zero categories,
        which reads downstream as "nothing was wrong" rather than
        "nothing was checked".
        """
        fn = getattr(client, fn_name, None)
        if fn is None:
            raise RuntimeError(
                f"AzureContentSafetyAdapter: {include_key!r} was requested but the "
                f"supplied client has no {fn_name!r} method. The GA "
                f"azure-ai-contentsafety ContentSafetyClient only exposes "
                f"analyze_text; {fn_name} is REST-only. Pass a client that "
                f"implements it, or drop {include_key!r} from `include`."
            )
        result = fn(options)
        if hasattr(result, "as_dict"):
            return result.as_dict()
        if isinstance(result, dict):
            return result
        raise TypeError(
            f"AzureContentSafetyAdapter: {fn_name} returned "
            f"{type(result).__name__}, expected a mapping or an object with as_dict()."
        )

    def scan_prompt(
        self, text: str, *, include: Optional[set[str]] = None,
        documents: Optional[list[str]] = None,
    ) -> ContentSafetyFinding:
        # "shield" is not in the default set: the GA client cannot serve
        # it, so defaulting it on made every out-of-the-box scan raise.
        include = include if include is not None else {"analyze_text"}
        analyze = shield = None
        if "analyze_text" in include:
            analyze = self._call("analyze_text", self._client, {"text": text},
                                 include_key="analyze_text")
        if "shield" in include:
            shield = self._call(
                "shield_prompt", self._shield_client,
                {"userPrompt": text, "documents": list(documents or [])},
                include_key="shield")
        return parse_responses(analyze_text=analyze, shield=shield,
                               scanned_role="prompt", block_threshold=self._block_threshold)

    def scan_response(
        self, text: str, *, include: Optional[set[str]] = None,
        groundedness_options: Optional[dict[str, Any]] = None,
    ) -> ContentSafetyFinding:
        include = include if include is not None else {"analyze_text"}
        analyze = protected = grounded = None
        if "analyze_text" in include:
            analyze = self._call("analyze_text", self._client, {"text": text},
                                 include_key="analyze_text")
        if "protected" in include:
            protected = self._call("detect_text_protected_material",
                                   self._protected_client, {"text": text},
                                   include_key="protected")
        if "grounded" in include:
            # Groundedness needs the task, domain and grounding sources;
            # text alone is not a valid request body.
            options = dict(groundedness_options or {})
            options.setdefault("text", text)
            grounded = self._call("detect_groundedness", self._groundedness_client,
                                  options, include_key="grounded")
        return parse_responses(analyze_text=analyze, protected=protected, grounded=grounded,
                               scanned_role="response", block_threshold=self._block_threshold)


__all__ = ["AzureContentSafetyAdapter", "parse_responses"]
