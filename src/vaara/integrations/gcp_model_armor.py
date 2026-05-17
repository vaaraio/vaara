"""GCP Model Armor adapter — upstream content-safety signal.

Wraps google-cloud-modelarmor's ``sanitize_user_prompt`` /
``sanitize_model_response``. Caller supplies a pre-configured client
and a template path; the adapter returns a ``ContentSafetyFinding``.

Confidence levels (LOW / MEDIUM_AND_ABOVE / HIGH) project onto
[0.0, 1.0]. Responsible-AI block threshold defaults to
``MEDIUM_AND_ABOVE``. CSAM, malicious URIs, prompt injection, and
SDP findings always block regardless of confidence.

Optional dep: ``pip install vaara[gcp-model-armor]``.
"""

from __future__ import annotations

from typing import Any, Optional

from vaara.integrations._content_safety_base import (
    ContentSafetyFinding, FindingCategory, build_finding, mapping_for,
)

_PROVIDER = "gcp-model-armor"
_CONFIDENCE_ORDER = {"LOW": 0, "MEDIUM_AND_ABOVE": 1, "HIGH": 2}
_CONFIDENCE_SEVERITY = {"LOW": 0.25, "MEDIUM_AND_ABOVE": 0.6, "HIGH": 0.9}


def _sev_str(v: float) -> str:
    return f"{max(0.0, min(1.0, v)):.4f}"


def _matched(node: Optional[dict[str, Any]]) -> bool:
    return bool(node) and node.get("matchState") == "MATCH_FOUND"


def _rai_action(confidence: str, block_threshold: str) -> str:
    cur = _CONFIDENCE_ORDER.get(confidence, -1)
    thr = _CONFIDENCE_ORDER.get(block_threshold, 1)
    if cur < 0:
        return "FLAGGED"
    return "BLOCKED" if cur >= thr else "FLAGGED"


def _rai_cats(rai: Optional[dict[str, Any]], block_threshold: str) -> list[FindingCategory]:
    if not rai:
        return []
    inner = rai.get("raiFilterResult") or rai
    if not _matched(inner):
        return []
    out: list[FindingCategory] = []
    for filter_type, result in (inner.get("raiFilterTypeResults") or {}).items():
        if not _matched(result):
            continue
        conf = result.get("confidenceLevel") or "MEDIUM_AND_ABOVE"
        key = f"responsible_ai.{filter_type}"
        out.append(FindingCategory(
            provider_category=key, severity_label=conf,
            normalized_severity=_sev_str(_CONFIDENCE_SEVERITY.get(conf, 0.6)),
            action=_rai_action(conf, block_threshold),
            mapping=mapping_for(_PROVIDER, key),
            evidence={"filter_type": filter_type},
        ))
    return out


def _pi_cats(pi: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not pi:
        return []
    inner = pi.get("piAndJailbreakFilterResult") or pi
    if not _matched(inner):
        return []
    conf = inner.get("confidenceLevel") or "MEDIUM_AND_ABOVE"
    return [FindingCategory(
        provider_category="pi_and_jailbreak", severity_label=conf,
        normalized_severity=_sev_str(_CONFIDENCE_SEVERITY.get(conf, 0.7)),
        action="BLOCKED",
        mapping=mapping_for(_PROVIDER, "pi_and_jailbreak"), evidence={},
    )]


def _malicious_cats(mu: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not mu:
        return []
    inner = mu.get("maliciousUriFilterResult") or mu
    if not _matched(inner):
        return []
    items = inner.get("maliciousUriMatchedItems") or []
    return [FindingCategory(
        provider_category="malicious_uris", severity_label="MATCH_FOUND",
        normalized_severity=_sev_str(0.9), action="BLOCKED",
        mapping=mapping_for(_PROVIDER, "malicious_uris"),
        evidence={"matched_count": len(items)},
    )]


def _sdp_cats(sdp: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not sdp:
        return []
    inner = sdp.get("sdpFilterResult") or sdp
    inspect = inner.get("inspectResult") or inner.get("deidentifyResult") or {}
    if not _matched(inspect):
        return []
    info_types = sorted({
        f.get("infoType") for f in (inspect.get("findings") or []) if f.get("infoType")
    })
    return [FindingCategory(
        provider_category="sdp", severity_label="MATCH_FOUND",
        normalized_severity=_sev_str(0.7), action="BLOCKED",
        mapping=mapping_for(_PROVIDER, "sdp"),
        evidence={"info_types": info_types},
    )]


def _csam_cats(csam: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not csam:
        return []
    inner = csam.get("csamFilterFilterResult") or csam.get("csamFilterResult") or csam
    if not _matched(inner):
        return []
    return [FindingCategory(
        provider_category="csam", severity_label="MATCH_FOUND",
        normalized_severity=_sev_str(1.0), action="BLOCKED",
        mapping=mapping_for(_PROVIDER, "csam"), evidence={},
    )]


def parse_sanitize_response(
    response: dict[str, Any], *, scanned_role: str = "",
    block_threshold: str = "MEDIUM_AND_ABOVE",
) -> ContentSafetyFinding:
    """Parse a sanitize_* response into a Finding."""
    sanit = response.get("sanitizationResult") or response
    fr = sanit.get("filterResults") or {}
    cats: list[FindingCategory] = []
    cats.extend(_rai_cats(fr.get("rai"), block_threshold))
    cats.extend(_pi_cats(fr.get("pi_and_jailbreak") or fr.get("piAndJailbreak")))
    cats.extend(_malicious_cats(fr.get("malicious_uris") or fr.get("maliciousUris")))
    cats.extend(_sdp_cats(fr.get("sdp")))
    cats.extend(_csam_cats(fr.get("csam")))
    return build_finding(provider=_PROVIDER, categories=cats, raw=response, scanned_role=scanned_role)


class GcpModelArmorAdapter:
    """Wraps a google-cloud-modelarmor client."""

    provider = _PROVIDER

    def __init__(self, client: Any, template: str, *,
                 block_threshold: str = "MEDIUM_AND_ABOVE") -> None:
        if not (hasattr(client, "sanitize_user_prompt") and hasattr(client, "sanitize_model_response")):
            raise TypeError(
                "GcpModelArmorAdapter: client must expose sanitize_user_prompt and "
                "sanitize_model_response (pass a google-cloud-modelarmor client).")
        if not template:
            raise ValueError("GcpModelArmorAdapter: template path is required.")
        self._client = client
        self._template = template
        self._block_threshold = block_threshold

    @staticmethod
    def _to_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, "to_dict"):
            return response.to_dict()
        return response if isinstance(response, dict) else {}

    def scan_prompt(self, text: str, **_: Any) -> ContentSafetyFinding:
        response = self._client.sanitize_user_prompt(request={
            "name": self._template, "user_prompt_data": {"text": text},
        })
        return parse_sanitize_response(
            self._to_dict(response), scanned_role="prompt",
            block_threshold=self._block_threshold,
        )

    def scan_response(self, text: str, **_: Any) -> ContentSafetyFinding:
        response = self._client.sanitize_model_response(request={
            "name": self._template, "model_response_data": {"text": text},
        })
        return parse_sanitize_response(
            self._to_dict(response), scanned_role="response",
            block_threshold=self._block_threshold,
        )


__all__ = ["GcpModelArmorAdapter", "parse_sanitize_response"]
