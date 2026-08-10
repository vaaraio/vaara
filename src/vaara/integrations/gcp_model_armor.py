# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GCP Model Armor adapter — upstream content-safety signal.

Wraps google-cloud-modelarmor's ``sanitize_user_prompt`` /
``sanitize_model_response``. Caller supplies a pre-configured client
and a template path; the adapter returns a ``ContentSafetyFinding``.

Confidence levels (LOW_AND_ABOVE / MEDIUM_AND_ABOVE / HIGH) project
onto [0.0, 1.0]. Responsible-AI block threshold defaults to
``MEDIUM_AND_ABOVE``. CSAM, malicious URIs, prompt injection, virus
scan, and SDP findings always block regardless of confidence.

Two response encodings are accepted, because the same API is reachable
two ways and they do not agree on spelling:

* The Python SDK returns proto-plus messages. ``to_dict`` lives on the
  message *metaclass*, not the instance, and its defaults emit
  snake_case keys with integer enums.
* The REST API returns JSON with camelCase keys and string enums.

Every field read here tolerates both spellings and both enum
encodings. Getting this wrong fails *open* — an unparsed response
yields zero categories and a verdict of "allow" — so the parser is
deliberately permissive about shape and strict about matching.

Optional dep: ``pip install vaara[gcp-model-armor]``.
"""

from __future__ import annotations

from typing import Any, Optional

from vaara.integrations._content_safety_base import (
    ContentSafetyFinding, FindingCategory, build_finding, mapping_for,
)

_PROVIDER = "gcp-model-armor"

# DetectionConfidenceLevel as the API actually spells it. "LOW" is kept
# as an alias because earlier Vaara releases used that name.
_CONFIDENCE_ORDER = {"LOW_AND_ABOVE": 0, "LOW": 0, "MEDIUM_AND_ABOVE": 1, "HIGH": 2}
_CONFIDENCE_SEVERITY = {
    "LOW_AND_ABOVE": 0.25, "LOW": 0.25, "MEDIUM_AND_ABOVE": 0.6, "HIGH": 0.9,
}

# FilterMatchState.MATCH_FOUND is 2. Integer enums arrive whenever a
# caller converted the proto with library defaults.
_MATCH_FOUND = ("MATCH_FOUND", 2)

# DetectionConfidenceLevel by ordinal, for the same integer-enum case.
_CONFIDENCE_BY_VALUE = {1: "LOW_AND_ABOVE", 2: "MEDIUM_AND_ABOVE", 3: "HIGH"}


def _sev_str(v: float) -> str:
    return f"{max(0.0, min(1.0, v)):.4f}"


def _get(node: Optional[dict[str, Any]], *names: str) -> Any:
    """First present key among ``names``. Accepts snake_case or camelCase."""
    if not isinstance(node, dict):
        return None
    for name in names:
        if name in node:
            return node[name]
    return None


def _matched(node: Optional[dict[str, Any]]) -> bool:
    if not isinstance(node, dict):
        return False
    return _get(node, "match_state", "matchState") in _MATCH_FOUND


def _confidence(node: Optional[dict[str, Any]], default: str = "MEDIUM_AND_ABOVE") -> str:
    raw = _get(node or {}, "confidence_level", "confidenceLevel")
    if raw in (None, "", 0, "DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED"):
        return default
    if isinstance(raw, int):
        return _CONFIDENCE_BY_VALUE.get(raw, default)
    return str(raw)


def _rai_action(confidence: str, block_threshold: str) -> str:
    cur = _CONFIDENCE_ORDER.get(confidence, -1)
    thr = _CONFIDENCE_ORDER.get(block_threshold, 1)
    if cur < 0:
        return "FLAGGED"
    return "BLOCKED" if cur >= thr else "FLAGGED"


def _rai_cats(rai: Optional[dict[str, Any]], block_threshold: str) -> list[FindingCategory]:
    if not rai:
        return []
    inner = _get(rai, "rai_filter_result", "raiFilterResult") or rai
    if not _matched(inner):
        return []
    out: list[FindingCategory] = []
    results = _get(inner, "rai_filter_type_results", "raiFilterTypeResults") or {}
    for filter_type, result in results.items():
        if not _matched(result):
            continue
        conf = _confidence(result)
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
    inner = _get(pi, "pi_and_jailbreak_filter_result", "piAndJailbreakFilterResult") or pi
    if not _matched(inner):
        return []
    conf = _confidence(inner)
    return [FindingCategory(
        provider_category="pi_and_jailbreak", severity_label=conf,
        normalized_severity=_sev_str(_CONFIDENCE_SEVERITY.get(conf, 0.7)),
        action="BLOCKED",
        mapping=mapping_for(_PROVIDER, "pi_and_jailbreak"), evidence={},
    )]


def _malicious_cats(mu: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not mu:
        return []
    inner = _get(mu, "malicious_uri_filter_result", "maliciousUriFilterResult") or mu
    if not _matched(inner):
        return []
    items = _get(inner, "malicious_uri_matched_items", "maliciousUriMatchedItems") or []
    return [FindingCategory(
        provider_category="malicious_uris", severity_label="MATCH_FOUND",
        normalized_severity=_sev_str(0.9), action="BLOCKED",
        mapping=mapping_for(_PROVIDER, "malicious_uris"),
        evidence={"matched_count": len(items)},
    )]


def _sdp_cats(sdp: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not sdp:
        return []
    inner = _get(sdp, "sdp_filter_result", "sdpFilterResult") or sdp
    inspect = (
        _get(inner, "inspect_result", "inspectResult")
        or _get(inner, "deidentify_result", "deidentifyResult")
        or {}
    )
    if not _matched(inspect):
        return []
    info_types = sorted({
        it for it in (
            _get(f, "info_type", "infoType")
            for f in (_get(inspect, "findings") or [])
        ) if it
    })
    # deidentify_result reports its classes in a flat info_types list
    # rather than per-finding.
    if not info_types:
        info_types = sorted({
            it for it in (_get(inspect, "info_types", "infoTypes") or []) if it
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
    inner = (
        _get(csam, "csam_filter_filter_result", "csamFilterFilterResult")
        or _get(csam, "csam_filter_result", "csamFilterResult")
        or csam
    )
    if not _matched(inner):
        return []
    return [FindingCategory(
        provider_category="csam", severity_label="MATCH_FOUND",
        normalized_severity=_sev_str(1.0), action="BLOCKED",
        mapping=mapping_for(_PROVIDER, "csam"), evidence={},
    )]


def _virus_cats(virus: Optional[dict[str, Any]]) -> list[FindingCategory]:
    if not virus:
        return []
    inner = _get(virus, "virus_scan_filter_result", "virusScanFilterResult") or virus
    if not _matched(inner):
        return []
    details = _get(inner, "virus_details", "virusDetails") or []
    return [FindingCategory(
        provider_category="virus_scan", severity_label="MATCH_FOUND",
        normalized_severity=_sev_str(1.0), action="BLOCKED",
        mapping=mapping_for(_PROVIDER, "virus_scan"),
        evidence={
            "virus_count": len(details),
            "scanned_content_type": _get(inner, "scanned_content_type", "scannedContentType"),
        },
    )]


def parse_sanitize_response(
    response: dict[str, Any], *, scanned_role: str = "",
    block_threshold: str = "MEDIUM_AND_ABOVE",
) -> ContentSafetyFinding:
    """Parse a sanitize_* response into a Finding.

    Accepts SDK (snake_case) and REST (camelCase) encodings alike.
    """
    sanit = _get(response, "sanitization_result", "sanitizationResult") or response
    fr = _get(sanit, "filter_results", "filterResults") or {}
    cats: list[FindingCategory] = []
    cats.extend(_rai_cats(_get(fr, "rai"), block_threshold))
    cats.extend(_pi_cats(_get(fr, "pi_and_jailbreak", "piAndJailbreak")))
    cats.extend(_malicious_cats(_get(fr, "malicious_uris", "maliciousUris")))
    cats.extend(_sdp_cats(_get(fr, "sdp")))
    cats.extend(_csam_cats(_get(fr, "csam")))
    cats.extend(_virus_cats(_get(fr, "virus_scan", "virusScan")))

    # Model Armor's own aggregate verdict. Same shape as the rebuff adapter's
    # injectionDetected cross-check. The six parsers above cover the filter
    # types that existed when they were written; filterMatchState is what
    # Google sets whenever any filter matched, including one this module does
    # not model. Reading only the known keys meant a new or renamed filter
    # produced no category, so a blocked call came back verdict "allow" and
    # was written to the trail as an upstream pass.
    top_state = _get(sanit, "filter_match_state", "filterMatchState")
    if top_state in _MATCH_FOUND and all(c.action == "NONE" for c in cats):
        cats.append(FindingCategory(
            provider_category="filter_match_state",
            severity_label="MATCH_FOUND",
            normalized_severity=_sev_str(0.9),
            action="BLOCKED",
            mapping=None,
            evidence={
                "reason": "provider reported filterMatchState=MATCH_FOUND with "
                          "no recognised filter above threshold",
                "parsed_filters": ",".join(sorted(fr.keys())) if isinstance(fr, dict) else "",
            },
        ))

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
        """Normalise an SDK response to a plain dict.

        proto-plus exposes ``to_dict`` on the metaclass, so
        ``hasattr(instance, "to_dict")`` is False and an instance-first
        check silently drops the entire response. Ask the type first,
        and pin the conversion flags: library defaults emit integer
        enums, which no downstream comparison here expects to be the
        only encoding.
        """
        if isinstance(response, dict):
            return response
        message_type = type(response)
        type_to_dict = getattr(message_type, "to_dict", None)
        if callable(type_to_dict):
            try:
                return type_to_dict(
                    response,
                    preserving_proto_field_name=True,
                    use_integers_for_enums=False,
                )
            except TypeError:
                # Not proto-plus, or a signature that predates the flags.
                try:
                    return type_to_dict(response)
                except TypeError:
                    pass
        instance_to_dict = getattr(response, "to_dict", None)
        if callable(instance_to_dict):
            result = instance_to_dict()
            if isinstance(result, dict):
                return result
        raise TypeError(
            "GcpModelArmorAdapter: cannot convert response of type "
            f"{message_type.__name__!r} to a dict. Pass a "
            "google-cloud-modelarmor response or a plain dict."
        )

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
