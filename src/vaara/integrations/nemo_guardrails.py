# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NVIDIA NeMo Guardrails adapter. Upstream rails signal.

Wraps a pre-configured ``LLMRails`` instance. NeMo Guardrails runs
input rails, dialog rails, output rails, and retrieval rails during
generation. Each activated rail is recorded in the response log. The
adapter parses that log into a ``ContentSafetyFinding`` the deployer
routes through ``InterceptionPipeline.intercept(context=...)`` so the
finding lands in Vaara's hash-chained audit trail and, when emitting
OVERT envelopes, in ``non_content_metadata``.

Two entry points:

* ``parse_generation_response(response)`` for callers running NeMo
  inline (most common). Parse the ``GenerationResponse`` after a call
  to ``LLMRails.generate_async``.
* ``NemoGuardrailsAdapter`` holds a configured ``LLMRails`` and
  exposes ``generate`` which returns ``(response_text, finding)`` in
  one call.

Optional dependency: ``pip install vaara[nemo-guardrails]``.
"""

from __future__ import annotations

from typing import Any

from vaara.integrations._content_safety_base import (
    ContentSafetyFinding,
    FindingCategory,
    build_finding,
    mapping_for,
)


_PROVIDER = "nvidia-nemo-guardrails"


def _sev_str(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.4f}"


_RAIL_TO_KEY = {
    "input": {
        "self check input": "input_rails.self_check",
        "jailbreak detection": "input_rails.jailbreak",
        "jailbreak detection heuristics": "input_rails.jailbreak",
    },
    "dialog": {
        "off topic": "dialog_rails.topic",
        "topical rail": "dialog_rails.topic",
    },
    "output": {
        "self check output": "output_rails.self_check",
        "self check facts": "output_rails.fact_check",
        "fact checking": "output_rails.fact_check",
        "check hallucination": "output_rails.fact_check",
        "sensitive data detection on output": "output_rails.sensitive_data",
        "sensitive data detection": "output_rails.sensitive_data",
    },
    "retrieval": {
        "check retrieval relevance": "retrieval_rails.relevance",
    },
}


def _normalize_rail(rail_type: str, rail_name: str) -> str:
    bucket = _RAIL_TO_KEY.get((rail_type or "").lower(), {})
    return bucket.get((rail_name or "").lower(), f"{rail_type}_rails.unmapped")


def _activated_to_category(activated: dict[str, Any]) -> FindingCategory:
    rail_type = activated.get("type") or activated.get("rail_type") or ""
    rail_name = activated.get("name") or activated.get("rail_name") or ""
    decisions = activated.get("decisions") or []
    stop = bool(activated.get("stop")) or any(
        (d or "").lower() in {"refuse", "stop", "abort", "block"} for d in decisions
    )
    altered = bool(activated.get("output_changed") or activated.get("altered"))
    action = "BLOCKED" if stop else ("FLAGGED" if altered else "NONE")
    key = _normalize_rail(rail_type, rail_name)
    return FindingCategory(
        provider_category=key,
        severity_label=(rail_name or rail_type or "rail").upper(),
        normalized_severity=_sev_str(0.9 if stop else (0.5 if altered else 0.0)),
        action=action,
        mapping=mapping_for(_PROVIDER, key),
        evidence={
            "rail_type": rail_type,
            "rail_name": rail_name,
            "decisions": list(decisions),
        },
    )


def _with_rails_log(options: Any) -> Any:
    """Return generation options that request the activated-rails log.

    Accepts ``None``, a dict, or a ``GenerationOptions``. Never imports
    nemoguardrails: a plain dict is a valid ``options`` value, so the
    optional dependency stays optional.
    """
    if options is None:
        return {"log": {"activated_rails": True}}
    if isinstance(options, dict):
        merged = dict(options)
        log = merged.get("log")
        if isinstance(log, dict):
            log = dict(log)
            log["activated_rails"] = True
            merged["log"] = log
        elif log is None:
            merged["log"] = {"activated_rails": True}
        else:  # a GenerationLogOptions instance
            try:
                log.activated_rails = True
            except (AttributeError, ValueError):
                pass
        return merged
    log = getattr(options, "log", None)
    try:
        if log is None:
            options.log = {"activated_rails": True}
        else:
            log.activated_rails = True
    except (AttributeError, ValueError):
        # Frozen or unexpected options object: leave it alone rather
        # than fail the generation. parse_generation_response still
        # handles an absent log.
        pass
    return options


def parse_generation_response(
    response: Any,
    *,
    scanned_role: str = "",
) -> ContentSafetyFinding:
    """Parse a NeMo ``GenerationResponse`` into a ``ContentSafetyFinding``.

    Accepts either the SDK object (with ``.log.activated_rails``) or a
    dict shape. Rails that fired without altering or stopping the
    generation are recorded with ``action="NONE"`` so the audit trail
    keeps a complete signal log.
    """
    log = getattr(response, "log", None)
    if log is None and isinstance(response, dict):
        log = response.get("log") or {}
    if log is None:
        log = {}

    activated = getattr(log, "activated_rails", None)
    if activated is None and isinstance(log, dict):
        activated = log.get("activated_rails") or []
    activated = activated or []

    cats: list[FindingCategory] = []
    for rail in activated:
        rail_dict = rail if isinstance(rail, dict) else rail.__dict__
        cats.append(_activated_to_category(rail_dict))

    raw = response if isinstance(response, dict) else {"response_repr": repr(response)}
    return build_finding(
        provider=_PROVIDER,
        categories=cats,
        raw=raw,
        scanned_role=scanned_role,
    )


class NemoGuardrailsAdapter:
    """Wraps a configured ``nemoguardrails.LLMRails`` instance."""

    provider = _PROVIDER

    def __init__(self, rails: Any) -> None:
        if not (hasattr(rails, "generate") or hasattr(rails, "generate_async")):
            raise TypeError(
                "NemoGuardrailsAdapter: rails must be a configured LLMRails "
                "instance (expose generate / generate_async)."
            )
        self._rails = rails

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> tuple[str, ContentSafetyFinding]:
        """Run NeMo generation and return ``(response_text, finding)``.

        The activated-rails log is requested explicitly. NeMo returns
        ``GenerationResponse.log = None`` unless the caller asks for it,
        and without that log there are no rails to parse, so a blocked
        generation would produce an empty finding and a verdict of
        "allow". A caller passing its own ``options`` keeps them, but
        gets ``log.activated_rails`` forced on for the same reason.
        """
        kwargs["options"] = _with_rails_log(kwargs.get("options"))
        response = self._rails.generate(messages=messages, **kwargs)
        text = response if isinstance(response, str) else (
            getattr(response, "response", None)
            or (response.get("response") if isinstance(response, dict) else "")
            or ""
        )
        finding = parse_generation_response(response, scanned_role="response")
        return text, finding

    def scan_response(self, response: Any, **_: Any) -> ContentSafetyFinding:
        """Convenience: parse an already-issued ``GenerationResponse``."""
        return parse_generation_response(response, scanned_role="response")


__all__ = ["NemoGuardrailsAdapter", "parse_generation_response"]
