"""Guardrails AI adapter. Upstream output-validation signal.

Wraps a pre-configured ``guardrails.Guard``. Each validator on the
Guard produces a pass/fail summary. The adapter parses the
``ValidationOutcome`` into a ``ContentSafetyFinding`` the deployer
routes through ``InterceptionPipeline.intercept(context=...)``.

Two entry points:

* ``parse_validation_outcome(outcome)`` for callers who already have
  a ``ValidationOutcome`` from ``Guard.parse`` or ``Guard.validate``.
* ``GuardrailsAIAdapter`` holds a configured ``Guard`` and exposes
  ``scan_response`` (text) plus ``parse`` (which returns the
  validated output alongside the finding).

Optional dependency: ``pip install vaara[guardrails-ai]``.
"""

from __future__ import annotations

from typing import Any

from vaara.integrations._content_safety_base import (
    ContentSafetyFinding,
    FindingCategory,
    build_finding,
    mapping_for,
)


_PROVIDER = "guardrails-ai"


def _sev_str(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.4f}"


def _validator_key(validator_name: str) -> str:
    """Normalise a validator name to the published mapping key.

    Guardrails Hub validators arrive in many casings and prefixes
    (``guardrails/detect_pii``, ``DetectPII``, ``detect-pii``). Map to
    PascalCase short names matching ``_content_safety_articles``.
    """
    name = (validator_name or "").rsplit("/", 1)[-1]
    # snake_case or kebab-case to PascalCase
    if "_" in name or "-" in name:
        parts = name.replace("-", "_").split("_")
        return "".join(p[:1].upper() + p[1:] for p in parts if p)
    return name


def _summary_to_category(summary: Any) -> FindingCategory:
    """Map one ``ValidationSummary`` (or dict) to a ``FindingCategory``."""
    get = (lambda k, d=None: summary.get(k, d)) if isinstance(summary, dict) else (
        lambda k, d=None: getattr(summary, k, d)
    )
    raw_name = get("validator_name") or get("name") or ""
    status = (get("validator_status") or get("status") or "").lower()
    failures = get("failures") or []
    failure_reason = get("failure_reason") or ""
    error_spans = get("error_spans") or []

    key = _validator_key(raw_name)
    failed = status in {"fail", "failed"} or bool(failures) or bool(failure_reason)
    return FindingCategory(
        provider_category=key,
        severity_label=("FAIL" if failed else "PASS"),
        normalized_severity=_sev_str(0.9 if failed else 0.0),
        action=("FLAGGED" if failed else "NONE"),
        mapping=mapping_for(_PROVIDER, key),
        evidence={
            "validator_name": raw_name,
            "status": status,
            "failure_reason": failure_reason,
            "error_span_count": len(error_spans) if hasattr(error_spans, "__len__") else 0,
        },
    )


def parse_validation_outcome(
    outcome: Any,
    *,
    scanned_role: str = "response",
) -> ContentSafetyFinding:
    """Parse a Guardrails AI ``ValidationOutcome`` into a finding.

    Accepts the SDK object or a dict shape. When no per-validator
    summary is available (older SDK paths), a single synthetic
    category is recorded against ``unmapped`` with the overall
    pass/fail status.
    """
    get = (lambda k, d=None: outcome.get(k, d)) if isinstance(outcome, dict) else (
        lambda k, d=None: getattr(outcome, k, d)
    )
    summaries = get("validation_summaries") or []
    passed = bool(get("validation_passed"))

    cats: list[FindingCategory] = []
    if summaries:
        for s in summaries:
            cats.append(_summary_to_category(s))
    elif not passed:
        cats.append(FindingCategory(
            provider_category="unmapped",
            severity_label="FAIL",
            normalized_severity=_sev_str(0.9),
            action="FLAGGED",
            mapping=None,
            evidence={"error": str(get("error") or "")},
        ))

    raw = outcome if isinstance(outcome, dict) else {
        "validation_passed": passed,
        "error": str(get("error") or ""),
    }
    return build_finding(
        provider=_PROVIDER,
        categories=cats,
        raw=raw,
        scanned_role=scanned_role,
    )


class GuardrailsAIAdapter:
    """Wraps a configured ``guardrails.Guard`` instance."""

    provider = _PROVIDER

    def __init__(self, guard: Any) -> None:
        if not (hasattr(guard, "parse") or hasattr(guard, "validate")):
            raise TypeError(
                "GuardrailsAIAdapter: guard must be a Guardrails AI Guard "
                "(expose parse / validate)."
            )
        self._guard = guard

    def parse(self, text: str, **kwargs: Any) -> tuple[Any, ContentSafetyFinding]:
        """Run ``Guard.parse(text)`` and return ``(validated_output, finding)``."""
        outcome = self._guard.parse(text, **kwargs)
        finding = parse_validation_outcome(outcome, scanned_role="response")
        validated = (
            outcome.get("validated_output") if isinstance(outcome, dict)
            else getattr(outcome, "validated_output", None)
        )
        return validated, finding

    def scan_response(self, text: str, **kwargs: Any) -> ContentSafetyFinding:
        outcome = self._guard.parse(text, **kwargs)
        return parse_validation_outcome(outcome, scanned_role="response")


__all__ = ["GuardrailsAIAdapter", "parse_validation_outcome"]
