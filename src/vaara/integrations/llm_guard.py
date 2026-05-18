"""LLM Guard adapter. Upstream scanner-library signal.

Wraps the ``llm_guard`` scanner pipeline. The deployer assembles a
list of input scanners (Anonymize, PromptInjection, Toxicity, ...)
and output scanners (Bias, Sensitive, MaliciousURLs, ...) and the
adapter forwards prompts and responses through them, parsing the
``(sanitized_text, results_valid, results_score)`` tuple into a
``ContentSafetyFinding`` the deployer routes through
``InterceptionPipeline.intercept(context=...)``.

Two entry points:

* ``parse_scan_result(results_valid, results_score, ...)`` for
  callers who already ran ``llm_guard.scan_prompt`` or ``scan_output``.
* ``LLMGuardAdapter`` holds pre-built scanner lists and exposes
  ``scan_prompt`` / ``scan_response``.

Optional dependency: ``pip install vaara[llm-guard]``.
"""

from __future__ import annotations

from typing import Any, Iterable

from vaara.integrations._content_safety_base import (
    ContentSafetyFinding,
    FindingCategory,
    build_finding,
    mapping_for,
)


_PROVIDER = "llm-guard"


def _sev_str(value: float) -> str:
    return f"{max(0.0, min(1.0, value)):.4f}"


def _scanner_key(scanner_name: str) -> str:
    """LLM Guard returns class names verbatim (``PromptInjection``).

    Strip any module prefix and return the class name unchanged.
    """
    name = (scanner_name or "").rsplit(".", 1)[-1]
    return name


def parse_scan_result(
    results_valid: dict[str, bool],
    results_score: dict[str, float],
    *,
    scanned_role: str,
    sanitized_text: str | None = None,
) -> ContentSafetyFinding:
    """Parse an ``llm_guard.scan_prompt``/``scan_output`` result triple.

    A scanner is "triggered" when ``results_valid[name] is False``.
    Score range is implementation-defined per scanner. LLM Guard
    typically returns ``0.0`` for pass and a non-zero risk score on
    failure.
    """
    cats: list[FindingCategory] = []
    for raw_name, valid in (results_valid or {}).items():
        key = _scanner_key(raw_name)
        score = float((results_score or {}).get(raw_name, 0.0) or 0.0)
        triggered = not bool(valid)
        cats.append(FindingCategory(
            provider_category=key,
            severity_label=("BLOCKED" if triggered else "PASS"),
            normalized_severity=_sev_str(score if triggered else 0.0),
            action=("BLOCKED" if triggered else "NONE"),
            mapping=mapping_for(_PROVIDER, key),
            evidence={
                "scanner": raw_name,
                "score": f"{score:.4f}",
                "valid": bool(valid),
            },
        ))

    raw: dict[str, Any] = {
        "results_valid": dict(results_valid or {}),
        "results_score": {k: f"{float(v):.4f}" for k, v in (results_score or {}).items()},
    }
    if sanitized_text is not None:
        raw["sanitized_text_length"] = len(sanitized_text)
    return build_finding(
        provider=_PROVIDER,
        categories=cats,
        raw=raw,
        scanned_role=scanned_role,
    )


class LLMGuardAdapter:
    """Wraps pre-built lists of ``llm_guard`` input/output scanners."""

    provider = _PROVIDER

    def __init__(
        self,
        input_scanners: Iterable[Any] | None = None,
        output_scanners: Iterable[Any] | None = None,
        *,
        scan_prompt_fn: Any = None,
        scan_output_fn: Any = None,
    ) -> None:
        self._input_scanners = list(input_scanners or [])
        self._output_scanners = list(output_scanners or [])
        if scan_prompt_fn is None or scan_output_fn is None:
            try:
                from llm_guard import scan_prompt as _scan_prompt
                from llm_guard import scan_output as _scan_output
            except ImportError as exc:
                raise ImportError(
                    "LLMGuardAdapter: install vaara[llm-guard] to use this adapter."
                ) from exc
            scan_prompt_fn = scan_prompt_fn or _scan_prompt
            scan_output_fn = scan_output_fn or _scan_output
        self._scan_prompt_fn = scan_prompt_fn
        self._scan_output_fn = scan_output_fn

    def scan_prompt(self, text: str, **_: Any) -> ContentSafetyFinding:
        sanitized, valid, score = self._scan_prompt_fn(self._input_scanners, text)
        return parse_scan_result(
            valid, score, scanned_role="prompt", sanitized_text=sanitized,
        )

    def scan_response(
        self,
        text: str,
        *,
        prompt: str = "",
        **_: Any,
    ) -> ContentSafetyFinding:
        sanitized, valid, score = self._scan_output_fn(
            self._output_scanners, prompt, text,
        )
        return parse_scan_result(
            valid, score, scanned_role="response", sanitized_text=sanitized,
        )


__all__ = ["LLMGuardAdapter", "parse_scan_result"]
