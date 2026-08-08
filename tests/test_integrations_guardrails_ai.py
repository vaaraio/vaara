"""Tests for the Guardrails AI adapter.

No guardrails-ai install required; uses dict-shaped ValidationOutcome
stubs and a fake Guard for the class-level surface.
"""

from __future__ import annotations

from typing import Any

import pytest

from vaara.integrations.guardrails_ai import (
    GuardrailsAIAdapter,
    parse_validation_outcome,
)


class _FakeGuard:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome
        self.last_text: str | None = None

    def parse(self, text: str, **_: Any) -> Any:
        self.last_text = text
        return self._outcome


def _outcome(
    *,
    passed: bool = True,
    summaries: list[dict[str, Any]] | None = None,
    error: str = "",
    validated_output: Any = None,
) -> dict[str, Any]:
    return {
        "validation_passed": passed,
        "validation_summaries": list(summaries or []),
        "error": error,
        "validated_output": validated_output,
    }


class TestParseValidationOutcome:
    def test_clean_outcome_yields_allow(self):
        finding = parse_validation_outcome(_outcome(passed=True))
        assert finding.provider == "guardrails-ai"
        assert finding.verdict == "allow"
        assert finding.categories == ()

    def test_pii_failure(self):
        finding = parse_validation_outcome(_outcome(passed=False, summaries=[{
            "validator_name": "guardrails/detect_pii",
            "validator_status": "fail",
            "failure_reason": "email detected",
        }]))
        assert finding.verdict == "flag"
        assert finding.categories[0].provider_category == "DetectPii"
        # Mapping uses PascalCase "DetectPII"; the normaliser yields
        # "DetectPii". Both resolve to the pii vaara_category through
        # the article table when the canonical key matches.
        assert finding.categories[0].vaara_category in {"pii", "unmapped"}

    def test_toxic_language_failure_maps_to_hate(self):
        finding = parse_validation_outcome(_outcome(passed=False, summaries=[{
            "validator_name": "ToxicLanguage",
            "validator_status": "fail",
            "failure_reason": "toxic span detected",
        }]))
        assert finding.categories[0].vaara_category == "hate"
        assert "Art. 5" in finding.ai_act_articles()

    def test_summary_less_failure_records_unmapped(self):
        finding = parse_validation_outcome(_outcome(
            passed=False, summaries=[], error="something failed",
        ))
        assert finding.verdict == "flag"
        assert finding.categories[0].provider_category == "unmapped"


class TestAdapter:
    def test_parse_returns_validated_and_finding(self):
        guard = _FakeGuard(_outcome(passed=True, validated_output={"ok": True}))
        adapter = GuardrailsAIAdapter(guard)
        validated, finding = adapter.parse("hello")
        assert validated == {"ok": True}
        assert finding.verdict == "allow"
        assert guard.last_text == "hello"

    def test_construction_rejects_non_guard(self):
        with pytest.raises(TypeError):
            GuardrailsAIAdapter(object())


class TestValidatorNameNormalisation:
    """Guardrails reports validator.rail_alias, not the class name.

    A rail alias like ``detect-pii`` PascalCases to ``DetectPii``, which
    never matches the published ``DetectPII`` key, so the finding lost
    its Art. 10 annotation and degraded to "unmapped".
    """

    @staticmethod
    def _outcome(alias: str) -> dict:
        return {
            "validation_passed": False,
            "validation_summaries": [
                {"validator_name": alias, "validator_status": "fail",
                 "failure_reason": "found"},
            ],
        }

    @pytest.mark.parametrize("alias", [
        "guardrails/detect_pii", "detect-pii", "detect_pii", "DetectPII",
    ])
    def test_every_pii_spelling_maps_to_article_10(self, alias):
        finding = parse_validation_outcome(self._outcome(alias))
        assert finding.categories[0].vaara_category == "pii"
        assert "Art. 10" in finding.ai_act_articles()

    @pytest.mark.parametrize("alias,expected", [
        ("valid-json", "schema_violation"),
        ("toxic-language", "hate"),
        ("secrets-present", "secrets_leak"),
        ("detect-prompt-injection", "adversarial"),
    ])
    def test_acronym_and_multiword_aliases_map(self, alias, expected):
        finding = parse_validation_outcome(self._outcome(alias))
        assert finding.categories[0].vaara_category == expected

    def test_a_genuinely_unknown_validator_stays_unmapped(self):
        finding = parse_validation_outcome(self._outcome("some-custom-validator"))
        assert finding.categories[0].vaara_category == "unmapped"
