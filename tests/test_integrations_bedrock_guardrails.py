"""Tests for the AWS Bedrock Guardrails adapter.

Mocked boto3 client; no AWS calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from vaara.integrations.bedrock_guardrails import (
    BedrockGuardrailsAdapter,
    parse_apply_guardrail_response,
)


class _FakeBedrock:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.last_request: dict[str, Any] = {}

    def apply_guardrail(self, **kwargs: Any) -> dict[str, Any]:
        self.last_request = kwargs
        return self._response


def _topic_response(action: str = "BLOCKED") -> dict[str, Any]:
    return {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{
            "topicPolicy": {
                "topics": [{"name": "FinancialAdvice", "type": "DENY",
                            "action": action, "detected": True}]
            }
        }],
    }


def _content_response(filter_type: str, confidence: str = "HIGH") -> dict[str, Any]:
    return {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{
            "contentPolicy": {
                "filters": [{"type": filter_type, "confidence": confidence,
                             "filterStrength": "HIGH", "action": "BLOCKED", "detected": True}]
            }
        }],
    }


def _pii_response() -> dict[str, Any]:
    return {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{
            "sensitiveInformationPolicy": {
                "piiEntities": [{"match": "alice@example.com", "type": "EMAIL",
                                 "action": "ANONYMIZED", "detected": True}]
            }
        }],
    }


def _grounding_response() -> dict[str, Any]:
    return {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [{
            "contextualGroundingPolicy": {
                "filters": [{"type": "GROUNDING", "threshold": 0.7, "score": 0.3,
                             "action": "BLOCKED", "detected": True}]
            }
        }],
    }


class TestParseApplyGuardrailResponse:
    def test_clean_response_yields_allow_verdict(self):
        finding = parse_apply_guardrail_response({"assessments": []})
        assert finding.verdict == "allow"
        assert finding.categories == ()
        assert finding.severity == "0.0000"

    def test_topic_policy_maps_to_article_5(self):
        finding = parse_apply_guardrail_response(_topic_response())
        assert finding.verdict == "block"
        assert "Art. 5" in finding.ai_act_articles()
        assert finding.categories[0].vaara_category == "prohibited_topic"
        assert finding.categories[0].evidence["topic_name"] == "FinancialAdvice"

    def test_content_policy_hate_high_confidence(self):
        finding = parse_apply_guardrail_response(_content_response("HATE", "HIGH"))
        assert finding.verdict == "block"
        cat = finding.categories[0]
        assert cat.provider_category == "contentPolicy.HATE"
        assert cat.vaara_category == "hate"
        assert cat.normalized_severity == "0.9000"

    def test_content_policy_prompt_attack_maps_to_article_15(self):
        finding = parse_apply_guardrail_response(_content_response("PROMPT_ATTACK"))
        assert finding.categories[0].vaara_category == "adversarial"
        assert "Art. 15" in finding.ai_act_articles()

    def test_pii_maps_to_article_10(self):
        finding = parse_apply_guardrail_response(_pii_response())
        assert finding.verdict == "flag"  # ANONYMIZED is not BLOCKED
        assert finding.categories[0].vaara_category == "pii"
        assert "Art. 10" in finding.ai_act_articles()

    def test_grounding_maps_to_articles_13_and_15(self):
        finding = parse_apply_guardrail_response(_grounding_response())
        articles = finding.ai_act_articles()
        assert "Art. 13" in articles
        assert "Art. 15" in articles
        assert finding.categories[0].vaara_category == "grounding"

    def test_undetected_findings_are_skipped(self):
        response = {
            "assessments": [{
                "contentPolicy": {
                    "filters": [{"type": "HATE", "confidence": "LOW",
                                 "action": "NONE", "detected": False}]
                }
            }]
        }
        finding = parse_apply_guardrail_response(response)
        assert finding.categories == ()

    def test_unmapped_category_uses_unmapped_vaara_label(self):
        # contentPolicy.UNKNOWN is not in the table.
        response = {
            "assessments": [{
                "contentPolicy": {
                    "filters": [{"type": "UNKNOWN", "confidence": "HIGH",
                                 "action": "BLOCKED", "detected": True}]
                }
            }]
        }
        finding = parse_apply_guardrail_response(response)
        assert finding.categories[0].vaara_category == "unmapped"
        assert finding.categories[0].ai_act_articles == ()

    def test_overt_metadata_uses_decimal_strings(self):
        finding = parse_apply_guardrail_response(_content_response("HATE", "HIGH"))
        meta = finding.to_overt_metadata()
        assert isinstance(meta["upstream_guardrail_severity"], str)
        assert "." in meta["upstream_guardrail_severity"]

    def test_audit_context_round_trip(self):
        finding = parse_apply_guardrail_response(_topic_response())
        ctx = finding.to_audit_context()
        assert ctx["upstream_guardrail"]["provider"] == "aws-bedrock-guardrails"
        assert ctx["upstream_guardrail"]["categories"][0]["vaara_category"] == "prohibited_topic"


class TestBedrockGuardrailsAdapter:
    def test_constructor_rejects_invalid_client(self):
        with pytest.raises(TypeError):
            BedrockGuardrailsAdapter(client=object(), guardrail_id="g1")

    def test_constructor_requires_guardrail_id(self):
        with pytest.raises(ValueError):
            BedrockGuardrailsAdapter(client=_FakeBedrock({}), guardrail_id="")

    def test_scan_prompt_passes_input_source(self):
        client = _FakeBedrock(_topic_response())
        adapter = BedrockGuardrailsAdapter(client, guardrail_id="g1", guardrail_version="DRAFT")
        finding = adapter.scan_prompt("buy AMD calls")
        assert client.last_request["source"] == "INPUT"
        assert client.last_request["guardrailIdentifier"] == "g1"
        assert finding.scanned_role == "prompt"
        assert finding.verdict == "block"

    def test_scan_response_passes_output_source(self):
        client = _FakeBedrock(_content_response("VIOLENCE", "MEDIUM"))
        adapter = BedrockGuardrailsAdapter(client, guardrail_id="g1")
        finding = adapter.scan_response("violent text")
        assert client.last_request["source"] == "OUTPUT"
        assert finding.scanned_role == "response"
        assert finding.categories[0].vaara_category == "violence"
