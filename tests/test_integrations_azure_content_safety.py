"""Tests for the Azure AI Content Safety adapter.

Mocked Azure client; no Azure calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from vaara.integrations.azure_content_safety import (
    AzureContentSafetyAdapter,
    parse_responses,
)


class _FakeAzure:
    """Duck-types analyze_text / shield_prompt / etc as dict-returning callables."""

    def __init__(self, **responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _bind(self, fn_name: str):
        # Every Content Safety endpoint takes ONE positional options
        # mapping, matching ContentSafetyClient.analyze_text(options).
        def _call(options: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((fn_name, options))
            return self._responses[fn_name]
        return _call

    def __getattr__(self, name: str):
        if name in self._responses:
            return self._bind(name)
        raise AttributeError(name)


def _analyze_response(*severities: tuple[str, int]) -> dict[str, Any]:
    return {
        "categoriesAnalysis": [
            {"category": cat, "severity": sev} for cat, sev in severities
        ]
    }


class TestParseResponses:
    def test_empty_response_allows(self):
        finding = parse_responses()
        assert finding.verdict == "allow"
        assert finding.categories == ()

    def test_severity_six_blocks(self):
        finding = parse_responses(analyze_text=_analyze_response(("Hate", 6)))
        assert finding.verdict == "block"
        assert finding.categories[0].normalized_severity == "1.0000"
        assert finding.categories[0].vaara_category == "hate"

    def test_severity_four_blocks_under_default_threshold(self):
        finding = parse_responses(analyze_text=_analyze_response(("Violence", 4)))
        assert finding.categories[0].action == "BLOCKED"

    def test_severity_two_flags_under_default_threshold(self):
        finding = parse_responses(analyze_text=_analyze_response(("Sexual", 2)))
        assert finding.verdict == "flag"
        assert finding.categories[0].action == "FLAGGED"

    def test_severity_zero_filtered_out_of_actions(self):
        finding = parse_responses(analyze_text=_analyze_response(("SelfHarm", 0)))
        assert finding.categories[0].action == "NONE"
        assert finding.verdict == "allow"

    def test_custom_block_threshold(self):
        finding = parse_responses(
            analyze_text=_analyze_response(("Hate", 2)),
            block_threshold=2,
        )
        assert finding.categories[0].action == "BLOCKED"
        assert finding.verdict == "block"

    def test_prompt_shield_user_attack(self):
        shield = {"userPromptAnalysis": {"attackDetected": True}}
        finding = parse_responses(shield=shield)
        assert finding.verdict == "block"
        assert finding.categories[0].vaara_category == "adversarial"
        assert "Art. 15" in finding.ai_act_articles()

    def test_prompt_shield_document_attack(self):
        shield = {
            "userPromptAnalysis": {"attackDetected": False},
            "documentsAnalysis": [
                {"attackDetected": False},
                {"attackDetected": True},
            ],
        }
        finding = parse_responses(shield=shield)
        assert len(finding.categories) == 1
        assert finding.categories[0].provider_category == "PromptShield.Documents"
        assert finding.categories[0].evidence["document_index"] == 1

    def test_protected_material_text_and_code(self):
        protected = {
            "protectedMaterialAnalysis": {"detected": True},
            "protectedMaterialCodeAnalysis": {"detected": True, "citation": {"license": "MIT"}},
        }
        finding = parse_responses(protected=protected)
        assert {c.provider_category for c in finding.categories} == {
            "ProtectedMaterial.Text", "ProtectedMaterial.Code",
        }
        assert "Art. 53" in finding.ai_act_articles()

    def test_groundedness_ungrounded(self):
        grounded = {"ungroundedDetected": True, "ungroundedPercentage": 0.42}
        finding = parse_responses(grounded=grounded)
        assert finding.categories[0].vaara_category == "grounding"
        articles = finding.ai_act_articles()
        assert "Art. 13" in articles
        assert "Art. 15" in articles

    def test_groundedness_grounded_emits_nothing(self):
        grounded = {"ungroundedDetected": False, "ungroundedPercentage": 0.0}
        finding = parse_responses(grounded=grounded)
        assert finding.categories == ()


class TestAzureContentSafetyAdapter:
    def test_constructor_rejects_invalid_client(self):
        with pytest.raises(TypeError):
            AzureContentSafetyAdapter(client=object())

    def test_scan_prompt_defaults_to_analyze_text_only(self):
        # shield_prompt does not exist on the GA ContentSafetyClient, so
        # it must not be in the default include set. Defaulting it on
        # made every out-of-the-box scan fail.
        client = _FakeAzure(analyze_text=_analyze_response(("Hate", 4)))
        adapter = AzureContentSafetyAdapter(client)
        finding = adapter.scan_prompt("text")
        assert {name for name, _ in client.calls} == {"analyze_text"}
        assert finding.scanned_role == "prompt"
        assert finding.verdict == "block"

    def test_endpoints_are_called_with_one_positional_options_mapping(self):
        client = _FakeAzure(analyze_text=_analyze_response(("Hate", 0)))
        AzureContentSafetyAdapter(client).scan_prompt("some text")
        assert client.calls == [("analyze_text", {"text": "some text"})]

    def test_shield_uses_rest_body_keys_when_a_client_supplies_it(self):
        client = _FakeAzure(
            analyze_text=_analyze_response(("Hate", 0)),
            shield_prompt={"userPromptAnalysis": {"attackDetected": True}},
        )
        adapter = AzureContentSafetyAdapter(client)
        finding = adapter.scan_prompt(
            "text", include={"shield"}, documents=["doc one"],
        )
        assert client.calls == [
            ("shield_prompt", {"userPrompt": "text", "documents": ["doc one"]}),
        ]
        assert finding.verdict == "block"

    def test_missing_endpoint_raises_instead_of_silently_allowing(self):
        # The regression this guards: a client without shield_prompt used
        # to yield a finding with zero categories and verdict "allow" —
        # a guardrail reporting clean because it never ran.
        client = _FakeAzure(analyze_text=_analyze_response(("Hate", 0)))
        adapter = AzureContentSafetyAdapter(client)
        with pytest.raises(RuntimeError, match="shield_prompt"):
            adapter.scan_prompt("text", include={"analyze_text", "shield"})

    def test_scan_response_protected_material(self):
        client = _FakeAzure(
            analyze_text=_analyze_response(("Hate", 0)),
            detect_text_protected_material={"protectedMaterialAnalysis": {"detected": True}},
        )
        adapter = AzureContentSafetyAdapter(client)
        finding = adapter.scan_response("text", include={"analyze_text", "protected"})
        assert {name for name, _ in client.calls} == {
            "analyze_text", "detect_text_protected_material",
        }
        assert finding.verdict == "flag"
        triggered = finding.triggered_categories()
        assert len(triggered) == 1
        assert triggered[0].vaara_category == "protected_material"

    def test_groundedness_options_are_merged_with_the_text(self):
        client = _FakeAzure(
            analyze_text=_analyze_response(("Hate", 0)),
            detect_groundedness={"ungroundedDetected": False},
        )
        adapter = AzureContentSafetyAdapter(client)
        adapter.scan_response(
            "answer", include={"grounded"},
            groundedness_options={
                "domain": "Generic", "task": "QnA",
                "groundingSources": ["source text"],
            },
        )
        _, options = client.calls[0]
        assert options == {
            "domain": "Generic", "task": "QnA",
            "groundingSources": ["source text"], "text": "answer",
        }

    def test_include_filter_skips_endpoints(self):
        client = _FakeAzure(analyze_text=_analyze_response(("Hate", 4)))
        adapter = AzureContentSafetyAdapter(client)
        adapter.scan_prompt("text", include={"analyze_text"})
        called = {name for name, _ in client.calls}
        assert called == {"analyze_text"}

    def test_response_as_dict_method_is_used(self):
        class _AsDict:
            def __init__(self, payload): self._payload = payload
            def as_dict(self): return self._payload

        class _SdkClient:
            def analyze_text(self, _options):
                return _AsDict(_analyze_response(("Hate", 6)))

        adapter = AzureContentSafetyAdapter(_SdkClient())
        finding = adapter.scan_prompt("text", include={"analyze_text"})
        assert finding.verdict == "block"

    def test_unparseable_response_raises_rather_than_allowing(self):
        class _BadClient:
            def analyze_text(self, _options):
                return "not a mapping"

        with pytest.raises(TypeError, match="expected a mapping"):
            AzureContentSafetyAdapter(_BadClient()).scan_prompt("text")
