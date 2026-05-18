"""Tests for the NVIDIA NeMo Guardrails adapter.

No nemoguardrails install required; uses dict-shaped response stubs
and a fake LLMRails for the class-level surface.
"""

from __future__ import annotations

from typing import Any

import pytest

from vaara.integrations.nemo_guardrails import (
    NemoGuardrailsAdapter,
    parse_generation_response,
)


class _FakeRails:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] | None = None

    def generate(self, messages: list[dict[str, str]], **_: Any) -> Any:
        self.last_messages = messages
        return self._response


def _response(
    *,
    response_text: str = "ok",
    activated: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "response": response_text,
        "log": {"activated_rails": list(activated or [])},
    }


class TestParseGenerationResponse:
    def test_no_activated_rails_yields_allow(self):
        finding = parse_generation_response(_response(activated=[]))
        assert finding.provider == "nvidia-nemo-guardrails"
        assert finding.verdict == "allow"
        assert finding.categories == ()

    def test_jailbreak_input_rail_blocks(self):
        finding = parse_generation_response(_response(activated=[{
            "type": "input",
            "name": "jailbreak detection",
            "stop": True,
            "decisions": ["refuse"],
        }]))
        assert finding.verdict == "block"
        assert finding.categories[0].provider_category == "input_rails.jailbreak"
        assert "Art. 15" in finding.ai_act_articles()

    def test_output_self_check_flag(self):
        finding = parse_generation_response(_response(activated=[{
            "type": "output",
            "name": "self check output",
            "altered": True,
            "decisions": [],
        }]))
        assert finding.verdict == "flag"
        assert finding.categories[0].provider_category == "output_rails.self_check"

    def test_sdk_object_shape_is_supported(self):
        class _Log:
            activated_rails = [{
                "type": "output",
                "name": "fact checking",
                "stop": True,
                "decisions": ["abort"],
            }]

        class _Response:
            response = "..."
            log = _Log()

        finding = parse_generation_response(_Response())
        assert finding.verdict == "block"
        assert finding.categories[0].provider_category == "output_rails.fact_check"


class TestAdapter:
    def test_generate_returns_text_and_finding(self):
        rails = _FakeRails(_response(
            response_text="hello",
            activated=[{"type": "dialog", "name": "off topic", "stop": True}],
        ))
        adapter = NemoGuardrailsAdapter(rails)
        text, finding = adapter.generate(messages=[{"role": "user", "content": "hi"}])
        assert text == "hello"
        assert finding.verdict == "block"
        assert rails.last_messages == [{"role": "user", "content": "hi"}]

    def test_construction_rejects_non_rails(self):
        with pytest.raises(TypeError):
            NemoGuardrailsAdapter(object())
