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

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.last_messages = messages
        self.last_options = kwargs.get("options")
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

    def test_generate_always_requests_the_activated_rails_log(self):
        # NeMo leaves GenerationResponse.log as None unless the caller
        # asks for it. Without the log there are no rails to parse, so a
        # blocked generation would report verdict "allow".
        rails = _FakeRails(_response(activated=[]))
        NemoGuardrailsAdapter(rails).generate(messages=[{"role": "user", "content": "hi"}])
        assert rails.last_options == {"log": {"activated_rails": True}}

    def test_caller_options_survive_with_the_log_forced_on(self):
        rails = _FakeRails(_response(activated=[]))
        NemoGuardrailsAdapter(rails).generate(
            messages=[{"role": "user", "content": "hi"}],
            options={"llm_output": True, "log": {"llm_calls": True}},
        )
        assert rails.last_options == {
            "llm_output": True,
            "log": {"llm_calls": True, "activated_rails": True},
        }

    def test_generation_options_object_is_mutated_not_replaced(self):
        class _LogOptions:
            def __init__(self): self.activated_rails = False

        class _Options:
            def __init__(self): self.llm_output = True; self.log = _LogOptions()

        rails = _FakeRails(_response(activated=[]))
        options = _Options()
        NemoGuardrailsAdapter(rails).generate(
            messages=[{"role": "user", "content": "hi"}], options=options,
        )
        assert rails.last_options is options
        assert options.llm_output is True
        assert options.log.activated_rails is True
