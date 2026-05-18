"""Tests for the LLM Guard adapter.

No llm_guard install required; the adapter takes scan callables via
constructor for injection-friendly testing.
"""

from __future__ import annotations

import pytest

from vaara.integrations.llm_guard import (
    LLMGuardAdapter,
    parse_scan_result,
)


class TestParseScanResult:
    def test_all_pass_yields_allow(self):
        finding = parse_scan_result(
            {"PromptInjection": True, "Toxicity": True},
            {"PromptInjection": 0.0, "Toxicity": 0.0},
            scanned_role="prompt",
        )
        assert finding.provider == "llm-guard"
        assert finding.verdict == "allow"

    def test_prompt_injection_triggered_blocks(self):
        finding = parse_scan_result(
            {"PromptInjection": False, "Toxicity": True},
            {"PromptInjection": 0.98, "Toxicity": 0.0},
            scanned_role="prompt",
        )
        assert finding.verdict == "block"
        triggered = finding.triggered_categories()
        assert triggered[0].provider_category == "PromptInjection"
        assert triggered[0].vaara_category == "adversarial"
        assert "Art. 15" in finding.ai_act_articles()

    def test_secrets_triggered_maps_to_secrets_leak(self):
        finding = parse_scan_result(
            {"Secrets": False},
            {"Secrets": 0.92},
            scanned_role="response",
        )
        assert finding.triggered_categories()[0].vaara_category == "secrets_leak"

    def test_severity_clamped_to_unit_interval(self):
        finding = parse_scan_result(
            {"PromptInjection": False},
            {"PromptInjection": 1.5},
            scanned_role="prompt",
        )
        assert finding.severity == "1.0000"


class TestAdapter:
    def test_scan_prompt_calls_injected_fn(self):
        def fake_scan_prompt(scanners, prompt):
            assert scanners == ["s1", "s2"]
            return prompt + "[s]", {"PromptInjection": False}, {"PromptInjection": 0.99}

        def fake_scan_output(scanners, prompt, output):
            return output, {}, {}

        adapter = LLMGuardAdapter(
            input_scanners=["s1", "s2"],
            scan_prompt_fn=fake_scan_prompt,
            scan_output_fn=fake_scan_output,
        )
        finding = adapter.scan_prompt("ignore previous")
        assert finding.verdict == "block"
        assert finding.scanned_role == "prompt"

    def test_scan_response_passes_prompt_through(self):
        seen: dict[str, str] = {}

        def fake_scan_output(scanners, prompt, output):
            seen["prompt"] = prompt
            seen["output"] = output
            return output, {"Bias": True}, {"Bias": 0.0}

        adapter = LLMGuardAdapter(
            output_scanners=["b"],
            scan_prompt_fn=lambda *_a, **_k: ("", {}, {}),
            scan_output_fn=fake_scan_output,
        )
        finding = adapter.scan_response("answer text", prompt="question")
        assert seen == {"prompt": "question", "output": "answer text"}
        assert finding.verdict == "allow"

    def test_missing_llm_guard_raises_when_no_fn_injected(self):
        # llm_guard is not installed in test env. When neither scan
        # function is injected, construction must raise ImportError.
        with pytest.raises(ImportError):
            LLMGuardAdapter()
