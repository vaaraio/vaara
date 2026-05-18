"""Tests for the Rebuff adapter.

No rebuff install required; uses a fake client with the documented
detect_injection / is_canary_word_leaked surface.
"""

from __future__ import annotations

from typing import Any

import pytest

from vaara.integrations.rebuff import (
    RebuffAdapter,
    parse_canary_leak,
    parse_detect_response,
)


class _FakeRebuff:
    def __init__(
        self,
        *,
        detect: dict[str, Any] | None = None,
        canary_leaked: bool = False,
    ) -> None:
        self._detect = detect or {}
        self._canary_leaked = canary_leaked
        self.last_detect_input: str | None = None

    def detect_injection(self, text: str, **_: Any) -> dict[str, Any]:
        self.last_detect_input = text
        return self._detect

    def is_canary_word_leaked(self, prompt: str, response: str, canary: str) -> bool:
        return self._canary_leaked


def _detect(
    *,
    heuristic: float = 0.0,
    model: float = 0.0,
    vector: float = 0.0,
    injection_detected: bool = False,
) -> dict[str, Any]:
    return {
        "heuristicScore": heuristic,
        "modelScore": model,
        "vectorScore": vector,
        "maxHeuristicScore": 0.75,
        "maxModelScore": 0.9,
        "maxVectorScore": 0.9,
        "runHeuristicCheck": True,
        "runLanguageModelCheck": True,
        "runVectorCheck": True,
        "injectionDetected": injection_detected,
    }


class TestParseDetectResponse:
    def test_clean_prompt_yields_allow(self):
        finding = parse_detect_response(_detect())
        assert finding.provider == "rebuff"
        assert finding.verdict == "allow"
        # All three layers recorded with NONE action.
        assert len(finding.categories) == 3
        assert {c.action for c in finding.categories} == {"NONE"}

    def test_heuristic_above_threshold_blocks(self):
        finding = parse_detect_response(_detect(heuristic=0.95))
        assert finding.verdict == "block"
        triggered = finding.triggered_categories()
        assert triggered[0].provider_category == "heuristic_injection"
        assert triggered[0].vaara_category == "adversarial"

    def test_vector_score_dict_shape_parsed(self):
        d = _detect()
        d["vectorScore"] = {"topScore": 0.95, "countOverMaxVectorScore": 3}
        finding = parse_detect_response(d)
        assert finding.verdict == "block"

    def test_skipped_layer_recorded(self):
        d = _detect()
        d["runVectorCheck"] = False
        finding = parse_detect_response(d)
        skipped = [c for c in finding.categories if c.provider_category == "vector_injection"]
        assert skipped[0].severity_label == "SKIPPED"


class TestParseCanaryLeak:
    def test_canary_leak_blocks(self):
        finding = parse_canary_leak(True, canary_word="abc123")
        assert finding.verdict == "block"
        assert finding.categories[0].vaara_category == "secrets_leak"

    def test_no_canary_leak_yields_allow(self):
        finding = parse_canary_leak(False, canary_word="abc123")
        assert finding.verdict == "allow"


class TestAdapter:
    def test_scan_prompt_uses_client(self):
        client = _FakeRebuff(detect=_detect(model=0.95))
        adapter = RebuffAdapter(client)
        finding = adapter.scan_prompt("ignore previous")
        assert finding.verdict == "block"
        assert client.last_detect_input == "ignore previous"

    def test_scan_response_requires_canary_word(self):
        adapter = RebuffAdapter(_FakeRebuff())
        with pytest.raises(ValueError):
            adapter.scan_response("text", prompt="p")

    def test_scan_response_canary_check(self):
        adapter = RebuffAdapter(_FakeRebuff(canary_leaked=True))
        finding = adapter.scan_response("leaked!", prompt="p", canary_word="abc123")
        assert finding.verdict == "block"
        assert finding.categories[0].provider_category == "canary_leak"

    def test_construction_rejects_non_rebuff(self):
        with pytest.raises(TypeError):
            RebuffAdapter(object())
