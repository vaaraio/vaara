"""Contract test for the LLM Guard adapter against the real library.

``test_integrations_llm_guard.py`` injects fake scan functions. This file
runs ``llm_guard.scan_prompt`` and ``scan_output`` themselves with scanners
that need no model download (BanSubstrings, Secrets), so the adapter parses
the tuple LLM Guard actually returns. Offline.

Skips when llm-guard is absent. The ``llm-guard-contract`` CI job installs
it and fails on a skip.
"""

from __future__ import annotations

import inspect

import pytest

llm_guard = pytest.importorskip("llm_guard", reason="pip install llm-guard")

from llm_guard.input_scanners import BanSubstrings, Secrets  # noqa: E402
from llm_guard.output_scanners import BanSubstrings as OutputBanSubstrings  # noqa: E402

from vaara.integrations.llm_guard import LLMGuardAdapter  # noqa: E402


@pytest.fixture
def adapter():
    # No injected functions: the adapter must find the real ones itself.
    return LLMGuardAdapter(
        [BanSubstrings(substrings=["rm -rf"]), Secrets()],
        [OutputBanSubstrings(substrings=["password:"])],
    )


def test_scan_signatures_match_the_adapter_calls():
    assert list(inspect.signature(llm_guard.scan_prompt).parameters)[:2] == [
        "scanners", "prompt"]
    assert list(inspect.signature(llm_guard.scan_output).parameters)[:3] == [
        "scanners", "prompt", "output"]


def test_clean_prompt_allows(adapter):
    finding = adapter.scan_prompt("hello")
    assert finding.verdict == "allow"
    assert {c.provider_category for c in finding.categories} == {"BanSubstrings", "Secrets"}


def test_banned_substring_blocks(adapter):
    finding = adapter.scan_prompt("please run rm -rf /")
    assert finding.verdict == "block"
    [hit] = finding.triggered_categories()
    assert hit.provider_category == "BanSubstrings"


def test_secret_in_prompt_blocks_as_secrets_leak(adapter):
    finding = adapter.scan_prompt("key AKIAIOSFODNN7EXAMPLE")
    [hit] = finding.triggered_categories()
    assert hit.provider_category == "Secrets"
    assert hit.vaara_category == "secrets_leak"


def test_passing_scanners_score_below_zero_and_record_zero_severity(adapter):
    finding = adapter.scan_prompt("hello")
    for category in finding.categories:
        assert float(category.evidence["score"]) < 0     # LLM Guard's -1.0 pass
        assert category.normalized_severity == "0.0000"


def test_output_scan_blocks(adapter):
    finding = adapter.scan_response("password: hunter2", prompt="q")
    assert finding.verdict == "block"
    assert finding.scanned_role == "response"
