"""Contract test for the Guardrails AI adapter against the real library.

``test_integrations_guardrails_ai.py`` builds outcomes by hand. This file
runs a real ``Guard`` with a locally registered validator, so the adapter
parses the ``ValidationOutcome`` and ``ValidationSummary`` Guardrails
actually returns. Offline: no hub validator, no model, metrics off.

Skips when guardrails-ai is absent. The ``guardrail-contracts`` CI job
installs it and fails on a skip.
"""

from __future__ import annotations

import pytest

guardrails = pytest.importorskip("guardrails", reason="pip install guardrails-ai")

from guardrails import Guard, OnFailAction  # noqa: E402
from guardrails.validators import (  # noqa: E402
    FailResult,
    PassResult,
    Validator,
    register_validator,
)

from vaara.integrations.guardrails_ai import GuardrailsAIAdapter  # noqa: E402


@register_validator(name="vaara-test/no-secret", data_type="string")
class NoSecret(Validator):
    def _validate(self, value, metadata):
        if "sk-" in value:
            return FailResult(error_message="secret found")
        return PassResult()


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    # Guard runs export telemetry to a hosted collector unless turned off.
    from guardrails.settings import settings

    monkeypatch.setattr(settings.rc, "enable_metrics", False)
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


def _adapter(on_fail=OnFailAction.NOOP):
    return GuardrailsAIAdapter(Guard().use(NoSecret(on_fail=on_fail)))


def test_a_failing_validator_flags_with_its_summary():
    validated, finding = _adapter().parse("here is sk-123")
    assert finding.verdict == "flag"
    [category] = finding.categories
    assert category.provider_category == "NoSecret"
    assert category.action == "FLAGGED"
    assert category.evidence["status"] == "fail"
    assert category.evidence["failure_reason"] == "secret found"
    assert validated == "here is sk-123"      # NOOP returns the text unchanged


def test_a_passing_guard_allows():
    finding = _adapter().scan_response("nothing to see")
    assert finding.verdict == "allow"


def test_a_fix_action_still_reports_the_failure():
    finding = _adapter(OnFailAction.FIX).scan_response("sk-1")
    assert finding.verdict == "flag"


def test_an_exception_action_propagates_rather_than_reading_as_a_pass():
    with pytest.raises(Exception, match="secret found"):
        _adapter(OnFailAction.EXCEPTION).scan_response("sk-1")
