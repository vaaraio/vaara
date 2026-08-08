# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""VaaraToolGuardrail must return the type the Agents SDK expects.

The guardrail imported ``GuardrailResult`` from ``agents``. That symbol
does not exist; the SDK's guardrail return type is
``GuardrailFunctionOutput``. Because the import sat inside a
``try/except ImportError`` that returned ``None``, a correctly installed
SDK produced a log line claiming the SDK was missing and every tool call
ran ungoverned.

The existing unit tests never caught it because they never had the real
``agents`` package present: with the SDK absent the no-op branch is
indistinguishable from correct behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vaara.integrations.openai_agents import VaaraToolGuardrail

agents = pytest.importorskip("agents", reason="pip install openai-agents")


class _ToolCall:
    id = "call_1"
    name = "transfer_funds"
    arguments = '{"to": "attacker", "amount": 999999}'


class _Output:
    tool_calls = [_ToolCall()]


class _Pipeline:
    def __init__(self, decision: str) -> None:
        self.decision = decision
        self.seen: list[dict] = []

    def intercept(self, **kwargs):
        self.seen.append(kwargs)
        return SimpleNamespace(
            decision=self.decision, action_id="a1", risk_score=0.97,
            reason="exfiltration pattern",
        )


def test_the_sdk_exports_guardrail_function_output_not_guardrail_result():
    """Pins the symbol the adapter must import."""
    assert hasattr(agents, "GuardrailFunctionOutput")
    assert not hasattr(agents, "GuardrailResult"), (
        "agents now exports GuardrailResult. Confirm which type the "
        "guardrail protocol expects before changing the adapter."
    )
    fields = agents.GuardrailFunctionOutput.__dataclass_fields__
    assert {"output_info", "tripwire_triggered"} <= set(fields)


def test_a_denied_tool_call_trips_the_tripwire():
    pipeline = _Pipeline("deny")
    result = VaaraToolGuardrail(pipeline)(context=None, agent=None, output=_Output())

    assert isinstance(result, agents.GuardrailFunctionOutput)
    assert result.tripwire_triggered is True
    assert result.output_info["blocked_tools"][0]["tool"] == "transfer_funds"
    assert "transfer_funds" in result.output_info["message"]


def test_an_allowed_tool_call_returns_a_real_verdict_not_none():
    """The regression: None meant "no verdict", which the SDK ignores."""
    result = VaaraToolGuardrail(_Pipeline("allow"))(
        context=None, agent=None, output=_Output(),
    )
    assert result is not None
    assert isinstance(result, agents.GuardrailFunctionOutput)
    assert result.tripwire_triggered is False


def test_tool_arguments_reach_the_pipeline_structured():
    pipeline = _Pipeline("allow")
    VaaraToolGuardrail(pipeline)(context=None, agent=None, output=_Output())

    parameters = pipeline.seen[0]["parameters"]
    assert parameters == {"to": "attacker", "amount": 999999}, (
        "arguments arrive as a JSON string and must be parsed, or the "
        "scorer sees one opaque blob instead of structured risk signals"
    )


def test_escalation_blocks_only_when_configured():
    allowed = VaaraToolGuardrail(_Pipeline("escalate"))(
        context=None, agent=None, output=_Output(),
    )
    assert allowed.tripwire_triggered is False

    blocked = VaaraToolGuardrail(_Pipeline("escalate"), block_on_escalate=True)(
        context=None, agent=None, output=_Output(),
    )
    assert blocked.tripwire_triggered is True
    assert "Escalated" in blocked.output_info["blocked_tools"][0]["reason"]
