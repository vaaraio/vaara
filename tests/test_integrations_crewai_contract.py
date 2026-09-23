"""Contract test for the CrewAI integration against the real library.

``test_integrations_crewai_governance.py`` drives the hooks with stand-in
context objects. This file uses CrewAI's own ``Agent``, ``Crew``, ``Task``,
``BaseTool``, ``@tool`` and ``ToolCallHookContext``. Offline: nothing calls
a model, the key is a dummy, telemetry is off.

It found one gap. ``@tool`` returns a ``Tool`` whose ``run`` calls ``func``
directly, and ``wrap_tools`` wrapped ``_run``, so calling ``tool.run(...)``
skipped Vaara. Such a tool is now wrapped at ``func``.

Skips when crewai is absent. The ``guardrail-contracts`` CI job installs it
and fails on a skip.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "sk-offline-contract-test")

crewai = pytest.importorskip("crewai", reason="pip install crewai")

from crewai import Agent, Crew, Task  # noqa: E402
from crewai.hooks import (  # noqa: E402
    ToolCallHookContext,
    clear_all_tool_call_hooks,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
)
from crewai.tools import BaseTool, tool  # noqa: E402

from vaara.integrations.crewai import (  # noqa: E402
    VaaraCrewGovernance,
    VaaraGovernance,
    register,
)
from vaara.integrations.langchain import ToolExecutionBlocked  # noqa: E402


class Transfer(BaseTool):
    name: str = "tx.transfer"
    description: str = "Move money."

    def _run(self, amount: int) -> str:
        return f"sent {amount}"


def _function_tool():
    @tool("fs.delete")
    def delete(path: str) -> str:
        """Delete a path."""
        return "deleted " + path

    return delete


class _Pipeline:
    """Records every intercept and returns a fixed decision."""

    def __init__(self, decision="allow"):
        self.decision = decision
        self.calls: list[str] = []

    def intercept(self, *, tool_name, **_):
        self.calls.append(tool_name)
        return SimpleNamespace(
            decision=self.decision, action_id=f"a{len(self.calls)}",
            risk_score=0.9, risk_interval=(0.8, 1.0), reason="test",
        )

    def report_outcome(self, *_a, **_k):
        pass


@pytest.mark.parametrize("make", [Transfer, _function_tool], ids=["BaseTool", "@tool"])
def test_every_call_path_is_intercepted_exactly_once(make):
    pipeline = _Pipeline()
    [wrapped] = VaaraCrewGovernance(pipeline).wrap_tools([make()])
    args = {"amount": 5} if wrapped.name == "tx.transfer" else {"path": "/x"}

    wrapped.run(**args)                              # direct call
    wrapped.to_structured_tool().invoke(args)        # the agent's path
    assert pipeline.calls == [wrapped.name, wrapped.name]


@pytest.mark.parametrize("make", [Transfer, _function_tool], ids=["BaseTool", "@tool"])
def test_a_deny_stops_the_tool_on_both_paths(make):
    pipeline = _Pipeline(decision="deny")
    [wrapped] = VaaraCrewGovernance(pipeline).wrap_tools([make()])
    args = {"amount": 5} if wrapped.name == "tx.transfer" else {"path": "/x"}
    with pytest.raises(ToolExecutionBlocked):
        wrapped.run(**args)
    with pytest.raises(ToolExecutionBlocked):
        wrapped.to_structured_tool().invoke(args)


def test_wrapping_twice_does_not_double_intercept():
    pipeline = _Pipeline()
    gov = VaaraCrewGovernance(pipeline)
    [once] = gov.wrap_tools([_function_tool()])
    [twice] = gov.wrap_tools([once])
    twice.run(path="/x")
    assert pipeline.calls == ["fs.delete"]


def test_hooks_read_the_real_context_and_close_the_run():
    agent = Agent(role="clerk", goal="g", backstory="b", llm="gpt-4o-mini")
    task = Task(description="d", expected_output="e", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    delete = _function_tool()
    ctx = ToolCallHookContext(
        tool_name="fs.delete", tool_input={"path": "/x"},
        tool=delete.to_structured_tool(), agent=agent, task=task, crew=crew,
    )

    gov = VaaraGovernance()
    assert gov.before_tool_call(ctx) is None
    ctx.raw_tool_result = "deleted /x"
    ctx.tool_result = "deleted /x"
    assert gov.after_tool_call(ctx) is None

    [decision] = gov.decisions()
    assert decision["agent_id"] == str(agent.id)
    assert decision["agent_role"] == "clerk"
    assert decision["extensions"]["vaara"]["completeness"]["boundaryId"] == str(crew.id)
    [outcome] = gov.outcomes()
    assert outcome["outcome"] == "executed"
    assert gov.verify_run(str(crew.id)).ok


def test_register_installs_both_hooks():
    clear_all_tool_call_hooks()
    try:
        gov = VaaraGovernance()
        register(gov)
        assert gov.before_tool_call in get_before_tool_call_hooks()
        assert gov.after_tool_call in get_after_tool_call_hooks()
    finally:
        clear_all_tool_call_hooks()
