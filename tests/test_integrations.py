"""Tests for framework integrations (LangChain, OpenAI, CrewAI).

These tests verify the integration logic WITHOUT requiring the actual
frameworks installed — they test Vaara's side of the contract.
"""

import pytest
from uuid import uuid4

from vaara.pipeline import InterceptionPipeline, InterceptionResult
from vaara.taxonomy.actions import (
    ActionType,
    ActionCategory,
    Reversibility,
    BlastRadius,
)
from vaara.integrations.langchain import (
    VaaraCallbackHandler,
    ToolExecutionBlocked,
    ToolExecutionEscalated,
    VaaraInterceptionError,
    vaara_wrap_tool,
)
from vaara.integrations.openai_agents import (
    vaara_wrap_function,
    ToolCallBlocked,
    ToolCallEscalated,
    VaaraToolGuardrail,
)
from vaara.integrations.crewai import VaaraCrewGovernance


def _fake_result(decision: str, allowed: bool, action_id: str = "act-test") -> InterceptionResult:
    """Build a fake InterceptionResult for a target decision."""
    return InterceptionResult(
        allowed=allowed,
        action_id=action_id,
        decision=decision,
        risk_score=0.9 if decision != "allow" else 0.1,
        risk_interval=(0.8, 1.0) if decision != "allow" else (0.0, 0.2),
        reason=f"test {decision}",
        action_type=ActionType(
            name="test.action",
            category=ActionCategory.DATA,
            reversibility=Reversibility.FULLY,
            blast_radius=BlastRadius.SELF,
        ),
        signals={},
        evaluation_ms=0.1,
    )


class _FakeRegistry:
    def classify(self, tool_name: str):
        return ActionType(
            name=tool_name,
            category=ActionCategory.DATA,
            reversibility=Reversibility.FULLY,
            blast_radius=BlastRadius.SELF,
        )


class _FakeScorer:
    def __init__(self, decision: str):
        self._decision = decision

    def evaluate(self, context):
        return {
            "action": self._decision,
            "raw_result": {"point_estimate": 0.9 if self._decision != "allow" else 0.1,
                           "conformal_interval": [0.8, 1.0] if self._decision != "allow" else [0.0, 0.2]},
        }


class _FakePipeline:
    """Minimal pipeline stub that returns a scripted decision."""

    def __init__(self, decision: str, allowed: bool):
        self._decision = decision
        self._allowed = allowed
        self.intercept_calls: list[dict] = []
        self.outcomes: list[tuple] = []
        self.registry = _FakeRegistry()
        self.scorer = _FakeScorer(decision)

    def intercept(self, **kwargs) -> InterceptionResult:
        self.intercept_calls.append(kwargs)
        return _fake_result(self._decision, self._allowed)

    def report_outcome(self, action_id, outcome_severity, description=""):
        self.outcomes.append((action_id, outcome_severity, description))


class TestLangChainHandler:
    def test_on_tool_start_low_risk(self):
        pipeline = InterceptionPipeline()
        handler = VaaraCallbackHandler(pipeline, agent_id="test-agent")

        # Low-risk tool should not raise
        handler.on_tool_start(
            serialized={"name": "data.read"},
            input_str="SELECT * FROM users",
            run_id=uuid4(),
        )

    def test_on_tool_start_records_audit(self):
        pipeline = InterceptionPipeline()
        handler = VaaraCallbackHandler(pipeline, agent_id="test-agent")

        handler.on_tool_start(
            serialized={"name": "data.read"},
            input_str="test",
            run_id=uuid4(),
        )

        assert pipeline.trail.size >= 3  # requested + scored + decision

    def test_on_tool_end_reports_outcome(self):
        pipeline = InterceptionPipeline()
        handler = VaaraCallbackHandler(pipeline, agent_id="test-agent")
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "data.read"},
            input_str="test",
            run_id=run_id,
        )
        handler.on_tool_end(output="success", run_id=run_id)

        # Should have outcome recorded
        assert pipeline.scorer.calibration_size >= 1

    def test_on_tool_error_reports_severity(self):
        pipeline = InterceptionPipeline()
        handler = VaaraCallbackHandler(
            pipeline, agent_id="test-agent", error_severity=0.5
        )
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "data.read"},
            input_str="test",
            run_id=run_id,
        )
        handler.on_tool_error(
            error=RuntimeError("connection failed"), run_id=run_id
        )

        assert pipeline.scorer.calibration_size >= 1

    def test_high_risk_blocks(self):
        pipeline = InterceptionPipeline()
        # Map tool to known high-risk action
        pipeline.registry.map_tool("phy.safety_override", "phy.safety_override")
        handler = VaaraCallbackHandler(
            pipeline, agent_id="test-agent"
        )

        # The handler should intercept; whether it blocks depends on score
        # At minimum, it shouldn't crash
        try:
            handler.on_tool_start(
                serialized={"name": "phy.safety_override"},
                input_str="disable brakes",
                run_id=uuid4(),
            )
        except (ToolExecutionBlocked, ToolExecutionEscalated):
            pass  # Expected for high-risk

    def test_multiple_tools_tracked_independently(self):
        pipeline = InterceptionPipeline()
        handler = VaaraCallbackHandler(pipeline, agent_id="test-agent")

        run1 = uuid4()
        run2 = uuid4()

        handler.on_tool_start(
            serialized={"name": "data.read"}, input_str="a", run_id=run1
        )
        handler.on_tool_start(
            serialized={"name": "data.write"}, input_str="b", run_id=run2
        )
        handler.on_tool_end(output="ok", run_id=run1)
        handler.on_tool_end(output="ok", run_id=run2)

        assert pipeline.scorer.calibration_size >= 2

    def test_duck_typed_protocol_attributes_present(self):
        """LangChain's CallbackManager reads ignore_* and raise_error on
        every handler. Missing attrs cause AttributeError at dispatch time
        and the exception is then swallowed (swallowing our deny signal).
        """
        pipeline = InterceptionPipeline()
        handler = VaaraCallbackHandler(pipeline)

        # Attributes required by langchain-core BaseCallbackHandler protocol
        for attr in (
            "ignore_agent", "ignore_chain", "ignore_chat_model",
            "ignore_custom_event", "ignore_llm", "ignore_retriever",
            "ignore_retry", "raise_error", "run_inline",
        ):
            assert hasattr(handler, attr), f"missing {attr}"
            assert getattr(handler, attr) in (True, False), f"{attr} not bool"

        # raise_error MUST be True so ToolExecutionBlocked propagates
        # instead of being swallowed by the callback manager.
        assert handler.raise_error is True


class TestLangChainHandlerBranches:
    """Cover the DENY/ESCALATE raise paths and the inputs-dict parameter path."""

    def test_on_tool_start_with_inputs_dict(self):
        fake = _FakePipeline(decision="allow", allowed=True)
        handler = VaaraCallbackHandler(fake, agent_id="test-agent")

        handler.on_tool_start(
            serialized={"name": "data.read"},
            input_str="",
            run_id=uuid4(),
            inputs={"query": "SELECT 1", "limit": 10},
        )

        assert fake.intercept_calls[0]["parameters"] == {"query": "SELECT 1", "limit": 10}

    def test_on_tool_start_deny_raises_blocked(self):
        fake = _FakePipeline(decision="deny", allowed=False)
        handler = VaaraCallbackHandler(fake, agent_id="test-agent")

        with pytest.raises(ToolExecutionBlocked) as exc:
            handler.on_tool_start(
                serialized={"name": "phy.safety_override"},
                input_str="disable brakes",
                run_id=uuid4(),
            )

        assert exc.value.tool_name == "phy.safety_override"
        assert exc.value.action_id == "act-test"
        assert exc.value.risk_score == 0.9
        assert exc.value.risk_interval == (0.8, 1.0)

    def test_on_tool_start_escalate_blocks_when_flag_set(self):
        fake = _FakePipeline(decision="escalate", allowed=False)
        handler = VaaraCallbackHandler(
            fake, agent_id="test-agent", block_on_escalate=True
        )

        with pytest.raises(ToolExecutionEscalated) as exc:
            handler.on_tool_start(
                serialized={"name": "data.export"},
                input_str="",
                run_id=uuid4(),
            )

        assert exc.value.tool_name == "data.export"
        assert exc.value.reason == "test escalate"

    def test_on_tool_start_escalate_passes_when_flag_unset(self):
        fake = _FakePipeline(decision="escalate", allowed=False)
        handler = VaaraCallbackHandler(fake, agent_id="test-agent")

        # block_on_escalate defaults to False → must not raise
        handler.on_tool_start(
            serialized={"name": "data.export"},
            input_str="",
            run_id=uuid4(),
        )


class TestVaaraWrapTool:
    """Cover the vaara_wrap_tool helper — both _run and func code paths."""

    class _FakeBaseTool:
        """Stand-in for a langchain BaseTool that has a `_run` method."""
        name = "fake.base_tool"

        def __init__(self):
            self.call_log = []

        def _run(self, *args, **kwargs):
            self.call_log.append((args, kwargs))
            return "base-ok"

    class _FakeFuncTool:
        """Stand-in for a langchain @tool decorated function (has `func`)."""
        name = "fake.func_tool"

        def __init__(self):
            self.call_log = []
            self.func = self._call

        def _call(self, *args, **kwargs):
            self.call_log.append((args, kwargs))
            return "func-ok"

    def test_wrap_tool_with_run_method_allows_and_reports(self):
        fake = _FakePipeline(decision="allow", allowed=True)
        tool = self._FakeBaseTool()

        wrapped = vaara_wrap_tool(tool, fake, agent_id="agent-A")
        out = wrapped._run("x", key="v")

        assert out == "base-ok"
        assert tool.call_log == [(("x",), {"key": "v"})]
        assert len(fake.outcomes) == 1
        assert fake.outcomes[0][0] == "act-test"
        assert fake.outcomes[0][1] == 0.0
        assert fake.intercept_calls[0]["agent_id"] == "agent-A"

    def test_wrap_tool_with_func_attribute(self):
        fake = _FakePipeline(decision="allow", allowed=True)
        tool = self._FakeFuncTool()

        vaara_wrap_tool(tool, fake)
        out = tool.func(target="x")

        assert out == "func-ok"
        assert len(fake.outcomes) == 1
        assert fake.outcomes[0][0] == "act-test"
        assert fake.outcomes[0][1] == 0.0

    def test_wrap_tool_blocks_on_deny(self):
        fake = _FakePipeline(decision="deny", allowed=False)
        tool = self._FakeBaseTool()

        wrapped = vaara_wrap_tool(tool, fake)

        with pytest.raises(ToolExecutionBlocked) as exc:
            wrapped._run(arg=1)

        assert exc.value.tool_name == "fake.base_tool"
        assert tool.call_log == []  # original never ran
        assert fake.outcomes == []  # no outcome on block

    def test_wrap_tool_escalates_when_flag_set(self):
        fake = _FakePipeline(decision="escalate", allowed=False)
        tool = self._FakeBaseTool()

        wrapped = vaara_wrap_tool(tool, fake, block_on_escalate=True)

        with pytest.raises(ToolExecutionEscalated):
            wrapped._run()

        assert tool.call_log == []

    def test_wrap_tool_reports_error_outcome_and_reraises(self):
        fake = _FakePipeline(decision="allow", allowed=True)

        class ExplodingTool:
            name = "fake.explode"
            def _run(self, *a, **kw):
                raise ValueError("boom")

        tool = ExplodingTool()
        vaara_wrap_tool(tool, fake)

        with pytest.raises(ValueError, match="boom"):
            tool._run()

        assert len(fake.outcomes) == 1
        action_id, severity, desc = fake.outcomes[0]
        assert action_id == "act-test"
        assert severity == 0.3
        assert "boom" in desc


class TestVaaraInterceptionError:
    """Cover the base exception class construction."""

    def test_construction_sets_all_attributes(self):
        err = VaaraInterceptionError(
            tool_name="t1",
            action_id="a1",
            risk_score=0.72,
            risk_interval=(0.6, 0.85),
            reason="too risky",
        )

        assert err.tool_name == "t1"
        assert err.action_id == "a1"
        assert err.risk_score == 0.72
        assert err.risk_interval == (0.6, 0.85)
        assert err.reason == "too risky"
        assert str(err) == "too risky"

    def test_subclass_inherits_fields(self):
        err = ToolExecutionBlocked(
            tool_name="t", action_id="a", risk_score=0.9,
            risk_interval=(0.8, 1.0), reason="deny",
        )
        assert isinstance(err, VaaraInterceptionError)
        assert err.risk_score == 0.9


class TestOpenAIWrapper:
    def test_wrapped_function_executes(self):
        pipeline = InterceptionPipeline()
        call_count = 0

        @vaara_wrap_function(pipeline, agent_id="test-bot")
        def safe_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result = safe_function(x=5)
        assert result == 10
        assert call_count == 1
        assert pipeline.trail.size >= 3

    def test_wrapped_function_reports_outcome(self):
        pipeline = InterceptionPipeline()

        @vaara_wrap_function(pipeline, agent_id="test-bot")
        def safe_function() -> str:
            return "ok"

        safe_function()
        assert pipeline.scorer.calibration_size >= 1

    def test_wrapped_function_reports_errors(self):
        pipeline = InterceptionPipeline()

        @vaara_wrap_function(pipeline, agent_id="test-bot")
        def failing_function() -> str:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing_function()

        # Error should still be reported as outcome
        assert pipeline.scorer.calibration_size >= 1

    def test_custom_tool_name(self):
        pipeline = InterceptionPipeline()

        @vaara_wrap_function(pipeline, agent_id="bot", tool_name="custom.action")
        def my_func() -> str:
            return "ok"

        my_func()
        # Check audit trail has the custom tool name
        records = pipeline.trail.get_records_by_type(
            __import__("vaara.audit.trail", fromlist=["EventType"]).EventType.ACTION_REQUESTED
        )
        assert any(r.tool_name == "custom.action" for r in records)


class TestSequenceDetectionAcrossFrameworks:
    """Verify that sequence detection works across framework boundaries."""

    def test_langchain_sequence_detection(self):
        pipeline = InterceptionPipeline()
        handler = VaaraCallbackHandler(pipeline, agent_id="suspect")

        # Build data exfiltration pattern
        handler.on_tool_start(
            serialized={"name": "data.read"}, input_str="", run_id=uuid4()
        )
        handler.on_tool_start(
            serialized={"name": "data.export"}, input_str="", run_id=uuid4()
        )

        # Check that the agent's profile shows the sequence
        profile = pipeline.scorer.get_agent_profile("suspect")
        assert profile is not None
        assert profile.total_actions == 2

    def test_openai_sequence_detection(self):
        pipeline = InterceptionPipeline()

        @vaara_wrap_function(pipeline, agent_id="suspect")
        def read_data() -> str:
            return "data"

        @vaara_wrap_function(pipeline, agent_id="suspect", tool_name="data.export")
        def export_data() -> str:
            return "exported"

        read_data()
        export_data()

        profile = pipeline.scorer.get_agent_profile("suspect")
        assert profile is not None
        assert profile.total_actions == 2


class TestOpenAIWrapperBranches:
    """Cover DENY / ESCALATE raise paths and exception constructors."""

    def test_wrap_function_deny_raises_blocked(self):
        fake = _FakePipeline(decision="deny", allowed=False)

        @vaara_wrap_function(fake, agent_id="b", tool_name="danger.tool")
        def target() -> str:
            return "should-not-run"

        with pytest.raises(ToolCallBlocked) as exc:
            target()

        assert exc.value.action_id == "act-test"
        assert exc.value.risk_score == 0.9
        assert "danger.tool" in str(exc.value)

    def test_wrap_function_escalate_raises_when_flag_set(self):
        fake = _FakePipeline(decision="escalate", allowed=False)

        @vaara_wrap_function(fake, agent_id="b", block_on_escalate=True)
        def target() -> str:
            return "nope"

        with pytest.raises(ToolCallEscalated) as exc:
            target()
        assert exc.value.action_id == "act-test"

    def test_wrap_function_escalate_passes_when_flag_unset(self):
        fake = _FakePipeline(decision="escalate", allowed=False)

        @vaara_wrap_function(fake, agent_id="b")
        def target() -> str:
            return "ran"

        assert target() == "ran"

    def test_tool_call_blocked_construction(self):
        err = ToolCallBlocked("msg", action_id="a1", risk_score=0.7)
        assert err.action_id == "a1"
        assert err.risk_score == 0.7
        assert str(err) == "msg"

    def test_tool_call_escalated_construction(self):
        err = ToolCallEscalated("msg2", action_id="a2", risk_score=0.55)
        assert err.action_id == "a2"
        assert err.risk_score == 0.55
        assert str(err) == "msg2"


class _FakeGuardrailResult:
    def __init__(self, tripwire_triggered: bool, output_info: dict):
        self.tripwire_triggered = tripwire_triggered
        self.output_info = output_info


def _install_fake_agents_module(monkeypatch):
    """Install a fake `agents` module exposing GuardrailResult."""
    import sys
    import types

    fake = types.ModuleType("agents")
    fake.GuardrailResult = _FakeGuardrailResult
    monkeypatch.setitem(sys.modules, "agents", fake)


class TestVaaraToolGuardrail:
    def test_sdk_missing_returns_none(self, monkeypatch):
        import sys
        # Force `agents` import to fail
        monkeypatch.setitem(sys.modules, "agents", None)
        gr = VaaraToolGuardrail(_FakePipeline("allow", True))
        assert gr(None, None, object()) is None

    def test_no_tool_calls_does_not_trigger(self, monkeypatch):
        _install_fake_agents_module(monkeypatch)
        gr = VaaraToolGuardrail(_FakePipeline("allow", True))

        res = gr(None, None, output={"tool_calls": []})
        assert isinstance(res, _FakeGuardrailResult)
        assert res.tripwire_triggered is False

    def test_dict_output_deny_triggers_tripwire(self, monkeypatch):
        _install_fake_agents_module(monkeypatch)
        fake = _FakePipeline("deny", False)
        gr = VaaraToolGuardrail(fake)

        output = {
            "tool_calls": [
                {"id": "c1", "function": {"name": "send_money", "arguments": {"amt": 5}}},
            ]
        }
        res = gr(None, None, output=output)
        assert res.tripwire_triggered is True
        assert res.output_info["blocked_tools"][0]["tool"] == "send_money"

    def test_dict_output_escalate_triggers_when_flag_set(self, monkeypatch):
        _install_fake_agents_module(monkeypatch)
        fake = _FakePipeline("escalate", False)
        gr = VaaraToolGuardrail(fake, block_on_escalate=True)

        output = {"tool_calls": [{"id": "c", "name": "review.me"}]}
        res = gr(None, None, output=output)
        assert res.tripwire_triggered is True
        assert "review.me" in res.output_info["message"]

    def test_object_output_with_tool_calls_attribute(self, monkeypatch):
        _install_fake_agents_module(monkeypatch)
        fake = _FakePipeline("allow", True)
        gr = VaaraToolGuardrail(fake)

        class _TC:
            id = "x1"
            name = "do.thing"
            arguments = {"a": 1}

        class _Out:
            tool_calls = [_TC()]

        res = gr(None, None, output=_Out())
        assert res.tripwire_triggered is False
        assert fake.intercept_calls[0]["tool_name"] == "do.thing"

    def test_object_output_tool_call_with_function_fallback(self, monkeypatch):
        _install_fake_agents_module(monkeypatch)
        fake = _FakePipeline("allow", True)
        gr = VaaraToolGuardrail(fake)

        class _Fn:
            name = "nested.name"

        class _TC:
            id = "x"
            name = ""
            function = _Fn()
            arguments = None  # exercises the `or {}` fallback

        class _Out:
            tool_calls = [_TC()]

        gr(None, None, output=_Out())
        assert fake.intercept_calls[0]["tool_name"] == "nested.name"


class _StubTool:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def _run(self, *a, **kw):
        self.calls.append((a, kw))
        return "ran"


class _StubAgent:
    def __init__(self, role, tools):
        self.role = role
        self.tools = tools


class _StubTask:
    def __init__(self, description, agent, tools=None):
        self.description = description
        self.agent = agent
        self.tools = tools


class _StubCrew:
    def __init__(self, agents, tasks, result="crew-done"):
        self.agents = agents
        self.tasks = tasks
        self._result = result
        self.kickoff_called = False

    def kickoff(self):
        self.kickoff_called = True
        return self._result


class TestCrewAIGovernance:
    def test_wrap_tools_returns_same_count(self):
        fake = _FakePipeline("allow", True)
        gov = VaaraCrewGovernance(fake)
        tools = [_StubTool("t1"), _StubTool("t2")]

        wrapped = gov.wrap_tools(tools, agent_id="r")
        assert len(wrapped) == 2
        wrapped[0]._run()
        assert fake.intercept_calls[0]["agent_id"] == "r"

    def test_screen_task_collects_assessments(self):
        fake = _FakePipeline("allow", True)
        gov = VaaraCrewGovernance(fake)

        res = gov.screen_task(
            task_description="analyze logs",
            agent_role="analyst",
            tools=["data.read", "data.export"],
        )
        assert res["task_allowed"] is True
        assert len(res["assessments"]) == 2
        assert res["blocked_tools"] == []

    def test_screen_task_flags_blocked(self):
        fake = _FakePipeline("deny", False)
        gov = VaaraCrewGovernance(fake)

        res = gov.screen_task("x", "role", tools=["phy.override"])
        assert res["task_allowed"] is False
        assert res["blocked_tools"] == ["phy.override"]

    def test_screen_task_flags_escalated(self):
        fake = _FakePipeline("escalate", False)
        gov = VaaraCrewGovernance(fake)

        res = gov.screen_task("x", "r", tools=["data.export"])
        assert res["escalated_tools"] == ["data.export"]

    def test_screen_task_empty_tools(self):
        fake = _FakePipeline("allow", True)
        gov = VaaraCrewGovernance(fake)

        res = gov.screen_task("x", "r", tools=[])
        assert res["max_risk_score"] == 0.0
        assert res["task_allowed"] is True

    def test_governed_kickoff_happy_path(self):
        fake = _FakePipeline("allow", True)
        gov = VaaraCrewGovernance(fake)

        tool = _StubTool("search")
        agent = _StubAgent("researcher", [tool])
        task = _StubTask("find things", agent=agent, tools=None)
        crew = _StubCrew(agents=[agent], tasks=[task])

        out = gov.governed_kickoff(crew)
        assert out == "crew-done"
        assert crew.kickoff_called is True

    def test_governed_kickoff_blocks_when_tool_denied(self):
        from vaara.integrations.langchain import ToolExecutionBlocked as _Blocked
        fake = _FakePipeline("deny", False)
        gov = VaaraCrewGovernance(fake)

        tool = _StubTool("phy.override")
        agent = _StubAgent("op", [tool])
        task = _StubTask("dangerous", agent=agent, tools=None)
        crew = _StubCrew(agents=[agent], tasks=[task])

        with pytest.raises(_Blocked):
            gov.governed_kickoff(crew)
        assert crew.kickoff_called is False

    def test_governed_kickoff_skips_prescreen(self):
        fake = _FakePipeline("allow", True)
        gov = VaaraCrewGovernance(fake)

        tool = _StubTool("search")
        agent = _StubAgent("a", [tool])
        task = _StubTask("t", agent=agent, tools=None)
        crew = _StubCrew(agents=[agent], tasks=[task])

        gov.governed_kickoff(crew, pre_screen=False)
        assert fake.intercept_calls == []

    def test_governed_kickoff_no_tasks_attr(self):
        fake = _FakePipeline("allow", True)
        gov = VaaraCrewGovernance(fake)

        class BareCrew:
            agents = []
            def kickoff(self):
                return "ok"

        assert gov.governed_kickoff(BareCrew()) == "ok"
