"""CrewAI integration — Vaara as task-level and tool-level governance.

CrewAI orchestrates multi-agent crews where agents have roles, goals, and
tools.  Vaara integrates at two levels:

1. **Tool-level** — wraps individual tools with risk scoring (same as
   LangChain since CrewAI uses LangChain tools internally).

2. **Task-level** — a pre-execution hook that scores the entire task
   context (agent role + task description + tools) before the crew
   starts working.  This catches dangerous task assignments before
   any tool calls happen.

Integration:

    from crewai import Crew, Agent, Task
    from vaara.integrations.crewai import VaaraCrewGovernance

    pipeline = InterceptionPipeline()
    gov = VaaraCrewGovernance(pipeline)

    # Wrap tools
    safe_tools = gov.wrap_tools(agent.tools, agent_id="research-agent")

    # Or wrap entire crew execution
    result = gov.governed_kickoff(crew)

Requires: crewai >= 0.30 (not a hard dependency — imported at runtime)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from vaara.pipeline import InterceptionPipeline
from vaara.integrations.langchain import (
    ToolExecutionBlocked,
    ToolExecutionEscalated,
)

logger = logging.getLogger(__name__)


class VaaraCrewGovernance:
    """Governance layer for CrewAI crews.

    Provides tool wrapping, task-level pre-screening, and crew-level
    execution governance.
    """

    def __init__(
        self,
        pipeline: InterceptionPipeline,
        block_on_escalate: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.block_on_escalate = block_on_escalate

    def wrap_tools(
        self,
        tools: list[Any],
        agent_id: str = "crewai-agent",
    ) -> list[Any]:
        """Wrap a list of CrewAI/LangChain tools with Vaara interception.

        CrewAI uses LangChain tools internally, so we reuse the
        LangChain tool wrapper.
        """
        from vaara.integrations.langchain import vaara_wrap_tool

        wrapped = []
        for tool in tools:
            wrapped.append(
                vaara_wrap_tool(
                    tool, self.pipeline, agent_id=agent_id,
                    block_on_escalate=self.block_on_escalate,
                )
            )
        return wrapped

    def screen_task(
        self,
        task_description: str,
        agent_role: str,
        tools: list[str],
        agent_id: str = "crewai-agent",
    ) -> dict:
        """Pre-screen a task before execution.

        Scores the task's risk based on what tools it might use and
        the agent's role.  Returns a dict with the assessment.

        This is a heuristic pre-check — the real scoring happens when
        tools are actually called.  But catching obviously dangerous
        task assignments early saves compute and reduces attack surface.
        """
        # Read-only scoring: hitting pipeline.intercept here would pollute
        # the audit trail with hypothetical events and bias the sequence
        # detector (a pre-screen that checks read → export would trip the
        # data_exfiltration pattern before any real action). Use the
        # scorer directly so no audit record is written.
        assessments = []
        for tool_name in tools:
            action_type = self.pipeline.registry.classify(tool_name)
            context = {
                "tool_name": tool_name,
                "agent_id": agent_id,
                "base_risk_score": action_type.base_risk_score,
                "reversibility": action_type.reversibility.value,
                "blast_radius": action_type.blast_radius.value,
                "framework": "crewai",
                "task_description": task_description[:500],
                "agent_role": agent_role,
                "prescreen": True,
            }
            # dry_run_evaluate is read-only: no profile mutation, no
            # sequence-detector warnings, no global history append.
            if hasattr(self.pipeline.scorer, "dry_run_evaluate"):
                scorer_result = self.pipeline.scorer.dry_run_evaluate(context)
            else:
                scorer_result = self.pipeline.scorer.evaluate(context)
            raw = scorer_result.get("raw_result", {})
            decision = scorer_result.get("action", "escalate")
            risk_score = raw.get("point_estimate", 0.5)
            assessments.append({
                "tool": tool_name,
                "decision": decision,
                "risk_score": risk_score,
                "action_id": None,  # no audit record for pre-screen
            })

        max_risk = max(a["risk_score"] for a in assessments) if assessments else 0.0
        any_blocked = any(a["decision"] == "deny" for a in assessments)
        any_escalated = any(a["decision"] == "escalate" for a in assessments)

        return {
            "task_allowed": not any_blocked,
            "max_risk_score": max_risk,
            "blocked_tools": [a["tool"] for a in assessments if a["decision"] == "deny"],
            "escalated_tools": [a["tool"] for a in assessments if a["decision"] == "escalate"],
            "assessments": assessments,
        }

    def governed_kickoff(
        self,
        crew: Any,
        pre_screen: bool = True,
    ) -> Any:
        """Execute a CrewAI crew with Vaara governance.

        Optionally pre-screens all tasks, then wraps all agent tools.
        Falls back gracefully if CrewAI is not installed.

        Args:
            crew: A CrewAI Crew instance.
            pre_screen: If True, pre-screen all tasks before execution.

        Returns:
            The crew's kickoff result.

        Raises:
            ToolExecutionBlocked: If a task's tools are all blocked.
        """
        # Pre-screen tasks
        if pre_screen and hasattr(crew, "tasks"):
            for task in crew.tasks:
                agent = getattr(task, "agent", None)
                agent_id = getattr(agent, "role", "crewai-agent") if agent else "crewai-agent"
                # Pick task.tools first; fall back to agent.tools only when
                # task.tools is absent (None). Using truthiness (`or`) would
                # treat tools=[] (explicitly no tools allowed) as absent,
                # silently inheriting the agent's full tool set — a governance
                # bypass for tasks intentionally scoped with no tools.
                task_tools = getattr(task, "tools", None)
                source_tools = (
                    task_tools if task_tools is not None
                    else (getattr(agent, "tools", []) if agent else [])
                ) or []
                tool_names = [getattr(t, "name", str(t)) for t in source_tools]
                if tool_names:
                    screening = self.screen_task(
                        task_description=getattr(task, "description", ""),
                        agent_role=agent_id,
                        tools=tool_names,
                        agent_id=agent_id,
                    )
                    if not screening["task_allowed"]:
                        raise ToolExecutionBlocked(
                            tool_name=", ".join(screening["blocked_tools"]),
                            action_id=screening["assessments"][0]["action_id"] or "prescreen",
                            risk_score=screening["max_risk_score"],
                            risk_interval=(0.0, 1.0),
                            reason=f"Task pre-screening blocked: {screening['blocked_tools']}",
                        )

        # Wrap all agent tools
        if hasattr(crew, "agents"):
            for agent in crew.agents:
                if hasattr(agent, "tools") and agent.tools:
                    agent_id = getattr(agent, "role", "crewai-agent")
                    agent.tools = self.wrap_tools(agent.tools, agent_id=agent_id)

        return crew.kickoff()
