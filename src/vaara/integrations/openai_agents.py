"""OpenAI Agents SDK integration — Vaara as a guardrail.

OpenAI's Agents SDK (March 2025) uses a guardrails pattern where
InputGuardrail / OutputGuardrail functions run before/after the agent.
Vaara plugs in as an OutputGuardrail on tool calls.

For the Responses API direct usage, Vaara wraps the function-call handler.

Integration patterns:

1. **Guardrail** (recommended for Agents SDK):

    from vaara.integrations.openai_agents import VaaraToolGuardrail
    from agents import Agent

    pipeline = InterceptionPipeline()
    guardrail = VaaraToolGuardrail(pipeline)

    agent = Agent(
        name="my-agent",
        tools=[...],
        output_guardrails=[guardrail],
    )

2. **Function wrapper** (for Responses API / custom loops):

    from vaara.integrations.openai_agents import vaara_wrap_function

    @vaara_wrap_function(pipeline, agent_id="my-bot")
    def transfer_funds(to: str, amount: float) -> str:
        return execute_transfer(to, amount)

Requires: openai >= 1.0 (not a hard dependency — imported at runtime)
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable, Optional

from vaara.pipeline import InterceptionPipeline
from vaara.integrations.langchain import _json_safe

logger = logging.getLogger(__name__)


def _parse_arguments(raw: Any) -> Any:
    """Normalize tool-call arguments into a structured value.

    OpenAI's Chat Completions and Responses APIs return ``function.arguments``
    as a **JSON string**, while the Agents SDK sometimes already parses it
    into a dict. We need structured params reaching the scorer and audit
    trail — otherwise the entire arg bag collapses into a single opaque
    string key (``{"args": "{...}"}``) and risk signals lose all structure.

    Falls back to ``{}`` on parse errors so a malformed string from a
    buggy client cannot crash the guardrail.
    """
    if raw is None:
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:
            return {}
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON — wrap so caller still sees the raw text.
            return {"raw_arguments": raw}
    return {"raw_arguments": raw}


class VaaraToolGuardrail:
    """OpenAI Agents SDK guardrail that intercepts tool calls.

    Implements the guardrail protocol: a callable that receives the
    agent output and returns a GuardrailResult (or raises to block).

    When used as an output_guardrail, it inspects the agent's planned
    tool calls and scores each one through Vaara before execution.
    """

    def __init__(
        self,
        pipeline: InterceptionPipeline,
        agent_id: str = "openai-agent",
        session_id: str = "",
        block_on_escalate: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.agent_id = agent_id
        self.session_id = session_id
        self.block_on_escalate = block_on_escalate

    def __call__(self, context: Any, agent: Any, output: Any) -> Any:
        """Guardrail evaluation — called by the Agents SDK runtime.

        Inspects tool_calls in the output and scores each one.
        Returns a GuardrailResult with tripwire_triggered=True to block.
        """
        try:
            from agents import GuardrailResult
        except ImportError:
            logger.warning("openai-agents SDK not installed, guardrail is a no-op")
            return None

        # Extract tool calls from the agent output
        tool_calls = self._extract_tool_calls(output)
        if not tool_calls:
            return GuardrailResult(tripwire_triggered=False, output_info={})

        blocked = []
        for tc in tool_calls:
            raw_args = tc.get("arguments", {}) or {}
            safe_args = (
                {str(k): _json_safe(v) for k, v in raw_args.items()}
                if isinstance(raw_args, dict) else {"args": _json_safe(raw_args)}
            )
            result = self.pipeline.intercept(
                agent_id=self.agent_id,
                tool_name=tc["name"],
                parameters=safe_args,
                context={
                    "framework": "openai_agents",
                    "call_id": tc.get("id", ""),
                },
                session_id=self.session_id,
            )

            if result.decision == "deny":
                blocked.append({
                    "tool": tc["name"],
                    "action_id": result.action_id,
                    "risk_score": result.risk_score,
                    "reason": result.reason,
                })
            elif result.decision == "escalate" and self.block_on_escalate:
                blocked.append({
                    "tool": tc["name"],
                    "action_id": result.action_id,
                    "risk_score": result.risk_score,
                    "reason": f"Escalated: {result.reason}",
                })

        if blocked:
            return GuardrailResult(
                tripwire_triggered=True,
                output_info={
                    "blocked_tools": blocked,
                    "message": (
                        f"Vaara blocked {len(blocked)} tool call(s): "
                        + ", ".join(b["tool"] for b in blocked)
                    ),
                },
            )

        return GuardrailResult(tripwire_triggered=False, output_info={})

    @staticmethod
    def _extract_tool_calls(output: Any) -> list[dict]:
        """Extract tool calls from various output formats.

        Handles both the Agents SDK object-style output and the Responses /
        Chat Completions dict-style output. In both cases ``arguments`` may
        arrive as a JSON string — parse it so structured params reach the
        scorer; otherwise the scorer sees only ``{"raw_arguments": "..."}``.
        """
        calls = []

        # OpenAI Agents SDK output format
        if hasattr(output, "tool_calls"):
            for tc in output.tool_calls:
                tc_function = getattr(tc, "function", None)
                # Arguments may live on tc directly (Agents SDK) or on
                # tc.function (Chat Completions style objects).
                raw_args = getattr(tc, "arguments", None)
                if raw_args is None and tc_function is not None:
                    raw_args = getattr(tc_function, "arguments", None)
                calls.append({
                    "id": getattr(tc, "id", "") or "",
                    "name": (
                        getattr(tc, "name", "")
                        or getattr(tc_function, "name", "")
                        or ""
                    ),
                    "arguments": _parse_arguments(raw_args),
                })
        # Responses / Chat Completions dict format
        elif isinstance(output, dict):
            for tc in output.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    fn = {}
                name = fn.get("name") or tc.get("name") or ""
                raw_args = fn.get("arguments")
                if raw_args is None:
                    raw_args = tc.get("arguments")
                calls.append({
                    "id": tc.get("id", "") or "",
                    "name": name,
                    "arguments": _parse_arguments(raw_args),
                })

        return calls


# ── Function wrapper ──────────────────────────────────────────────────────

def vaara_wrap_function(
    pipeline: InterceptionPipeline,
    agent_id: str = "openai-agent",
    tool_name: Optional[str] = None,
    block_on_escalate: bool = False,
) -> Callable:
    """Decorator that wraps a function with Vaara interception.

    For use with the Responses API or custom tool-calling loops.

    Example::

        @vaara_wrap_function(pipeline, agent_id="my-bot")
        def send_funds(to: str, amount: float) -> str:
            return blockchain.transfer(to, amount)

        # Now send_funds checks Vaara before executing
        result = send_funds(to="0xabc", amount=1000)
    """
    def decorator(func: Callable) -> Callable:
        name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Capture both positional and keyword args; the previous
            # `if kwargs else {args: ...}` branch silently dropped
            # positionals whenever any kwarg was present.
            safe_params: dict = {
                str(k): _json_safe(v) for k, v in kwargs.items()
            }
            if args:
                safe_params["args"] = _json_safe(list(args))
            result = pipeline.intercept(
                agent_id=agent_id,
                tool_name=name,
                parameters=safe_params,
                context={"framework": "openai_responses", "wrapper": True},
            )

            if result.decision == "deny":
                raise ToolCallBlocked(
                    f"Vaara blocked {name}: {result.reason}",
                    action_id=result.action_id,
                    risk_score=result.risk_score,
                )
            if result.decision == "escalate" and block_on_escalate:
                raise ToolCallEscalated(
                    f"Vaara escalated {name}: {result.reason}",
                    action_id=result.action_id,
                    risk_score=result.risk_score,
                )

            # Both branches isolate report_outcome failures. A pipeline /
            # backend error inside report_outcome must NOT shadow the
            # wrapped tool's return value (success path) or mask the
            # tool's original exception (error path) — the caller only
            # cares about the tool outcome; a failed learning-signal
            # write is a logger.exception concern, not a raise concern.
            try:
                output = func(*args, **kwargs)
            except Exception as e:
                try:
                    pipeline.report_outcome(result.action_id, 0.3, str(e))
                except Exception:
                    logger.exception(
                        "report_outcome failed for action_id=%s; "
                        "original tool exception preserved",
                        result.action_id,
                    )
                raise
            try:
                pipeline.report_outcome(result.action_id, 0.0)
            except Exception:
                logger.exception(
                    "report_outcome failed for action_id=%s; "
                    "tool return value preserved",
                    result.action_id,
                )
            return output

        return wrapper
    return decorator


# ── Exceptions ────────────────────────────────────────────────────────────

class ToolCallBlocked(Exception):
    """Raised when Vaara blocks a function call."""
    def __init__(self, message: str, action_id: str, risk_score: float):
        self.action_id = action_id
        self.risk_score = risk_score
        super().__init__(message)


class ToolCallEscalated(Exception):
    """Raised when Vaara escalates a function call."""
    def __init__(self, message: str, action_id: str, risk_score: float):
        self.action_id = action_id
        self.risk_score = risk_score
        super().__init__(message)
