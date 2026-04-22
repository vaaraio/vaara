"""LangChain integration — Vaara as a tool-calling callback handler.

LangChain's callback system fires events for every tool invocation.
This integration intercepts those events, scores them through Vaara's
pipeline, and blocks high-risk actions before they execute.

Two integration patterns:

1. **CallbackHandler** (recommended) — automatic, intercepts all tool calls:

    from vaara.integrations.langchain import VaaraCallbackHandler
    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline()
    handler = VaaraCallbackHandler(pipeline, agent_id="my-agent")

    agent = create_react_agent(llm, tools)
    result = agent.invoke(
        {"messages": [("user", "...")]},
        config={"callbacks": [handler]},
    )

2. **Tool wrapper** — wraps individual tools for fine-grained control:

    from vaara.integrations.langchain import vaara_wrap_tool

    safe_tool = vaara_wrap_tool(dangerous_tool, pipeline, agent_id="my-agent")

Both patterns produce the same audit trail and risk scoring.

Requires: langchain-core >= 0.2 (not a hard dependency — imported at runtime)
"""

from __future__ import annotations

import functools
import logging
import threading
from collections import OrderedDict
from typing import Any, Optional, Union
from uuid import UUID

from vaara._sanitize import json_safe as _json_safe
from vaara.pipeline import InterceptionPipeline

logger = logging.getLogger(__name__)

# Cap on in-flight run_id → action_id mappings. Orphan tool_starts
# (cancelled runs, streaming aborts, crashes) would otherwise leak
# unboundedly in a long-running agent. 10_000 covers realistic
# concurrency while capping worst-case memory at ~1MB.
_MAX_PENDING = 10_000


class VaaraCallbackHandler:
    """LangChain callback handler that intercepts tool calls via Vaara.

    Implements the LangChain BaseCallbackHandler protocol without importing
    it — duck typing is sufficient and avoids a hard dependency.

    On tool_start:
      - Classifies the tool call through Vaara's taxonomy
      - Scores risk with conformal prediction interval
      - If DENY: raises ToolExecutionBlocked (LangChain will catch it)
      - If ESCALATE: raises ToolExecutionEscalated (agent can retry or stop)
      - If ALLOW: proceeds normally, records in audit trail

    On tool_end:
      - Reports successful outcome (severity 0.0) to close the feedback loop

    On tool_error:
      - Reports error outcome (severity configurable) to the scorer
    """

    # ── LangChain BaseCallbackHandler protocol attributes ────────
    # Set as class defaults so we stay duck-typed (no hard dependency
    # on langchain-core) but still satisfy its CallbackManager, which
    # reads these flags on every handler it owns.
    ignore_agent: bool = False
    ignore_chain: bool = False
    ignore_chat_model: bool = False
    ignore_custom_event: bool = False
    ignore_llm: bool = False
    ignore_retriever: bool = False
    ignore_retry: bool = False
    # raise_error MUST be True: Vaara signals deny/escalate by raising
    # ToolExecutionBlocked / ToolExecutionEscalated from on_tool_start.
    # With raise_error=False (the LangChain default) the exception is
    # swallowed by the callback manager and the tool executes anyway.
    raise_error: bool = True
    run_inline: bool = False

    def __init__(
        self,
        pipeline: InterceptionPipeline,
        agent_id: str = "langchain-agent",
        session_id: str = "",
        error_severity: float = 0.3,
        block_on_escalate: bool = False,
    ) -> None:
        """
        Args:
            pipeline: The Vaara interception pipeline.
            agent_id: Identifier for this agent instance.
            session_id: Session identifier for grouping actions.
            error_severity: Default outcome severity for tool errors (0.0-1.0).
            block_on_escalate: If True, escalated actions are also blocked.
        """
        self.pipeline = pipeline
        self.agent_id = agent_id
        self.session_id = session_id
        self.error_severity = error_severity
        self.block_on_escalate = block_on_escalate
        # OrderedDict for FIFO eviction when _MAX_PENDING is exceeded.
        # Lock guards against free-threaded Python (3.13t+) where dict
        # ops are no longer GIL-atomic.
        self._pending: "OrderedDict[str, str]" = OrderedDict()
        self._pending_lock = threading.Lock()

    # ── LangChain callback protocol ──────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Intercept tool call before execution."""
        tool_name = serialized.get("name", "unknown_tool")

        # Build parameters from available sources, coercing to JSON-safe
        # forms so the audit trail hash (strict json.dumps) cannot crash.
        parameters: dict = {}
        if inputs:
            parameters = {str(k): _json_safe(v) for k, v in inputs.items()}
        elif input_str:
            parameters = {"input": _json_safe(input_str)}

        result = self.pipeline.intercept(
            agent_id=self.agent_id,
            tool_name=tool_name,
            parameters=parameters,
            context={
                "framework": "langchain",
                "tags": tags or [],
                "metadata": metadata or {},
            },
            session_id=self.session_id,
            parent_action_id=str(parent_run_id) if parent_run_id else None,
        )

        # Store action_id for outcome reporting. Cap size to protect
        # against orphan tool_starts (cancelled runs, streaming aborts)
        # from leaking memory in long-running agents.
        with self._pending_lock:
            self._pending[str(run_id)] = result.action_id
            while len(self._pending) > _MAX_PENDING:
                evicted_rid, _ = self._pending.popitem(last=False)
                logger.warning(
                    "VaaraCallbackHandler._pending full (%d); evicted run_id %s",
                    _MAX_PENDING, evicted_rid,
                )

        if not result.allowed:
            if result.decision == "deny":
                raise ToolExecutionBlocked(
                    tool_name=tool_name,
                    action_id=result.action_id,
                    risk_score=result.risk_score,
                    risk_interval=result.risk_interval,
                    reason=result.reason,
                )
            elif result.decision == "escalate" and self.block_on_escalate:
                raise ToolExecutionEscalated(
                    tool_name=tool_name,
                    action_id=result.action_id,
                    risk_score=result.risk_score,
                    risk_interval=result.risk_interval,
                    reason=result.reason,
                )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Report successful tool execution."""
        with self._pending_lock:
            action_id = self._pending.pop(str(run_id), None)
        if action_id:
            # raise_error=True is required for ToolExecutionBlocked to
            # surface through LangChain's callback manager. That same
            # flag causes ANY exception inside an on_* handler to
            # propagate and break the parent dispatch (chain, agent,
            # streaming token). A report_outcome failure (DB / scorer
            # transient) would then kill the enclosing run.
            try:
                self.pipeline.report_outcome(
                    action_id=action_id,
                    outcome_severity=0.0,
                    description="Tool completed successfully",
                )
            except Exception:
                logger.exception(
                    "report_outcome failed in on_tool_end for action_id=%s",
                    action_id,
                )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Report tool execution error."""
        with self._pending_lock:
            action_id = self._pending.pop(str(run_id), None)
        if action_id:
            # Same raise_error=True isolation as on_tool_end — never
            # shadow the tool's own error with a report_outcome failure.
            try:
                self.pipeline.report_outcome(
                    action_id=action_id,
                    outcome_severity=self.error_severity,
                    description=f"Tool error: {error}",
                )
            except Exception:
                logger.exception(
                    "report_outcome failed in on_tool_error for action_id=%s",
                    action_id,
                )

    # Unused callbacks — must exist for protocol compliance.
    # LangChain's _ahandle_event_for_handler does getattr(handler, event_name)
    # and any AttributeError is caught, logged, and RE-RAISED when
    # raise_error=True (which we need for ToolExecutionBlocked). Every
    # BaseCallbackHandler event LangChain dispatches must therefore exist
    # as a no-op, or the first chat model / retriever / stream-token
    # event will blow up the agent. Keep this list aligned with
    # langchain_core.callbacks.BaseCallbackHandler's event surface.
    def on_llm_start(self, *args, **kwargs): pass
    def on_llm_end(self, *args, **kwargs): pass
    def on_llm_error(self, *args, **kwargs): pass
    def on_llm_new_token(self, *args, **kwargs): pass
    def on_chat_model_start(self, *args, **kwargs): pass
    def on_chain_start(self, *args, **kwargs): pass
    def on_chain_end(self, *args, **kwargs): pass
    def on_chain_error(self, *args, **kwargs): pass
    def on_agent_action(self, *args, **kwargs): pass
    def on_agent_finish(self, *args, **kwargs): pass
    def on_retriever_start(self, *args, **kwargs): pass
    def on_retriever_end(self, *args, **kwargs): pass
    def on_retriever_error(self, *args, **kwargs): pass
    def on_text(self, *args, **kwargs): pass
    def on_retry(self, *args, **kwargs): pass
    def on_custom_event(self, *args, **kwargs): pass


# ── Tool wrapper ──────────────────────────────────────────────────────────

def vaara_wrap_tool(
    tool: Any,
    pipeline: InterceptionPipeline,
    agent_id: str = "langchain-agent",
    block_on_escalate: bool = False,
) -> Any:
    """Wrap a LangChain tool with Vaara interception.

    Returns a new tool that checks Vaara before executing.
    Works with both @tool functions and BaseTool subclasses.

    Example::

        from langchain_core.tools import tool
        from vaara.integrations.langchain import vaara_wrap_tool

        @tool
        def send_email(to: str, body: str) -> str:
            '''Send an email.'''
            return do_send(to, body)

        safe_email = vaara_wrap_tool(send_email, pipeline)
    """
    # Idempotency: re-wrapping the same tool instance would stack
    # interceptors and double every scoring + outcome call, skewing
    # MWU weights. CrewAI's governed_kickoff can re-enter this path
    # when a crew is kicked off twice — guard against it here.
    if getattr(tool, "_vaara_wrapped", False):
        return tool

    has_run = hasattr(tool, "_run")
    has_func = hasattr(tool, "func")
    has_arun = hasattr(tool, "_arun")
    if not (has_run or has_func or has_arun):
        raise TypeError(
            f"vaara_wrap_tool: {tool!r} has no _run, _arun, or func attribute; "
            "pass a LangChain BaseTool subclass or @tool-decorated function."
        )

    original_run = tool._run if has_run else (tool.func if has_func else None)
    original_arun = tool._arun if has_arun else None

    # Fallback sentinel path: when tool is a frozen pydantic model we
    # stamp _vaara_wrapped onto the wrapper function instead of the
    # tool, so the tool-level check above misses it on re-entry. Check
    # the current _run/_arun/func too before re-wrapping.
    if original_run is not None and getattr(original_run, "_vaara_wrapped", False):
        return tool
    if original_arun is not None and getattr(original_arun, "_vaara_wrapped", False):
        return tool

    tool_name_str = getattr(tool, "name", str(tool))

    def _build_params(args: tuple, kwargs: dict) -> dict:
        # Capture positional AND keyword args. The previous
        # `if kwargs else {args: ...}` branch silently dropped
        # positionals whenever any kwarg was present — fine for
        # LangChain's _run(**inputs) convention but wrong for direct
        # callers.
        safe_params: dict = {
            str(k): _json_safe(v) for k, v in kwargs.items()
        }
        if args:
            safe_params["args"] = _json_safe(list(args))
        return safe_params

    def _pre_intercept(args: tuple, kwargs: dict):
        result = pipeline.intercept(
            agent_id=agent_id,
            tool_name=tool_name_str,
            parameters=_build_params(args, kwargs),
            context={"framework": "langchain", "wrapper": True},
        )
        if result.decision == "deny":
            raise ToolExecutionBlocked(
                tool_name=tool_name_str,
                action_id=result.action_id,
                risk_score=result.risk_score,
                risk_interval=result.risk_interval,
                reason=result.reason,
            )
        if result.decision == "escalate" and block_on_escalate:
            raise ToolExecutionEscalated(
                tool_name=tool_name_str,
                action_id=result.action_id,
                risk_score=result.risk_score,
                risk_interval=result.risk_interval,
                reason=result.reason,
            )
        return result

    def _safe_report(action_id: str, severity: float, desc: str = "") -> None:
        # Isolate report_outcome failures. A pipeline/backend error must
        # NOT shadow the wrapped tool's return value (success) or mask
        # the tool's original exception (error). Callers of governance-
        # wrapped tools only care about the tool outcome; a failed
        # learning-signal write is a logger.exception concern.
        try:
            pipeline.report_outcome(action_id, severity, desc) if desc else pipeline.report_outcome(action_id, severity)
        except Exception:
            logger.exception(
                "report_outcome failed for action_id=%s; "
                "preserving tool outcome",
                action_id,
            )

    def wrapped_run(*args, **kwargs):
        if original_run is None:
            raise AttributeError(f"{tool_name_str}: no sync _run/func to invoke")
        result = _pre_intercept(args, kwargs)
        try:
            output = original_run(*args, **kwargs)
        except Exception as e:
            _safe_report(result.action_id, 0.3, str(e))
            raise
        _safe_report(result.action_id, 0.0)
        return output

    async def wrapped_arun(*args, **kwargs):
        # Mirror sync wrapper for async LangChain paths (agent.ainvoke,
        # AgentExecutor async mode). Without this, any tool with a
        # non-trivial _arun silently bypasses Vaara governance entirely.
        if original_arun is None:
            # Fall back to sync if caller invokes async path on a
            # sync-only tool; LangChain normally wouldn't, but defensive.
            if original_run is None:
                raise AttributeError(f"{tool_name_str}: no _arun or _run to invoke")
            return wrapped_run(*args, **kwargs)
        result = _pre_intercept(args, kwargs)
        try:
            output = await original_arun(*args, **kwargs)
        except Exception as e:
            _safe_report(result.action_id, 0.3, str(e))
            raise
        _safe_report(result.action_id, 0.0)
        return output

    # Stamp wrappers BEFORE attaching them to the tool. Without this, two
    # concurrent wraps (e.g., two CrewAI `governed_kickoff` calls on crews
    # sharing a tool instance) can double-wrap: T1 assigns tool._run =
    # wrapped_T1 but hasn't stamped anything yet, T2 reads tool._run →
    # wrapped_T1, fails the idempotency checks at lines above (both
    # tool._vaara_wrapped and wrapped_T1._vaara_wrapped are False),
    # builds wrapped_T2 around wrapped_T1, and stores it — giving two
    # layers of interception that MWU-weight and outcome-report twice.
    # Stamping wrappers first guarantees any concurrent re-read of
    # tool._run sees an already-marked wrapper and short-circuits.
    wrapped_run = functools.wraps(original_run)(wrapped_run) if original_run is not None else wrapped_run
    wrapped_arun = functools.wraps(original_arun)(wrapped_arun) if original_arun is not None else wrapped_arun
    wrapped_run._vaara_wrapped = True  # type: ignore[attr-defined]
    wrapped_arun._vaara_wrapped = True  # type: ignore[attr-defined]

    if has_run:
        tool._run = wrapped_run
    elif has_func:
        tool.func = wrapped_run
    if has_arun:
        tool._arun = wrapped_arun

    try:
        tool._vaara_wrapped = True
    except AttributeError:
        # Frozen pydantic models (rare on BaseTool) can't take new
        # attributes; the wrapper-level stamps above are sufficient.
        pass
    return tool


# ── Exceptions ────────────────────────────────────────────────────────────

class VaaraInterceptionError(Exception):
    """Base exception for Vaara interception events."""
    def __init__(
        self,
        tool_name: str,
        action_id: str,
        risk_score: float,
        risk_interval: tuple[float, float],
        reason: str,
    ):
        self.tool_name = tool_name
        self.action_id = action_id
        self.risk_score = risk_score
        self.risk_interval = risk_interval
        self.reason = reason
        super().__init__(reason)


class ToolExecutionBlocked(VaaraInterceptionError):
    """Raised when Vaara blocks a tool call due to high risk."""
    pass


class ToolExecutionEscalated(VaaraInterceptionError):
    """Raised when Vaara escalates a tool call for human review."""
    pass
