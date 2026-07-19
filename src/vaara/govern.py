# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The one-liner.

A whole governance engine behind a single embeddable call: one decorator turns
any Python function into a governed tool call — classified, risk-scored with a
conformal interval, decided allow/deny/escalate, written to a hash-chained
audit trail — with zero setup.

    import vaara

    @vaara.govern
    def transfer_funds(to: str, amount: float) -> str:
        ...

The first decorated call lazily builds a process-wide default
``InterceptionPipeline`` (the same engine the framework adapters and the MCP
proxy drive). It fails closed: any non-``allow`` decision (``deny`` or
``escalate``) raises ``vaara.Blocked`` before the body runs — check
``.decision`` to route an escalation to review. An ``allow`` runs the function
and reports the outcome back so the scorer keeps calibrating. Reach for the
explicit ``InterceptionPipeline`` (signed receipts,
the MCP proxy, the credential gateway, compliance export) when you need more.
This is the floor, not the ceiling.
"""

from __future__ import annotations

import functools
import inspect
import threading
from typing import Any, Callable, Optional, TypeVar, cast

from vaara.pipeline import InterceptionPipeline, InterceptionResult

__all__ = ["govern", "Blocked", "default_pipeline", "set_default_pipeline"]

F = TypeVar("F", bound=Callable[..., Any])


class Blocked(Exception):
    """Raised when the governed decision is ``deny``.

    Carries the fields the audit trail recorded (``decision``, ``reason``,
    ``risk_score``, ``risk_interval``, ``action_id``) so a caller that catches
    it can surface the reason without reaching into the pipeline.
    """

    def __init__(self, result: InterceptionResult) -> None:
        self.result = result
        self.action_id = result.action_id
        self.decision = result.decision
        self.reason = result.reason
        self.risk_score = result.risk_score
        self.risk_interval = result.risk_interval
        atype = getattr(result.action_type, "name", result.action_type)
        super().__init__(
            f"vaara blocked {atype} ({result.decision}, "
            f"risk={result.risk_score:.3f}): {result.reason}"
        )


# Process-wide default pipeline, built on first use so importing vaara stays
# cheap and side-effect free. The first governed call builds it exactly once.
_default_pipeline: Optional[InterceptionPipeline] = None
_default_lock = threading.Lock()


def default_pipeline() -> InterceptionPipeline:
    """Return the lazily-built process-wide pipeline used by ``@govern``."""
    global _default_pipeline
    if _default_pipeline is None:
        with _default_lock:
            if _default_pipeline is None:
                _default_pipeline = InterceptionPipeline()
    return _default_pipeline


def set_default_pipeline(pipeline: InterceptionPipeline) -> None:
    """Wire a configured pipeline (signing trail, tuned scorer) under every
    ``@govern`` in the process, once at startup, without touching decorators."""
    global _default_pipeline
    with _default_lock:
        _default_pipeline = pipeline


def _arguments(func: Callable[..., Any], args: tuple, kwargs: dict) -> dict:
    """Best-effort bound-argument map for the audit record. Drops a leading
    ``self``/``cls``; falls back to positional indices when the signature
    cannot be bound. The pipeline sanitizes and size-caps whatever it gets."""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        params = dict(bound.arguments)
        first = next(iter(params), None)
        if first in ("self", "cls"):
            params.pop(first)
        return params
    except (TypeError, ValueError):
        return {"args": list(args), **kwargs}


_shadow_singleton: Optional[InterceptionPipeline] = None
_shadow_lock = threading.Lock()


def _shadow_pipeline() -> InterceptionPipeline:
    global _shadow_singleton
    if _shadow_singleton is None:
        with _shadow_lock:
            if _shadow_singleton is None:
                _shadow_singleton = InterceptionPipeline(enforce=False)
    return _shadow_singleton


def govern(
    func: Optional[F] = None,
    *,
    agent_id: str = "default",
    tool_name: Optional[str] = None,
    shadow: bool = False,
    pipeline: Optional[InterceptionPipeline] = None,
) -> Any:
    """Govern a function with one decorator.

    Bare, zero config::

        @vaara.govern
        def send_email(to, body): ...

    Or configured::

        @vaara.govern(agent_id="billing-agent", tool_name="tx.transfer")
        def transfer(to, amount): ...

    Each call is classified, scored, and decided before the body runs. ``deny``
    raises ``vaara.Blocked``; ``allow`` runs the function and reports the
    outcome back to the scorer. Either way the decision lands in the trail.

    Args:
        agent_id: Identity recorded against the action ("default" if unset).
        tool_name: Name classified and recorded; defaults to the function's
            qualified name. Pass a taxonomy name (``tx.transfer``) to route it.
        shadow: Non-enforcing mode — classify, score, and audit every call but
            block nothing, so you can collect evidence of what *would* be
            denied before flipping to enforcing. Ignored if ``pipeline`` given.
        pipeline: Explicit ``InterceptionPipeline`` instead of the default.
    """

    def decorate(fn: F) -> F:
        resolved_name = tool_name or cast(str, getattr(fn, "__qualname__", "function"))

        def _pipe() -> InterceptionPipeline:
            if pipeline is not None:
                return pipeline
            return _shadow_pipeline() if shadow else default_pipeline()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            pipe = _pipe()
            result = pipe.intercept(
                agent_id=agent_id,
                tool_name=resolved_name,
                parameters=_arguments(fn, args, kwargs),
            )
            if not result.allowed:
                raise Blocked(result)
            try:
                outcome = fn(*args, **kwargs)
            except Exception:
                # Action ran and failed: report high severity so the scorer
                # learns this call shape went wrong, then re-raise untouched.
                pipe.report_outcome(result.action_id, 1.0, description="raised exception")
                raise
            pipe.report_outcome(result.action_id, 0.0)
            return outcome

        return wrapper  # type: ignore[return-value]

    return decorate(func) if func is not None else decorate
