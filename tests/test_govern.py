"""Unit tests for the ``@vaara.govern`` one-liner decorator.

The decorator's contract: classify-score-decide before the body runs, fail
closed on any non-``allow`` decision, run and report the outcome on ``allow``.
These tests pin that contract with an injected fake pipeline so the decision is
deterministic, then a real-pipeline smoke test in shadow mode proves the actual
``import vaara`` wiring runs end to end.
"""

from __future__ import annotations

import pytest

import vaara
from vaara.govern import Blocked, govern, set_default_pipeline


class _FakeResult:
    """Minimal stand-in for InterceptionResult carrying the fields the
    decorator and ``Blocked`` read."""

    def __init__(self, allowed: bool, decision: str, reason: str) -> None:
        self.allowed = allowed
        self.action_id = "act-test"
        self.decision = decision
        self.reason = reason
        self.risk_score = 0.5
        self.risk_interval = (0.4, 0.6)
        self.action_type = "tool_call"


class _FakePipeline:
    """Records what it was asked to intercept/report and returns a fixed
    decision, so the decorator contract can be tested in isolation."""

    def __init__(self, allowed: bool) -> None:
        self._result = _FakeResult(
            allowed,
            decision="allow" if allowed else "deny",
            reason="ok" if allowed else "policy denied",
        )
        self.reported: list = []
        self.last: dict = {}

    def intercept(self, *, agent_id, tool_name, parameters):
        self.last = {"agent_id": agent_id, "tool_name": tool_name, "parameters": parameters}
        return self._result

    def report_outcome(self, action_id, severity, description=None):
        self.reported.append((action_id, severity, description))


@pytest.fixture(autouse=True)
def _reset_module_singletons():
    """Keep tests isolated: the default and shadow pipelines are process-wide
    singletons built lazily, so reset them around each test."""
    import vaara.govern as g

    g._default_pipeline = None
    g._shadow_singleton = None
    yield
    g._default_pipeline = None
    g._shadow_singleton = None


def test_govern_and_blocked_exposed_on_package():
    # This is the export that was sitting uncommitted: `import vaara` must
    # surface the decorator and the exception.
    assert callable(vaara.govern)
    assert issubclass(vaara.Blocked, Exception)


def test_allow_runs_body_and_reports_success():
    pipe = _FakePipeline(allowed=True)

    @govern(pipeline=pipe)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert pipe.reported == [("act-test", 0.0, None)]
    assert pipe.last["tool_name"].endswith("add")
    assert pipe.last["parameters"] == {"a": 2, "b": 3}


def test_deny_raises_blocked_and_body_never_runs():
    pipe = _FakePipeline(allowed=False)
    ran: list = []

    @govern(pipeline=pipe)
    def danger():
        ran.append(True)
        return "should not happen"

    with pytest.raises(Blocked) as excinfo:
        danger()

    assert ran == []  # fails closed: body never executed
    assert excinfo.value.decision == "deny"
    assert "policy denied" in excinfo.value.reason
    assert pipe.reported == []  # nothing to report when blocked


def test_failing_body_reports_high_severity_then_reraises():
    pipe = _FakePipeline(allowed=True)

    @govern(pipeline=pipe)
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        boom()

    assert pipe.reported == [("act-test", 1.0, "raised exception")]


def test_bare_decorator_form_uses_default_pipeline():
    pipe = _FakePipeline(allowed=True)
    set_default_pipeline(pipe)

    @govern
    def greet(name):
        return f"hi {name}"

    assert greet("h") == "hi h"
    assert pipe.reported  # routed through the wired default pipeline


def test_shadow_mode_runs_with_real_pipeline():
    # shadow=True builds a real InterceptionPipeline(enforce=False); it must
    # never block, so the wrapped function runs end to end through real code.
    @govern(shadow=True)
    def compute():
        return 7

    assert compute() == 7
