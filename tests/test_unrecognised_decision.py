"""A verdict the pipeline cannot read is refused, and the record names it.

From an implementer's report (agentbaseline#49): a response builder that
branched on two of four verdicts passed a refusal spelled in another
vocabulary through as a success. Every spelling of a refusal must refuse,
anything outside the vocabulary must refuse, and the record must say which
value was not recognised, so a refusal spelled differently can be told
apart from no answer at all.
"""

from __future__ import annotations

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline

_MISSING = object()


class _Fixed:
    def __init__(self, action) -> None:
        self._action = action

    def evaluate(self, context):
        out = {"reason": "stub", "raw_result": {
            "point_estimate": 0.1, "conformal_interval": [0.0, 0.2], "signals": {}}}
        if self._action is not _MISSING:
            out["action"] = self._action
        return out


def _run(action):
    trail = SQLiteAuditBackend(":memory:").load_trail()
    result = InterceptionPipeline(trail=trail, scorer=_Fixed(action)).intercept(
        agent_id="a", tool_name="data.read", parameters={})
    record = [r for r in trail._records if r.event_type == EventType.ACTION_BLOCKED][-1]
    return result, record


@pytest.mark.parametrize("action", ["deny", "DENY", "Deny", " deny "])
def test_every_spelling_of_a_refusal_refuses(action):
    result, record = _run(action)
    assert not result.allowed
    assert record.data["violation_type"] != "invalid_decision"


@pytest.mark.parametrize("action,named", [
    ("block", "'block'"), ("maybe", "'maybe'"), ("", "''"), (None, "no decision"),
    (_MISSING, "no decision"), (7, "'7'"),
])
def test_an_unreadable_verdict_refuses_and_is_named(action, named):
    result, record = _run(action)
    assert not result.allowed
    assert record.data["violation_type"] == "invalid_decision"
    assert named in record.data["reason"]
    assert "scorer: stub" in record.data["reason"]
