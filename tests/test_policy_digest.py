"""Every decision names the decision state it was made under.

A receipt that says "the policy decided" cannot be checked against anything.
Each decision record carries ``policy_digest`` over the engine's decision
state, and the first decision after that state changes carries the state
itself, so a reader holding the trail can recompute every digest.
"""

from __future__ import annotations

from vaara.audit.decision_receipts import DecisionReceiptSink
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline
from vaara.policy_state import SCHEMA, digest

_DECISIONS = (EventType.DECISION_MADE, EventType.ACTION_BLOCKED)


def _pipeline(**kw):
    return InterceptionPipeline(trail=SQLiteAuditBackend(":memory:").load_trail(), **kw)


def _decisions(pipe):
    return [r for r in pipe.trail._records if r.event_type in _DECISIONS]


def test_a_decision_carries_the_digest_and_the_first_carries_the_state():
    pipe = _pipeline()
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "y"})
    first, second = _decisions(pipe)
    assert first.data["policy_digest"] == second.data["policy_digest"]
    assert first.data["policy_state"]["schema"] == SCHEMA
    assert "policy_state" not in second.data


def test_the_recorded_state_recomputes_to_the_digest():
    pipe = _pipeline()
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    record = _decisions(pipe)[0]
    assert digest(record.data["policy_state"]) == record.data["policy_digest"]


def test_a_threshold_change_writes_a_new_digest_and_the_new_state():
    from vaara.policy.loader import SCHEMA_VERSION, from_dict

    pipe = _pipeline()
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    pipe.scorer.apply_policy(from_dict({
        "version": SCHEMA_VERSION, "thresholds": {"default": {"escalate": 0.2, "deny": 0.5}}}))
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "y"})
    before, after = _decisions(pipe)
    assert before.data["policy_digest"] != after.data["policy_digest"]
    state = after.data["policy_state"]
    assert state["scorer"]["thresholds"] == {"allow": 0.2, "deny": 0.5}
    assert digest(state) == after.data["policy_digest"]


def test_shadow_and_enforce_are_different_states():
    enforced, shadow = _pipeline(), _pipeline(enforce=False)
    for pipe in (enforced, shadow):
        pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    assert (_decisions(enforced)[0].data["policy_digest"]
            != _decisions(shadow)[0].data["policy_digest"])


def test_the_receipt_carries_the_digest(tmp_path):
    pipe = _pipeline()
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    record = _decisions(pipe)[0]
    evidence = DecisionReceiptSink(trail_dir=tmp_path).mint(record)["evidence"]
    assert evidence["policyDigest"] == record.data["policy_digest"]


def test_a_scorer_that_cannot_describe_itself_is_named():
    class _Stub:
        def evaluate(self, context):
            return {"action": "allow", "reason": "stub", "raw_result": {
                "point_estimate": 0.1, "conformal_interval": [0.0, 0.2], "signals": {}}}

    pipe = _pipeline(scorer=_Stub())
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    assert _decisions(pipe)[0].data["policy_state"]["scorer"] == {"scorer": "_Stub"}


def test_a_state_that_cannot_be_digested_never_blocks_the_decision():
    class _Broken:
        def policy_state(self):
            raise RuntimeError("no")

        def evaluate(self, context):
            return {"action": "allow", "reason": "stub", "raw_result": {
                "point_estimate": 0.1, "conformal_interval": [0.0, 0.2], "signals": {}}}

    pipe = _pipeline(scorer=_Broken())
    result = pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    assert result.decision == "allow"
    assert "policy_digest" not in _decisions(pipe)[0].data
