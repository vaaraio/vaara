"""Every deny on the chain names the policy that denied it and the ground.

OVERT TOOL-1.3 asks for a denial receipt with a policy reference and a
violation type. The reason string alone is prose a relying party would have
to parse, so the deny record carries both as fields.
"""

from __future__ import annotations

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.credential import Capability
from vaara.pipeline import InterceptionPipeline


class _FixedScorer:
    def __init__(self, action: str, **extra) -> None:
        self._payload = {
            "action": action,
            "reason": f"stub {action}",
            "raw_result": {
                "point_estimate": 0.5,
                "conformal_interval": [0.4, 0.6],
                "signals": {},
            },
            **extra,
        }

    def evaluate(self, context):
        return dict(self._payload)


class _BrokenScorer:
    def evaluate(self, context):
        raise RuntimeError("model bundle corrupt")


def _pipeline(scorer):
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail, scorer=scorer)


def _blocked(pipe, action_id=None):
    records = [r for r in pipe.trail.get_records_by_type(EventType.ACTION_BLOCKED)
               if action_id is None or r.action_id == action_id]
    assert records, "no deny on the chain"
    return records[-1].data


def test_a_scorer_deny_names_the_scorer_and_the_threshold():
    pipe = _pipeline(_FixedScorer("deny"))
    result = pipe.intercept(agent_id="a", tool_name="tx.transfer")
    data = _blocked(pipe, result.action_id)
    assert data["policy_id"] == "scorer:_FixedScorer"
    assert data["violation_type"] == "risk_threshold"


def test_a_scorer_can_name_its_own_policy():
    pipe = _pipeline(_FixedScorer("deny", policy_id="payments-v3",
                                  violation_type="amount_over_limit"))
    result = pipe.intercept(agent_id="a", tool_name="tx.transfer")
    data = _blocked(pipe, result.action_id)
    assert data["policy_id"] == "payments-v3"
    assert data["violation_type"] == "amount_over_limit"


def test_an_unknown_scorer_verdict_is_named_as_such():
    pipe = _pipeline(_FixedScorer("launch"))
    result = pipe.intercept(agent_id="a", tool_name="tx.transfer")
    assert _blocked(pipe, result.action_id)["violation_type"] == "invalid_decision"


def test_a_rule_deny_names_the_rule():
    pipe = _pipeline(_FixedScorer("allow"))
    result = pipe.intercept(
        agent_id="a", tool_name="Bash", policy_decision="deny",
        policy_reason="deny rule rm-home: rm -rf on a home path",
        policy_id="deny_rule:rm-home",
    )
    data = _blocked(pipe, result.action_id)
    assert data["policy_id"] == "deny_rule:rm-home"
    assert data["violation_type"] == "policy_rule"


def test_an_attenuation_deny_names_the_attenuation():
    pipe = _pipeline(_FixedScorer("allow"))
    parent = pipe.intercept(
        agent_id="planner", tool_name="plan_task",
        capabilities=(Capability("amount", "le", "500"),),
    )
    child = pipe.intercept(
        agent_id="worker", tool_name="read_document",
        parent_action_id=parent.action_id,
        capabilities=(Capability("amount", "le", "5000"),),
    )
    data = _blocked(pipe, child.action_id)
    assert data["policy_id"] == "capability_attenuation"
    assert data["violation_type"] == "privilege_attenuation"


def test_a_scorer_failure_deny_names_the_failure():
    pipe = _pipeline(_BrokenScorer())
    with pytest.raises(RuntimeError):
        pipe.intercept(agent_id="a", tool_name="tx.transfer")
    data = _blocked(pipe)
    assert data["policy_id"] == "scorer:_BrokenScorer"
    assert data["violation_type"] == "scorer_failure"


@pytest.mark.parametrize("verdict", ["allow", "escalate"])
def test_a_non_deny_gains_no_keys(verdict):
    pipe = _pipeline(_FixedScorer(verdict))
    result = pipe.intercept(agent_id="a", tool_name="data.read")
    [record] = [r for r in pipe.trail.get_records_by_type(EventType.DECISION_MADE)
                if r.action_id == result.action_id]
    assert "policy_id" not in record.data
    assert "violation_type" not in record.data


def test_the_chain_verifies_with_the_new_keys():
    pipe = _pipeline(_FixedScorer("deny"))
    pipe.intercept(agent_id="a", tool_name="tx.transfer")
    assert pipe.trail.verify_chain() is None


def test_the_hook_deny_rule_path_names_the_rule(tmp_path, monkeypatch):
    from vaara.integrations import claude_code_hooks as hooks

    monkeypatch.delenv("VAARA_PLUGIN_AUDIT_DB", raising=False)

    cfg = {"audit_db": str(tmp_path / "audit.db")}
    hooks._record_call(
        cfg, "claude-code", "Bash", {"command": "rm -rf ~"},
        {"rule_id": "rm-home", "rule_message": "rm -rf on a home path"},
    )
    trail = SQLiteAuditBackend(str(tmp_path / "audit.db")).load_trail()
    [record] = trail.get_records_by_type(EventType.ACTION_BLOCKED)
    assert record.data["policy_id"] == "deny_rule:rm-home"
    assert record.data["violation_type"] == "policy_rule"
