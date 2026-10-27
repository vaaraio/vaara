"""A delegating action names who may act under it; nobody else can.

Negative vector for the delegation key-substitution gap (oauth WG thread on
signed delegation receipts, October 2026): a parent that names no delegatee
lets anyone who knows the parent's id act as its child, inheriting the
parent's lineage and its grant ceiling. Three-hop chain planner -> researcher
-> executor, the executor's hop taken by another agent with a narrower grant.

Before the binding, attenuation passed and the delegation graph attributed the
substitute's action to the planner. With the binding the gate refuses it on
its own ground, and the read side names the hop as substituted.
"""

from __future__ import annotations

import uuid

from vaara.audit.delegation import build_delegation_graph, graph_from_trail
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditRecord, EventType
from vaara.credential import Capability
from vaara.pipeline import InterceptionPipeline
from vaara.policy_state import engine_state

ROOT_GRANT = (Capability("amount", "le", "1000"),)
HOP1_GRANT = (Capability("amount", "le", "500"),)
HOP2_GRANT = (Capability("amount", "le", "100"),)


class _AllowScorer:
    def evaluate(self, context):
        return {
            "action": "allow",
            "reason": "stub allow",
            "raw_result": {
                "point_estimate": 0.1,
                "conformal_interval": [0.0, 0.2],
                "signals": {},
            },
        }


def _pipeline(**kwargs) -> InterceptionPipeline:
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail, scorer=_AllowScorer(), **kwargs)


def _blocked(pipe, action_id):
    records = [r for r in pipe.trail.get_records_by_type(EventType.ACTION_BLOCKED)
               if r.action_id == action_id]
    assert records, "no deny on the chain"
    return records[-1].data


def _requested(pipe, action_id):
    return next(r for r in pipe.trail.get_records_by_type(EventType.ACTION_REQUESTED)
                if r.action_id == action_id)


def _bound_chain(pipe):
    a = pipe.intercept(agent_id="planner", tool_name="plan_task",
                       capabilities=ROOT_GRANT, delegatees=["researcher"])
    b = pipe.intercept(agent_id="researcher", tool_name="query_db",
                       parent_action_id=a.action_id, capabilities=HOP1_GRANT,
                       delegatees=["executor"])
    return a, b


# ── The vector: substitution at hop 2 of a three-hop chain ────────────────

def test_bound_three_hop_chain_is_allowed():
    pipe = _pipeline()
    a, b = _bound_chain(pipe)
    c = pipe.intercept(agent_id="executor", tool_name="write_record",
                       parent_action_id=b.action_id, capabilities=HOP2_GRANT)
    assert a.allowed and b.allowed and c.allowed
    assert "delegation binding" not in c.reason


def test_substitute_at_hop_two_is_denied_on_its_own_ground():
    pipe = _pipeline()
    _, b = _bound_chain(pipe)
    # Narrower grant, so attenuation alone would let it through.
    sub = pipe.intercept(agent_id="mallory", tool_name="write_record",
                         parent_action_id=b.action_id, capabilities=HOP2_GRANT)
    assert sub.allowed is False
    assert sub.decision == "deny"
    assert "delegation binding violation (delegatee_not_bound)" in sub.reason
    data = _blocked(pipe, sub.action_id)
    assert data["policy_id"] == "delegatee_binding"
    assert data["violation_type"] == "delegation_binding"


def test_substitute_without_a_grant_is_denied_too():
    pipe = _pipeline()
    _, b = _bound_chain(pipe)
    sub = pipe.intercept(agent_id="mallory", tool_name="write_record",
                         parent_action_id=b.action_id)
    assert sub.allowed is False
    assert "delegatee_not_bound" in sub.reason


def test_bound_delegate_still_runs_after_a_refused_substitute():
    pipe = _pipeline()
    _, b = _bound_chain(pipe)
    pipe.intercept(agent_id="mallory", tool_name="write_record",
                   parent_action_id=b.action_id, capabilities=HOP2_GRANT)
    c = pipe.intercept(agent_id="executor", tool_name="write_record",
                       parent_action_id=b.action_id, capabilities=HOP2_GRANT)
    assert c.allowed


def test_substitute_is_named_on_the_read_side():
    pipe = _pipeline()
    a, b = _bound_chain(pipe)
    sub = pipe.intercept(agent_id="mallory", tool_name="write_record",
                         parent_action_id=b.action_id, capabilities=HOP2_GRANT)
    c = pipe.intercept(agent_id="executor", tool_name="write_record",
                       parent_action_id=b.action_id, capabilities=HOP2_GRANT)
    g = graph_from_trail(pipe.trail)
    assert g.binding_of(a.action_id) == "root"
    assert g.binding_of(b.action_id) == "bound"
    assert g.binding_of(c.action_id) == "bound"
    assert g.binding_of(sub.action_id) == "substituted"
    assert g.substituted == {sub.action_id}
    assert g.unbound == set()


def test_broadening_substitute_names_both_grounds():
    pipe = _pipeline()
    _, b = _bound_chain(pipe)
    sub = pipe.intercept(agent_id="mallory", tool_name="write_record",
                         parent_action_id=b.action_id,
                         capabilities=(Capability("amount", "le", "9000"),))
    assert "delegatee_not_bound" in sub.reason
    assert "broadens:amount" in sub.reason
    assert _blocked(pipe, sub.action_id)["violation_type"] == "delegation_binding"


# ── A parent with no binding ──────────────────────────────────────────────

def test_unbound_parent_is_unchanged_by_default():
    pipe = _pipeline()
    a = pipe.intercept(agent_id="planner", tool_name="plan_task",
                       capabilities=ROOT_GRANT)
    child = pipe.intercept(agent_id="anyone", tool_name="query_db",
                           parent_action_id=a.action_id, capabilities=HOP1_GRANT)
    assert child.allowed
    assert "delegation binding" not in child.reason


def test_unbound_parent_is_reported_never_as_bound():
    pipe = _pipeline()
    a = pipe.intercept(agent_id="planner", tool_name="plan_task")
    child = pipe.intercept(agent_id="anyone", tool_name="query_db",
                           parent_action_id=a.action_id)
    g = graph_from_trail(pipe.trail)
    assert g.binding_of(child.action_id) == "unbound"
    assert g.unbound == {child.action_id}


def test_unbound_parent_is_denied_when_binding_is_required():
    pipe = _pipeline(require_delegatee_binding=True)
    a = pipe.intercept(agent_id="planner", tool_name="plan_task",
                       capabilities=ROOT_GRANT)
    child = pipe.intercept(agent_id="anyone", tool_name="query_db",
                           parent_action_id=a.action_id, capabilities=HOP1_GRANT)
    assert child.allowed is False
    assert "delegation binding violation (parent_unbound)" in child.reason
    assert _blocked(pipe, child.action_id)["violation_type"] == "delegation_binding"


def test_unknown_parent_is_denied_when_binding_is_required():
    pipe = _pipeline(require_delegatee_binding=True)
    child = pipe.intercept(agent_id="anyone", tool_name="query_db",
                           parent_action_id="never-recorded")
    assert child.allowed is False
    assert "delegation binding violation (parent_unknown)" in child.reason


def test_required_binding_leaves_roots_and_bound_hops_alone():
    pipe = _pipeline(require_delegatee_binding=True)
    a, b = _bound_chain(pipe)
    c = pipe.intercept(agent_id="executor", tool_name="write_record",
                       parent_action_id=b.action_id, capabilities=HOP2_GRANT)
    assert a.allowed and b.allowed and c.allowed


# ── The record ────────────────────────────────────────────────────────────

def test_binding_is_recorded_in_the_hashed_request():
    pipe = _pipeline()
    a = pipe.intercept(agent_id="planner", tool_name="plan_task",
                       delegatees=["researcher", "analyst", "researcher"])
    assert _requested(pipe, a.action_id).data["delegatees"] == ["analyst", "researcher"]


def test_request_without_binding_has_no_delegatees_key():
    pipe = _pipeline()
    a = pipe.intercept(agent_id="planner", tool_name="plan_task")
    assert "delegatees" not in _requested(pipe, a.action_id).data


def test_rewriting_the_binding_breaks_the_chain():
    pipe = _pipeline()
    a, _ = _bound_chain(pipe)
    assert pipe.trail.verify_chain() is None
    _requested(pipe, a.action_id).data["delegatees"] = ["mallory"]
    assert pipe.trail.verify_chain() is not None


def test_policy_state_is_unchanged_unless_binding_is_required():
    plain = engine_state(version="x", enforce=True, scorer=_AllowScorer())
    assert "require_delegatee_binding" not in plain
    strict = engine_state(version="x", enforce=True, scorer=_AllowScorer(),
                          require_delegatee_binding=True)
    assert strict["require_delegatee_binding"] is True


# ── Read side over hand-built records ─────────────────────────────────────

def _req(action_id, parent, agent, delegatees=None):
    data = {"parent_action_id": parent}
    if delegatees is not None:
        data["delegatees"] = delegatees
    return AuditRecord(
        record_id=str(uuid.uuid4()), action_id=action_id,
        event_type=EventType.ACTION_REQUESTED, timestamp=0.0,
        agent_id=agent, tool_name="tool", data=data,
    )


def test_read_side_vector_three_hops_substituted_at_hop_two():
    g = build_delegation_graph([
        _req("A", None, "planner", ["researcher"]),
        _req("B", "A", "researcher", ["executor"]),
        _req("C", "B", "mallory"),
    ])
    assert g.chain_for("C") == ["A", "B", "C"]
    assert [g.binding_of(x) for x in g.chain_for("C")] == ["root", "bound", "substituted"]
    assert g.substituted == {"C"}


def test_read_side_malformed_binding_reads_as_unbound():
    g = build_delegation_graph([
        _req("A", None, "planner", "researcher"),  # a string, not a list
        _req("B", "A", "researcher"),
    ])
    assert g.binding_of("B") == "unbound"


def test_read_side_dangling_and_unknown_nodes():
    g = build_delegation_graph([_req("B", "ghost", "researcher")])
    assert g.binding_of("B") == "dangling"
    assert g.binding_of("nope") == "unknown"
