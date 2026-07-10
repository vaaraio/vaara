"""Reconstruction of multi-agent delegation chains + tamper-evidence.

Covers the read-side reconstruction in vaara.audit.delegation (linear,
branching, dangling, cycle, self-parent, multi-event, empty) and the end-to-
end guarantee that a forged parent link is caught by the existing hash chain.
"""

from __future__ import annotations

import uuid

from vaara.audit.delegation import build_delegation_graph, graph_from_trail
from vaara.audit.trail import AuditRecord, EventType


def _req(action_id: str, parent, agent: str = "agent", tool: str = "tool") -> AuditRecord:
    """An action_requested record carrying a parent_action_id in data."""
    return AuditRecord(
        record_id=str(uuid.uuid4()),
        action_id=action_id,
        event_type=EventType.ACTION_REQUESTED,
        timestamp=0.0,
        agent_id=agent,
        tool_name=tool,
        data={"parent_action_id": parent},
    )


def _evt(action_id: str, event_type: EventType) -> AuditRecord:
    """A non-action_requested lifecycle event for an existing action."""
    return AuditRecord(
        record_id=str(uuid.uuid4()),
        action_id=action_id,
        event_type=event_type,
        timestamp=0.0,
        agent_id="agent",
        tool_name="tool",
        data={},
    )


# ── Linear chain A -> B -> C ──────────────────────────────────────────────

def test_linear_chain_reconstructs_root_to_leaf():
    g = build_delegation_graph([
        _req("A", None, agent="planner"),
        _req("B", "A", agent="researcher"),
        _req("C", "B", agent="executor"),
    ])
    assert g.chain_for("C") == ["A", "B", "C"]
    assert g.root_of("C") == "A"
    assert g.depth_of("C") == 2
    assert g.depth_of("A") == 0
    assert g.roots() == ["A"]
    assert g.descendants("A") == ["B", "C"]
    assert g.descendants("C") == []
    assert g.agent_of["C"] == "executor"


# ── Branching A -> {B, C} ─────────────────────────────────────────────────

def test_branching_blast_radius():
    g = build_delegation_graph([
        _req("A", None),
        _req("B", "A"),
        _req("C", "A"),
    ])
    assert g.roots() == ["A"]
    assert sorted(g.descendants("A")) == ["B", "C"]
    assert g.chain_for("B") == ["A", "B"]
    assert g.chain_for("C") == ["A", "C"]
    assert g.depth_of("B") == 1


# ── Dangling parent (forged / pruned ancestor) ────────────────────────────

def test_dangling_parent_is_surfaced_not_dropped():
    g = build_delegation_graph([
        _req("B", "ghost"),  # parent never appears
    ])
    assert "B" in g
    assert "B" in g.dangling
    assert g.root_of("B") == "B"
    assert g.chain_for("B") == ["B"]
    assert g.depth_of("B") == 0


# ── Cycle A -> B -> A ─────────────────────────────────────────────────────

def test_cycle_is_detected_and_terminates():
    g = build_delegation_graph([
        _req("A", "B"),
        _req("B", "A"),
    ])
    assert "A" in g.cycles and "B" in g.cycles
    # Traversals must terminate and not raise.
    assert g.root_of("A") in {"A", "B"}
    assert isinstance(g.depth_of("A"), int)
    assert isinstance(g.chain_for("B"), list)


def test_self_parent_is_a_cycle():
    g = build_delegation_graph([_req("A", "A")])
    assert "A" in g.cycles
    assert g.root_of("A") == "A"
    assert g.depth_of("A") == 0


# ── Multiple events per action collapse to one node ───────────────────────

def test_multiple_events_one_node():
    g = build_delegation_graph([
        _req("A", None),
        _req("B", "A"),
        _evt("B", EventType.DECISION_MADE),
        _evt("B", EventType.ACTION_EXECUTED),
    ])
    assert g.chain_for("B") == ["A", "B"]
    assert g.descendants("A") == ["B"]


def test_action_with_only_later_events_is_a_root():
    # A decision event with no preceding action_requested (e.g. skeleton
    # reload) still yields a node, as a root with unknown parent.
    g = build_delegation_graph([_evt("X", EventType.DECISION_MADE)])
    assert "X" in g
    assert g.root_of("X") == "X"


# ── Empty ─────────────────────────────────────────────────────────────────

def test_empty_trail():
    g = build_delegation_graph([])
    assert g.roots() == []
    assert g.chain_for("nope") == ["nope"]
    assert g.descendants("nope") == []


# ── End-to-end via the pipeline + tamper-evidence ─────────────────────────

def test_pipeline_chain_reconstructs_and_tamper_is_caught():
    from vaara import Pipeline

    pipe = Pipeline()
    a = pipe.intercept(agent_id="planner", tool_name="plan_task", parameters={})
    b = pipe.intercept(
        agent_id="researcher", tool_name="search_docs",
        parameters={}, parent_action_id=a.action_id,
    )
    c = pipe.intercept(
        agent_id="executor", tool_name="write_record",
        parameters={}, parent_action_id=b.action_id,
    )

    # Chain intact and reconstructable end to end.
    assert pipe.trail.verify_chain() is None
    g = graph_from_trail(pipe.trail)
    assert g.chain_for(c.action_id) == [a.action_id, b.action_id, c.action_id]
    assert g.root_of(c.action_id) == a.action_id
    assert set(g.descendants(a.action_id)) == {b.action_id, c.action_id}

    # Forge the authorization trail: rewrite who spawned C. Because
    # parent_action_id is hash-covered, verify_chain must now fail.
    target = next(
        r for r in pipe.trail.snapshot()
        if r.event_type == EventType.ACTION_REQUESTED
        and r.data.get("parent_action_id") == b.action_id
    )
    target.data["parent_action_id"] = "forged-authorizer"
    assert pipe.trail.verify_chain() is not None
