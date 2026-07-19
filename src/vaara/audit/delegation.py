# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reconstruct multi-agent delegation chains from the audit trail.

Read-side only. The delegation edge — ``parent_action_id`` — is captured at
ingest (``Pipeline.intercept`` threads it, the LangChain and MCP proxy
integrations populate it automatically) and lands inside the hash-covered
``data`` of each ``action_requested`` event. Because ``compute_hash`` covers
``data``, tampering with a parent link breaks ``verify_chain()``: the edge is
already tamper-evident.

What was missing is the read back. When agent A delegates to agent B which
delegates to agent C, nothing walked those edges into a chain, so an auditor
could not reconstruct "who authorized C's action" as a single lineage. This
module does exactly that, over a list of records or a whole trail, without
touching the stored schema (``root`` and ``depth`` are computed, not stored).

The reconstruction is deliberately defensive: a forged, pruned, or cyclic
parent link never crashes or hangs the walk. Dangling and cyclic action_ids
are surfaced on the graph rather than silently dropped, so an anomaly in the
delegation structure is itself visible evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from vaara.audit.trail import AuditRecord, EventType

# Upper bound on a single parent walk. Guards against pathological depth and
# is the backstop that makes cycle handling terminate even if the pre-computed
# cycle set is somehow bypassed. Delegation trees in practice are shallow;
# 10k is far beyond any real agent topology.
_MAX_WALK = 10_000


@dataclass
class DelegationGraph:
    """Reconstructed delegation forest over a set of audit records.

    Nodes are ``action_id`` strings. ``effective_parent`` is acyclic by
    construction — a recorded parent that is unknown (dangling) or part of a
    cycle is cut to ``None`` (making that node a local root) and reported in
    :attr:`dangling` / :attr:`cycles`. Every traversal method therefore
    terminates.
    """

    effective_parent: dict[str, Optional[str]] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    dangling: set[str] = field(default_factory=set)
    cycles: set[str] = field(default_factory=set)
    agent_of: dict[str, str] = field(default_factory=dict)
    tool_of: dict[str, str] = field(default_factory=dict)

    def __contains__(self, action_id: str) -> bool:
        return action_id in self.effective_parent

    def roots(self) -> list[str]:
        """Action_ids with no (effective) parent, in first-seen order."""
        return [a for a, p in self.effective_parent.items() if p is None]

    def root_of(self, action_id: str) -> str:
        """Top of ``action_id``'s delegation chain. Self if unknown or root."""
        cur = action_id
        for _ in range(_MAX_WALK):
            parent = self.effective_parent.get(cur)
            if parent is None:
                return cur
            cur = parent
        return cur

    def depth_of(self, action_id: str) -> int:
        """Hops from ``action_id`` up to its root. A root has depth 0."""
        depth = 0
        cur = action_id
        for _ in range(_MAX_WALK):
            parent = self.effective_parent.get(cur)
            if parent is None:
                return depth
            depth += 1
            cur = parent
        return depth

    def chain_for(self, action_id: str) -> list[str]:
        """Ordered root -> ... -> ``action_id`` path of action_ids.

        Returns ``[action_id]`` for an unknown node, so a caller always gets
        the node itself back rather than an empty list.
        """
        if action_id not in self.effective_parent:
            return [action_id]
        reverse: list[str] = []
        cur: Optional[str] = action_id
        for _ in range(_MAX_WALK):
            if cur is None:
                break
            reverse.append(cur)
            cur = self.effective_parent.get(cur)
        reverse.reverse()
        return reverse

    def descendants(self, action_id: str) -> list[str]:
        """All action_ids delegated (transitively) from ``action_id``.

        Breadth-first, excluding ``action_id`` itself. This is the blast
        radius of an action: everything downstream it set in motion.
        """
        out: list[str] = []
        seen = {action_id}
        queue = list(self.children.get(action_id, []))
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            out.append(node)
            queue.extend(self.children.get(node, []))
        return out


def _parent_link(record: AuditRecord) -> Optional[str]:
    """Read the recorded parent_action_id off an action_requested record."""
    parent = record.data.get("parent_action_id")
    if isinstance(parent, str) and parent:
        return parent
    return None


def build_delegation_graph(records: Iterable[AuditRecord]) -> DelegationGraph:
    """Reconstruct the delegation forest from an iterable of audit records.

    A node exists for every ``action_id`` seen on any event type, so an action
    that only carries later-lifecycle events (e.g. a skeleton reload) still
    appears — as a root, since only ``action_requested`` carries the parent
    link. Recorded parents that point at an unknown action are cut and listed
    in ``dangling``; cyclic and self-parent links are cut and listed in
    ``cycles``. The resulting ``effective_parent`` map is acyclic.
    """
    seen: dict[str, None] = {}  # ordered set of every action_id
    raw_parent: dict[str, Optional[str]] = {}
    agent_of: dict[str, str] = {}
    tool_of: dict[str, str] = {}

    for rec in records:
        aid = rec.action_id
        seen.setdefault(aid, None)
        if rec.event_type == EventType.ACTION_REQUESTED:
            # The action_requested event is authoritative for the edge and the
            # originating agent. If two somehow exist for one action_id, the
            # first wins (append order), matching the trail's own ordering.
            if aid not in raw_parent:
                raw_parent[aid] = _parent_link(rec)
                agent_of[aid] = rec.agent_id
                tool_of[aid] = rec.tool_name

    dangling: set[str] = set()
    cycles: set[str] = set()
    effective_parent: dict[str, Optional[str]] = {}

    for aid in seen:
        parent = raw_parent.get(aid)
        if parent is None:
            effective_parent[aid] = None
        elif parent not in seen:
            # Forged or pruned ancestor: keep the node, expose the break.
            dangling.add(aid)
            effective_parent[aid] = None
        else:
            effective_parent[aid] = parent

    # Cycle detection over the functional parent graph (each node has <= 1
    # parent). Walk up from each node; a revisit means a loop. Every node on
    # the loop is recorded and cut to a root so traversals stay finite.
    for start in seen:
        path: list[str] = []
        path_set: set[str] = set()
        cur: Optional[str] = start
        steps = 0
        while cur is not None and steps < _MAX_WALK:
            if cur in cycles:
                break
            if cur in path_set:
                idx = path.index(cur)
                for node in path[idx:]:
                    cycles.add(node)
                    effective_parent[node] = None
                break
            path.append(cur)
            path_set.add(cur)
            cur = effective_parent.get(cur)
            steps += 1

    children: dict[str, list[str]] = {aid: [] for aid in seen}
    for aid in seen:
        parent = effective_parent.get(aid)
        if parent is not None:
            children[parent].append(aid)

    return DelegationGraph(
        effective_parent=effective_parent,
        children=children,
        dangling=dangling,
        cycles=cycles,
        agent_of=agent_of,
        tool_of=tool_of,
    )


def graph_from_trail(trail) -> DelegationGraph:
    """Reconstruct the delegation forest from a live :class:`AuditTrail`."""
    return build_delegation_graph(trail.snapshot())
