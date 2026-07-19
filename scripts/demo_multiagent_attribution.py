#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Demo: reconstruct a multi-agent delegation chain and prove it can't be forged.

Runs a three-agent chain through Vaara's governance pipeline:

    planner (A)  ->  researcher (B)  ->  executor (C)

The executor calls a high-blast-radius "production" tool. Vaara records every
step in its hash-chained audit trail, carrying the delegation edge
(parent_action_id) in the hash-covered data of each request.

The demo then does two things no runtime-enforcement tool does:

  1. Reconstructs the delegation chain from the trail — "who authorized the
     production write?" answered as a single lineage, root to leaf.
  2. Forges the authorization trail (rewrites who spawned the executor) and
     shows Vaara's existing hash chain catch the tamper immediately.

This is the evidence wedge: not blocking the action, but proving — after the
fact, tamper-evidently — exactly who set it in motion.

Run:  python scripts/demo_multiagent_attribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vaara import Pipeline  # noqa: E402
from vaara.audit.delegation import graph_from_trail  # noqa: E402
from vaara.audit.trail import EventType  # noqa: E402
from vaara.credential import Capability, chain_is_attenuating  # noqa: E402


def _short(action_id: str) -> str:
    return action_id[:8]


def _render_tree(graph, action_id: str, indent: int = 0) -> None:
    agent = graph.agent_of.get(action_id, "?")
    tool = graph.tool_of.get(action_id, "?")
    prefix = "    " * indent + ("└─ " if indent else "")
    print(f"{prefix}{agent} :: {tool}  [{_short(action_id)}]")
    for child in graph.children.get(action_id, []):
        _render_tree(graph, child, indent + 1)


def main() -> int:
    pipe = Pipeline()

    print("== 1. Three agents act, each delegating to the next ==\n")
    a = pipe.intercept(agent_id="planner", tool_name="plan_incident_response",
                       parameters={"ticket": "INC-4471"})
    b = pipe.intercept(agent_id="researcher", tool_name="query_customer_db",
                       parameters={"scope": "affected_accounts"},
                       parent_action_id=a.action_id)
    c = pipe.intercept(agent_id="executor", tool_name="update_production_record",
                       parameters={"table": "accounts", "op": "write"},
                       parent_action_id=b.action_id)
    print(f"  planner    -> {_short(a.action_id)}  ({a.decision})")
    print(f"  researcher -> {_short(b.action_id)}  ({b.decision})")
    print(f"  executor   -> {_short(c.action_id)}  ({c.decision})")

    graph = graph_from_trail(pipe.trail)

    print("\n== 2. Reconstructed delegation tree ==\n")
    for root in graph.roots():
        _render_tree(graph, root)

    print("\n== 3. Who authorized the production write? ==\n")
    chain = graph.chain_for(c.action_id)
    lineage = " -> ".join(
        f"{graph.agent_of.get(aid, '?')}({_short(aid)})" for aid in chain
    )
    print(f"  {lineage}")
    print(f"  depth: {graph.depth_of(c.action_id)} hops from root "
          f"{graph.agent_of.get(graph.root_of(c.action_id), '?')}")
    print(f"  blast radius of planner's action: "
          f"{[graph.agent_of.get(x, '?') for x in graph.descendants(a.action_id)]}")

    print("\n== 4. Try to forge the authorization trail ==\n")
    print(f"  chain intact before tamper:  {pipe.trail.verify_chain() is None}")
    target = next(
        r for r in pipe.trail.snapshot()
        if r.event_type == EventType.ACTION_REQUESTED
        and r.data.get("parent_action_id") == b.action_id
    )
    print(f"  rewriting executor's parent {_short(b.action_id)} -> 'forged-authorizer'")
    target.data["parent_action_id"] = "forged-authorizer"
    err = pipe.trail.verify_chain()
    print(f"  chain intact after tamper:   {err is None}")
    print(f"  tamper caught:               {err is not None}")
    if err is not None:
        print(f"  verifier says: {err.splitlines()[0]}")

    print("\nThe forged authorization did not survive the hash chain.")

    # ── Second act: privilege cannot grow as it is delegated ──────────────
    print("\n== 5. Delegated-privilege attenuation ==\n")
    grants = {
        a.action_id: (
            Capability("amount", "le", "1000"),
            Capability("vendor", "in", ("acme", "globex")),
        ),
        b.action_id: (  # researcher: narrower -> ok
            Capability("amount", "le", "500"),
            Capability("vendor", "eq", "acme"),
        ),
        c.action_id: (  # executor: tries to raise its own ceiling -> broadens
            Capability("amount", "le", "900"),
            Capability("vendor", "eq", "acme"),
        ),
    }
    chain = graph.chain_for(c.action_id)
    print("  granted amount ceilings down the chain:")
    for aid in chain:
        ceiling = next(cap.value for cap in grants[aid] if cap.arg == "amount")
        print(f"    {graph.agent_of.get(aid, '?'):11} le {ceiling}")

    report = chain_is_attenuating([grants[aid] for aid in chain])
    if report.ok:
        print("  chain attenuates cleanly: authority only narrows.")
    else:
        hop = chain[report.first_broadening_index]
        print(f"  VIOLATION: {graph.agent_of.get(hop, '?')} broadened its grant "
              f"({report.reason}) beyond what it was delegated.")

    # And a compliant executor (stays within researcher's grant) passes.
    grants[c.action_id] = (
        Capability("amount", "le", "100"),
        Capability("vendor", "eq", "acme"),
    )
    fixed = chain_is_attenuating([grants[aid] for aid in chain])
    print(f"  with executor kept within scope (le 100): "
          f"{'clean' if fixed.ok else 'still violating'}")

    print("\nAuthority never grew as it was handed down — and when it tried, "
          "Vaara caught it.")
    return 0 if (err is not None and not report.ok and fixed.ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
