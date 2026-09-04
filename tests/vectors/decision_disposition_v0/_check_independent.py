#!/usr/bin/env python3
"""Independent conformance checker for the decision_disposition_v0 vectors.

Imports the standard library only. It does not import Vaara. A second
implementation can run this file to confirm the disposition rules are
consumable from the committed bytes alone.

The property under test is that three ALLOW-shaped outcomes are separable by
field rather than by prose:

  1. policy allowed the action outright
  2. a human approved it at escalation
  3. a prior human approval was replayed by cache lookup inside its window

(3) is a policy disposition wearing a human's earlier decision. All three carry
``decision: "allow"``. Only the disposition fields tell them apart, and a
relying party that branches on ``human_disposed`` must not have to read the
``reason`` string to get the right answer.

Two rules are reproduced from scratch:

**The closed set.** ``approver`` is exactly "human" or "policy". An unknown
value is non-conforming rather than tolerated, because a value an
implementation may invent is a value a relying party cannot branch on.

**The honest flag.** ``human_disposed`` is true ONLY when a human actually
acted. It must be a real boolean, so a truthy 1 or "yes" is non-conforming and
never becomes a human claim. ``human_disposed: true`` requires
``approver: "human"``: a producer must not claim a human disposed what a policy
did. The converse is legal, since a human approver with a false flag only
narrows the claim.

Absent disposition is conforming. A record written before this vocabulary
existed carries neither key and must keep hashing as it did, so silence is a
valid state and is not read as "a human acted".

Exit 0 when every case agrees with expected.json, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

APPROVER = frozenset({"human", "policy"})


def evaluate(case: dict) -> dict:
    """Return the conformance verdict for one case, from its bytes alone."""
    approver = case.get("approver")
    flag = case.get("human_disposed")

    # Silence is a valid state: neither key present.
    if approver is None and flag is None:
        return {
            "conforming": True,
            "keys_present": False,
            "approver": None,
            "human_disposed": None,
        }

    # A flag standing on its own is the claim with nothing behind it.
    if approver is None:
        return {"conforming": False, "reason_class": "human_claim_by_policy"}

    if not isinstance(approver, str):
        return {"conforming": False, "reason_class": "approver_not_in_closed_set"}

    normalized = approver.strip().lower()
    if normalized not in APPROVER:
        return {"conforming": False, "reason_class": "approver_not_in_closed_set"}

    # Checked before truthiness is ever consulted. `1` is truthy and is not a
    # boolean, and letting it through here is exactly how a policy disposition
    # becomes a human claim.
    if not isinstance(flag, bool):
        return {"conforming": False, "reason_class": "flag_not_boolean"}

    if flag and normalized != "human":
        return {"conforming": False, "reason_class": "human_claim_by_policy"}

    return {
        "conforming": True,
        "keys_present": True,
        "approver": normalized,
        "human_disposed": flag,
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    cases = json.loads((here / "cases.json").read_text())
    expected = json.loads((here / "expected.json").read_text())

    if set(cases) != set(expected):
        print("case names do not match expected names", file=sys.stderr)
        return 1

    failures = 0
    for name in sorted(cases):
        got = evaluate(cases[name])
        want = expected[name]
        # Compare on the keys the expectation states, so an implementation
        # carrying extra diagnostic keys is not penalised for them.
        narrowed = {k: got.get(k) for k in want}
        if narrowed != want:
            failures += 1
            print(f"FAIL {name}\n  expected {want}\n  got      {narrowed}")
        else:
            print(f"ok   {name}  {want}")

    # The property the suite exists for, asserted rather than implied.
    live = evaluate(cases["human_approved_at_escalation"])
    replayed = evaluate(cases["replayed_prior_approval"])
    if cases["human_approved_at_escalation"]["decision"] != "allow" or \
            cases["replayed_prior_approval"]["decision"] != "allow":
        failures += 1
        print("FAIL both comparison cases must carry decision 'allow'")
    elif live["human_disposed"] == replayed["human_disposed"]:
        failures += 1
        print("FAIL a live approval and a replayed one are indistinguishable")
    else:
        print("ok   live and replayed approvals separate on human_disposed")

    print(f"\n{len(cases)} cases, {failures} disagreement(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
