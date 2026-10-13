# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check a recorded trail against a grant, offline, after the fact.

Internal module. Public surface is in ``vaara.credential``.

``verify_grant`` answers for one call at the moment it runs. This module
answers the question an auditor asks later, holding a trail and the grant the
principal signed: of every call the trail recorded for the granted tool,
which stayed inside the grant, which went outside it, and what the gate did
with the ones that went outside.

A call outside the grant is only a finding when the trail shows it allowed.
A refusal of the same call is evidence the gate held, so it is reported apart
rather than mixed into the violations. Refusals are on the record because the
trail writes a decision for every request, denied or not.

Each recorded call lands in exactly one state:

* ``within``: the arguments satisfy the grant.
* ``violated``: the arguments fall outside the grant and the call was allowed.
* ``refused``: outside the grant, and the decision was a deny.
* ``held``: outside the grant, and the decision was an escalation still open.
* ``undecided``: no decision was recorded for the request.
* ``unverifiable``: the arguments were truncated when recorded, so nobody can
  say whether they fit.
* ``outside_window``: recorded before the grant was issued or after it expired;
  this grant says nothing about those calls.

The chain is verified first and the grant signature second. If either fails,
no call is judged, because a verdict over records or a grant that cannot be
trusted would read as evidence it is not.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from vaara.attestation._attest_canonical import iso8601_to_epoch, make_args_digest
from vaara.credential._grant_capability import evaluate
from vaara.credential._grant_emit import verify_grant_signature
from vaara.credential._grant_types import BrokeredCredential

#: States that keep ``ok`` true. Everything else is something to look at.
_CLEAN_STATES = frozenset({"within", "refused", "held", "outside_window"})


@dataclass(frozen=True)
class GrantedCall:
    """One recorded request for the granted tool, and where it landed."""

    action_id: str
    timestamp: float
    agent_id: str
    state: str
    decision: str
    reason: str


@dataclass(frozen=True)
class TrailGrantReport:
    """The verdict over a whole trail. ``ok`` is false on any finding."""

    tool_name: str
    tenant_id: str
    calls: tuple[GrantedCall, ...] = ()
    chain_problem: str = ""
    grant_problem: str = ""
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        if self.chain_problem or self.grant_problem:
            return False
        return all(c.state in _CLEAN_STATES for c in self.calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "toolName": self.tool_name,
            "tenantId": self.tenant_id,
            "chainProblem": self.chain_problem,
            "grantProblem": self.grant_problem,
            "counts": dict(self.counts),
            "calls": [
                {
                    "actionId": c.action_id,
                    "timestamp": c.timestamp,
                    "agentId": c.agent_id,
                    "state": c.state,
                    "decision": c.decision,
                    "reason": c.reason,
                }
                for c in self.calls
            ],
        }


def _fits(credential: BrokeredCredential, args: Any) -> tuple[bool, str]:
    if credential.capabilities:
        return evaluate(credential.capabilities, args)
    try:
        digest = make_args_digest(args).projection_digest
    except Exception:
        return (False, "scope_mismatch")
    if digest != credential.scope.args_commitment:
        return (False, "scope_mismatch")
    return (True, "ok")


def _last_decision(records: list) -> str:
    decision = ""
    for rec in records:
        value = rec.data.get("decision") if isinstance(rec.data, dict) else None
        if value:
            decision = str(value)
    return decision


def check_trail_against_grant(
    trail: Any,
    credential: BrokeredCredential,
    *,
    verifying_material: Any,
    verify_chain: bool = True,
    clock_skew_seconds: int = 30,
) -> TrailGrantReport:
    """Judge every recorded call to the granted tool against the grant."""
    scope = credential.scope

    if verify_chain:
        broken = trail.verify_chain()
        if broken:
            return TrailGrantReport(scope.tool_name, scope.tenant_id, chain_problem=str(broken))

    if not verify_grant_signature(credential, verifying_material=verifying_material):
        return TrailGrantReport(scope.tool_name, scope.tenant_id, grant_problem="bad_signature")
    iat = iso8601_to_epoch(credential.asserted.iat)
    if iat is None:
        return TrailGrantReport(scope.tool_name, scope.tenant_id, grant_problem="unreadable_iat")
    start = iat - clock_skew_seconds
    end = iat + credential.asserted.exp_seconds + clock_skew_seconds

    from vaara.audit.trail import EventType

    calls: list[GrantedCall] = []
    for rec in list(trail._records):
        if rec.event_type != EventType.ACTION_REQUESTED:
            continue
        if rec.tool_name != scope.tool_name or (rec.tenant_id or "") != scope.tenant_id:
            continue
        decision = _last_decision(trail.get_action_trail(rec.action_id)[1:])
        params = rec.data.get("parameters") if isinstance(rec.data, dict) else None

        if not start <= rec.timestamp <= end:
            state, reason = "outside_window", ""
        elif isinstance(params, dict) and params.get("_truncated"):
            state, reason = "unverifiable", "parameters_truncated"
        else:
            fits, reason = _fits(credential, params if params is not None else {})
            if fits:
                state = "within" if decision else "undecided"
            elif decision == "deny":
                state = "refused"
            elif decision == "escalate":
                state = "held"
            elif decision:
                state = "violated"
            else:
                state = "undecided"
        calls.append(GrantedCall(
            action_id=rec.action_id,
            timestamp=rec.timestamp,
            agent_id=rec.agent_id,
            state=state,
            decision=decision,
            reason=reason if reason != "ok" else "",
        ))

    counts = dict(Counter(c.state for c in calls))
    return TrailGrantReport(scope.tool_name, scope.tenant_id, calls=tuple(calls), counts=counts)
