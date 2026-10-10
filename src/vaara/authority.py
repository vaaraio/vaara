# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Authority decay: a session's room to act shrinks as its risk adds up.

Each session (one agent, one ``session_id``) starts with a budget. Every
decision in it spends some: an allowed call the part of its risk score above
a floor, so routine calls cost nothing, and an escalated or denied call its
whole score, since asking for something risky is itself the signal. Spent
budget comes back by half every ``half_life_s``. Once what is left would fall
below ``low``, a call the scorer allowed is escalated instead: a person has to
approve it. A person approving an escalation in the session refills the
budget, which is the re-authorisation.

Decay only tightens. It turns an allow into an escalate and never loosens a
decision.

The budget is not stored anywhere. It is replayed from the session's own
records on the trail (decisions, their risk scores, human resolutions, and
their timestamps), so a process that starts fresh on every call, such as a
hook, sees the same budget a long-running proxy does, and a reviewer can
recompute from the chain why a call was escalated. The replay reads the
whole session, never a window of recent records: a window would let a
session outrun its own spend by making calls.

One caveat across processes: a long-running proxy replays from the records
it loaded at start plus the ones it wrote. Records another process (a hook
in the same session) wrote to the same store after that are not in its
replay until it reloads, while a process that starts fresh per call reads
them all. Two processes deciding for one session can therefore see
different budgets for a while; both are honest about the records they hold.

``VAARA_AUTHORITY_BUDGET`` sets the budget for every pipeline that is not
given a policy explicitly; ``0`` or ``off`` turns decay off. Calls with no
``session_id`` are never decayed: without a session there is nothing to add
up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

POLICY_ID = "vaara-authority-decay"


@dataclass(frozen=True)
class AuthorityPolicy:
    budget: float = 3.0
    low: float = 1.0
    half_life_s: float = 3600.0
    floor: float = 0.3
    enabled: bool = True


def policy_from_env(env: Optional[dict] = None) -> AuthorityPolicy:
    raw = (os.environ if env is None else env).get("VAARA_AUTHORITY_BUDGET", "").strip().lower()
    if not raw:
        return AuthorityPolicy()
    if raw in ("0", "off", "false", "no"):
        return AuthorityPolicy(enabled=False)
    try:
        budget = float(raw)
    except ValueError:
        return AuthorityPolicy()
    if not budget > 0:
        return AuthorityPolicy(enabled=False)
    default = AuthorityPolicy()
    # Keep the escalation line at the same share of the budget.
    return AuthorityPolicy(budget=budget, low=budget * default.low / default.budget)


def spend(decision: str, risk: float, policy: AuthorityPolicy) -> float:
    """What one decision costs the session."""
    risk = max(0.0, min(1.0, float(risk)))
    if decision == "allow":
        return max(0.0, risk - policy.floor)
    return risk


def _value(record: Any, name: str) -> Any:
    return getattr(record, name, None) if not isinstance(record, dict) else record.get(name)


def remaining(
    records: Iterable[Any],
    session_id: str,
    policy: AuthorityPolicy,
    now: float,
    *,
    exclude_action: str = "",
) -> float:
    """The session's budget left at ``now``, replayed from its records.

    ``records`` are one agent's trail records in order (``AuditRecord`` or the
    same fields as a dict). ``exclude_action`` leaves out the call being
    decided, whose own decision is not on the trail yet.
    """
    actions: set[str] = set()
    events: list[tuple[float, str, Any]] = []
    for r in records:
        kind = _value(r, "event_type")
        kind = getattr(kind, "value", kind)
        action = _value(r, "action_id") or ""
        data = _value(r, "data") or {}
        if action == exclude_action:
            continue
        if kind == "action_requested":
            if data.get("session_id") == session_id:
                actions.add(action)
        elif action in actions and (kind == "escalation_resolved" or "decision" in data):
            events.append((float(_value(r, "timestamp") or 0.0), kind, data))
    deficit = 0.0
    last: Optional[float] = None
    for ts, kind, data in sorted(events, key=lambda e: e[0]):
        if last is not None and ts > last:
            deficit *= 0.5 ** ((ts - last) / policy.half_life_s)
        last = ts if last is None else max(last, ts)
        if kind != "escalation_resolved":
            score = data.get("risk_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                deficit += spend(str(data.get("decision", "")), score, policy)
        elif data.get("resolution") == "allow" and data.get("human_disposed") is True:
            deficit = 0.0
    if last is not None and now > last:
        deficit *= 0.5 ** ((now - last) / policy.half_life_s)
    return policy.budget - deficit


def check(
    records: Iterable[Any],
    session_id: str,
    risk: float,
    policy: AuthorityPolicy,
    now: float,
    *,
    exclude_action: str = "",
) -> Optional[str]:
    """The reason to escalate an allowed call, or None to leave it allowed."""
    if not policy.enabled or not session_id:
        return None
    left = remaining(records, session_id, policy, now, exclude_action=exclude_action)
    after = left - spend("allow", risk, policy)
    if after >= policy.low:
        return None
    return (f"authority decay: this session has {max(after, 0.0):.2f} of "
            f"{policy.budget:g} left after this call (escalates below {policy.low:g}); "
            f"a person's approval restores it")
