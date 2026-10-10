# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Authority decay: a session's allows escalate once its risk has added up."""

from __future__ import annotations

from vaara import authority
from vaara.authority import AuthorityPolicy


def _session(*events, session="s1"):
    """Records for one session: (ts, decision, risk) or (ts, "approved")."""
    out = []
    for n, ev in enumerate(events):
        aid = f"a{n}"
        out.append({"event_type": "action_requested", "action_id": aid,
                    "data": {"session_id": session}, "timestamp": ev[0]})
        if ev[1] == "approved":
            out.append({"event_type": "escalation_resolved", "action_id": aid, "timestamp": ev[0],
                        "data": {"resolution": "allow", "human_disposed": True}})
        else:
            out.append({"event_type": "decision_made", "action_id": aid, "timestamp": ev[0],
                        "data": {"decision": ev[1], "risk_score": ev[2]}})
    return out


P = AuthorityPolicy()


def test_routine_calls_cost_nothing():
    recs = _session(*[(i, "allow", 0.1) for i in range(200)])
    assert authority.remaining(recs, "s1", P, 200) == P.budget
    assert authority.check(recs, "s1", 0.1, P, 200) is None


def test_risky_calls_add_up_to_an_escalation():
    recs = _session((0, "allow", 0.9), (1, "deny", 0.95), (2, "escalate", 0.8))
    reason = authority.check(recs, "s1", 0.6, P, 3)
    assert reason is not None and "authority decay" in reason


def test_a_person_approving_refills_the_budget():
    recs = _session((0, "deny", 0.95), (1, "deny", 0.95), (2, "approved"))
    assert authority.remaining(recs, "s1", P, 3) == P.budget


def test_spent_budget_comes_back_with_time():
    recs = _session((0, "deny", 1.0), (0, "deny", 1.0))
    assert authority.remaining(recs, "s1", P, P.half_life_s) == P.budget - 1.0


def test_other_sessions_and_no_session_are_untouched():
    recs = _session((0, "deny", 1.0), (0, "deny", 1.0), (0, "deny", 1.0), session="other")
    assert authority.check(recs, "s1", 0.9, P, 1) is None
    assert authority.check(recs, "", 0.9, P, 1) is None


def test_env_switch():
    assert authority.policy_from_env({"VAARA_AUTHORITY_BUDGET": "off"}).enabled is False
    assert authority.policy_from_env({"VAARA_AUTHORITY_BUDGET": "6"}).low == 2.0
    assert authority.policy_from_env({}) == AuthorityPolicy()


def test_pipeline_escalates_an_allow_late_in_a_risky_session():
    from vaara.audit.trail import AuditTrail
    from vaara.pipeline import InterceptionPipeline

    class Scorer:
        def __init__(self):
            self.next = ("allow", 0.1)

        def evaluate(self, _ctx):
            action, risk = self.next
            return {"action": action, "raw_result": {"point_estimate": risk}}

    scorer = Scorer()
    pipe = InterceptionPipeline(scorer=scorer, trail=AuditTrail(),
                                authority=AuthorityPolicy(budget=1.5, low=0.5))
    assert pipe.intercept("ag", "fs.read", {}, session_id="s").decision == "allow"
    scorer.next = ("deny", 0.9)
    pipe.intercept("ag", "shell.exec", {"cmd": "x"}, session_id="s")
    scorer.next = ("allow", 0.5)
    late = pipe.intercept("ag", "net.fetch", {"u": "y"}, session_id="s")
    assert late.decision == "escalate"
    assert "authority decay" in late.reason
    other = pipe.intercept("ag", "net.fetch", {"u": "y"}, session_id="fresh")
    assert other.decision == "allow"


def test_pipeline_replays_the_whole_session_not_a_record_window():
    # Audit 2026-10-10 finding 2: the budget was replayed from the agent's
    # last 100 records, four to five per action, so spend older than ~20
    # actions was forgotten and a session outran its own decay by calling.
    from vaara.audit.trail import AuditTrail
    from vaara.pipeline import InterceptionPipeline

    class Scorer:
        def __init__(self):
            self.next = ("allow", 0.0)

        def evaluate(self, _ctx):
            action, risk = self.next
            return {"action": action, "raw_result": {"point_estimate": risk}}

    scorer = Scorer()
    trail = AuditTrail()
    pipe = InterceptionPipeline(scorer=scorer, trail=trail,
                                authority=AuthorityPolicy(budget=1.5, low=0.5,
                                                          half_life_s=1e9))
    scorer.next = ("deny", 0.9)
    pipe.intercept("ag", "shell.exec", {"cmd": "x"}, session_id="s")
    scorer.next = ("deny", 0.5)
    pipe.intercept("ag", "shell.exec", {"cmd": "y"}, session_id="s")
    scorer.next = ("allow", 0.0)
    for n in range(40):
        r = pipe.intercept("ag", "fs.read", {"n": n}, session_id="s")
        assert r.decision == "escalate", f"call {n} forgot the session's spend"
    assert len(trail.get_agent_records("ag", limit=10**6)) > 100
    # A session that spent nothing is not touched by the first one's records.
    assert pipe.intercept("ag", "fs.read", {}, session_id="fresh").decision == "allow"
