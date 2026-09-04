# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three allow-shaped cases must separate by FIELD, not by prose.

Before this vocabulary, a relying party holding the trail saw `allow` in all
three of these and could only tell them apart by reading the English in
`reason`:

  a) policy allowed it outright
  b) a human approved it at escalation
  c) a prior human approval was replayed by cache lookup inside its window

(c) is a policy disposition wearing a human's earlier decision, and it is the
one worth catching. Assertions below compare field identity, never truthiness.
"""

import pytest

from vaara._disposition import DispositionError
from vaara.audit.trail import AuditTrail, EventType


def _trail():
    return AuditTrail()


def _data(trail, event_type):
    records = trail.get_records_by_type(event_type)
    if not records:
        raise AssertionError(f"no {event_type} record")
    return records[-1].data or {}


class TestTheThreeCasesSeparate:
    def test_a_plain_policy_allow_carries_no_disposition_keys(self):
        t = _trail()
        t.record_decision(
            action_id="act-a", agent_id="a1", tool_name="fs.read",
            decision="allow", reason="low risk", risk_score=0.1,
        )
        data = _data(t, EventType.DECISION_MADE)
        assert "approver" not in data
        assert "human_disposed" not in data

    def test_b_human_approval_says_human_and_true(self):
        t = _trail()
        t.record_decision(
            action_id="act-b", agent_id="a1", tool_name="tx.transfer",
            decision="allow", reason="approved at escalation", risk_score=0.6,
            approver="human", human_disposed=True,
        )
        data = _data(t, EventType.DECISION_MADE)
        assert data["approver"] == "human"
        assert data["human_disposed"] is True

    def test_c_replayed_approval_says_policy_and_false(self):
        t = _trail()
        t.record_decision(
            action_id="act-c", agent_id="a1", tool_name="tx.transfer",
            decision="allow",
            reason="auto-allowed by prior approval (action_id=act-b, 123.0)",
            risk_score=0.6,
            approver="policy", human_disposed=False,
        )
        data = _data(t, EventType.DECISION_MADE)
        assert data["approver"] == "policy"
        assert data["human_disposed"] is False

    def test_b_and_c_are_distinguishable_without_reading_reason(self):
        """The whole point. Same decision word, different disposition."""
        t = _trail()
        t.record_decision(
            action_id="act-b", agent_id="a1", tool_name="tx.transfer",
            decision="allow", reason="x", risk_score=0.6,
            approver="human", human_disposed=True,
        )
        live = _data(t, EventType.DECISION_MADE)
        t.record_decision(
            action_id="act-c", agent_id="a1", tool_name="tx.transfer",
            decision="allow", reason="y", risk_score=0.6,
            approver="policy", human_disposed=False,
        )
        replayed = _data(t, EventType.DECISION_MADE)

        assert live["decision"] == replayed["decision"] == "allow"
        assert live["human_disposed"] != replayed["human_disposed"]
        assert live["approver"] != replayed["approver"]


class TestTheRecordCannotOverstate:
    def test_policy_claiming_a_human_raises(self):
        t = _trail()
        with pytest.raises(DispositionError, match="must not claim a human"):
            t.record_decision(
                action_id="act-x", agent_id="a1", tool_name="t",
                decision="allow", reason="r", risk_score=0.1,
                approver="policy", human_disposed=True,
            )

    def test_bare_flag_with_no_approver_raises(self):
        t = _trail()
        with pytest.raises(DispositionError, match="requires approver"):
            t.record_decision(
                action_id="act-y", agent_id="a1", tool_name="t",
                decision="allow", reason="r", risk_score=0.1,
                human_disposed=True,
            )

    def test_unknown_approver_raises_rather_than_being_dropped(self):
        """Contrast with decision_detail, which is dropped with a warning.
        A bad disposition overstates; it must not degrade to silence."""
        t = _trail()
        with pytest.raises(DispositionError, match="closed"):
            t.record_decision(
                action_id="act-z", agent_id="a1", tool_name="t",
                decision="allow", reason="r", risk_score=0.1,
                approver="counterparty", human_disposed=False,
            )


class TestHashesDoNotMove:
    def test_a_record_without_a_disposition_hashes_as_before(self):
        """Every existing trail must verify unchanged, so the no-disposition
        path has to produce byte-identical data to the pre-vocabulary code."""
        a, b = _trail(), _trail()
        for t in (a, b):
            t.record_decision(
                action_id="act-1", agent_id="a1", tool_name="fs.read",
                decision="allow", reason="low risk", risk_score=0.25,
            )
        rec_a = _data(a, EventType.DECISION_MADE)
        rec_b = _data(b, EventType.DECISION_MADE)
        assert rec_a == rec_b
        assert set(rec_a) == {"decision", "reason", "risk_score"}

    def test_empty_approver_adds_nothing_even_when_passed_explicitly(self):
        t = _trail()
        t.record_decision(
            action_id="act-2", agent_id="a1", tool_name="fs.read",
            decision="allow", reason="low risk", risk_score=0.25,
            approver="", human_disposed=False,
        )
        assert set(_data(t, EventType.DECISION_MADE)) == {
            "decision", "reason", "risk_score",
        }

    def test_the_chain_still_verifies_with_dispositions_present(self):
        t = _trail()
        t.record_decision(
            action_id="act-3", agent_id="a1", tool_name="t",
            decision="allow", reason="r", risk_score=0.1,
            approver="human", human_disposed=True,
        )
        t.record_decision(
            action_id="act-4", agent_id="a1", tool_name="t",
            decision="allow", reason="r", risk_score=0.1,
            approver="policy", human_disposed=False,
        )
        # verify_chain returns None when the chain is intact, or a string
        # describing the first break.
        assert t.verify_chain() is None
