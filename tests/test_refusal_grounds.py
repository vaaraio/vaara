"""A deny says whether a check ran and failed, or never completed.

A refusal word alone cannot tell a control that held from a control that was
never reached: a deny from a scorer that could not be reached reads the same
as a deny from one that scored the action too risky. The trail records which,
and the signed receipt carries it as a ``refusal`` block, naming the input
that was missing where a check could not complete. A deny whose ground the
trail does not record is ``undetermined``; the receipt never claims a check
ran when the trail does not say so.
"""

import pytest

from vaara.audit.decision_receipts import build_evidence
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail
from vaara.pipeline import InterceptionPipeline
from vaara.scorer.composite import CompositeScorer
from vaara.scorer.composition import ExternalScorer

_DEAD = "http://user:secret@127.0.0.1:1/score?token=abc"


class _FixedScorer:
    def __init__(self, **payload) -> None:
        self._payload = {
            "reason": "stub",
            "raw_result": {"point_estimate": 0.9, "conformal_interval": [0.85, 0.95],
                           "signals": {}},
            **payload,
        }

    def evaluate(self, context):
        return dict(self._payload)


class _RaisingScorer:
    def evaluate(self, context):
        raise RuntimeError("model bundle corrupt")


class _Member:
    """A composite member answering in the scorer-protocol shape."""

    def __init__(self, score: float, decision: str, name: str) -> None:
        self._score, self._decision, self.name = score, decision, name

    def evaluate(self, context):
        return {"point_estimate": self._score, "decision": self._decision,
                "backend": self.name,
                "raw_result": {"point_estimate": self._score,
                               "conformal_interval": [self._score, self._score]}}


def _pipeline(scorer):
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail, scorer=scorer)


def _decision(pipeline):
    records = [r for r in pipeline.trail._records if "decision" in (r.data or {})]
    assert records, "no decision record on the trail"
    return records[-1]


def _refusal(scorer):
    pipeline = _pipeline(scorer)
    try:
        pipeline.intercept(agent_id="a", tool_name="tx.transfer")
    except RuntimeError:
        pass  # the scorer-exception path re-raises after recording the deny
    record = _decision(pipeline)
    assert record.data["decision"] == "deny"
    return record, build_evidence(record)["refusal"]


# ── A check that never completed ──────────────────────────────────────────

def test_a_scorer_that_raised_is_not_a_judgement():
    record, refusal = _refusal(_RaisingScorer())
    assert refusal["basis"] == "not_evaluated"
    assert refusal["violationType"] == "scorer_failure"
    assert refusal["unavailableInput"] == "scorer:_RaisingScorer"
    assert record.data["unavailable_input"] == "scorer:_RaisingScorer"


def test_an_unreachable_external_scorer_is_not_a_risk_threshold():
    """The case that was recorded as `risk_threshold` with risk 1.0."""
    _, refusal = _refusal(ExternalScorer(_DEAD, timeout=0.05))
    assert refusal["basis"] == "not_evaluated"
    assert refusal["violationType"] == "scorer_unavailable"
    assert refusal["unavailableInput"] == "scorer:127.0.0.1:1"


def test_the_missing_input_never_carries_the_scorer_url():
    """The name defaults to the URL; credentials and query stay off the record."""
    record, refusal = _refusal(ExternalScorer(_DEAD, timeout=0.05))
    for text in (refusal["unavailableInput"], record.data["unavailable_input"]):
        assert "secret" not in text and "token" not in text and "/score" not in text


def test_a_scorer_answer_with_no_decision_is_not_evaluated():
    _, refusal = _refusal(_FixedScorer())
    assert refusal["basis"] == "not_evaluated"
    assert refusal["violationType"] == "invalid_decision"
    assert refusal["unavailableInput"] == "scorer:_FixedScorer.action"


def test_a_decision_outside_the_vocabulary_is_not_evaluated():
    _, refusal = _refusal(_FixedScorer(action="maybe"))
    assert refusal["basis"] == "not_evaluated"
    assert refusal["violationType"] == "invalid_decision"


def test_a_composite_with_a_dead_member_keeps_the_ground():
    """Mean mode builds a new result; the dead member's ground must survive it."""
    live = _Member(0.2, "allow", "local")
    dead = ExternalScorer(_DEAD, name="peer", timeout=0.05)
    _, refusal = _refusal(CompositeScorer([live, dead], combine="mean"))
    assert refusal["basis"] == "not_evaluated"
    assert refusal["unavailableInput"] == "scorer:peer"


# ── A check that ran and failed the action ────────────────────────────────

def test_a_scored_deny_is_evaluated():
    _, refusal = _refusal(_FixedScorer(action="deny"))
    assert refusal == {"basis": "evaluated", "violationType": "risk_threshold",
                       "policyId": "scorer:_FixedScorer"}


def test_a_live_external_scorer_decision_reaches_the_pipeline():
    """The external scorer's verdict is read, not replaced by a deny."""
    pipeline = _pipeline(CompositeScorer([_Member(0.1, "allow", "local")]))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    assert result.decision == "allow"


# ── What the receipt will not claim ───────────────────────────────────────

def test_a_deny_with_no_recorded_ground_is_undetermined():
    trail = AuditTrail()
    trail.record_decision(action_id="x", agent_id="a", tool_name="t",
                          decision="deny", reason="custom code", risk_score=0.9)
    refusal = build_evidence(trail._records[-1])["refusal"]
    assert refusal == {"basis": "undetermined"}


def test_an_unknown_ground_is_undetermined():
    trail = AuditTrail()
    trail.record_decision(action_id="x", agent_id="a", tool_name="t",
                          decision="deny", reason="r", risk_score=0.9,
                          violation_type="vendor_specific")
    assert build_evidence(trail._records[-1])["refusal"]["basis"] == "undetermined"


def test_an_input_is_not_named_on_a_check_that_ran():
    trail = AuditTrail()
    trail.record_decision(action_id="x", agent_id="a", tool_name="t",
                          decision="deny", reason="r", risk_score=0.9,
                          violation_type="policy_rule", unavailable_input="scorer:x")
    assert "unavailableInput" not in build_evidence(trail._records[-1])["refusal"]


@pytest.mark.parametrize("decision", ["allow", "escalate"])
def test_only_a_deny_carries_a_refusal(decision):
    trail = AuditTrail()
    trail.record_decision(action_id="x", agent_id="a", tool_name="t",
                          decision=decision, reason="r", risk_score=0.1,
                          unavailable_input="scorer:x")
    record = trail._records[-1]
    assert "refusal" not in build_evidence(record)
    assert "unavailable_input" not in record.data


# ── The published vectors carry all three ─────────────────────────────────

def test_the_trail_decision_vectors_cover_every_basis():
    import json
    from pathlib import Path

    valid = Path(__file__).parent / "vectors" / "trail_decision_v0" / "valid"
    bases = {}
    for path in sorted(valid.glob("*.json")):
        evidence = json.loads(path.read_text())["evidence"]
        if evidence["decision"] == "deny":
            bases[path.name] = evidence["refusal"]["basis"]
        else:
            assert "refusal" not in evidence
    assert set(bases.values()) == {"evaluated", "not_evaluated", "undetermined"}
