"""The five-light decision vocabulary and its projection onto the wire.

AARM Core R4 asks for ALLOW, DENY, MODIFY, STEP_UP and DEFER. The signed
decision record cannot carry five: published conformance checkers hold a
closed set of ``{allow, block, escalate}`` and grade anything else
non-conforming, and six independent parties already run those checkers.

So the refinements live in the policy layer and project down. These tests pin
both halves: the new names exist and behave, and nothing that reaches the
chain or an existing integration changed shape.
"""

import pytest

from vaara.audit.receipts import _verdict_to_wire
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import _FINE_TO_COARSE, InterceptionPipeline
from vaara.scorer.adaptive import Decision, RiskAssessment


class _FixedScorer:
    """Scorer stub returning one fixed decision dict."""

    def __init__(self, action: str, **extra) -> None:
        self._payload = {
            "action": action,
            "reason": f"stub {action}",
            "raw_result": {
                "point_estimate": 0.5,
                "conformal_interval": [0.4, 0.6],
                "signals": {},
            },
            **extra,
        }

    def evaluate(self, context):
        return dict(self._payload)


def _pipeline(scorer):
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail, scorer=scorer)


def _decision_record(pipeline, action_id):
    for record in pipeline.trail.get_action_trail(action_id):
        if "decision" in (record.data or {}):
            return record
    raise AssertionError("no decision record on the trail")


# ── The enum ──────────────────────────────────────────────────────────────

def test_the_three_published_values_are_unchanged():
    """These strings are on other people's disks. They do not move."""
    assert Decision.ALLOW.value == "allow"
    assert Decision.DENY.value == "deny"
    assert Decision.ESCALATE.value == "escalate"


def test_r4_refinements_exist():
    assert Decision.MODIFY.value == "modify"
    assert Decision.STEP_UP.value == "step_up"
    assert Decision.DEFER.value == "defer"


def test_enum_has_exactly_six_members():
    assert len(Decision) == 6


@pytest.mark.parametrize("decision", list(Decision))
def test_allowed_is_true_for_allow_and_nothing_else(decision):
    """The one rule every relying party already knows, kept intact."""
    assessment = RiskAssessment(
        action_name="t", agent_id="a", point_estimate=0.5,
        conformal_lower=0.4, conformal_upper=0.6, decision=decision,
        signals={}, mwu_weights={}, threshold_allow=0.4, threshold_deny=0.7,
        sequence_risk=0.0, calibration_size=0,
    )
    backend = assessment.to_backend_decision()
    assert backend["allowed"] is (decision is Decision.ALLOW)
    assert backend["action"] == decision.value


# ── The projection ────────────────────────────────────────────────────────

def test_coarse_names_project_to_themselves():
    assert _FINE_TO_COARSE["allow"] == "allow"
    assert _FINE_TO_COARSE["deny"] == "deny"
    assert _FINE_TO_COARSE["escalate"] == "escalate"


def test_refinements_project_onto_the_coarse_three():
    assert _FINE_TO_COARSE["step_up"] == "escalate"
    assert _FINE_TO_COARSE["defer"] == "escalate"
    assert _FINE_TO_COARSE["modify"] == "deny"


def test_every_enum_member_has_a_projection():
    # list(Decision) rather than iterating the class directly: CodeQL does not
    # model EnumMeta.__iter__ and reports py/non-iterable-in-for-loop.
    for decision in list(Decision):
        assert decision.value in _FINE_TO_COARSE


def test_wire_projection_covers_the_refinements():
    """receipts._VERDICT_TO_WIRE is the second, independent projection."""
    assert _verdict_to_wire("step_up") == "escalate"
    assert _verdict_to_wire("defer") == "escalate"
    assert _verdict_to_wire("modify") == "block"
    # And the ones that were already there.
    assert _verdict_to_wire("deny") == "block"
    assert _verdict_to_wire("escalate") == "escalate"
    assert _verdict_to_wire("allow") == "allow"


# ── Through the pipeline ──────────────────────────────────────────────────

def test_step_up_holds_the_action_and_reads_as_escalate():
    pipeline = _pipeline(_FixedScorer("step_up"))
    result = pipeline.intercept(agent_id="a", tool_name="tx.sign")

    assert result.allowed is False
    assert result.decision == "escalate"      # what old integrations see
    assert result.decision_detail == "step_up"  # what R4 asks for


def test_defer_holds_the_action_and_reads_as_escalate():
    pipeline = _pipeline(_FixedScorer("defer"))
    result = pipeline.intercept(agent_id="a", tool_name="tx.sign")

    assert result.allowed is False
    assert result.decision == "escalate"
    assert result.decision_detail == "defer"


def test_modify_blocks_the_original_and_hands_back_the_change():
    pipeline = _pipeline(_FixedScorer(
        "modify", modified_parameters={"amount": 5000},
    ))
    result = pipeline.intercept(
        agent_id="a", tool_name="tx.transfer", parameters={"amount": 50000},
    )

    assert result.allowed is False
    assert result.decision == "deny"          # the original never runs
    assert result.decision_detail == "modify"
    assert result.modified_parameters == {"amount": 5000}


def test_modify_without_a_modification_fails_closed_to_plain_deny():
    """A scorer saying 'modify' but supplying nothing is malformed."""
    pipeline = _pipeline(_FixedScorer("modify"))
    result = pipeline.intercept(agent_id="a", tool_name="tx.transfer")

    assert result.allowed is False
    assert result.decision == "deny"
    assert result.decision_detail is None
    assert result.modified_parameters is None


def test_modify_with_a_non_dict_modification_fails_closed():
    pipeline = _pipeline(_FixedScorer("modify", modified_parameters=["nope"]))
    result = pipeline.intercept(agent_id="a", tool_name="tx.transfer")

    assert result.decision == "deny"
    assert result.decision_detail is None


def test_unknown_decision_still_fails_closed():
    """Regression: the fail-closed invariant predates this vocabulary."""
    pipeline = _pipeline(_FixedScorer("maybe"))
    result = pipeline.intercept(agent_id="a", tool_name="tx.sign")

    assert result.allowed is False
    assert result.decision == "deny"
    assert result.decision_detail is None


def test_plain_decisions_carry_no_detail():
    pipeline = _pipeline(_FixedScorer("allow"))
    result = pipeline.intercept(agent_id="a", tool_name="data.read")

    assert result.allowed is True
    assert result.decision == "allow"
    assert result.decision_detail is None


# ── What lands on the chain ───────────────────────────────────────────────

def test_the_chain_records_the_coarse_decision():
    """data.decision stays inside the documented allow/escalate/deny enum."""
    pipeline = _pipeline(_FixedScorer("step_up"))
    result = pipeline.intercept(agent_id="a", tool_name="tx.sign")

    record = _decision_record(pipeline, result.action_id)
    assert record.data["decision"] == "escalate"


def test_the_chain_also_records_the_refinement():
    """Unknown keys are permitted by the 1.0 schema, so the detail is evidence."""
    pipeline = _pipeline(_FixedScorer("defer"))
    result = pipeline.intercept(agent_id="a", tool_name="tx.sign")

    record = _decision_record(pipeline, result.action_id)
    assert record.data["decision_detail"] == "defer"


def test_a_plain_decision_record_gains_no_new_keys():
    """Existing records must stay byte-identical, or every hash moves."""
    pipeline = _pipeline(_FixedScorer("allow"))
    result = pipeline.intercept(agent_id="a", tool_name="data.read")

    record = _decision_record(pipeline, result.action_id)
    assert set(record.data) == {"decision", "reason", "risk_score"}


def test_modify_records_what_it_proposed():
    pipeline = _pipeline(_FixedScorer(
        "modify", modified_parameters={"amount": 5000},
    ))
    result = pipeline.intercept(
        agent_id="a", tool_name="tx.transfer", parameters={"amount": 50000},
    )

    record = _decision_record(pipeline, result.action_id)
    assert record.data["decision"] == "deny"
    assert record.data["decision_detail"] == "modify"
    assert record.data["modified_parameters"] == {"amount": 5000}


# ── The re-decision loop, which is the point of shape (a) ─────────────────

def test_the_retry_is_a_separate_decision_bound_to_its_own_arguments():
    """No approval is ever bound to arguments other than the ones that ran."""
    scorer = _FixedScorer("modify", modified_parameters={"amount": 5000})
    pipeline = _pipeline(scorer)

    first = pipeline.intercept(
        agent_id="a", tool_name="tx.transfer", parameters={"amount": 50000},
    )
    assert first.allowed is False

    # The caller resubmits what the gate proposed. Fresh scorer verdict.
    pipeline.scorer = _FixedScorer("allow")
    second = pipeline.intercept(
        agent_id="a", tool_name="tx.transfer",
        parameters=first.modified_parameters,
        parent_action_id=first.action_id,
    )

    assert second.allowed is True
    assert second.action_id != first.action_id

    blocked = _decision_record(pipeline, first.action_id)
    permitted = _decision_record(pipeline, second.action_id)
    assert blocked.data["decision"] == "deny"
    assert permitted.data["decision"] == "allow"
