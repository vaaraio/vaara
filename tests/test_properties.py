"""Property-based tests (Hypothesis) for core invariants.

These fuzz the interception pipeline and scorer primitives with
arbitrary inputs to catch crashes and invariant violations that
example-based tests miss. Agent tool-call inputs in the wild vary
widely; the intercept boundary must hold under any shape of dict.
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vaara.pipeline import InterceptionPipeline
from vaara.scorer.adaptive import ConformalCalibrator, MWUExperts


json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**31), max_value=2**31 - 1)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(max_size=64)
)

json_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=32),
    values=json_scalars,
    max_size=8,
)

finite_floats_unit = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

signal_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=16),
    values=st.floats(allow_nan=False, allow_infinity=False, width=32),
    min_size=1,
    max_size=6,
)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    tool_name=st.text(min_size=1, max_size=64),
    params=json_dicts,
    context=json_dicts,
    confidence=st.one_of(st.none(), finite_floats_unit),
)
def test_intercept_never_crashes_and_score_bounded(tool_name, params, context, confidence):
    """For any reasonable tool call, intercept() returns a bounded score
    and a valid decision label. No arbitrary payload should crash the
    boundary: that is the whole point of the interception layer."""
    pipeline = InterceptionPipeline()
    result = pipeline.intercept(
        agent_id="fuzz-agent",
        tool_name=tool_name,
        parameters=params,
        context=context,
        agent_confidence=confidence,
    )
    assert 0.0 <= result.risk_score <= 1.0
    assert result.decision in ("allow", "escalate", "deny")
    assert result.evaluation_ms >= 0.0
    assert result.action_id


@settings(max_examples=200, deadline=None)
@given(signals=signal_dicts)
def test_mwu_predict_stays_in_unit_interval(signals):
    """MWUExperts.predict must clamp any signal vector into [0,1].
    Scorer outputs flow into decision thresholds; an out-of-band score
    would poison every downstream gate comparison."""
    mwu = MWUExperts(list(signals.keys()))
    score = mwu.predict(signals)
    assert 0.0 <= score <= 1.0


@settings(max_examples=200, deadline=None)
@given(score=finite_floats_unit)
def test_conformal_interval_is_valid(score):
    """Pre-calibration conformal intervals must still form a valid
    [lower, upper] pair inside [0,1] for any score. If this ever
    inverts, downstream risk-gate logic compares against nonsense."""
    cc = ConformalCalibrator(min_calibration=30)
    lower, upper = cc.predict_interval(score)
    assert 0.0 <= lower <= upper <= 1.0
