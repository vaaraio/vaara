"""Invariants from the 2026-09-26 review, each pinned by a test that failed
before the fix.

- An outcome exists only for an action that could run.
- One action gets at most one outcome, across threads and processes.
- One escalation gets at most one resolution.
- Conformal coverage is judged against the interval the prediction used,
  and the quantile is the k-th smallest residual, k = ceil((1-alpha)(n+1)).
"""

import math
import threading
import time

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline
from vaara.scorer.adaptive import ConformalCalibrator


def _pipeline(db=":memory:", enforce=True):
    backend = SQLiteAuditBackend(db)
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail, enforce=enforce)


def _count(pipeline, action_id, event_type):
    return sum(r.event_type == event_type for r in pipeline.trail.get_action_trail(action_id))


def _slow_trail_writes(pipeline, method):
    orig = getattr(pipeline.trail, method)

    def slow(**kw):
        time.sleep(0.2)
        return orig(**kw)

    setattr(pipeline.trail, method, slow)


class TestOutcomeOnlyForActionsThatCanRun:
    def test_denied_action_takes_no_outcome(self):
        p = _pipeline()
        r = p.intercept(agent_id="a", tool_name="x.y", policy_decision="deny", policy_reason="t")
        assert r.allowed is False
        p.report_outcome(r.action_id, 0.0)
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 0
        assert p.trail._backend.get_pending_outcome(r.action_id) is None

    def test_shadow_mode_deny_still_runs_so_it_takes_an_outcome(self):
        p = _pipeline(enforce=False)
        r = p.intercept(agent_id="a", tool_name="x.y", policy_decision="deny", policy_reason="t")
        assert r.allowed is True and r.decision == "deny"
        p.report_outcome(r.action_id, 0.2)
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 1

    def test_allowed_action_takes_an_outcome(self):
        p = _pipeline()
        r = p.intercept(agent_id="a", tool_name="x.y", policy_decision="allow", policy_reason="t")
        p.report_outcome(r.action_id, 0.1)
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 1

    def test_escalation_denied_by_reviewer_drops_its_outcome(self):
        p = _pipeline()
        r = p.intercept(agent_id="a", tool_name="x.y", policy_decision="escalate", policy_reason="t")
        assert r.decision == "escalate"
        p.resolve_escalation(r.action_id, "deny", reviewer="r")
        p.report_outcome(r.action_id, 0.0)
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 0

    def test_escalation_approved_keeps_its_outcome(self):
        p = _pipeline()
        r = p.intercept(agent_id="a", tool_name="x.y", policy_decision="escalate", policy_reason="t")
        p.resolve_escalation(r.action_id, "allow", reviewer="r")
        p.report_outcome(r.action_id, 0.0)
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 1


class TestOneOutcomePerAction:
    def test_two_threads_record_one_outcome(self):
        p = _pipeline()
        r = p.intercept(agent_id="a", tool_name="x.y", policy_decision="allow", policy_reason="t")
        _slow_trail_writes(p, "record_outcome")
        ts = [threading.Thread(target=p.report_outcome, args=(r.action_id, 0.1)) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 1

    def test_two_processes_on_one_database_record_one_outcome(self, tmp_path):
        db = tmp_path / "trail.db"
        checker = _pipeline(db)
        r = checker.intercept(agent_id="a", tool_name="x.y", policy_decision="allow", policy_reason="t")
        # Two fresh pipelines stand in for two `vaara outcome` processes:
        # neither holds the entry in memory, both find the database row.
        one, two = _pipeline(db), _pipeline(db)
        _slow_trail_writes(one, "record_outcome")
        _slow_trail_writes(two, "record_outcome")
        ts = [threading.Thread(target=x.report_outcome, args=(r.action_id, 0.1)) for x in (one, two)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        written = sum(_count(x, r.action_id, EventType.OUTCOME_RECORDED) for x in (one, two))
        assert written == 1

    def test_failed_trail_write_leaves_the_outcome_for_a_retry(self):
        p = _pipeline()
        r = p.intercept(agent_id="a", tool_name="x.y", policy_decision="allow", policy_reason="t")
        orig = p.trail.record_outcome
        calls = {"n": 0}

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk full")
            return orig(**kw)

        p.trail.record_outcome = flaky
        p.report_outcome(r.action_id, 0.1)
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 0
        assert p.trail._backend.get_pending_outcome(r.action_id) is not None
        p.report_outcome(r.action_id, 0.1)
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 1
        p.report_outcome(r.action_id, 0.1)
        assert _count(p, r.action_id, EventType.OUTCOME_RECORDED) == 1


class TestOneResolutionPerEscalation:
    def test_two_threads_record_one_resolution(self):
        p = _pipeline()
        r = p.intercept(agent_id="a", tool_name="x.y", policy_decision="escalate", policy_reason="t")
        _slow_trail_writes(p, "record_escalation_resolved")
        ts = [
            threading.Thread(target=p.resolve_escalation, args=(r.action_id, v, "r"))
            for v in ("allow", "deny", "allow")
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert _count(p, r.action_id, EventType.ESCALATION_RESOLVED) == 1


class TestConformalQuantile:
    def _calibrated(self, residuals, alpha):
        c = ConformalCalibrator(alpha=alpha, min_calibration=10**6)
        for x in residuals:
            c.add_calibration_point(x, 0.0)
        return c

    def test_quantile_is_the_kth_smallest_residual(self):
        for n in (1, 2, 3, 10, 29, 30, 31, 100, 257):
            residuals = [i / 1000 for i in range(1, n + 1)]
            for alpha in (0.01, 0.05, 0.1, 0.2):
                c = self._calibrated(residuals, alpha)
                k = math.ceil((1 - alpha) * (n + 1))
                want = residuals[min(k, n) - 1]
                assert c._get_quantile() == want, (n, alpha)

    def test_n30_alpha10_is_the_28th_value_not_the_29th(self):
        residuals = [i / 100 for i in range(1, 31)]
        assert self._calibrated(residuals, 0.1)._get_quantile() == 0.28


class TestConformalOrdering:
    def test_new_residual_is_judged_against_the_interval_it_was_predicted_with(self):
        # Ten residuals of 0 give an interval of width 0. A residual of 0.5 is
        # outside it: a miss, so alpha must fall. Judged against a set that
        # already held the 0.5, it read as covered and alpha rose.
        c = ConformalCalibrator(alpha=0.1, min_calibration=10, gamma=0.05)
        for _ in range(10):
            c.add_calibration_point(0.0, 0.0)
        before = c.effective_alpha
        c.add_calibration_point(0.5, 0.0)
        assert c.effective_alpha < before

    def test_a_covered_residual_raises_alpha(self):
        c = ConformalCalibrator(alpha=0.1, min_calibration=10, gamma=0.05)
        for _ in range(10):
            c.add_calibration_point(0.3, 0.0)
        before = c.effective_alpha
        c.add_calibration_point(0.1, 0.0)
        assert c.effective_alpha > before


class TestLongRunCoverage:
    """The README's promise: the miss rate settles at alpha whatever the
    input sequence does (FACI, docs/formal_specification.md 5.2). Checked on
    sequences built to break a fixed-alpha interval."""

    T = 20_000

    def _miss_rate(self, seq, alpha):
        c = ConformalCalibrator(alpha=alpha)
        miss = n = 0
        for r in seq:
            if c.is_calibrated_for(None):
                miss += r > c._get_quantile()
                n += 1
            c.add_calibration_point(r, 0.0)
        return miss / n

    def _sequences(self):
        import random
        rng = random.Random(7)
        T = self.T
        yield "abrupt shift", [rng.random() * (0.15 if t < T // 2 else 0.75) for t in range(T)]
        yield "regime switching", [rng.random() * (0.1 if (t // 2000) % 2 else 0.9) for t in range(T)]
        adversarial, top = [], 0.001
        for t in range(T):
            # Every 500 steps, 50 residuals each above everything seen so far.
            if t % 500 < 50:
                top = min(1.0, top * 1.3)
                adversarial.append(top)
            else:
                adversarial.append(rng.random() * top)
        yield "always above the last maximum", adversarial

    def test_miss_rate_settles_at_alpha_under_shift(self):
        for name, seq in self._sequences():
            for alpha in (0.1, 0.05):
                rate = self._miss_rate(seq, alpha)
                assert abs(rate - alpha) < 0.01, (name, alpha, rate)
class TestJsonSafeKeepsEveryKey:
    def test_colliding_keys_are_both_kept(self):
        from vaara._sanitize import json_safe
        out = json_safe({1: "A", "1": "B"})
        assert sorted(out.values()) == ["A", "B"]
        assert json_safe({True: "x", "True": "y"}) == {"True": "x", "True<str>": "y"}

    def test_a_dict_without_collisions_is_unchanged(self):
        from vaara._sanitize import json_safe
        assert json_safe({"a": 1, 2: "b", None: 3}) == {"a": 1, "2": "b", "None": 3}
