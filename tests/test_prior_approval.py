"""Regression tests for prior-approval auto-allow (2026-08 audit C2).

Three invariants pin the audited defects:
1. Approving ``tx.transfer`` amount=10 must NOT auto-allow amount=999999
   (the args-digest guard was vacuous because the digest was never stored).
2. An auto-allowed action's trail must show decision=allow with NO
   ESCALATION_SENT — the record and the behaviour must agree.
3. Tenant A's approval must never auto-allow tenant B's same-named
   agent + tool (the lookup was not tenant-scoped).
"""

import hashlib
import json

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline


@pytest.fixture
def pipeline():
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail)


def _escalate_and_approve(pipeline, amount, *, agent="agent-1", tenant=""):
    result = pipeline.intercept(
        agent_id=agent,
        tool_name="tx.transfer",
        parameters={"amount": amount},
        tenant_id=tenant,
    )
    assert result.decision == "escalate", (
        f"precondition: tx.transfer amount={amount} must escalate, "
        f"got {result.decision}"
    )
    pipeline.resolve_escalation(
        action_id=result.action_id,
        resolution="allow",
        reviewer="admin@example.com",
        justification="approved in review",
    )
    return result


def _expected_digest(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()


class TestPriorApprovalAutoAllow:
    def test_same_shape_auto_allows(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
        )
        assert second.allowed is True
        assert second.decision == "allow"
        assert "prior approval" in second.reason

    def test_different_amount_does_not_auto_allow(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 999999},
        )
        assert second.allowed is False
        assert second.decision == "escalate"

    def test_auto_allow_trail_agrees_with_behaviour(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
        )
        trail = pipeline.trail.get_action_trail(second.action_id)
        event_types = [r.event_type for r in trail]
        decisions = [
            r for r in trail if r.event_type == EventType.DECISION_MADE
        ]
        # Caller was allowed; the chain must record allow, never a
        # dangling escalate with no ESCALATION_SENT behind it.
        assert EventType.ESCALATION_SENT not in event_types
        assert len(decisions) == 1
        assert decisions[0].data["decision"] == "allow"
        assert "prior approval" in decisions[0].data["reason"]

    def test_real_escalation_still_records_sent(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 999999},
        )
        trail = pipeline.trail.get_action_trail(second.action_id)
        event_types = [r.event_type for r in trail]
        assert EventType.ESCALATION_SENT in event_types

    def test_tenant_isolation(self, pipeline):
        _escalate_and_approve(pipeline, amount=10, tenant="tenant-a")
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
            tenant_id="tenant-b",
        )
        assert second.decision == "escalate"
        assert second.allowed is False

    def test_resolution_record_carries_args_digest(self, pipeline):
        approved = _escalate_and_approve(pipeline, amount=10)
        trail = pipeline.trail.get_action_trail(approved.action_id)
        resolved = [
            r for r in trail
            if r.event_type == EventType.ESCALATION_RESOLVED
        ]
        assert len(resolved) == 1
        assert resolved[0].data["args_digest"] == _expected_digest(
            {"amount": 10}
        )

    def test_digest_persists_across_restart(self, tmp_path):
        db = str(tmp_path / "audit.db")
        backend = SQLiteAuditBackend(db)
        trail = backend.load_trail()
        trail._on_record = backend.write_record
        p1 = InterceptionPipeline(trail=trail)
        _escalate_and_approve(p1, amount=10)

        backend2 = SQLiteAuditBackend(db)
        trail2 = backend2.load_trail()
        trail2._on_record = backend2.write_record
        p2 = InterceptionPipeline(trail=trail2)
        second = p2.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
        )
        assert second.decision == "allow"
        # ...and the guard survives the restart too:
        third = p2.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 999999},
        )
        assert third.decision == "escalate"


class TestPriorApprovalScanCost:
    """The lookup must not walk the whole trail (2026-09-08 benchmark).

    `AuditTrail.find_prior_approval` accounted for 62 percent of a 3,000 call
    `bench/latency.py` run: 5.676 s of 9.199 s, in its own frame. It copied the
    entire trail on every intercept and its reverse scan used `continue` where
    the time ordering permits `break`, so a call with no prior approval walked
    the whole history every time. Cost per call therefore rose with the number
    of calls already made, and `unknown.tool` fell from 13,052 ops/sec at 500
    calls to 152 at 10,000.

    A missed approval leaves `decision_str == "escalate"` (pipeline.py:689), so
    this lookup fails CLOSED. Any fix can only cost an extra human review, never
    an auto-allow, which is what makes the area safe to change.

    CORRECTED 2026-09-08 after a failed first attempt, recorded so the next
    reader does not repeat it. Changing the window check from `continue` to
    `break` looks right, because records append in time order. It buys nothing
    under the load that matters. At eHealth rates, 23,000 services an hour, a
    full 24 hour window holds about 552,000 records and EVERY one of them is
    inside the window, so the early exit never fires. Measured: with all
    records in-window the scan stayed dead linear, and moving it under the lock
    made contention worse.

    The real fix is an index of ESCALATION_RESOLVED records. Those are rare
    even over a busy day, so scanning only them is cheap regardless of trail
    length. Queued as `perf-find-prior-approval-scan`.
    """

    def test_approval_older_than_the_window_does_not_auto_allow(self, pipeline):
        """The property the early exit could break if ordering ever slipped."""
        first = _escalate_and_approve(pipeline, amount=10)
        # Age every record in the trail past the 24 hour window.
        with pipeline.trail._lock:
            for record in pipeline.trail._records:
                record.timestamp -= 25 * 3600
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
        )
        assert second.allowed is False, (
            "an approval older than window_hours must not auto-allow"
        )
        assert second.decision == "escalate"
        assert first.action_id not in (second.reason or "")

    def test_approval_inside_the_window_still_auto_allows(self, pipeline):
        """The other side of the boundary, so the fix cannot over-tighten."""
        _escalate_and_approve(pipeline, amount=10)
        with pipeline.trail._lock:
            for record in pipeline.trail._records:
                record.timestamp -= 23 * 3600
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
        )
        assert second.allowed is True
        assert second.decision == "allow"

    def test_lookup_cost_does_not_grow_with_trail_length(self, pipeline):
        """A long trail must answer in time comparable to a short one.

        This is the regression the benchmark found. It measures the lookup
        directly rather than through `intercept`, so a change in scoring cost
        cannot mask a change in scan cost. Every filler record is left INSIDE
        the 24 hour window on purpose, because that is the shape of a real
        high-throughput day and it is the case a window-edge early exit cannot
        help with.
        """
        import time as _time

        trail = pipeline.trail

        def _elapsed_for_lookup() -> float:
            start = _time.perf_counter()
            for _ in range(200):
                trail.find_prior_approval(
                    agent_id="absent-agent", tool_name="tx.transfer",
                    args_digest="deadbeef",
                )
            return _time.perf_counter() - start

        _escalate_and_approve(pipeline, amount=10)
        short_trail = _elapsed_for_lookup()

        # Grow the trail well past anything the lookup should care about.
        # Grow _records directly by cloning a real record. The scan reads
        # _records and nothing else, so this isolates trail LENGTH from every
        # other cost in the intercept path.
        import copy as _copy
        with trail._lock:
            template = trail._records[0]
            for i in range(20000):
                filler = _copy.copy(template)
                filler.agent_id = f"filler-{i}"
                trail._records.append(filler)

        long_trail = _elapsed_for_lookup()

        assert len(trail._records) > 20000, "precondition: the trail grew"
        # Generous bound. Before the fix this ratio was in the tens; the point
        # is to catch a return to linear scanning, not to police jitter.
        assert long_trail < short_trail * 5 + 0.005, (
            f"lookup cost grew with trail length: {short_trail:.4f}s over a "
            f"short trail, {long_trail:.4f}s over {len(trail._records)} "
            f"records. The scan is walking the whole history again."
        )


class TestApprovalIndexStaysComplete:
    """The index must never be a subset of the resolutions in the trail.

    `find_prior_approval` scans `_resolved_approvals` instead of `_records`,
    which is what makes it cheap. The cost of that is a second structure that
    can drift, and drift here is SILENT and fails CLOSED: a missing entry means
    the action escalates to a human a second time rather than being
    auto-allowed. Nothing raises, nothing logs, and the only symptom is a
    reviewer being asked twice.

    The path that would have been forgotten is `load_trail`, because reloaded
    records never pass through `_append`. These tests exist so that forgetting
    it is a red build rather than a support ticket.
    """

    def _resolutions_in(self, trail):
        return [r for r in trail._records
                if r.event_type == EventType.ESCALATION_RESOLVED]

    def test_index_matches_the_trail_after_appends(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        _escalate_and_approve(pipeline, amount=20)
        trail = pipeline.trail
        assert self._resolutions_in(trail) == trail._resolved_approvals, (
            "the index and the trail disagree about which records are "
            "resolutions"
        )
        assert len(trail._resolved_approvals) == 2

    def test_index_is_rebuilt_when_a_trail_is_reloaded(self, tmp_path):
        """The regression. Reloaded records bypass _append entirely."""
        db = tmp_path / "trail.db"
        backend = SQLiteAuditBackend(str(db))
        trail = backend.load_trail()
        trail._on_record = backend.write_record
        pipe = InterceptionPipeline(trail=trail)
        _escalate_and_approve(pipe, amount=10)
        assert len(trail._resolved_approvals) == 1, "precondition"

        reloaded = SQLiteAuditBackend(str(db)).load_trail()
        assert self._resolutions_in(reloaded) == reloaded._resolved_approvals
        assert len(reloaded._resolved_approvals) == 1, (
            "a reloaded trail lost its approval index, so every prior "
            "approval is invisible and every action re-escalates"
        )

    def test_prior_approval_still_found_after_a_reload(self, tmp_path):
        """The behaviour the index exists to serve, across a restart.

        The docstring on find_prior_approval promises that approved action
        shapes survive process restarts within the time window. That promise
        now depends on load_trail maintaining the index.
        """
        db = tmp_path / "trail.db"
        backend = SQLiteAuditBackend(str(db))
        trail = backend.load_trail()
        trail._on_record = backend.write_record
        pipe = InterceptionPipeline(trail=trail)
        first = _escalate_and_approve(pipe, amount=10)

        reloaded = SQLiteAuditBackend(str(db)).load_trail()
        found = reloaded.find_prior_approval(
            agent_id="agent-1", tool_name="tx.transfer",
            args_digest=_expected_digest({"amount": 10}),
        )
        assert found is not None, (
            "a prior approval must survive a restart; it did not, which means "
            "the reviewer is asked to approve the same action again"
        )
        assert found.event_type == EventType.ESCALATION_RESOLVED
        assert (found.data or {}).get("resolution") == "allow"
        assert first.action_id

    def test_index_holds_only_resolutions(self, pipeline):
        """A wider index would reintroduce the cost this change removed."""
        _escalate_and_approve(pipeline, amount=10)
        pipeline.intercept(agent_id="agent-1", tool_name="db.read",
                           parameters={"table": "t"})
        trail = pipeline.trail
        assert len(trail._records) > len(trail._resolved_approvals)
        assert all(r.event_type == EventType.ESCALATION_RESOLVED
                   for r in trail._resolved_approvals)

    def test_records_is_only_appended_through_index_record(self):
        """Structural guard, because a future bulk loader is the real risk.

        Documentation did not stop `load_trail` from appending by hand the
        first time. This fails the build if a new writer does the same.
        """
        import pathlib
        import re
        root = pathlib.Path(__file__).resolve().parent.parent / "src" / "vaara"
        offenders = []
        for path in (root / "audit" / "trail.py",
                     root / "audit" / "sqlite_backend.py"):
            for number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"_records\.append\(", line):
                    # The one legitimate appender.
                    if path.name == "trail.py" and "self._records.append" in line:
                        continue
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        assert not offenders, (
            "these append to _records without going through _index_record, so "
            "the approval index will silently miss those records:\n  "
            + "\n  ".join(offenders)
        )
