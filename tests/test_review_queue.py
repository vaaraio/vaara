"""Tests for the human-in-the-loop review queue (EU AI Act Article 14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaara.audit.review_queue import (
    InvalidTransitionError,
    ItemNotFoundError,
    RESOLUTION_ABSTAIN,
    RESOLUTION_ALLOW,
    RESOLUTION_DENY,
    ReviewQueue,
    STATUS_CLAIMED,
    STATUS_EXPIRED,
    STATUS_PENDING,
    STATUS_RESOLVED,
)


def _enqueue_one(q: ReviewQueue, **kw) -> str:
    base = dict(
        action_id="act-1", agent_id="agent-007", tool_name="fs.write",
        risk_score=0.62, conformal_lower=0.55, conformal_upper=0.71,
        action_type="fs.write_file", bucket_category="filesystem",
        reason="borderline score",
        parameters={"path": "/etc/foo"}, context={"session": "s1"},
        signals={"heuristic": 0.4, "classifier": 0.6},
    )
    base.update(kw)
    return q.enqueue(**base)


def test_enqueue_then_get_round_trips_all_fields() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q)
    item = q.get(qid)
    assert item.queue_id == qid
    assert item.action_id == "act-1"
    assert item.agent_id == "agent-007"
    assert item.tool_name == "fs.write"
    assert item.action_type == "fs.write_file"
    assert item.risk_score == pytest.approx(0.62)
    assert item.conformal_lower == pytest.approx(0.55)
    assert item.conformal_upper == pytest.approx(0.71)
    assert item.bucket_category == "filesystem"
    assert item.reason == "borderline score"
    assert item.parameters == {"path": "/etc/foo"}
    assert item.context == {"session": "s1"}
    assert item.signals == {"heuristic": 0.4, "classifier": 0.6}
    assert item.status == STATUS_PENDING
    assert item.interval_width == pytest.approx(0.16)


def test_get_unknown_raises_item_not_found() -> None:
    q = ReviewQueue(":memory:")
    with pytest.raises(ItemNotFoundError):
        q.get("does-not-exist")


def test_list_items_defaults_to_pending() -> None:
    q = ReviewQueue(":memory:")
    qid1 = _enqueue_one(q, action_id="a1", now=100.0)
    qid2 = _enqueue_one(q, action_id="a2", now=200.0)
    items = q.list_items()
    assert [i.queue_id for i in items] == [qid1, qid2]
    assert all(i.status == STATUS_PENDING for i in items)


def test_list_items_status_none_returns_all() -> None:
    q = ReviewQueue(":memory:")
    qid1 = _enqueue_one(q, action_id="a1", now=100.0)
    qid2 = _enqueue_one(q, action_id="a2", now=200.0)
    q.resolve(qid1, reviewer="alice", resolution=RESOLUTION_DENY)
    items = q.list_items(status=None)
    statuses = {i.queue_id: i.status for i in items}
    assert statuses[qid1] == STATUS_RESOLVED
    assert statuses[qid2] == STATUS_PENDING


def test_list_items_filter_by_agent_id() -> None:
    q = ReviewQueue(":memory:")
    _enqueue_one(q, action_id="a1", agent_id="alpha")
    _enqueue_one(q, action_id="a2", agent_id="beta")
    out = q.list_items(agent_id="beta")
    assert len(out) == 1
    assert out[0].agent_id == "beta"


def test_claim_marks_pending_as_claimed_with_reviewer() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q)
    item = q.claim(qid, reviewer="alice", now=500.0)
    assert item.status == STATUS_CLAIMED
    assert item.claimed_by == "alice"
    assert item.claimed_at == pytest.approx(500.0)


def test_double_claim_raises_invalid_transition() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q)
    q.claim(qid, reviewer="alice")
    with pytest.raises(InvalidTransitionError):
        q.claim(qid, reviewer="bob")


def test_resolve_writes_terminal_state_and_resolution() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q)
    item = q.resolve(
        qid, reviewer="alice", resolution=RESOLUTION_ALLOW,
        justification="reviewed parameters", now=900.0,
    )
    assert item.status == STATUS_RESOLVED
    assert item.resolution == RESOLUTION_ALLOW
    assert item.resolved_by == "alice"
    assert item.justification == "reviewed parameters"
    assert item.resolved_at == pytest.approx(900.0)


def test_resolve_after_claim_is_allowed() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q)
    q.claim(qid, reviewer="alice")
    item = q.resolve(qid, reviewer="alice", resolution=RESOLUTION_DENY)
    assert item.status == STATUS_RESOLVED
    assert item.resolution == RESOLUTION_DENY


def test_resolve_already_resolved_raises() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q)
    q.resolve(qid, reviewer="alice", resolution=RESOLUTION_ABSTAIN)
    with pytest.raises(InvalidTransitionError):
        q.resolve(qid, reviewer="bob", resolution=RESOLUTION_DENY)


def test_resolve_rejects_unknown_resolution() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q)
    with pytest.raises(ValueError):
        q.resolve(qid, reviewer="alice", resolution="maybe")


def test_resolve_writes_escalation_resolved_when_trail_supplied() -> None:
    from vaara.audit.trail import AuditTrail, EventType
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q)
    trail = AuditTrail()
    q.resolve(
        qid, reviewer="alice", resolution=RESOLUTION_ALLOW,
        justification="ok", trail=trail,
    )
    resolved = [r for r in trail._records
                if r.event_type == EventType.ESCALATION_RESOLVED]
    assert len(resolved) == 1
    assert resolved[0].action_id == "act-1"
    assert resolved[0].data["resolution"] == RESOLUTION_ALLOW
    assert resolved[0].data["reviewer"] == "alice"
    assert resolved[0].data["justification"] == "ok"


def test_expire_stale_marks_old_pending_items_expired() -> None:
    q = ReviewQueue(":memory:")
    qid_old = _enqueue_one(q, action_id="old", now=1000.0)
    qid_new = _enqueue_one(q, action_id="new", now=2000.0)
    n = q.expire_stale(timeout_seconds=500, now=2000.0)
    assert n == 1
    assert q.get(qid_old).status == STATUS_EXPIRED
    assert q.get(qid_new).status == STATUS_PENDING


def test_expire_dry_run_does_not_mutate() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q, now=1000.0)
    n = q.expire_stale(timeout_seconds=500, now=2000.0, dry_run=True)
    assert n == 1
    assert q.get(qid).status == STATUS_PENDING


def test_expire_leaves_claimed_items_alone() -> None:
    q = ReviewQueue(":memory:")
    qid = _enqueue_one(q, now=1000.0)
    q.claim(qid, reviewer="alice", now=1100.0)
    n = q.expire_stale(timeout_seconds=10, now=9999.0)
    assert n == 0
    assert q.get(qid).status == STATUS_CLAIMED


def test_expire_requires_positive_timeout() -> None:
    q = ReviewQueue(":memory:")
    with pytest.raises(ValueError):
        q.expire_stale(timeout_seconds=0)


def test_counts_partitions_by_status() -> None:
    q = ReviewQueue(":memory:")
    q1 = _enqueue_one(q, action_id="a1", now=100.0)
    q2 = _enqueue_one(q, action_id="a2", now=110.0)
    _enqueue_one(q, action_id="a3", now=120.0)
    q.claim(q1, reviewer="alice")
    q.resolve(q2, reviewer="alice", resolution=RESOLUTION_DENY)
    counts = q.counts()
    assert counts == {
        STATUS_PENDING: 1, STATUS_CLAIMED: 1,
        STATUS_RESOLVED: 1, STATUS_EXPIRED: 0,
    }


def test_persistence_round_trip_via_file(tmp_path: Path) -> None:
    db = tmp_path / "queue.db"
    q1 = ReviewQueue(db)
    qid = _enqueue_one(q1)
    q1.close()
    q2 = ReviewQueue(db)
    item = q2.get(qid)
    assert item.action_id == "act-1"
    assert item.status == STATUS_PENDING
    q2.close()


def test_enqueue_caps_huge_signals_blob() -> None:
    q = ReviewQueue(":memory:")
    big = {f"k{i}": "x" * 1000 for i in range(200)}
    qid = _enqueue_one(q, signals=big)
    stored = q.get(qid).signals
    assert stored.get("_truncated") is True
    assert stored.get("_cap_bytes") == 64 * 1024


def test_pipeline_intercept_enqueues_on_escalate() -> None:
    from vaara.pipeline import InterceptionPipeline

    class _FixedScorer:
        def evaluate(self, ctx):
            return {
                "action": "escalate",
                "reason": "fixed for test",
                "raw_result": {
                    "point_estimate": 0.6,
                    "conformal_interval": (0.55, 0.71),
                    "signals": {"sig1": 0.4},
                    "calibration_size": 50,
                    "effective_alpha": 0.1,
                    "bucket_category": "filesystem",
                },
            }

    queue = ReviewQueue(":memory:")
    pipe = InterceptionPipeline(scorer=_FixedScorer(), review_queue=queue)
    result = pipe.intercept(
        agent_id="agent-007", tool_name="fs.write_file",
        parameters={"path": "/etc/foo"},
    )
    assert result.decision == "escalate"
    items = queue.list_items()
    assert len(items) == 1
    item = items[0]
    assert item.action_id == result.action_id
    assert item.agent_id == "agent-007"
    assert item.tool_name == "fs.write_file"
    assert item.risk_score == pytest.approx(0.6)
    assert item.conformal_lower == pytest.approx(0.55)
    assert item.conformal_upper == pytest.approx(0.71)
    assert item.signals == {"sig1": 0.4}


def test_pipeline_does_not_enqueue_on_allow() -> None:
    from vaara.pipeline import InterceptionPipeline

    class _AllowScorer:
        def evaluate(self, ctx):
            return {
                "action": "allow", "reason": "ok",
                "raw_result": {
                    "point_estimate": 0.1,
                    "conformal_interval": (0.05, 0.15),
                    "signals": {}, "calibration_size": 50,
                    "effective_alpha": 0.1, "bucket_category": None,
                },
            }

    queue = ReviewQueue(":memory:")
    pipe = InterceptionPipeline(scorer=_AllowScorer(), review_queue=queue)
    pipe.intercept(agent_id="a", tool_name="data.read_file", parameters={})
    assert queue.list_items() == []


def test_cli_list_show_claim_resolve(tmp_path: Path, capsys) -> None:
    from vaara.cli import main
    db = tmp_path / "queue.db"
    q = ReviewQueue(db)
    qid = _enqueue_one(q)
    q.close()

    rc = main(["review", "list", "--db", str(db)])
    out = capsys.readouterr().out
    assert rc == 0
    assert qid[:8] in out

    rc = main(["review", "show", "--db", str(db), "--queue-id", qid])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["queue_id"] == qid

    rc = main([
        "review", "claim", "--db", str(db),
        "--queue-id", qid, "--reviewer", "alice",
    ])
    assert rc == 0

    rc = main([
        "review", "resolve", "--db", str(db),
        "--queue-id", qid, "--reviewer", "alice",
        "--resolution", "allow",
        "--justification", "low blast radius",
    ])
    assert rc == 0

    q = ReviewQueue(db)
    item = q.get(qid)
    assert item.status == STATUS_RESOLVED
    assert item.resolution == RESOLUTION_ALLOW
    assert item.justification == "low blast radius"
    q.close()


def test_cli_resolve_with_audit_db_writes_escalation_resolved(
    tmp_path: Path,
) -> None:
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.audit.trail import EventType
    from vaara.cli import main

    queue_db = tmp_path / "queue.db"
    audit_db = tmp_path / "audit.db"

    q = ReviewQueue(queue_db)
    qid = _enqueue_one(q)
    q.close()

    rc = main([
        "review", "resolve", "--db", str(queue_db),
        "--queue-id", qid, "--reviewer", "alice",
        "--resolution", "deny", "--justification", "blast radius too wide",
        "--audit-db", str(audit_db),
    ])
    assert rc == 0

    backend = SQLiteAuditBackend(audit_db)
    try:
        records = backend.query_by_event_type(EventType.ESCALATION_RESOLVED)
    finally:
        backend.close()
    assert len(records) == 1
    assert records[0].data["resolution"] == RESOLUTION_DENY
    assert records[0].data["reviewer"] == "alice"


def test_cli_expire_dry_run(tmp_path: Path, capsys) -> None:
    from vaara.cli import main
    db = tmp_path / "queue.db"
    q = ReviewQueue(db)
    _enqueue_one(q, now=1000.0)
    q.close()
    rc = main([
        "review", "expire", "--db", str(db),
        "--timeout-seconds", "1", "--dry-run",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Would expire" in out
