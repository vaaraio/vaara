"""Segment-indexed verification: prove one record without walking the trail.

The question a supervisor actually asks is not "is my trail broadly sound", it
is "show me that this action's record is what it was when it was written".
Anchors already make that answerable exactly: a segment between two anchors is
independently verifiable, because ``previous_hash`` is part of the hashed
content, so a record cannot be re-parented onto a different hash without
breaking its own hash.

The load-bearing test here is ``test_tamper_before_the_lower_anchor_is_out_of
_scope``. If that one ever fails, segment verification has stopped being
segment-scoped and the feature has no reason to exist.

Anchoring is opt-in and no TSA is configured by default, so the unanchored case
is the COMMON one. It must fail loudly rather than degrade to a genesis walk,
which is what ``test_unanchored_trail_fails_loudly`` pins.
"""

from __future__ import annotations

import pytest

from tests.test_timeanchor import _GEN_TIME, _local_tsa_client
from vaara.audit.trail import AuditTrail, EventType
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    Reversibility,
)

_ACTION = ActionType(
    name="t",
    category=ActionCategory.DATA,
    reversibility=Reversibility.FULLY,
    blast_radius=BlastRadius.LOCAL,
)


def _add(trail: AuditTrail, n: int = 1) -> list[str]:
    """Append ``n`` action records. Returns their action ids."""
    return [
        trail.record_action_requested(ActionRequest(
            action_type=_ACTION, tool_name="t", agent_id="agent", parameters={"i": i},
        ))
        for i in range(n)
    ]


def _anchored_trail(client=None):
    """A trail of 30 records with anchors at positions 9 and 19.

    Leaves 10 records past the last anchor, so the open-segment case is
    reachable without rebuilding.
    """
    trail = AuditTrail()
    client = client or _local_tsa_client()
    _add(trail, 10)
    trail.anchor_head(client)          # position 9
    _add(trail, 10)
    trail.anchor_head(client)          # position 19
    _add(trail, 10)
    return trail


def _rid(trail: AuditTrail, index: int) -> str:
    return trail._records[index].record_id


# ── The reason the feature exists ───────────────────────────────────────────

def test_tamper_before_the_lower_anchor_is_out_of_scope():
    """A record rewritten BELOW the segment does not fail the segment.

    This is the whole claim. If verifying record 15 required records 0..14 to
    be intact, segment verification would be a genesis walk wearing a hat.
    """
    trail = _anchored_trail()
    trail._records[3].data = {"rewritten": True}   # breaks record 3's own hash

    result = trail.verify_segment(record_id=_rid(trail, 15))

    assert result.ok, result.reason
    assert result.closed
    # And the whole-chain walk DOES see it, so the tamper is real.
    assert trail.verify_chain() is not None


def test_segment_walk_is_bounded_by_the_anchors():
    trail = _anchored_trail()
    result = trail.verify_segment(record_id=_rid(trail, 15))

    assert result.ok, result.reason
    assert result.lower_anchor_position == 9
    assert result.upper_anchor_position == 19
    # Records 10..19 inclusive, not the 30 in the trail.
    assert result.records_verified == 10
    assert result.records_verified < trail.size


# ── Failing loudly, which matters more than passing ─────────────────────────

def test_unanchored_trail_fails_loudly():
    """No anchors means no answer. It must not look like a pass."""
    trail = AuditTrail()
    _add(trail, 5)

    result = trail.verify_segment(record_id=_rid(trail, 2))

    assert not result.ok
    assert "anchor" in result.reason.lower()
    assert result.records_verified == 0


def test_tamper_inside_the_segment_is_caught():
    trail = _anchored_trail()
    trail._records[15].data = {"rewritten": True}

    result = trail.verify_segment(record_id=_rid(trail, 15))

    assert not result.ok
    assert "15" in result.reason


def test_tamper_inside_segment_but_not_the_named_record_is_caught():
    """The segment is the unit. A neighbour rewritten inside it still fails."""
    trail = _anchored_trail()
    trail._records[12].data = {"rewritten": True}

    result = trail.verify_segment(record_id=_rid(trail, 15))

    assert not result.ok


def test_unknown_record_id_fails():
    trail = _anchored_trail()
    result = trail.verify_segment(record_id="no-such-record")

    assert not result.ok
    assert "not found" in result.reason.lower()


def test_rewritten_anchor_target_is_caught():
    """An anchor pointing at a hash the trail no longer has is a rewrite."""
    trail = _anchored_trail()
    trail.anchors[1].chain_head_hash = "0" * 64

    result = trail.verify_segment(record_id=_rid(trail, 15))

    assert not result.ok


def test_tampered_anchor_token_is_caught():
    trail = _anchored_trail()
    good = trail.anchors[0].token_b64
    trail.anchors[0].token_b64 = "!!!not base64!!!"

    result = trail.verify_segment(record_id=_rid(trail, 15))

    assert not result.ok
    assert "anchor" in result.reason.lower()
    trail.anchors[0].token_b64 = good


# ── Edge cases the block named ──────────────────────────────────────────────

def test_record_after_the_last_anchor_is_open_not_closed():
    """Verifiable against the chain, not closed by an anchor. Say so."""
    trail = _anchored_trail()
    result = trail.verify_segment(record_id=_rid(trail, 25))

    assert result.ok, result.reason
    assert not result.closed
    assert result.lower_anchor_position == 19
    assert result.upper_anchor_position is None
    assert "not" in result.reason.lower()   # the report explains the gap


def test_record_before_the_first_anchor_walks_from_genesis():
    trail = _anchored_trail()
    result = trail.verify_segment(record_id=_rid(trail, 3))

    assert result.ok, result.reason
    assert result.closed
    assert result.lower_anchor_position is None    # genesis, not an anchor
    assert result.upper_anchor_position == 9
    assert result.records_verified == 10           # 0..9


def test_record_before_first_anchor_still_catches_a_tamper():
    trail = _anchored_trail()
    trail._records[3].data = {"rewritten": True}

    result = trail.verify_segment(record_id=_rid(trail, 3))

    assert not result.ok


def test_anchor_gap_marker_in_the_segment_is_reported():
    """``_record_anchor_gap`` writes a chained marker when a TSA is down.

    It verifies like any other record, so the walk passes. The report has to
    name it anyway: it means external attestation is missing across that
    stretch, which is exactly what a segment claim rests on.
    """
    trail = AuditTrail()
    client = _local_tsa_client()
    _add(trail, 10)
    trail.anchor_head(client)                      # position 9
    _add(trail, 3)
    trail._record_anchor_gap(len(trail._records) - 1,
                             trail._records[-1].record_hash, "tsa unreachable", client)
    _add(trail, 3)
    trail.anchor_head(client)

    target = next(i for i, r in enumerate(trail._records)
                  if r.event_type is EventType.ANCHOR_GAP)
    result = trail.verify_segment(record_id=_rid(trail, target + 1))

    assert result.ok, result.reason
    assert result.anchor_gaps == [target]


def test_anchor_positions_report_attested_times():
    trail = _anchored_trail()
    result = trail.verify_segment(record_id=_rid(trail, 15))

    assert result.ok, result.reason
    assert result.upper_attested_time == _GEN_TIME.isoformat()


# ── Lookup by action id ─────────────────────────────────────────────────────

def test_action_id_resolves_to_its_segment():
    trail = _anchored_trail()
    action_id = trail._records[15].action_id

    result = trail.verify_segment(action_id=action_id)

    assert result.ok, result.reason
    assert result.lower_anchor_position == 9
    assert result.upper_anchor_position == 19


def test_action_id_spanning_two_segments_widens_the_walk():
    """An action's records can straddle an anchor. Cover all of them or lie."""
    trail = AuditTrail()
    client = _local_tsa_client()
    _add(trail, 5)
    action_id = trail.record_action_requested(ActionRequest(
        action_type=_ACTION, tool_name="t", agent_id="agent", parameters={},
    ))
    trail.anchor_head(client)                      # anchor lands mid-action
    trail.record_outcome(action_id, agent_id="agent", tool_name="t",
                         outcome_severity=0.0)
    _add(trail, 3)
    trail.anchor_head(client)

    result = trail.verify_segment(action_id=action_id)

    assert result.ok, result.reason
    indices = [i for i, r in enumerate(trail._records) if r.action_id == action_id]
    assert len(indices) >= 2
    assert result.upper_anchor_position == trail.anchors[-1].chain_position


def test_requires_exactly_one_selector():
    trail = _anchored_trail()

    with pytest.raises(ValueError):
        trail.verify_segment()
    with pytest.raises(ValueError):
        trail.verify_segment(record_id="a", action_id="b")
