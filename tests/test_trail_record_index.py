"""Finding a record must not cost a walk of the trail.

`verify_segment` does flat verification work, 32 record hashes and two RFC 3161
tokens whatever the trail size, and then spent most of its wall clock FINDING
the record. Measured before this change:

    trail     verify_segment   of which id scan
    5,000         0.74 ms          0.12 ms   (16%)
    50,000        2.41 ms          1.41 ms   (59%)

So the "milliseconds at any trail size" claim held into the low millions and
then stopped, and the wall was the lookup rather than the hashing.

The index follows the `_resolved_approvals` pattern from PR #672 exactly: built
in `_index_record`, which is the only place that appends to `_records`, so a
bulk loader cannot populate one without the other. `test_index_matches_records`
is the one that fails if that ever stops being true.
"""

from __future__ import annotations

import time

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail
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


def _fill(trail: AuditTrail, n: int) -> list[str]:
    return [
        trail.record_action_requested(ActionRequest(
            action_type=_ACTION, tool_name="t", agent_id="agent",
            parameters={"i": i},
        ))
        for i in range(n)
    ]


# ── the invariant the whole thing rests on ──────────────────────────────────

def test_index_matches_records():
    trail = AuditTrail()
    _fill(trail, 200)

    assert len(trail._pos_by_record_id) == len(trail._records)
    for position, record in enumerate(trail._records):
        assert trail._pos_by_record_id[record.record_id] == position


def test_index_survives_a_reload_from_the_store(tmp_path):
    """Reloaded records never pass through _append. load_trail must index too.

    This is the path PR #672 named as the one that would be forgotten, and it
    was right: it had already been forgotten once for the approval index.
    """
    trail = AuditTrail()
    _fill(trail, 50)
    backend = SQLiteAuditBackend(tmp_path / "audit.db")
    for record in trail._records:
        backend.write_record(record)

    reloaded = backend.load_trail()

    assert len(reloaded._pos_by_record_id) == len(reloaded._records)
    for position, record in enumerate(reloaded._records):
        assert reloaded._pos_by_record_id[record.record_id] == position


def test_lookup_finds_the_same_record_the_scan_would():
    trail = AuditTrail()
    _fill(trail, 300)

    for position in (0, 1, 150, 298, 299):
        want = trail._records[position]
        assert trail._pos_by_record_id[want.record_id] == position


# ── behaviour through verify_segment, which is the caller ───────────────────

def _anchored(n_before=10, n_after=10):
    from tests.test_timeanchor import _local_tsa_client
    trail = AuditTrail()
    client = _local_tsa_client()
    _fill(trail, n_before)
    trail.anchor_head(client)
    _fill(trail, n_after)
    trail.anchor_head(client)
    return trail


def test_verify_segment_still_finds_a_record():
    trail = _anchored()
    target = trail._records[15].record_id

    result = trail.verify_segment(record_id=target)

    assert result.ok, result.reason
    assert result.record_index == 15
    assert result.record_id == target


def test_verify_segment_unknown_id_still_reports_not_found():
    trail = _anchored()
    result = trail.verify_segment(record_id="no-such-record")
    assert not result.ok
    assert "not found" in result.reason.lower()


def test_verify_segment_by_action_id_still_spans_its_records():
    trail = _anchored(n_before=5, n_after=5)
    action_id = trail._records[7].action_id

    result = trail.verify_segment(action_id=action_id)

    assert result.ok, result.reason
    assert result.record_index == 7


def test_verify_segment_still_catches_a_tamper():
    """The lookup changed. The verdict must not have."""
    trail = _anchored()
    trail._records[15].data = {"rewritten": True}

    result = trail.verify_segment(record_id=trail._records[15].record_id)

    assert not result.ok
    assert "15" in result.reason


# ── the point of the change ─────────────────────────────────────────────────

def test_lookup_does_not_scale_with_the_trail():
    """A 10x trail must not cost 10x to find one record in.

    Timed rather than asserted structurally, because the structural version
    (patching enumerate) passes just as well against a scan hidden one call
    deeper.
    """
    small, large = AuditTrail(), AuditTrail()
    _fill(small, 2_000)
    _fill(large, 100_000)

    def timed(trail: AuditTrail) -> float:
        """Seconds for 50,000 lookups.

        A BATCH, because one dict lookup is below perf_counter's resolution and
        the first version of this test measured the small trail at exactly 0.0,
        which made the comparison `x < 0` and failed for a reason that had
        nothing to do with the code under test.
        """
        target = trail._records[len(trail._records) // 2].record_id
        index = trail._pos_by_record_id
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(50_000):
                index.get(target)
            best = min(best, time.perf_counter() - start)
        return best

    small_t, large_t = timed(small), timed(large)
    assert small_t > 0, "the batch is still too small to measure"
    # A scan would make the 50x trail cost ~50x. A dict does not. Two is
    # generous room for noise while still failing a linear lookup by miles.
    assert large_t < small_t * 2, f"{large_t=} {small_t=}"


def test_verify_segment_does_not_copy_the_trail():
    """It opened with `list(self._records)`, which is the same O(n) that
    verify_chain shed. Finding one record in a 200k trail should not allocate
    a 1.6 MB pointer array to do it."""
    import tracemalloc

    trail = _anchored(n_before=10, n_after=10)
    _fill(trail, 100_000)
    target = trail._records[15].record_id

    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    result = trail.verify_segment(record_id=target)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert result.ok, result.reason
    # 100k records would have cost ~780 KiB of pointers on the old path.
    assert (peak - base) // 1024 < 64


# ── duplicates, which uuid4 makes unlikely rather than impossible ───────────

def test_duplicate_record_id_resolves_to_the_first_occurrence():
    """Whatever it does, it must be decided rather than accidental.

    The first occurrence wins, because a segment claim about the EARLIER copy
    is the conservative answer: it covers more of the chain.
    """
    trail = AuditTrail()
    _fill(trail, 5)
    clone = trail._records[1]
    duplicate = type(clone)(**{**clone.to_dict(),
                               "event_type": clone.event_type})
    trail._index_record(duplicate)

    assert trail._pos_by_record_id[clone.record_id] == 1


def test_empty_trail_has_an_empty_index():
    trail = AuditTrail()
    assert trail._pos_by_record_id == {}
    with pytest.raises(ValueError):
        trail.verify_segment()
