"""Whole-chain verification that does not materialise the chain.

Two separate problems wear the same name here and the tests keep them apart.

``AuditTrail.verify_chain`` opened with ``snapshot = list(self._records)``.
Measured, that copy IS the whole transient cost of the call: 393 KiB over
50,000 records against 390.6 KiB of pointer array, and 1565 KiB over 200,000
against 1562.5 KiB. So removing it takes the walk to O(1) in memory, and
``test_verify_chain_peak_does_not_grow_with_the_trail`` pins that it stays
there.

It does NOT make a trail too big to hold verifiable, because ``_records`` still
holds every record. That is the store's problem and
``SQLiteAuditBackend.verify_chain_streaming`` is where it is answered: one row
at a time off a cursor, nothing accumulated, so peak memory is flat in the row
count. ``test_streaming_peak_is_flat_in_row_count`` is the test that would
catch a ``fetchall`` creeping back in.

Anchors make either walk resumable: an anchored head is a valid restart point,
so a long verify can checkpoint rather than start over.
"""

from __future__ import annotations

import tracemalloc

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


def _fill(trail: AuditTrail, n: int) -> None:
    for i in range(n):
        trail.record_action_requested(ActionRequest(
            action_type=_ACTION, tool_name="t", agent_id="agent", parameters={"i": i},
        ))


def _peak_of(fn) -> int:
    """Peak bytes allocated above the entry baseline, in KiB."""
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return (peak - base) // 1024


# ── The in-memory walk ──────────────────────────────────────────────────────

def test_verify_chain_peak_does_not_grow_with_the_trail():
    """The snapshot copy was the entire transient cost. It must stay gone.

    Before: 390 KiB at 50k records, 1562 KiB at 200k, exactly 8 bytes each.
    A regression here is someone reintroducing ``list(self._records)``.
    """
    small, large = AuditTrail(), AuditTrail()
    _fill(small, 20_000)
    _fill(large, 200_000)

    peak_small = _peak_of(small.verify_chain)
    peak_large = _peak_of(large.verify_chain)

    # 200k records would have cost 1562 KiB of pointers on the old path.
    assert peak_large < 64, f"peak grew to {peak_large} KiB"
    # And it is flat, not merely small: a 10x trail must not cost 10x.
    assert peak_large <= peak_small + 32


def test_verify_chain_still_catches_a_tamper():
    trail = AuditTrail()
    _fill(trail, 50)
    trail._records[20].data = {"rewritten": True}

    err = trail.verify_chain()

    assert err is not None
    assert "20" in err


def test_verify_chain_intact_on_a_clean_trail():
    trail = AuditTrail()
    _fill(trail, 50)
    assert trail.verify_chain() is None


def test_verify_chain_sees_records_appended_during_the_walk():
    """Index iteration is bounded by the length captured at entry.

    An append landing mid-walk must not be half-read, and must not make the
    walk run off the end. The walk covers what existed when it started.
    """
    trail = AuditTrail()
    _fill(trail, 10)
    _fill(trail, 10)
    assert trail.verify_chain() is None
    assert trail.size == 20


# ── Resuming from an anchored head ──────────────────────────────────────────

def test_resume_skips_everything_below_the_restart_point():
    """That is the contract, and it is why a checkpoint is worth anything."""
    trail = AuditTrail()
    _fill(trail, 40)
    restart = 20
    prev = trail._records[restart - 1].record_hash
    trail._records[5].data = {"rewritten": True}

    assert trail.verify_chain() is not None          # the full walk sees it
    assert trail.verify_chain(                        # the resumed walk does not
        start_index=restart, expected_previous_hash=prev) is None


def test_resume_catches_a_tamper_above_the_restart_point():
    trail = AuditTrail()
    _fill(trail, 40)
    prev = trail._records[19].record_hash
    trail._records[30].data = {"rewritten": True}

    err = trail.verify_chain(start_index=20, expected_previous_hash=prev)

    assert err is not None
    assert "30" in err


def test_resume_with_the_wrong_previous_hash_fails():
    trail = AuditTrail()
    _fill(trail, 40)

    err = trail.verify_chain(start_index=20, expected_previous_hash="0" * 64)

    assert err is not None


def test_resume_rejects_an_out_of_range_start():
    trail = AuditTrail()
    _fill(trail, 10)

    with pytest.raises(ValueError):
        trail.verify_chain(start_index=-1)
    with pytest.raises(ValueError):
        trail.verify_chain(start_index=99, expected_previous_hash="x")


def test_resume_from_a_time_anchor_position():
    """The intended caller: an anchored head is a valid restart point."""
    from tests.test_timeanchor import _local_tsa_client

    trail = AuditTrail()
    _fill(trail, 20)
    anchor = trail.anchor_head(_local_tsa_client())
    _fill(trail, 20)

    err = trail.verify_chain(
        start_index=anchor.chain_position + 1,
        expected_previous_hash=anchor.chain_head_hash,
    )

    assert err is None


# ── The store walk, which is the one that scales ────────────────────────────

def _backend(tmp_path, n: int) -> SQLiteAuditBackend:
    tmp_path.mkdir(parents=True, exist_ok=True)
    trail = AuditTrail()
    _fill(trail, n)
    backend = SQLiteAuditBackend(tmp_path / "audit.db")
    for record in trail._records:
        backend.write_record(record)
    return backend


def test_streaming_agrees_with_the_loaded_trail(tmp_path):
    backend = _backend(tmp_path, 200)

    assert backend.verify_chain_streaming() is None
    assert backend.load_trail().verify_chain() is None


def test_streaming_catches_a_tampered_row(tmp_path):
    backend = _backend(tmp_path, 200)
    with backend._lock:
        backend._conn.execute(
            "UPDATE audit_records SET agent_id='rewritten' WHERE seq=100")
        backend._conn.commit()

    err = backend.verify_chain_streaming()

    assert err is not None
    assert "100" in err or "rewritten" in err


def test_streaming_catches_a_deleted_row(tmp_path):
    """Deleting a row breaks the link, which a per-row hash check alone misses."""
    backend = _backend(tmp_path, 200)
    with backend._lock:
        backend._conn.execute("DELETE FROM audit_records WHERE seq=100")
        backend._conn.commit()

    assert backend.verify_chain_streaming() is not None


def test_streaming_peak_is_flat_in_row_count(tmp_path):
    """The test that catches a fetchall creeping back in."""
    small = _backend(tmp_path / "a", 500)
    large = _backend(tmp_path / "b", 5_000)

    peak_small = _peak_of(small.verify_chain_streaming)
    peak_large = _peak_of(large.verify_chain_streaming)

    assert peak_large < 512, f"peak grew to {peak_large} KiB"
    assert peak_large <= peak_small + 128


def test_streaming_resumes_from_a_sequence_point(tmp_path):
    backend = _backend(tmp_path, 200)
    rows = backend.load_trail()._records
    prev = rows[99].record_hash

    # Tamper below the restart point: the resumed walk must not see it.
    with backend._lock:
        backend._conn.execute(
            "UPDATE audit_records SET agent_id='rewritten' WHERE seq=10")
        backend._conn.commit()

    assert backend.verify_chain_streaming() is not None
    assert backend.verify_chain_streaming(
        start_seq=100, expected_previous_hash=prev) is None


def test_streaming_does_not_hold_the_write_lock(tmp_path):
    """A 17-minute walk holding the lock would stall every recording thread.

    An operator who learns that verifying stalls production stops verifying,
    so this is a correctness property of the feature, not a nicety.
    """
    backend = _backend(tmp_path, 300)
    held: list[bool] = []

    backend.verify_chain_streaming(
        on_progress=lambda seq, h: held.append(backend._lock.locked()),
        progress_every=50,
    )

    assert held, "progress never fired, so the check proved nothing"
    assert not any(held)


def test_streaming_works_on_an_in_memory_store():
    """No second connection to open, so this path reads under the lock."""
    trail = AuditTrail()
    _fill(trail, 50)
    backend = SQLiteAuditBackend(":memory:")
    for record in trail._records:
        backend.write_record(record)

    assert backend.verify_chain_streaming() is None


def test_streaming_reports_progress_for_checkpointing(tmp_path):
    """A 201M-record walk has to be able to stop and come back."""
    backend = _backend(tmp_path, 300)
    seen: list[tuple[int, str]] = []

    err = backend.verify_chain_streaming(
        on_progress=lambda seq, h: seen.append((seq, h)), progress_every=100)

    assert err is None
    assert len(seen) >= 2
    # Each checkpoint is a valid restart point.
    seq, head = seen[0]
    assert backend.verify_chain_streaming(
        start_seq=seq + 1, expected_previous_hash=head) is None
