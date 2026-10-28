"""Contiguity verification: a dropped receipt inside a boundary is a provable gap.

These tests pin the property that answers "completeness, not just non-inclusion":
given only the receipts a third party holds, a missing sequence number and the
signed running count make absence self-evident, with no issuer access and no
external witness.
"""

from __future__ import annotations

import pytest

from vaara.credential import (
    ClassGateDecision,
    ContiguityReport,
    enforce_on_sealed_class,
    verify_contiguity,
)

BOUNDARY = "vaara-mcp-proxy"


def _ev(seq: int, running_count: int, boundary: str = BOUNDARY) -> dict:
    """An authorization-evidence record carrying just a completeness block."""
    return {
        "schema": "vaara.authorization/v0",
        "completeness": {
            "boundaryId": boundary,
            "seq": seq,
            "runningCount": running_count,
        },
    }


def _stream(n: int, boundary: str = BOUNDARY) -> list[dict]:
    return [_ev(i, i + 1, boundary) for i in range(n)]


def test_complete_stream_verifies():
    report = verify_contiguity(_stream(5))
    assert isinstance(report, ContiguityReport)
    assert report.ok
    assert report.present == 5
    assert report.expected == 5
    assert report.missing_seqs == []
    assert "contiguous" in report.gap_report()


def test_dropped_middle_receipt_is_a_provable_gap():
    stream = _stream(5)
    del stream[2]  # drop seq 2
    report = verify_contiguity(stream)
    assert not report.ok
    assert report.missing_seqs == [2]
    assert report.present == 4
    assert report.expected == 5
    assert "missing seq: 2" in report.gap_report()


def test_gap_exposed_when_a_later_receipt_is_held():
    # Hold seqs 0,1,2,4 (seq 3 dropped). The held seq-4 receipt carries
    # runningCount 5, so the boundary asserts 5 exist and seq 3 is the gap.
    stream = [_ev(0, 1), _ev(1, 2), _ev(2, 3), _ev(4, 5)]
    report = verify_contiguity(stream)
    assert not report.ok
    assert report.expected == 5
    assert report.missing_seqs == [3]


def test_pure_tail_truncation_needs_an_anchor():
    # Honest limit: holding 0,1,2 with nothing after, the latest held
    # runningCount is 3, so the held set cannot tell that later receipts ever
    # existed. A pure tail truncation is invisible to contiguity alone; closing
    # it is the job of an rfc3161 anchor over the running count (which attests
    # "at time T, N receipts existed under this boundary"). Documented, not a bug.
    report = verify_contiguity(_stream(3))
    assert report.ok
    assert report.expected == 3


def test_count_mismatch_is_flagged():
    stream = [_ev(0, 1), _ev(1, 9), _ev(2, 3)]  # seq 1 claims runningCount 9
    report = verify_contiguity(stream)
    assert not report.ok
    assert {"seq": 1, "runningCount": 9} in report.count_mismatches
    assert "count mismatch at seq 1" in report.gap_report()


def test_duplicate_seq_is_flagged():
    stream = _stream(3) + [_ev(1, 2)]  # seq 1 appears twice
    report = verify_contiguity(stream)
    assert not report.ok
    assert report.duplicate_seqs == [1]
    assert "duplicate seq: 1" in report.gap_report()


def test_empty_or_no_completeness_is_vacuously_complete():
    assert verify_contiguity([]).ok
    plain = [{"schema": "vaara.authorization/v0", "verdict": "allow"}]
    report = verify_contiguity(plain)
    assert report.ok
    assert report.expected == 0
    assert "no completeness asserted" in report.gap_report()


def test_multiple_boundaries_requires_disambiguation():
    mixed = _stream(2, "boundary-a") + _stream(3, "boundary-b")
    with pytest.raises(ValueError, match="multiple boundaries"):
        verify_contiguity(mixed)


def test_boundary_id_scopes_the_check():
    mixed = _stream(2, "boundary-a") + _stream(3, "boundary-b")
    # boundary-b alone is a complete 0..2 run.
    report = verify_contiguity(mixed, boundary_id="boundary-b")
    assert report.ok
    assert report.boundary_id == "boundary-b"
    assert report.expected == 3
    # boundary-a alone is a complete 0..1 run.
    assert verify_contiguity(mixed, boundary_id="boundary-a").ok


def test_gap_report_compacts_ranges():
    stream = _stream(10)
    for seq in (3, 4, 5, 8):
        stream = [ev for ev in stream if ev["completeness"]["seq"] != seq]
    report = verify_contiguity(stream)
    assert not report.ok
    assert report.missing_seqs == [3, 4, 5, 8]
    assert "missing seq: 3-5, 8" in report.gap_report()


def _seal(total: int, boundary: str = BOUNDARY, max_class: str | None = None) -> dict:
    """A terminal sealing record pinning the run length (and optional max class)."""
    completeness: dict = {"boundaryId": boundary, "sealed": True, "total": total}
    if max_class is not None:
        completeness["maxClass"] = max_class
    return {"schema": "vaara.authorization/v0", "completeness": completeness}


def test_seal_max_class_bounds_a_gap_worst_case():
    # Seal asserts 5 receipts and that the boundary's highest action class is
    # "transfer". Drop the tail (seqs 3,4): the held set alone now both proves
    # the gap and bounds its worst case.
    stream = _stream(3) + [_seal(5, max_class="transfer")]
    report = verify_contiguity(stream)
    assert not report.ok
    assert report.missing_seqs == [3, 4]
    assert report.worst_case_class == "transfer"
    assert "gap worst-case: action class up to 'transfer'" in report.gap_report()


def test_seal_without_max_class_leaves_worst_case_unset():
    stream = _stream(3) + [_seal(5)]
    report = verify_contiguity(stream)
    assert not report.ok
    assert report.worst_case_class is None
    assert "worst-case" not in report.gap_report()


def test_max_class_surfaces_on_a_complete_run():
    # A seal's class rides along even when the run is whole; there is just no gap
    # for it to bound, and the gap-report stays quiet.
    stream = _stream(5) + [_seal(5, max_class="refund")]
    report = verify_contiguity(stream)
    assert report.ok
    assert report.worst_case_class == "refund"
    assert "worst-case" not in report.gap_report()


# -- enforce_on_sealed_class: the gate that consumes the sealed class ----------

PERMITTED = ["data.read", "data.write"]


def test_gate_permits_when_sealed_class_is_permitted():
    stream = _stream(3) + [_seal(3, max_class="data.read")]
    decision = enforce_on_sealed_class(stream, PERMITTED)
    assert isinstance(decision, ClassGateDecision)
    assert decision.permit
    assert decision.reason == "permitted"
    assert decision.worst_case_class == "data.read"
    assert "PERMIT" in decision.gate_report()


def test_gate_permits_over_a_gap_the_seal_bounds():
    # The load-bearing case: seq 1 is dropped, so the run has a provable gap, yet
    # the gate permits because the sealed data.read bounds the gap's worst case.
    stream = [_ev(0, 1), _ev(2, 3), _seal(3, max_class="data.read")]
    decision = enforce_on_sealed_class(stream, PERMITTED)
    assert verify_contiguity(stream).ok is False
    assert decision.permit
    assert decision.reason == "permitted"


def test_gate_denies_a_class_outside_the_permitted_set():
    stream = _stream(3) + [_seal(3, max_class="tx.transfer")]
    decision = enforce_on_sealed_class(stream, PERMITTED)
    assert not decision.permit
    assert decision.reason == "class_not_permitted"
    assert decision.worst_case_class == "tx.transfer"


def test_gate_fails_closed_when_no_class_is_sealed():
    # No maxClass: a gap's worst case is unbounded, so the gate denies even though
    # the stream is otherwise well-formed.
    stream = [_ev(0, 1), _ev(2, 3), _seal(3)]
    decision = enforce_on_sealed_class(stream, PERMITTED)
    assert not decision.permit
    assert decision.reason == "unbounded_no_sealed_class"
    assert decision.worst_case_class is None
    assert "fail-closed" in decision.gate_report()


def test_gate_fails_closed_on_a_complete_run_with_no_sealed_class():
    # Even a whole run denies when no class is sealed: there is no bound to consume.
    stream = _stream(3) + [_seal(3)]
    decision = enforce_on_sealed_class(stream, PERMITTED)
    assert not decision.permit
    assert decision.reason == "unbounded_no_sealed_class"


# --- what an ok verdict does and does not say -------------------------------


def test_a_cut_tail_reads_consistent_never_complete():
    # Five receipts issued, the last two removed. The held prefix 0..2 is
    # contiguous and its running counts agree, so ok stays true, but nothing in
    # the held set can say the run was longer: the line must not say "complete".
    report = verify_contiguity(_stream(3))
    text = report.gap_report()
    assert report.ok
    assert not report.sealed
    assert "complete" not in text.replace("completeness", "")
    assert "consistent" in text
    assert "seq 0..2 contiguous" in text
    assert "highest count held 3" in text
    assert "no seal held" in text
    assert "no anchor checked" in text


def test_a_held_seal_is_named_in_an_ok_line():
    report = verify_contiguity(_stream(3) + [_seal(3)])
    text = report.gap_report()
    assert report.ok
    assert report.sealed
    assert "no seal held" not in text
    assert "seal held" in text
    assert "no anchor checked" in text


def test_a_seal_over_a_cut_tail_is_still_incomplete():
    report = verify_contiguity(_stream(3) + [_seal(5)])
    assert not report.ok
    assert report.sealed
    assert report.gap_report().startswith(f"boundary {BOUNDARY!r}: INCOMPLETE")


def test_two_different_records_at_one_seq_are_a_fork():
    other = _ev(1, 2)
    other["decision"] = "a different record at the same seq"
    report = verify_contiguity(_stream(3) + [other])
    assert not report.ok
    assert report.duplicates == [{"seq": 1, "kind": "fork"}]
    assert "duplicate seq: 1 (different records)" in report.gap_report()


def test_a_byte_identical_copy_at_one_seq_is_a_replay_and_still_fails():
    report = verify_contiguity(_stream(3) + [_ev(1, 2)])
    assert not report.ok
    assert report.duplicates == [{"seq": 1, "kind": "replay"}]
    assert "duplicate seq: 1 (same record)" in report.gap_report()


def test_a_replay_beside_a_fork_at_one_seq_reads_as_a_fork():
    other = _ev(1, 2)
    other["decision"] = "different"
    report = verify_contiguity(_stream(3) + [_ev(1, 2), other])
    assert report.duplicates == [{"seq": 1, "kind": "fork"}]


def test_forks_and_replays_are_named_per_seq():
    other = _ev(0, 1)
    other["decision"] = "different"
    report = verify_contiguity(_stream(4) + [other, _ev(2, 3)])
    assert report.duplicate_seqs == [0, 2]
    assert report.duplicates == [
        {"seq": 0, "kind": "fork"},
        {"seq": 2, "kind": "replay"},
    ]
    text = report.gap_report()
    assert "duplicate seq: 0 (different records)" in text
    assert "duplicate seq: 2 (same record)" in text


def test_whole_records_decide_the_kind_when_given():
    """Same evidence under two different signed records is a fork, not a replay."""
    stream = _stream(3) + [_ev(1, 2)]
    wholes = [{"record": {"sig": f"s{i}"}, "evidence": ev} for i, ev in enumerate(stream)]
    report = verify_contiguity(stream, records=wholes)
    assert report.duplicates == [{"seq": 1, "kind": "fork"}]
    wholes[3] = wholes[1]
    report = verify_contiguity(stream, records=wholes)
    assert report.duplicates == [{"seq": 1, "kind": "replay"}]


def test_records_must_match_the_evidence_list():
    with pytest.raises(ValueError, match="records"):
        verify_contiguity(_stream(2), records=[{}])
