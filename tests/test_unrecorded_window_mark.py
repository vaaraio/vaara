# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A trail that stopped recording says so on the chain when it starts again.

When the audit store refuses a write the call still runs, the chain on disk
stays contiguous (the refused records never took a place in it), and the only
trace is ``<db>.write-failure.json`` beside the database. An exported trail
looked clean. The first write that lands after the refused ones is now
followed by a ``trail_not_recording`` record: how many writes the store
refused, from when to when, and which record landed first afterwards.
``verify_signed`` lists it, ``vaara trail verify`` and the standalone script
print it, and a record that carries nonsense fails the export.

The limits are part of the claim. Nothing is written if the store never
recovers, and a process that dies before the first good write loses the
marker, so the mark narrows the hole and does not close it.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from tests.test_cage_lifecycle import _rechain  # noqa: E402
from vaara.audit import verify as lib  # noqa: E402
from vaara.audit import write_failure as wf  # noqa: E402
from vaara.audit.export import export_signed  # noqa: E402
from vaara.audit.sqlite_backend import SQLiteAuditBackend  # noqa: E402
from vaara.audit.trail import SYSTEM_EVENT_TYPES, AuditTrail, EventType  # noqa: E402
from vaara.pipeline import InterceptionPipeline  # noqa: E402

MARK = "trail_not_recording"


class _Gate:
    """Switches a store between refusing writes and keeping them."""

    def __init__(self) -> None:
        self.refuse = False
        self.allow = 0  # writes let through while ``refuse`` is on


def _outage(tmp_path: Path, monkeypatch):
    """A pipeline on a SQLite store whose inserts fail the way 2026-08-22's did."""
    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    real = backend.append_record
    gate = _Gate()

    def append_record(record, stamp):
        if gate.refuse:
            if gate.allow <= 0:
                stamp("")
                raise sqlite3.DatabaseError("database disk image is malformed")
            gate.allow -= 1
        return real(record, stamp)

    monkeypatch.setattr(backend, "append_record", append_record)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    return db, backend, pipeline, gate


def _call(pipeline: InterceptionPipeline, n: int) -> None:
    pipeline.intercept(agent_id="outage-test", tool_name="file.read",
                       parameters={"path": f"/srv/{n}.txt"})


def _on_disk(db: Path) -> tuple[AuditTrail, SQLiteAuditBackend]:
    backend = SQLiteAuditBackend(db)
    return backend.load_trail(), backend


def _marks(trail: AuditTrail) -> list:
    return [r for r in trail._records if r.event_type.value == MARK]


def _recovered_after_two_refused_calls(tmp_path: Path, monkeypatch):
    """Call 0 lands, calls 1 and 2 are refused, call 3 lands."""
    db, backend, pipeline, gate = _outage(tmp_path, monkeypatch)
    _call(pipeline, 0)
    gate.refuse = True
    _call(pipeline, 1)
    _call(pipeline, 2)
    refused = wf.active_failure(db)
    gate.refuse = False
    _call(pipeline, 3)
    backend.close()
    return db, refused


# ── The trail writes the mark ──────────────────────────────────────────

class TestTheTrailWritesTheMark:
    def test_it_is_a_system_record_only_the_trail_writes(self):
        assert EventType.TRAIL_NOT_RECORDING.value == MARK
        assert EventType.TRAIL_NOT_RECORDING in SYSTEM_EVENT_TYPES

    def test_the_first_write_after_refused_writes_is_followed_by_a_mark(self, tmp_path, monkeypatch):
        db, refused = _recovered_after_two_refused_calls(tmp_path, monkeypatch)
        assert refused["count"] >= 2

        trail, backend = _on_disk(db)
        (mark,) = _marks(trail)
        backend.close()
        assert mark.data["unrecorded_writes"] == refused["count"]
        assert mark.data["first_failure_utc"] == refused["first_failure_utc"]
        assert mark.data["last_failure_utc"] == refused["last_failure_utc"]
        assert mark.data["stage"] == "append_record"
        assert "malformed" in mark.data["error"]

    def test_the_mark_names_the_record_that_landed_first_and_follows_it(self, tmp_path, monkeypatch):
        db, _ = _recovered_after_two_refused_calls(tmp_path, monkeypatch)

        trail, backend = _on_disk(db)
        records = trail._records
        at = next(i for i, r in enumerate(records) if r.event_type.value == MARK)
        backend.close()
        assert records[at - 1].record_id == records[at].data["resumed_record_id"]
        assert records[at - 1].event_type == EventType.ACTION_REQUESTED

    def test_the_chain_on_disk_stays_contiguous_across_the_window(self, tmp_path, monkeypatch):
        db, _ = _recovered_after_two_refused_calls(tmp_path, monkeypatch)

        trail, backend = _on_disk(db)
        try:
            assert trail.verify_chain() is None
            assert len(_marks(trail)) == 1
        finally:
            backend.close()

    def test_a_trail_that_never_failed_writes_no_mark(self, tmp_path, monkeypatch):
        db, backend, pipeline, _ = _outage(tmp_path, monkeypatch)
        for n in range(3):
            _call(pipeline, n)
        backend.close()

        trail, backend = _on_disk(db)
        backend.close()
        assert _marks(trail) == []

    def test_one_outage_gets_one_mark(self, tmp_path, monkeypatch):
        db, backend, pipeline, gate = _outage(tmp_path, monkeypatch)
        gate.refuse = True
        _call(pipeline, 0)
        gate.refuse = False
        for n in range(1, 5):
            _call(pipeline, n)
        backend.close()

        trail, backend = _on_disk(db)
        backend.close()
        assert len(_marks(trail)) == 1

    def test_a_second_outage_gets_its_own_mark_and_its_own_count(self, tmp_path, monkeypatch):
        db, backend, pipeline, gate = _outage(tmp_path, monkeypatch)
        _call(pipeline, 0)
        gate.refuse = True
        _call(pipeline, 1)
        first = wf.active_failure(db)["count"]
        gate.refuse = False
        _call(pipeline, 2)
        gate.refuse = True
        _call(pipeline, 3)
        _call(pipeline, 4)
        second = wf.active_failure(db)["count"]
        gate.refuse = False
        _call(pipeline, 5)
        backend.close()

        trail, backend = _on_disk(db)
        backend.close()
        counts = [m.data["unrecorded_writes"] for m in _marks(trail)]
        assert counts == [first, second]
        assert second > first, "the second outage must not inherit the first one's count"

    def test_the_marker_is_resolved_once_the_mark_has_landed(self, tmp_path, monkeypatch):
        db, _ = _recovered_after_two_refused_calls(tmp_path, monkeypatch)

        assert wf.active_failure(db) is None
        assert wf.read_marker(db)["resolved"] is True

    def test_a_refused_mark_leaves_the_outage_open_for_the_next_good_write(self, tmp_path, monkeypatch):
        db, backend, pipeline, gate = _outage(tmp_path, monkeypatch)
        gate.refuse = True
        _call(pipeline, 0)
        first_failure = wf.active_failure(db)["first_failure_utc"]
        # One write lands, then the store refuses again: the mark is refused.
        gate.allow = 1
        _call(pipeline, 1)
        still_open = wf.active_failure(db)
        assert still_open is not None, "a window nobody wrote down must stay open"
        assert still_open["first_failure_utc"] == first_failure

        gate.refuse = False
        _call(pipeline, 2)
        backend.close()

        trail, backend = _on_disk(db)
        backend.close()
        (mark,) = _marks(trail)
        assert mark.data["first_failure_utc"] == first_failure
        assert mark.data["unrecorded_writes"] == wf.read_marker(db)["count"]
        assert wf.active_failure(db) is None

    def test_a_marker_a_hook_process_left_is_reported_by_the_next_trail(self, tmp_path):
        db = tmp_path / "audit.db"
        for _ in range(3):
            wf.record_failure(db, RuntimeError("cannot open the trail"), stage="append")

        backend = SQLiteAuditBackend(db)
        pipeline = InterceptionPipeline(trail=backend.load_trail())
        _call(pipeline, 0)
        backend.close()

        trail, backend = _on_disk(db)
        backend.close()
        (mark,) = _marks(trail)
        assert mark.data["unrecorded_writes"] == 3
        assert mark.data["stage"] == "append"
        assert "cannot open the trail" in mark.data["error"]

    def test_a_marker_edited_by_hand_still_gives_a_well_formed_mark(self, tmp_path):
        db = tmp_path / "audit.db"
        wf.marker_path(db).write_text(json.dumps({"count": "lots", "resolved": False}))

        backend = SQLiteAuditBackend(db)
        pipeline = InterceptionPipeline(trail=backend.load_trail())
        _call(pipeline, 0)
        backend.close()

        trail, backend = _on_disk(db)
        backend.close()
        (mark,) = _marks(trail)
        assert mark.data["unrecorded_writes"] == 1
        assert mark.data["first_failure_utc"] == "unknown"

    def test_an_in_memory_trail_has_nowhere_to_keep_a_window_and_writes_none(self):
        trail = AuditTrail()
        InterceptionPipeline(trail=trail).intercept(
            agent_id="mem", tool_name="file.read", parameters={"path": "/srv/a.txt"})
        assert _marks(trail) == []


# ── The signed-export check ────────────────────────────────────────────

def _outage_trail(tmp_path: Path, monkeypatch):
    db, _ = _recovered_after_two_refused_calls(tmp_path, monkeypatch)
    return _on_disk(db)


def _export(trail: AuditTrail, tmp_path: Path) -> Path:
    out = tmp_path / "trail.zip"
    export_signed(trail, out, Ed25519PrivateKey.generate())
    return out


def _outage_export(tmp_path: Path, monkeypatch) -> Path:
    trail, backend = _outage_trail(tmp_path, monkeypatch)
    try:
        return _export(trail, tmp_path)
    finally:
        backend.close()


def _forged_export(tmp_path: Path, monkeypatch, mutate) -> Path:
    """An outage export whose mark is rewritten and re-hashed, then re-signed."""
    trail, backend = _outage_trail(tmp_path, monkeypatch)
    try:
        at = next(i for i, r in enumerate(trail._records) if r.event_type.value == MARK)
        mutate(trail, at)
        _rechain(trail)
        return _export(trail, tmp_path)
    finally:
        backend.close()


def test_verify_lists_the_window_and_the_export_still_passes(tmp_path, monkeypatch):
    out = _outage_export(tmp_path, monkeypatch)

    result = lib.verify_signed(out)

    assert result.ok, result.errors
    (window,) = result.unrecorded_windows
    assert window["unrecorded_writes"] >= 2
    assert window["resumed_index"] == window["after_index"] + 1
    assert window["index"] == window["resumed_index"] + 1
    assert window["stage"] == "append_record"


def test_the_line_says_how_many_writes_between_which_records(tmp_path, monkeypatch):
    out = _outage_export(tmp_path, monkeypatch)
    result = lib.verify_signed(out)
    (window,) = result.unrecorded_windows

    (line,) = lib.describe_unrecorded_windows(result)

    assert line.startswith(f"record {window['index']} trail not recording: ")
    assert f"the store refused at least {window['unrecorded_writes']} write(s) " in line
    assert (f"between record {window['after_index']} and record {window['resumed_index']}"
            in line)
    assert window["first_failure_utc"] in line and window["last_failure_utc"] in line
    assert "append_record" in line and "malformed" in line
    assert "calls in that window ran with no record" in line


def test_an_export_with_no_outage_lists_nothing(tmp_path):
    trail = AuditTrail()
    InterceptionPipeline(trail=trail).intercept(
        agent_id="clean", tool_name="file.read", parameters={"path": "/srv/a.txt"})

    result = lib.verify_signed(_export(trail, tmp_path))

    assert result.ok, result.errors
    assert result.unrecorded_windows == []
    assert lib.describe_unrecorded_windows(result) == []


def test_an_independent_reader_with_no_vaara_import_finds_the_same_window(tmp_path, monkeypatch):
    out = _outage_export(tmp_path, monkeypatch)
    (window,) = lib.verify_signed(out).unrecorded_windows

    # zipfile and json only: where the mark sits, how many writes it counts,
    # and the two adjacent records the window lies between.
    with zipfile.ZipFile(out) as z:
        rows = [json.loads(ln) for ln in z.read("trail.jsonl").splitlines() if ln.strip()]
    position = {row["record_id"]: i for i, row in enumerate(rows)}
    found = [(i, row["data"]["unrecorded_writes"],
              position[row["data"]["resumed_record_id"]] - 1,
              position[row["data"]["resumed_record_id"]])
             for i, row in enumerate(rows) if row["event_type"] == MARK]

    assert found == [(window["index"], window["unrecorded_writes"],
                      window["after_index"], window["resumed_index"])]
    assert rows[found[0][2]]["record_hash"] == rows[found[0][3]]["previous_hash"]


@pytest.mark.parametrize("field,value", [
    ("unrecorded_writes", 0),
    ("unrecorded_writes", -3),
    ("unrecorded_writes", True),
    ("unrecorded_writes", "6"),
    ("first_failure_utc", ""),
    ("first_failure_utc", 5),
    ("last_failure_utc", None),
    ("resumed_record_id", ""),
    ("resumed_record_id", 7),
])
def test_a_mark_with_a_nonsense_field_fails_the_export(tmp_path, monkeypatch, field, value):
    def corrupt(trail, at):
        trail._records[at].data[field] = value

    result = lib.verify_signed(_forged_export(tmp_path, monkeypatch, corrupt))

    assert not result.ok
    assert any(e.startswith("trail_not_recording record ") and field in e
               for e in result.errors), result.errors


def test_a_mark_that_names_a_later_record_fails_the_export(tmp_path, monkeypatch):
    def point_forward(trail, at):
        trail._records[at].data["resumed_record_id"] = trail._records[at + 1].record_id

    result = lib.verify_signed(_forged_export(tmp_path, monkeypatch, point_forward))

    assert not result.ok
    assert any("names record" in e and "which is not an earlier record" in e
               for e in result.errors), result.errors


def test_a_resumed_record_outside_the_export_is_reported_and_does_not_fail(tmp_path, monkeypatch):
    def somewhere_else(trail, at):
        trail._records[at].data["resumed_record_id"] = "not-in-this-export"

    result = lib.verify_signed(_forged_export(tmp_path, monkeypatch, somewhere_else))

    assert result.ok, result.errors
    (window,) = result.unrecorded_windows
    assert window["resumed_index"] is None and window["after_index"] is None
    (line,) = lib.describe_unrecorded_windows(result)
    assert "before this export" in line


def test_an_export_that_starts_at_the_resumed_record_says_before_record_zero(tmp_path, monkeypatch):
    def cut_earlier_records(trail, at):
        resumed = at - 1
        del trail._records[:resumed]

    result = lib.verify_signed(_forged_export(tmp_path, monkeypatch, cut_earlier_records))

    assert result.ok, result.errors
    (window,) = result.unrecorded_windows
    assert window["resumed_index"] == 0 and window["after_index"] is None
    (line,) = lib.describe_unrecorded_windows(result)
    assert " before record 0," in line


def test_two_windows_in_one_export_are_both_listed(tmp_path, monkeypatch):
    db, backend, pipeline, gate = _outage(tmp_path, monkeypatch)
    for outage in range(2):
        _call(pipeline, outage * 10)
        gate.refuse = True
        _call(pipeline, outage * 10 + 1)
        gate.refuse = False
    _call(pipeline, 99)
    backend.close()
    trail, backend = _on_disk(db)
    try:
        result = lib.verify_signed(_export(trail, tmp_path))
    finally:
        backend.close()

    assert result.ok, result.errors
    assert len(result.unrecorded_windows) == 2
    assert [w["index"] for w in result.unrecorded_windows] == sorted(
        w["index"] for w in result.unrecorded_windows)
    assert len(lib.describe_unrecorded_windows(result)) == 2


def test_a_malformed_line_and_a_non_object_line_do_not_crash_the_check():
    no_data = json.dumps({"record_id": "m1", "event_type": MARK}).encode()

    windows, errors = lib._check_unrecorded_windows([b"{not json", b"[1, 2]", no_data])

    assert windows == [], "a mark that cannot be read is an error, not a window"
    assert any("record 2" in e and "unrecorded_writes" in e for e in errors), errors


# ── The two commands ───────────────────────────────────────────────────

def test_vaara_trail_verify_prints_the_window(tmp_path, monkeypatch, capsys):
    from vaara.cli import main

    out = _outage_export(tmp_path, monkeypatch)
    assert main(["trail", "verify", "--zip", str(out)]) == 0
    text = capsys.readouterr().out

    assert "trail not recording: the store refused at least" in text
    assert "calls in that window ran with no record" in text


def test_vaara_audit_verify_json_carries_the_window(tmp_path, monkeypatch, capsys):
    from vaara.audit_cli import main

    out = _outage_export(tmp_path, monkeypatch)
    assert main(["verify", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    (window,) = payload["unrecorded_windows"]
    assert window["unrecorded_writes"] >= 2
    assert window["resumed_index"] == window["after_index"] + 1


def test_vaara_audit_verify_text_lists_the_window(tmp_path, monkeypatch, capsys):
    from vaara.audit_cli import main

    out = _outage_export(tmp_path, monkeypatch)
    assert main(["verify", str(out)]) == 0
    text = capsys.readouterr().out

    assert "unrecorded windows:" in text
    assert "trail not recording: the store refused at least" in text
