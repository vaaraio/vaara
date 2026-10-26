"""The standalone verifier reports an unrecorded window the way the library does.

When the audit store refuses writes and then takes them again, the trail
appends a `trail_not_recording` record saying how many writes it refused and
between which two records. `vaara trail verify` lists it and fails a record
whose fields are nonsense. `scripts/verify_vaara_trail.py` imports no Vaara
code and used to pass over the record, so an auditor holding only the script
saw a clean chain where the library saw a hole. It now gives the same verdicts
and prints the same lines.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from tests.test_unrecorded_window_mark import (  # noqa: E402
    MARK,
    _export,
    _forged_export,
    _outage_export,
)
from vaara.audit import verify as lib  # noqa: E402
from vaara.audit.trail import AuditTrail  # noqa: E402
from vaara.pipeline import InterceptionPipeline  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_vaara_trail.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_vaara_trail_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script = _load_script()


def _run(zip_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(zip_path), *extra],
                          capture_output=True, text=True)


# --- the whole script, on real exports ------------------------------------------


def test_an_outage_export_verifies_and_the_script_prints_the_library_lines(tmp_path, monkeypatch):
    out = _outage_export(tmp_path, monkeypatch)

    run = _run(out)

    assert run.returncode == 0, run.stderr
    expected = lib.describe_unrecorded_windows(lib.verify_signed(out))
    assert len(expected) == 1
    assert f"  {expected[0]}" in run.stdout.splitlines()
    assert "trail not recording: the store refused at least" in run.stdout


def test_a_mark_with_a_zero_count_fails_the_export(tmp_path, monkeypatch):
    def zero(trail, at):
        trail._records[at].data["unrecorded_writes"] = 0

    run = _run(_forged_export(tmp_path, monkeypatch, zero))

    assert run.returncode == 1
    assert "Verification: FAILED" in run.stderr
    assert "unrecorded_writes 0 is not a whole number of at least 1" in run.stderr
    assert "trail not recording" not in run.stdout


def test_a_mark_that_names_a_later_record_fails_the_export(tmp_path, monkeypatch):
    def point_forward(trail, at):
        trail._records[at].data["resumed_record_id"] = trail._records[at + 1].record_id

    run = _run(_forged_export(tmp_path, monkeypatch, point_forward))

    assert run.returncode == 1
    assert "which is not an earlier record" in run.stderr


def test_a_resumed_record_outside_the_export_is_reported_and_does_not_fail(tmp_path, monkeypatch):
    def somewhere_else(trail, at):
        trail._records[at].data["resumed_record_id"] = "not-in-this-export"

    run = _run(_forged_export(tmp_path, monkeypatch, somewhere_else))

    assert run.returncode == 0, run.stderr
    assert "refused at least" in run.stdout and "before this export" in run.stdout


def test_an_export_with_no_outage_prints_no_window_line(tmp_path):
    trail = AuditTrail()
    InterceptionPipeline(trail=trail).intercept(
        agent_id="clean", tool_name="file.read", parameters={"path": "/srv/a.txt"})

    run = _run(_export(trail, tmp_path))

    assert run.returncode == 0, run.stderr
    assert "trail not recording" not in run.stdout


# --- the check and the lines, record for record against the library -------------


def _rec(record_id: str, event_type: str, data: dict) -> bytes:
    return json.dumps({"record_id": record_id, "event_type": event_type,
                       "timestamp": 1760000000.0, "data": data}).encode()


def _plain(record_id: str) -> bytes:
    return _rec(record_id, "action_requested", {"n": 1})


def _mark(record_id: str, resumed: object = "r2", **more) -> bytes:
    data = {"unrecorded_writes": 6, "first_failure_utc": "2026-10-26T10:00:00Z",
            "last_failure_utc": "2026-10-26T10:00:03Z", "resumed_record_id": resumed,
            "stage": "append_record",
            "error": "DatabaseError: database disk image is malformed", **more}
    return _rec(record_id, MARK, {k: v for k, v in data.items() if v is not ...})


_SCENARIOS = {
    "one-window": [_plain("r1"), _plain("r2"), _mark("m1")],
    "window-at-the-start": [_plain("r2"), _mark("m1")],
    "resumed-outside-export": [_mark("m1", "gone")],
    "no-stage-no-error": [_plain("r1"), _plain("r2"), _mark("m1", stage="", error="")],
    "stage-only": [_plain("r1"), _plain("r2"), _mark("m1", error="")],
    "error-only": [_plain("r1"), _plain("r2"), _mark("m1", stage="")],
    "long-error": [_plain("r1"), _plain("r2"), _mark("m1", error="x" * 300)],
    "two-windows": [_plain("r1"), _plain("r2"), _mark("m1"), _plain("r3"),
                    _mark("m2", "r3", unrecorded_writes=2)],
    "count-zero": [_plain("r2"), _mark("m1", unrecorded_writes=0)],
    "count-negative": [_plain("r2"), _mark("m1", unrecorded_writes=-4)],
    "count-bool": [_plain("r2"), _mark("m1", unrecorded_writes=True)],
    "count-string": [_plain("r2"), _mark("m1", unrecorded_writes="6")],
    "count-float": [_plain("r2"), _mark("m1", unrecorded_writes=2.5)],
    "count-missing": [_plain("r2"), _mark("m1", unrecorded_writes=...)],
    "first-failure-missing": [_plain("r2"), _mark("m1", first_failure_utc=...)],
    "last-failure-not-text": [_plain("r2"), _mark("m1", last_failure_utc=7)],
    "resumed-empty": [_plain("r2"), _mark("m1", "")],
    "resumed-not-text": [_plain("r2"), _mark("m1", 5)],
    "names-a-later-record": [_mark("m1", "r9"), _plain("r9")],
    "names-itself": [_plain("r1"), _mark("m1", "m1")],
    "malformed-and-non-dict-lines": [b"{not json", b"[1, 2]", _plain("r1"), _plain("r2"),
                                     _mark("m1")],
    "repeated-member-line": [b'{"event_type": "trail_not_recording", "event_type": "x"}',
                             _plain("r2"), _mark("m1")],
    "no-data": [_rec("m1", MARK, {}), b'{"event_type": "trail_not_recording"}'],
}


@pytest.mark.parametrize("lines", _SCENARIOS.values(), ids=list(_SCENARIOS))
def test_the_check_gives_the_librarys_verdict(lines):
    assert script._check_unrecorded_windows(lines) == lib._check_unrecorded_windows(lines)


@pytest.mark.parametrize("lines", _SCENARIOS.values(), ids=list(_SCENARIOS))
def test_the_lines_are_the_librarys_lines(lines):
    windows, _ = lib._check_unrecorded_windows(lines)
    result = lib.VerifyResult(ok=True, unrecorded_windows=windows)

    assert script.describe_unrecorded_windows(windows) == lib.describe_unrecorded_windows(result)


def test_the_scenarios_reach_every_outcome():
    verdicts = {name: script._check_unrecorded_windows(lines)
                for name, lines in _SCENARIOS.items()}

    (window,), errors = verdicts["one-window"]
    assert errors == []
    assert (window["after_index"], window["resumed_index"]) == (0, 1)
    assert verdicts["window-at-the-start"][0][0]["after_index"] is None
    assert verdicts["resumed-outside-export"][0][0]["resumed_index"] is None
    assert len(verdicts["two-windows"][0]) == 2
    for name in ("count-zero", "count-negative", "count-bool", "count-string",
                 "count-float", "count-missing"):
        assert verdicts[name][0] == [] and "unrecorded_writes" in verdicts[name][1][0], name
    for name in ("first-failure-missing", "last-failure-not-text", "resumed-empty",
                 "resumed-not-text"):
        assert verdicts[name][0] == [] and "is missing or not text" in verdicts[name][1][0], name
    assert "which is not an earlier record" in verdicts["names-a-later-record"][1][0]
    assert "which is not an earlier record" in verdicts["names-itself"][1][0]
    assert verdicts["malformed-and-non-dict-lines"][1] == []
    assert len(script.describe_unrecorded_windows(verdicts["long-error"][0])[0]) < 400
