"""The standalone verifier pairs cage stops with their starts the way the library does.

`vaara cage run`, `vaara cage stop` and `vaara run` write a `cage_lifecycle`
record when a launch starts and when it ends, and `vaara trail verify` pairs
each stop with the start it names: a stop that names another launch's start,
or one written before its start, fails the export, a start with no stop is
listed as still open, and a stop whose start is outside the export is listed
as unpaired. `scripts/verify_vaara_trail.py` imports no Vaara code and used
to skip all of it, so an auditor holding only the script passed a trail a
signer had forged a stop into. It now gives the same verdicts and prints the
same lines.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from tests.test_cage_lifecycle import (  # noqa: E402
    _forge_export,
    _paired_export,
    _rechain,
)
from tests.test_erasure_record_signed import _build, _export  # noqa: E402
from vaara.audit import verify as lib  # noqa: E402

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


def test_a_paired_launch_verifies_and_the_script_prints_the_library_lines(tmp_path: Path):
    zp = _paired_export(tmp_path)

    run = _run(zp)

    assert run.returncode == 0, run.stderr
    expected = lib.describe_cage_lifecycle(lib.verify_signed(zp))
    assert len(expected) == 3
    printed = run.stdout.splitlines()
    for line in expected:
        assert f"  {line}" in printed
    assert "stopped (operator, 2 processes ended, exit 0)" in run.stdout
    assert "cage gvisor/still-running: started, no stop in this export" in run.stdout


def test_a_stop_that_names_another_launchs_start_fails_the_export(tmp_path: Path):
    def point_at_other(trail):
        trail._records[2].data["started_record_id"] = trail._records[1].record_id
        _rechain(trail)

    run = _run(_forge_export(tmp_path, point_at_other))

    assert run.returncode == 1
    assert "Verification: FAILED" in run.stderr
    assert ("cage_lifecycle record 2: names record 1, which is not the start of "
            "openshell/demo") in run.stderr
    assert "cage openshell/demo" not in run.stdout


def test_a_stop_written_before_its_start_fails_the_export(tmp_path: Path):
    def stop_first(trail):
        trail._records.insert(0, trail._records.pop())
        _rechain(trail)

    run = _run(_forge_export(tmp_path, stop_first))

    assert run.returncode == 1
    assert "cage_lifecycle record 0: a stop written before its start (record 1)" in run.stderr


def test_a_start_outside_the_export_is_reported_and_does_not_fail(tmp_path: Path):
    def drop_start(trail):
        del trail._records[0]
        _rechain(trail)

    run = _run(_forge_export(tmp_path, drop_start))

    assert run.returncode == 0, run.stderr
    assert "cage openshell/demo: stopped" in run.stdout
    assert "start not in this export" in run.stdout


def test_an_export_without_cage_records_prints_no_cage_line(tmp_path: Path):
    path, _ = _build(tmp_path)

    run = _run(_export(path, tmp_path))

    assert run.returncode == 0, run.stderr
    assert "cage " not in run.stdout


# --- the pairing and the lines, record for record against the library -----------


def _rec(record_id: str, event_type: str, data: dict) -> bytes:
    return json.dumps({"record_id": record_id, "event_type": event_type,
                       "timestamp": 1760000000.0, "data": data}).encode()


def _start(record_id: str, driver: str = "openshell", name: str = "demo", **more) -> bytes:
    return _rec(record_id, "cage_lifecycle", {"action": "started", "driver": driver,
                                              "name": name, "policy_digest": "sha256:11",
                                              "actor": "henri", **more})


def _stop(record_id: str, started: object = "s1", driver: str = "openshell",
          name: str = "demo", **more) -> bytes:
    data = {"action": "stopped", "driver": driver, "name": name, **more}
    if started is not None:
        data["started_record_id"] = started
    return _rec(record_id, "cage_lifecycle", data)


_SCENARIOS = {
    "paired": [_start("s1"), _stop("e1", processes_ended=2, exit_code=0, reason="operator")],
    "open-start": [_start("s1"), _start("s2", "gvisor", "other")],
    "stop-without-pointer": [_stop("e1", started=None)],
    "start-outside-export": [_stop("e1", started="gone", reason="guard_gone")],
    "other-launchs-start": [_start("s1"), _start("s2", "gvisor", "other"), _stop("e1", "s2")],
    "driver-differs": [_start("s1", "gvisor"), _stop("e1", "s1", driver="openshell")],
    "names-a-plain-record": [_rec("p1", "action_requested", {"n": 1}), _stop("e1", "p1")],
    "names-another-stop": [_start("s1"), _stop("e1", "s1"), _stop("e2", "e1")],
    "stop-before-start": [_stop("e1", "s1"), _start("s1")],
    "unknown-action": [_rec("c1", "cage_lifecycle", {"action": "paused", "driver": "x",
                                                     "name": "y"})],
    "pointer-not-a-string": [_start("s1"), _stop("e1", started=5)],
    "malformed-and-non-dict-lines": [b"{not json", b"[1, 2]", _start("s1"), _stop("e1", "s1")],
    "two-stops-one-start": [_start("s1"), _stop("e1", "s1"), _stop("e2", "s1")],
    "no-data": [_rec("c1", "cage_lifecycle", {}), b'{"event_type": "cage_lifecycle"}'],
}


@pytest.mark.parametrize("lines", _SCENARIOS.values(), ids=list(_SCENARIOS))
def test_the_pairing_gives_the_librarys_verdict(lines):
    assert script._check_cage_lifecycle(lines) == lib._check_cage_lifecycle(lines)


@pytest.mark.parametrize("lines", _SCENARIOS.values(), ids=list(_SCENARIOS))
def test_the_lines_are_the_librarys_lines(lines):
    records, cage_open, unpaired, _ = lib._check_cage_lifecycle(lines)
    result = lib.VerifyResult(ok=True, cage_lifecycle=records, cage_open=cage_open,
                              cage_unpaired_stops=unpaired)

    assert script.describe_cage_lifecycle(records, cage_open, unpaired) == \
        lib.describe_cage_lifecycle(result)


def test_the_scenarios_reach_every_outcome():
    verdicts = {name: script._check_cage_lifecycle(lines) for name, lines in _SCENARIOS.items()}

    assert verdicts["paired"][1:] == ([], [], [])
    assert [e["record_id"] for e in verdicts["open-start"][1]] == ["s1", "s2"]
    assert [e["record_id"] for e in verdicts["start-outside-export"][2]] == ["e1"]
    assert "not the start of openshell/demo" in verdicts["other-launchs-start"][3][0]
    assert "a stop written before its start" in verdicts["stop-before-start"][3][0]
    assert "unknown action 'paused'" in verdicts["unknown-action"][3][0]
