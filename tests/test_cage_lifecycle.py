# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A cage's start and stop on the trail, and the signed-export check on them.

`vaara cage run`, `vaara cage stop` and `vaara run` write a cage_lifecycle
record when a launch starts and when it ends: driver, name, policy digest,
who, and on a stop the processes ended, the exit code and why. The record is
an ordinary chained record, so it is covered by the export signature;
`verify_signed` lists the records and fails an export whose stop names a
record that is not the start of the same launch, or one that comes after it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import SYSTEM_EVENT_TYPES, AuditTrail, EventType


def _trail_at(db: Path) -> AuditTrail:
    return SQLiteAuditBackend(str(db)).load_trail()


def _lifecycle(db: Path) -> list:
    return [r for r in _trail_at(db).get_records_by_type(EventType.CAGE_LIFECYCLE)]


# ── The trail record ───────────────────────────────────────────────────

class TestRecord:
    def test_start_and_stop_are_chained_records(self):
        trail = AuditTrail()
        start = trail.record_cage_lifecycle(
            "started", "openshell", "demo", policy_digest="sha256:" + "ab" * 32, actor="henri")
        stop = trail.record_cage_lifecycle(
            "stopped", "openshell", "demo", actor="henri", started_record_id=start,
            processes_ended=3, exit_code=0, reason="operator")
        recs = trail.get_records_by_type(EventType.CAGE_LIFECYCLE)
        assert [r.record_id for r in recs] == [start, stop]
        assert recs[0].data == {"action": "started", "driver": "openshell", "name": "demo",
                                "policy_digest": "sha256:" + "ab" * 32, "actor": "henri"}
        assert recs[1].data == {"action": "stopped", "driver": "openshell", "name": "demo",
                                "policy_digest": "", "actor": "henri",
                                "started_record_id": start, "processes_ended": 3,
                                "exit_code": 0, "reason": "operator"}
        assert recs[0].tool_name == "cage:openshell"
        assert trail.verify_chain() is None

    def test_unknown_action_and_missing_driver_are_refused(self):
        trail = AuditTrail()
        with pytest.raises(ValueError, match="action"):
            trail.record_cage_lifecycle("paused", "openshell", "demo")
        with pytest.raises(ValueError, match="driver"):
            trail.record_cage_lifecycle("started", "", "demo")

    def test_a_stop_cannot_name_a_record_that_is_not_this_launchs_start(self):
        trail = AuditTrail()
        other = trail.record_cage_lifecycle("started", "gvisor", "other")
        with pytest.raises(ValueError, match="start"):
            trail.record_cage_lifecycle("stopped", "openshell", "demo",
                                        started_record_id=other)
        with pytest.raises(ValueError, match="start"):
            trail.record_cage_lifecycle("stopped", "openshell", "demo",
                                        started_record_id="no-such-record")

    def test_only_the_trail_writes_it(self):
        assert EventType.CAGE_LIFECYCLE in SYSTEM_EVENT_TYPES

    def test_open_start_finds_the_newest_unstopped_start(self):
        trail = AuditTrail()
        first = trail.record_cage_lifecycle("started", "openshell", "demo")
        trail.record_cage_lifecycle("stopped", "openshell", "demo", started_record_id=first)
        second = trail.record_cage_lifecycle("started", "openshell", "demo")
        trail.record_cage_lifecycle("started", "openshell", "other")
        assert trail.open_cage_start("openshell", "demo") == second
        trail.record_cage_lifecycle("stopped", "openshell", "demo", started_record_id=second)
        assert trail.open_cage_start("openshell", "demo") is None


# ── vaara cage ─────────────────────────────────────────────────────────

class _ChildDriver:
    """A process-backed driver: the launch is this process's child."""

    def __init__(self, code: int = 3):
        from vaara.cage._cli import ChildLaunches

        self._launches = ChildLaunches()
        self.code = code
        self.stopped: list[str] = []

    def start(self, agent, policy, name=None):
        from vaara.cage import CageState
        from vaara.cage._cli import ChildLaunch
        from vaara.cage.driver import CageLaunch

        proc = subprocess.Popen([sys.executable, "-c", f"import sys; sys.exit({self.code})"],
                                stderr=subprocess.PIPE)
        state = CageState(driver="fake", name=name or "n1", config_digest="sha256:cc")
        self._launches.add(ChildLaunch(name or "n1", proc, state))
        return CageLaunch(driver="fake", name=name or "n1", state=state, pid=proc.pid)

    def stop(self, name):
        self.stopped.append(name)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "audit.db"
    monkeypatch.setenv("VAARA_DB", str(path))
    return path


def test_cage_run_writes_the_start_and_the_end_of_a_child_launch(db, tmp_path, monkeypatch):
    from vaara.cage import cli as cage_cli

    policy = tmp_path / "policy.yaml"
    policy.write_bytes(b"network: deny\n")
    monkeypatch.setattr(cage_cli, "load_driver", lambda name: _ChildDriver(code=3))
    code = cage_cli.main(["run", "--driver", "codex", "--policy", str(policy),
                          "--name", "n1", "--", "python", "agent.py"])
    assert code == 3
    start, stop = _lifecycle(db)
    assert start.data["action"] == "started" and start.data["driver"] == "codex"
    assert start.data["name"] == "n1"
    assert start.data["policy_digest"] == "sha256:" + hashlib.sha256(b"network: deny\n").hexdigest()
    assert stop.data["action"] == "stopped"
    assert stop.data["started_record_id"] == start.record_id
    assert stop.data["exit_code"] == 3 and stop.data["reason"] == "exited"


def test_cage_run_without_a_policy_records_the_cages_own_digest(db, monkeypatch):
    from vaara.cage import cli as cage_cli

    monkeypatch.setattr(cage_cli, "load_driver", lambda name: _ChildDriver(code=0))
    assert cage_cli.main(["run", "--driver", "codex", "--", "agent"]) == 0
    assert _lifecycle(db)[0].data["policy_digest"] == "sha256:cc"


def test_cage_stop_closes_the_open_start(db, monkeypatch):
    from vaara.cage import cli as cage_cli

    trail = _trail_at(db)
    start = trail.record_cage_lifecycle("started", "openshell", "demo")
    driver = _ChildDriver()
    monkeypatch.setattr(cage_cli, "load_driver", lambda name: driver)
    assert cage_cli.main(["stop", "--driver", "openshell", "demo"]) == 0
    assert driver.stopped == ["demo"]
    stop = _lifecycle(db)[-1]
    assert stop.data["action"] == "stopped" and stop.data["reason"] == "operator"
    assert stop.data["started_record_id"] == start


def test_cage_stop_of_a_launch_started_elsewhere_is_still_recorded(db, monkeypatch):
    from vaara.cage import cli as cage_cli

    monkeypatch.setattr(cage_cli, "load_driver", lambda name: _ChildDriver())
    assert cage_cli.main(["stop", "--driver", "openshell", "demo"]) == 0
    (stop,) = _lifecycle(db)
    assert "started_record_id" not in stop.data


def test_a_trail_that_cannot_be_written_does_not_stop_the_cage(tmp_path, monkeypatch, capsys):
    from vaara.cage import cli as cage_cli

    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("VAARA_DB", str(blocker / "audit.db"))
    driver = _ChildDriver()
    monkeypatch.setattr(cage_cli, "load_driver", lambda name: driver)
    assert cage_cli.main(["stop", "--driver", "openshell", "demo"]) == 0
    assert driver.stopped == ["demo"]
    assert "lifecycle record not written" in capsys.readouterr().err


# ── vaara run ──────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != "linux",
                    reason="vaara run cages on Linux only (cgroups, AppArmor, an abstract socket)")
def test_vaara_run_records_the_launch_and_the_guard_kill(db, tmp_path, monkeypatch):
    """The guard goes away: the launch is ended, and the stop says how many processes."""
    from vaara.oslayer import run as run_mod

    monkeypatch.setattr(run_mod.floor, "apparmor_enabled", lambda: True)
    monkeypatch.setattr(run_mod.sys, "platform", "linux")
    group = tmp_path / "launch-1"
    group.mkdir()
    (group / "cgroup.procs").write_text("4242\n4243\n")
    killed = threading.Event()

    class Launch:
        cgroup = str(group)
        netlock = False

        def wait_closed(self):
            return True

        def close(self):
            pass

    def fake_kill(pid, sig):
        killed.set()

    def fake_waitpid(pid, flags):
        killed.wait(5)
        return pid, 9  # ended by SIGKILL

    import vaara.oslayer.client as client
    monkeypatch.setattr(client, "request", lambda payload, **kw: {
        "ok": True, "profile_digest": "sha256:11", "profile_loaded": True})
    monkeypatch.setattr(client, "open_launch", lambda payload, **kw: Launch())
    monkeypatch.setattr("vaara.cage.vaara_cage.apparmor_version", lambda: "apparmor 4.0.1")
    monkeypatch.setattr(run_mod, "resolve", lambda agent: ([agent], agent))
    monkeypatch.setattr(run_mod.os, "fork", lambda: 4242)
    monkeypatch.setattr(run_mod.os, "kill", fake_kill)
    monkeypatch.setattr(run_mod.os, "waitpid", fake_waitpid)
    monkeypatch.setattr(run_mod.cgroup, "kill", lambda path: None)
    # With fork faked there is no child to read the go byte; a second read
    # end keeps the pipe open for it.
    r, w = os.pipe()
    reader = os.dup(r)
    monkeypatch.setattr(run_mod.os, "pipe", lambda: (r, w))
    with mock.patch.dict(os.environ), \
            mock.patch.object(run_mod.signal, "signal", lambda *a: None):
        assert run_mod.run("reviewer", ["claude"]) == run_mod.EXIT_GUARD_GONE
    os.close(reader)
    start, stop = _lifecycle(db)
    assert start.data["driver"] == "vaara-cage" and start.data["name"] == "reviewer"
    assert start.data["policy_digest"] == "sha256:11"
    assert stop.data["started_record_id"] == start.record_id
    assert stop.data["reason"] == "guard_gone" and stop.data["processes_ended"] == 2


# ── The signed export ──────────────────────────────────────────────────

def _export(trail: AuditTrail, tmp_path: Path) -> Path:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from vaara.audit.export import export_signed

    out = tmp_path / "trail.zip"
    export_signed(trail, out, Ed25519PrivateKey.generate())
    return out


def test_verify_lists_the_lifecycle_and_passes_a_paired_launch(tmp_path):
    from vaara.audit.verify import verify_signed

    trail = AuditTrail()
    start = trail.record_cage_lifecycle("started", "openshell", "demo", actor="henri")
    trail.record_cage_lifecycle("stopped", "openshell", "demo", started_record_id=start,
                                processes_ended=1, reason="operator")
    trail.record_cage_lifecycle("started", "gvisor", "still-running")
    result = verify_signed(_export(trail, tmp_path))
    assert result.ok, result.errors
    assert [(c["action"], c["driver"], c["name"]) for c in result.cage_lifecycle] == [
        ("started", "openshell", "demo"), ("stopped", "openshell", "demo"),
        ("started", "gvisor", "still-running")]
    assert [c["name"] for c in result.cage_open] == ["still-running"]


def _forge_export(tmp_path: Path, mutate) -> Path:
    """An export whose stop record is rewritten and re-hashed, then re-signed.

    The chain and the signature hold, so only the pairing check can see it:
    this is a signer who wrote a stop for a launch it never recorded starting.
    """
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from vaara.audit.export import export_signed

    trail = AuditTrail()
    start = trail.record_cage_lifecycle("started", "openshell", "demo")
    trail.record_cage_lifecycle("started", "gvisor", "other")
    trail.record_cage_lifecycle("stopped", "openshell", "demo", started_record_id=start)
    mutate(trail)
    out = tmp_path / "forged.zip"
    export_signed(trail, out, Ed25519PrivateKey.generate())
    return out


def _rechain(trail: AuditTrail) -> None:
    prev = ""
    for r in trail._records:
        r.previous_hash = prev
        r.record_hash = r.compute_hash()
        prev = r.record_hash


def test_verify_fails_a_stop_that_names_another_launchs_start(tmp_path):
    from vaara.audit.verify import verify_signed

    def point_at_other(trail):
        other = trail._records[1].record_id
        trail._records[2].data["started_record_id"] = other
        _rechain(trail)

    result = verify_signed(_forge_export(tmp_path, point_at_other))
    assert not result.ok
    assert any("cage_lifecycle" in e and "not the start" in e for e in result.errors)


def test_verify_fails_a_stop_written_before_its_start(tmp_path):
    from vaara.audit.verify import verify_signed

    def stop_first(trail):
        trail._records.insert(0, trail._records.pop())
        _rechain(trail)

    result = verify_signed(_forge_export(tmp_path, stop_first))
    assert not result.ok
    assert any("cage_lifecycle" in e and "before" in e for e in result.errors)


def test_verify_reports_a_start_outside_the_export_without_failing(tmp_path):
    from vaara.audit.verify import verify_signed

    def drop_start(trail):
        del trail._records[0]
        _rechain(trail)

    result = verify_signed(_forge_export(tmp_path, drop_start))
    assert result.ok, result.errors
    assert [c["name"] for c in result.cage_unpaired_stops] == ["demo"]


# ── What the verify commands print ─────────────────────────────────────

def _paired_export(tmp_path: Path) -> Path:
    trail = AuditTrail()
    start = trail.record_cage_lifecycle("started", "openshell", "demo", actor="henri")
    trail.record_cage_lifecycle("stopped", "openshell", "demo", started_record_id=start,
                                processes_ended=2, exit_code=0, reason="operator")
    trail.record_cage_lifecycle("started", "gvisor", "still-running")
    return _export(trail, tmp_path)


def test_vaara_trail_verify_prints_the_cage_launches(tmp_path, capsys):
    from vaara.cli import main

    out = _paired_export(tmp_path)
    assert main(["trail", "verify", "--zip", str(out)]) == 0
    text = capsys.readouterr().out
    assert "cage openshell/demo: started" in text
    assert "stopped (operator, 2 processes ended, exit 0)" in text
    assert "cage gvisor/still-running: started, no stop in this export" in text


def test_vaara_audit_verify_json_carries_the_cage_launches(tmp_path, capsys):
    from vaara.audit_cli import main

    out = _paired_export(tmp_path)
    assert main(["verify", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [(c["action"], c["name"]) for c in payload["cage_lifecycle"]] == [
        ("started", "demo"), ("stopped", "demo"), ("started", "still-running")]
    assert [c["name"] for c in payload["cage_open"]] == ["still-running"]
    assert payload["cage_unpaired_stops"] == []


def test_vaara_audit_verify_text_lists_the_cage_launches(tmp_path, capsys):
    from vaara.audit_cli import main

    out = _paired_export(tmp_path)
    assert main(["verify", str(out)]) == 0
    text = capsys.readouterr().out
    assert "cage launches:" in text
    assert "openshell/demo: started" in text
    assert "gvisor/still-running: started, no stop in this export" in text
