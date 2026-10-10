# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The cage layer: the block every decision carries, and the two drivers."""

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

from vaara import cage
from vaara.audit.decision_receipts import build_evidence
from vaara.audit.trail import AuditTrail, EventType
from vaara.cage.driver import CageError, CageLaunch
from vaara.cage.openshell import OpenShellDriver, parse_log_line
from vaara.cage.vaara_cage import VaaraCageDriver, profile_digest

DECLARED = {
    cage.CAGE_ENV: "openshell",
    cage.DIGEST_ENV: "sha256:" + "ab" * 32,
    cage.UPSTREAM_ENV: "openshell 0.1.5",
    cage.NAME_ENV: "demo",
}


# ── The block ──────────────────────────────────────────────────────────

class TestObserve:
    def test_no_declaration_is_none_and_two_keys(self):
        state = cage.observe({})
        assert state.driver == "none" and not state.confirmed
        assert state.to_record() == {"driver": "none", "confirmed": False}

    def test_declared_openshell_confirmed_by_the_kernel(self, monkeypatch):
        monkeypatch.setattr(cage, "seccomp_filter_on", lambda: True)
        monkeypatch.setitem(cage._CONFIRM, "openshell", ((cage.seccomp_filter_on, cage.BASIS_SECCOMP),))
        block = cage.observe(DECLARED).to_record()
        assert block == {
            "driver": "openshell", "confirmed": True, "upstream": "openshell 0.1.5",
            "config_digest": "sha256:" + "ab" * 32, "basis": "seccomp_filter", "name": "demo",
        }

    def test_declared_but_not_held_stays_unconfirmed(self, monkeypatch):
        monkeypatch.setitem(cage._CONFIRM, "openshell", ((lambda: False, cage.BASIS_SECCOMP),))
        block = cage.observe(DECLARED).to_record()
        assert block["confirmed"] is False and block["basis"] == "declared"
        assert block["driver"] == "openshell"

    def test_declared_vaara_cage_confirmed_by_the_apparmor_label(self, monkeypatch):
        monkeypatch.setitem(cage._CONFIRM, "vaara-cage", ((lambda: True, cage.BASIS_APPARMOR),))
        env = {cage.CAGE_ENV: "vaara-cage", cage.DIGEST_ENV: "sha256:00",
               cage.UPSTREAM_ENV: "apparmor 4.0.1"}
        state = cage.observe(env)
        assert state.confirmed and state.basis == "apparmor_label"
        assert "name" not in state.to_record()

    def test_a_probe_that_raises_never_confirms(self, monkeypatch):
        def boom():
            raise OSError("no /proc")
        monkeypatch.setitem(cage._CONFIRM, "openshell", ((boom, cage.BASIS_SECCOMP),))
        assert cage.observe(DECLARED).confirmed is False

    def test_unknown_driver_is_declared_only(self):
        state = cage.observe({cage.CAGE_ENV: "firecracker-thing"})
        assert state.driver == "firecracker-thing" and state.basis == "declared"

    def test_environ_round_trip(self):
        state = cage.CageState(driver="openshell", upstream="openshell 0.1.5",
                               config_digest="sha256:cd", name="x")
        assert cage.declared(cage.environ_for(state)).to_record() == {
            "driver": "openshell", "confirmed": False, "upstream": "openshell 0.1.5",
            "config_digest": "sha256:cd", "basis": "declared", "name": "x",
        }

    def test_seccomp_probe_reads_proc_status(self, monkeypatch):
        monkeypatch.setattr(cage, "_proc_status", lambda pid="self": {"Seccomp": "2", "NoNewPrivs": "1"})
        assert cage.seccomp_filter_on()
        monkeypatch.setattr(cage, "_proc_status", lambda pid="self": {"Seccomp": "0", "NoNewPrivs": "1"})
        assert not cage.seccomp_filter_on()
        monkeypatch.setattr(cage, "_proc_status", lambda pid="self": {})
        assert not cage.seccomp_filter_on()

    def test_registry(self):
        assert isinstance(cage.load_driver("openshell", binary="/nonexistent/openshell"), OpenShellDriver)
        with pytest.raises(ValueError):
            cage.load_driver("chroot")


# ── On the record and in the receipt ──────────────────────────────────

class TestOnTheRecord:
    def test_every_decision_carries_a_cage_block(self, monkeypatch):
        monkeypatch.delenv(cage.CAGE_ENV, raising=False)
        trail = AuditTrail()
        trail.record_decision(action_id="a", agent_id="x", tool_name="t",
                              decision="allow", reason="r", risk_score=0.1)
        trail.record_decision(action_id="b", agent_id="x", tool_name="t",
                              decision="deny", reason="r", risk_score=0.9)
        allowed = trail.get_records_by_type(EventType.DECISION_MADE)[0]
        denied = trail.get_records_by_type(EventType.ACTION_BLOCKED)[0]
        assert allowed.data["cage"] == {"driver": "none", "confirmed": False}
        assert denied.data["cage"] == {"driver": "none", "confirmed": False}

    def test_the_trail_observes_the_declaration(self, monkeypatch):
        for k, v in DECLARED.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setitem(cage._CONFIRM, "openshell", ((lambda: True, cage.BASIS_SECCOMP),))
        trail = AuditTrail()
        trail.record_decision(action_id="a", agent_id="x", tool_name="t",
                              decision="allow", reason="r", risk_score=0.1)
        rec = trail.get_records_by_type(EventType.DECISION_MADE)[0]
        assert rec.data["cage"]["driver"] == "openshell"
        assert rec.data["cage"]["confirmed"] is True

    def test_an_explicit_block_is_written_as_given(self):
        trail = AuditTrail()
        block = {"driver": "vaara-cage", "confirmed": True, "upstream": "apparmor 4.0.1",
                 "config_digest": "sha256:ff", "basis": "apparmor_label"}
        trail.record_decision(action_id="a", agent_id="x", tool_name="t",
                              decision="allow", reason="r", risk_score=0.1, cage=block)
        rec = trail.get_records_by_type(EventType.DECISION_MADE)[0]
        assert rec.data["cage"] == block

    def test_the_receipt_lifts_the_block(self):
        trail = AuditTrail()
        block = {"driver": "vaara-cage", "confirmed": True, "upstream": "apparmor 4.0.1",
                 "config_digest": "sha256:" + "ff" * 32, "basis": "apparmor_label", "name": "claude"}
        trail.record_decision(action_id="a", agent_id="x", tool_name="t",
                              decision="allow", reason="r", risk_score=0.1, cage=block)
        rec = trail.get_records_by_type(EventType.DECISION_MADE)[0]
        evidence = build_evidence(rec)
        assert evidence["cage"] == {
            "driver": "vaara-cage", "confirmed": True, "upstream": "apparmor 4.0.1",
            "configDigest": "sha256:" + "ff" * 32, "basis": "apparmor_label", "name": "claude",
        }

    @pytest.mark.parametrize("block, want", [
        # A digest that is not sha256 hex is left out of the receipt.
        ({"driver": "openshell", "confirmed": False, "config_digest": "sha256:ff",
          "basis": "declared"},
         {"driver": "openshell", "confirmed": False, "basis": "declared"}),
        # Confirmed on the launcher's word alone is written as unconfirmed.
        ({"driver": "openshell", "confirmed": True, "basis": "declared"},
         {"driver": "openshell", "confirmed": False, "basis": "declared"}),
        ({"driver": "openshell", "confirmed": True},
         {"driver": "openshell", "confirmed": False, "basis": "declared"}),
        # No cage cannot be a confirmed cage, and carries nothing else.
        ({"driver": "none", "confirmed": True, "basis": "apparmor_label"},
         {"driver": "none", "confirmed": False}),
    ])
    def test_the_receipt_never_overstates_a_hand_built_block(self, block, want):
        trail = AuditTrail()
        trail.record_decision(action_id="a", agent_id="x", tool_name="t",
                              decision="allow", reason="r", risk_score=0.1, cage=block)
        rec = trail.get_records_by_type(EventType.DECISION_MADE)[0]
        assert rec.data["cage"] == block
        assert build_evidence(rec)["cage"] == want

    def test_an_unconfined_receipt_says_so(self, monkeypatch):
        monkeypatch.delenv(cage.CAGE_ENV, raising=False)
        trail = AuditTrail()
        trail.record_decision(action_id="a", agent_id="x", tool_name="t",
                              decision="allow", reason="r", risk_score=0.1)
        rec = trail.get_records_by_type(EventType.DECISION_MADE)[0]
        assert build_evidence(rec)["cage"] == {"driver": "none", "confirmed": False}

    def test_a_record_without_a_block_leaves_the_receipt_silent(self):
        from vaara.audit.trail import AuditRecord
        rec = AuditRecord(record_id="r", action_id="a", event_type=EventType.DECISION_MADE,
                          timestamp=1.0, agent_id="x", tool_name="t",
                          data={"decision": "allow", "reason": "r", "risk_score": 0.1},
                          regulatory_articles=[], record_hash="", previous_hash="")
        assert "cage" not in build_evidence(rec)


# ── OpenShell driver, against a fake CLI ──────────────────────────────

FAKE_OPENSHELL = r'''#!/usr/bin/env python3
import json, os, sys
log = os.environ["FAKE_LOG"]
with open(log, "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\n")
args = sys.argv[1:]
if args == ["--version"]:
    print("openshell 0.1.5"); sys.exit(0)
if args[:2] == ["sandbox", "create"]:
    sys.exit(0)
if args[:2] == ["sandbox", "get"] and "--policy-only" in args:
    sys.stdout.write("version: 1\nnetwork:\n  policies: []\n"); sys.exit(0)
if args[:2] == ["sandbox", "get"]:
    print(json.dumps({"id": "sb-1", "name": args[2], "phase": os.environ.get("FAKE_PHASE", "Ready"),
                      "current_policy_version": 2, "policy_source": "sandbox", "revision": 2,
                      "configuration_admission": {"state": "accepted", "policy_hash": "sha256:cafe"},
                      "exit_code": None})); sys.exit(0)
if args[:2] == ["sandbox", "stop"] or args[:2] == ["sandbox", "delete"]:
    sys.exit(0)
if args[:1] == ["logs"]:
    print("[12.345] [sandbox] [INFO ] [ocsf] CONNECT action=allow dst_host=api.example port=443")
    print("[12.400] [sandbox] [INFO ] [supervisor] policy loaded version=2")
    print("[12.500] [sandbox] [WARN ] [ocsf] CONNECT action=deny dst_host=evil.example port=443 reason=\"no rule\"")
    print("not a log line")
    sys.exit(0)
sys.stderr.write("fake openshell: unknown " + " ".join(args) + "\n"); sys.exit(1)
'''


@pytest.fixture
def fake_openshell(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("the fake openshell is an executable script, which Windows does not run")
    binary = tmp_path / "openshell"
    binary.write_text(FAKE_OPENSHELL)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("FAKE_PHASE", raising=False)

    def calls():
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []

    return OpenShellDriver(binary=str(binary)), calls


class TestOpenShellDriver:
    def test_start_declares_the_cage_to_the_sandbox(self, fake_openshell, tmp_path):
        driver, calls = fake_openshell
        policy = tmp_path / "policy.yaml"
        policy.write_text("version: 1\n")
        launch = driver.start(["claude", "-p", "hi"], policy, name="demo", image="img:1",
                              providers=["anthropic"])
        assert isinstance(launch, CageLaunch) and launch.name == "demo"
        create = next(c for c in calls() if c[:2] == ["sandbox", "create"])
        assert create[2:4] == ["--name", "demo"]
        assert "--detach" in create and "--policy" in create and "--from" in create
        assert create[create.index("--from") + 1] == "img:1"
        assert create[create.index("--provider") + 1] == "anthropic"
        envs = [create[i + 1] for i, a in enumerate(create) if a == "--env"]
        digest = "sha256:" + __import__("hashlib").sha256(b"version: 1\n").hexdigest()
        assert "VAARA_CAGE=openshell" in envs
        assert f"VAARA_CAGE_DIGEST={digest}" in envs
        assert "VAARA_CAGE_UPSTREAM=openshell 0.1.5" in envs
        assert "VAARA_CAGE_NAME=demo" in envs
        assert create[create.index("--"):] == ["--", "claude", "-p", "hi"]
        assert launch.detail["declared"]["config_digest"] == digest
        # After create, the gateway's view
        assert launch.state.confirmed and launch.state.basis == "gateway_status"

    def test_enforcement_state_reads_the_gateway(self, fake_openshell):
        driver, _ = fake_openshell
        state = driver.enforcement_state("demo")
        assert state.driver == "openshell" and state.confirmed
        assert state.upstream == "openshell 0.1.5"
        assert state.config_digest.startswith("sha256:")
        assert state.detail["policy_hash"] == "sha256:cafe"
        assert state.detail["current_policy_version"] == 2

    def test_a_sandbox_that_is_not_ready_is_not_confirmed(self, fake_openshell, monkeypatch):
        driver, _ = fake_openshell
        monkeypatch.setenv("FAKE_PHASE", "Provisioning")
        state = driver.enforcement_state("demo")
        assert not state.confirmed and state.detail["phase"] == "Provisioning"

    def test_enforcement_state_needs_a_name(self, fake_openshell):
        driver, _ = fake_openshell
        with pytest.raises(CageError):
            driver.enforcement_state()

    def test_events_parse_the_ocsf_lines(self, fake_openshell):
        driver, calls = fake_openshell
        events = list(driver.events("demo", since=__import__("time").time() - 600))
        assert [e["target"] for e in events] == ["ocsf", "supervisor", "ocsf"]
        assert events[0]["kind"] == "CONNECT"
        assert events[0]["fields"] == {"action": "allow", "dst_host": "api.example", "port": "443"}
        assert events[2]["fields"]["reason"] == "no rule"
        logs = next(c for c in calls() if c[:1] == ["logs"])
        assert logs[1:4] == ["demo", "--source", "sandbox"] and logs[4] == "--since"
        assert logs[5].endswith("s")

    def test_stop_and_delete(self, fake_openshell):
        driver, calls = fake_openshell
        driver.stop("demo")
        driver.delete("demo")
        assert ["sandbox", "stop", "demo"] in calls()
        assert ["sandbox", "delete", "demo"] in calls()

    def test_missing_binary_is_a_plain_error(self, tmp_path):
        driver = OpenShellDriver(binary=str(tmp_path / "nope"))
        with pytest.raises(CageError, match="not found"):
            driver.upstream_version()

    def test_a_failing_command_reports_stderr(self, fake_openshell):
        driver, _ = fake_openshell
        with pytest.raises(CageError, match="unknown"):
            driver._run("sandbox", "frobnicate")


def test_parse_log_line():
    assert parse_log_line("garbage") is None
    event = parse_log_line('[1.5] [gateway] [OCSF ] [ocsf] NET:OPEN a=1 b="two words"')
    assert event["source"] == "gateway" and event["level"] == "OCSF"
    assert event["kind"] == "NET:OPEN" and event["fields"] == {"a": "1", "b": "two words"}


# ── Vaara cage driver, against a fake guard ───────────────────────────

@pytest.fixture
def fake_guard(tmp_path):
    """A unix socket that answers `status` like the guard does."""
    path = tmp_path / "guard.sock"
    reply = {"ok": True, "pid": 1, "user": "op", "profile": "vaara-agent",
             "profile_loaded": True, "profile_digest": profile_digest("profile text"),
             "trail": str(tmp_path / "missing.db"), "folders": [], "apps": [],
             "ask_timeout": 30, "launches": [{"launch": "l1", "agent": "claude", "pid": 42,
                                              "binary": "/usr/bin/claude", "started": 1.0}],
             "waiting": 0}
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    stop = threading.Event()

    def serve():
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.recv(65536)
                conn.sendall((json.dumps(reply) + "\n").encode())

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    yield path, reply
    stop.set()
    t.join(timeout=2)
    server.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the guard socket is Linux")
class TestVaaraCageDriver:
    def test_enforcement_state_from_the_guard(self, fake_guard, monkeypatch):
        path, reply = fake_guard
        monkeypatch.setattr("vaara.cage.vaara_cage.apparmor_version", lambda: "apparmor 4.0.1")
        driver = VaaraCageDriver(socket_path=path)
        state = driver.enforcement_state()
        assert state.driver == "vaara-cage" and state.confirmed
        assert state.config_digest == reply["profile_digest"]
        assert state.upstream == "apparmor 4.0.1" and state.basis == "guard_status"
        assert driver.enforcement_state("claude").confirmed
        assert not driver.enforcement_state("codex").confirmed

    def test_status_names_the_launch(self, fake_guard):
        path, _ = fake_guard
        driver = VaaraCageDriver(socket_path=path)
        assert driver.status("claude")["running"] is True
        assert driver.status("codex")["running"] is False

    def test_policy_file_is_refused(self, fake_guard, tmp_path):
        path, _ = fake_guard
        driver = VaaraCageDriver(socket_path=path)
        with pytest.raises(CageError, match="OS-layer selection"):
            driver.start(["claude"], tmp_path / "p.yaml")

    def test_no_guard_is_a_plain_error(self, tmp_path):
        driver = VaaraCageDriver(socket_path=tmp_path / "none.sock")
        with pytest.raises(CageError, match="not running"):
            driver.enforcement_state()

    def test_events_with_no_trail_file_is_empty(self, fake_guard):
        path, _ = fake_guard
        assert list(VaaraCageDriver(socket_path=path).events("claude")) == []

    def test_start_spawns_vaara_run(self, fake_guard, tmp_path, monkeypatch):
        path, _ = fake_guard
        monkeypatch.setattr("vaara.cage.vaara_cage.apparmor_version", lambda: "apparmor 4.0.1")
        marker = tmp_path / "argv.json"
        fake = tmp_path / "vaara"
        fake.write_text(f"#!/usr/bin/env python3\nimport json,sys\n"
                        f"json.dump(sys.argv[1:], open({str(marker)!r}, 'w'))\n")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        driver = VaaraCageDriver(socket_path=path, vaara_argv=[str(fake)])
        launch = driver.start(["claude", "-p", "hi"], name="reviewer")
        driver._launches["reviewer"].wait(timeout=10)
        assert json.loads(marker.read_text(encoding="utf-8")) == ["run", "--name", "reviewer", "--", "claude", "-p", "hi"]
        assert launch.state.confirmed and launch.state.name == "reviewer"
        driver.stop("reviewer")
        with pytest.raises(CageError):
            driver.stop("reviewer")


# ── CLI ───────────────────────────────────────────────────────────────

def test_cli_drivers_and_observe(capsys, monkeypatch):
    from vaara.cage.cli import main
    monkeypatch.delenv(cage.CAGE_ENV, raising=False)
    assert main(["drivers"]) == 0
    heads = [line.split()[0] for line in capsys.readouterr().out.splitlines()
             if line and not line[0].isspace()]
    assert heads == list(cage.DRIVERS)
    assert main(["observe"]) == 0
    assert json.loads(capsys.readouterr().out) == {"driver": "none", "confirmed": False}


def test_cli_status_through_the_fake_openshell(fake_openshell, capsys, monkeypatch):
    from vaara.cage.cli import main
    driver, _ = fake_openshell
    monkeypatch.setenv("OPENSHELL_BIN", driver._binary)
    assert main(["status", "--driver", "openshell", "demo"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["driver"] == "openshell" and out["confirmed"] is True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the guard socket is Linux")
def test_vaara_run_declares_the_cage(monkeypatch):
    """`vaara run` hands the tree the declaration from the guard's status."""
    from vaara.oslayer import run as run_mod
    monkeypatch.setattr(run_mod.floor, "apparmor_enabled", lambda: True)
    monkeypatch.setattr(run_mod.sys, "platform", "linux")
    seen = {}

    def fake_request(payload, *, socket_path=None, timeout=10.0):
        return {"ok": True, "profile_digest": "sha256:11", "profile_loaded": True}

    class StopHere(Exception):
        pass

    def fake_open_launch(payload, *, socket_path=None):
        seen.update({k: os.environ.get(k) for k in (cage.CAGE_ENV, cage.DIGEST_ENV,
                                                    cage.UPSTREAM_ENV, cage.NAME_ENV)})
        raise StopHere()

    import vaara.oslayer.client as client
    monkeypatch.setattr(client, "request", fake_request)
    monkeypatch.setattr(client, "open_launch", fake_open_launch)
    monkeypatch.setattr("vaara.cage.vaara_cage.apparmor_version", lambda: "apparmor 4.0.1")
    monkeypatch.setattr(run_mod, "resolve", lambda agent: ([agent], agent))
    monkeypatch.setattr(run_mod.os, "fork", lambda: 4242)
    monkeypatch.setattr(run_mod.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(run_mod.os, "waitpid", lambda pid, flags: (pid, 0))
    # run() writes the declaration into its own environment before the fork;
    # with fork faked that is this process, and every later test would see it.
    with mock.patch.dict(os.environ), pytest.raises(StopHere):
        run_mod.run("reviewer", ["claude"])
    assert seen == {cage.CAGE_ENV: "vaara-cage", cage.DIGEST_ENV: "sha256:11",
                    cage.UPSTREAM_ENV: "apparmor 4.0.1", cage.NAME_ENV: "reviewer"}


def test_cage_run_keeps_a_dash_dash_inside_the_agent_and_follows_a_child(monkeypatch, capsys):
    """Only the leading separator goes, and a process-backed launch is waited for."""
    import subprocess

    from vaara.cage import cli as cage_cli
    from vaara.cage._cli import ChildLaunch, ChildLaunches
    from vaara.cage.driver import CageLaunch

    class Fake:
        def __init__(self):
            self._launches = ChildLaunches()

        def start(self, agent, policy, name=None):
            self.argv = agent
            proc = subprocess.Popen([sys.executable, "-c",
                                     "import sys; print('from the cage', file=sys.stderr); sys.exit(3)"],
                                    stderr=subprocess.PIPE)
            from vaara.cage import CageState
            state = CageState(driver="fake", name="n1")
            self._launches.add(ChildLaunch("n1", proc, state))
            return CageLaunch(driver="fake", name="n1", state=state, pid=proc.pid)

    fake = Fake()
    monkeypatch.setattr(cage_cli, "load_driver", lambda name: fake)
    code = cage_cli.main(["run", "--driver", "codex", "--", "git", "log", "--", "src/"])
    assert fake.argv == ["git", "log", "--", "src/"]
    assert code == 3
    assert "from the cage" in capsys.readouterr().err


def test_a_relayed_decision_confirms_the_vaara_cage_on_the_asking_agent(monkeypatch):
    """The hook runs outside the floor; the agent that asked carries the label."""
    import vaara.cage as cage

    seen = []

    def label(pid="self"):
        seen.append(pid)
        return pid == 4242

    monkeypatch.setattr(cage, "apparmor_agent_label", label)
    monkeypatch.setitem(cage._CONFIRM, "vaara-cage", ((label, cage.BASIS_APPARMOR),))
    env = {cage.CAGE_ENV: "vaara-cage", cage.DIGEST_ENV: "sha256:aa"}
    assert cage.observe(env).basis == cage.BASIS_DECLARED
    got = cage.observe({**env, cage.PEER_ENV: "4242"})
    assert got.confirmed and got.basis == cage.BASIS_APPARMOR
    assert not cage.observe({**env, cage.PEER_ENV: "77"}).confirmed
    # Another cage's declaration is never confirmed through the peer.
    monkeypatch.setitem(cage._CONFIRM, "codex", ((lambda: False, cage.BASIS_SECCOMP),))
    assert not cage.observe({cage.CAGE_ENV: "codex", cage.PEER_ENV: "4242"}).confirmed


def test_the_relay_hands_the_hook_the_asking_pid(monkeypatch, tmp_path):
    import subprocess as sp

    from vaara.cage import PEER_ENV
    from vaara.oslayer import forward

    captured = {}

    def fake_run(argv, **kw):
        captured.update(kw["env"])
        return sp.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(forward.subprocess, "run", fake_run)
    monkeypatch.setenv(PEER_ENV, "999")  # never inherited from vaara run itself
    server = forward.HookServer.__new__(forward.HookServer)
    server._hook_cmd = ["vaara"]
    server.answer({"argv": ["pre-tool-use"], "stdin": ""}, peer=1234)
    assert captured[PEER_ENV] == "1234"
    captured.clear()
    server.answer({"argv": ["pre-tool-use"], "stdin": ""})
    assert PEER_ENV not in captured


# ── The public vectors ────────────────────────────────────────────────

CAGE_VECTORS = Path(__file__).parent / "vectors" / "cage_v0"


def test_the_independent_checker_matches_expected():
    run = subprocess.run([sys.executable, str(CAGE_VECTORS / "_check_independent.py")],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize(
    "name", sorted(json.loads((CAGE_VECTORS / "expected.json").read_text(encoding="utf-8")))
)
def test_the_engine_verifier_agrees_on_signature_and_evidence(name):
    from vaara.audit import decision_receipts as dr

    want = json.loads((CAGE_VECTORS / "expected.json").read_text(encoding="utf-8"))[name]
    c = dr.verify_receipt_file(
        CAGE_VECTORS / name, public_key_pem=(CAGE_VECTORS / dr.PUBKEY_NAME).read_bytes(),
    )
    assert (c.signature_ok, c.evidence_ok) == (want["signature"], want["evidence"])


@pytest.mark.parametrize("name", sorted(p.name for p in (CAGE_VECTORS / "invalid").glob("*.json")))
def test_the_engine_would_have_written_each_bad_block_as_a_good_one(name):
    from vaara.audit.decision_receipts import _cage_block

    sys.path.insert(0, str(CAGE_VECTORS))
    try:
        import _check_independent as checker
    finally:
        sys.path.remove(str(CAGE_VECTORS))
    wire = json.loads((CAGE_VECTORS / "invalid" / name).read_text(encoding="utf-8"))["evidence"]["cage"]
    trail_form = {("config_digest" if k == "configDigest" else k): v for k, v in wire.items()}
    assert checker._cage_ok({"cage": _cage_block(trail_form, "r")})
