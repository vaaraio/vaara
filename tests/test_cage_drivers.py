# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The nine drivers added after OpenShell, each against a fake of the cage's
own tool: the command line Vaara issues, the declaration it hands in, and
the state it reads back."""

from __future__ import annotations

import hashlib
import json
import stat
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vaara import cage
from vaara.cage.driver import CageError

# ── A fake tool that records its argv and answers from a script ────────

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(json.dumps({"argv": sys.argv[1:], "stdin": sys.stdin.read() if not sys.stdin.isatty() and os.environ.get("FAKE_READ_STDIN") else "", "env": {k: v for k, v in os.environ.items() if k.startswith("VAARA_CAGE")}}) + "\n")
answers = json.loads(os.environ["FAKE_ANSWERS"])
args = sys.argv[1:]
for key, answer in answers:
    if args[:len(key)] == key:
        sys.stdout.write(answer if isinstance(answer, str) else json.dumps(answer))
        sys.exit(0)
if args and args[0] == "--version":
    print(os.environ.get("FAKE_VERSION", "fake 1.2.3")); sys.exit(0)
if os.environ.get("FAKE_SLEEP") and args and args[0] == os.environ.get("FAKE_SLEEP"):
    import time; sys.stderr.write("fake: running\n"); sys.stderr.flush(); time.sleep(30); sys.exit(0)
sys.stderr.write("fake: unknown " + " ".join(args) + "\n"); sys.exit(1)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A factory: fake(name, answers) writes an executable and returns
    (path, calls) where calls() lists what it was invoked with."""
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))

    def make(name: str, answers: list, version: str = "fake 1.2.3", sleep_on: str = ""):
        binary = tmp_path / name
        binary.write_text(FAKE)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("FAKE_ANSWERS", json.dumps(answers))
        monkeypatch.setenv("FAKE_VERSION", version)
        if sleep_on:
            monkeypatch.setenv("FAKE_SLEEP", sleep_on)

        def calls():
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text().splitlines()]

        return str(binary), calls

    return make


def _call(calls, *prefix):
    """The first recorded call starting with ``prefix``; a spawned fake may
    still be writing its line, so wait up to two seconds for it."""
    deadline = time.time() + 2.0
    while True:
        for c in calls():
            if c["argv"][:len(prefix)] == list(prefix):
                return c
        if time.time() > deadline:
            raise AssertionError(f"no call starting with {prefix}: {[c['argv'] for c in calls()]}")
        time.sleep(0.05)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ── Codex ──────────────────────────────────────────────────────────────

class TestCodex:
    def test_start_passes_the_sandbox_state_and_the_declaration(self, fake, tmp_path):
        binary, calls = fake("codex", [], version="codex-cli 0.49.0", sleep_on="sandbox")
        from vaara.cage.codex import CodexSandboxDriver
        policy = tmp_path / "state.json"
        policy.write_text('{"writable_roots": ["/work"], "network": false}')
        d = CodexSandboxDriver(binary=binary)
        launch = d.start(["claude", "-p", "hi"], policy, name="rev")
        try:
            c = _call(calls, "sandbox")
            assert c["argv"] == ["sandbox", "--sandbox-state-json", policy.read_text(),
                                 "--", "claude", "-p", "hi"]
            assert c["env"][cage.CAGE_ENV] == "codex"
            assert c["env"][cage.DIGEST_ENV] == _sha(policy.read_bytes())
            assert c["env"][cage.UPSTREAM_ENV] == "codex 0.49.0"
            assert c["env"][cage.NAME_ENV] == "rev"
            assert launch.state.confirmed and launch.state.basis == "process_alive"
            assert d.status("rev")["running"] is True
            time.sleep(0.3)
            assert any("running" in e["message"] for e in d.events("rev"))
        finally:
            d.stop("rev")
        with pytest.raises(CageError):
            d.status("rev")

    def test_profile_instead_of_a_file(self, fake, tmp_path):
        binary, calls = fake("codex", [], sleep_on="sandbox")
        from vaara.cage.codex import CodexSandboxDriver
        d = CodexSandboxDriver(binary=binary)
        d.start(["agent"], None, name="p", permission_profile="strict", cwd=tmp_path)
        try:
            c = _call(calls, "sandbox")
            assert c["argv"][:5] == ["sandbox", "--permission-profile", "strict", "--cd", str(tmp_path)]
            assert c["env"][cage.DIGEST_ENV] == _sha(b"permission-profile:strict")
        finally:
            d.stop("p")

    def test_non_json_policy_is_refused(self, fake, tmp_path):
        binary, _ = fake("codex", [])
        from vaara.cage.codex import CodexSandboxDriver
        policy = tmp_path / "state.yaml"
        policy.write_text("writable: yes\n")
        with pytest.raises(CageError, match="JSON"):
            CodexSandboxDriver(binary=binary).start(["agent"], policy)


# ── sandbox-runtime ────────────────────────────────────────────────────

class TestSandboxRuntime:
    def test_start_passes_settings(self, fake, tmp_path):
        binary, calls = fake("srt", [], version="0.0.79", sleep_on="--settings")
        from vaara.cage.sandbox_runtime import SandboxRuntimeDriver
        settings = tmp_path / "srt.json"
        settings.write_text('{"network": {"allowedDomains": ["api.anthropic.com"]}}')
        d = SandboxRuntimeDriver(binary=binary)
        launch = d.start(["claude"], settings, name="c")
        try:
            c = _call(calls, "--settings")
            assert c["argv"] == ["--settings", str(settings), "--", "claude"]
            assert c["env"][cage.CAGE_ENV] == "sandbox-runtime"
            assert c["env"][cage.UPSTREAM_ENV] == "sandbox-runtime 0.0.79"
            assert c["env"][cage.DIGEST_ENV] == _sha(settings.read_bytes())
            assert launch.state.confirmed
        finally:
            d.stop("c")

    def test_the_inside_check_is_bwrap_as_init(self, monkeypatch):
        monkeypatch.setattr(cage, "bwrap_is_init", lambda: True)
        monkeypatch.setitem(cage._CONFIRM, "sandbox-runtime", ((cage.bwrap_is_init, cage.BASIS_BWRAP),))
        state = cage.observe({cage.CAGE_ENV: "sandbox-runtime", cage.DIGEST_ENV: "sha256:aa"})
        assert state.confirmed and state.basis == "bwrap_init"


# ── nono ───────────────────────────────────────────────────────────────

NONO_PS = [{"session_id": "20261009-1", "name": "rev", "status": "Running", "child_pid": 7,
            "profile": "opencode", "network": "proxy", "exit_code": None, "command": ["agent"]}]


class TestNono:
    def test_start_status_events_stop(self, fake, tmp_path):
        binary, calls = fake("nono", [
            [["run"], ""],
            [["ps"], NONO_PS],
            [["logs"], '{"ts": 1.0, "message": "landlock applied"}\n{"message": "denied /etc/shadow"}\n'],
            [["stop"], ""],
        ], version="nono 0.14.0")
        from vaara.cage.nono import NonoDriver
        profile = tmp_path / "profile.toml"
        profile.write_text("[filesystem]\nallow = ['/work']\n")
        d = NonoDriver(binary=binary)
        launch = d.start(["agent", "--x"], profile, name="rev", allow=["/work"])
        c = _call(calls, "run")
        assert c["argv"] == ["run", "--name", "rev", "--detached", "--profile", str(profile),
                             "--allow", "/work", "--", "agent", "--x"]
        assert c["env"][cage.CAGE_ENV] == "nono"
        assert c["env"][cage.DIGEST_ENV] == _sha(profile.read_bytes())
        assert launch.state.confirmed and launch.state.basis == "control_plane"
        assert launch.state.detail["session_id"] == "20261009-1"
        events = list(d.events("rev"))
        assert events[0]["message"] == "landlock applied" and events[1]["ts"]
        assert _call(calls, "logs")["argv"] == ["logs", "20261009-1", "--json"]
        d.stop("rev")
        assert _call(calls, "stop")["argv"] == ["stop", "rev"]

    def test_catalogue_profile_is_digested_by_name(self, fake):
        binary, calls = fake("nono", [[["run"], ""], [["ps"], NONO_PS]])
        from vaara.cage.nono import NonoDriver
        NonoDriver(binary=binary).start(["agent"], Path("nolabs-ai/opencode"), name="rev")
        c = _call(calls, "run")
        assert c["argv"][4:6] == ["--profile", "nolabs-ai/opencode"]
        assert c["env"][cage.DIGEST_ENV] == _sha(b"profile:nolabs-ai/opencode")

    def test_unknown_session(self, fake):
        binary, _ = fake("nono", [[["ps"], []]])
        from vaara.cage.nono import NonoDriver
        with pytest.raises(CageError, match="no session"):
            NonoDriver(binary=binary).status("ghost")

    def test_inside_falls_back_from_seccomp_to_no_new_privs(self, monkeypatch):
        monkeypatch.setitem(cage._CONFIRM, "nono", ((lambda: False, cage.BASIS_SECCOMP),
                                                   (lambda: True, cage.BASIS_NO_NEW_PRIVS)))
        state = cage.observe({cage.CAGE_ENV: "nono"})
        assert state.confirmed and state.basis == "no_new_privs"


# ── gVisor and Kata through an engine ──────────────────────────────────

INSPECT = [{"Id": "abc123", "State": {"Status": "running", "Running": True, "ExitCode": 0},
            "HostConfig": {"Runtime": "runsc", "SecurityOpt": None},
            "Config": {"Image": "python:3.12", "Cmd": ["python", "agent.py"]}}]


class TestEngineDrivers:
    def test_gvisor_runs_with_runsc(self, fake, tmp_path):
        engine, calls = fake("docker", [
            [["run"], "abc123\n"], [["inspect"], INSPECT], [["stop"], ""],
            [["logs"], "2026-10-09T20:00:00.000000000Z hello from the sandbox\n"],
        ])
        runsc = tmp_path / "runsc"
        runsc.write_text("#!/bin/sh\necho 'runsc version release-20261001.0'\necho 'spec: 1.1.0'\n")
        runsc.chmod(runsc.stat().st_mode | stat.S_IXUSR)
        from vaara.cage.gvisor import GVisorDriver
        d = GVisorDriver(engine=engine, runsc=str(runsc))
        launch = d.start(["python", "agent.py"], None, name="g", image="python:3.12")
        c = _call(calls, "run")
        assert c["argv"][:4] == ["run", "-d", "--name", "g"]
        assert "--runtime=runsc" in c["argv"]
        envs = [c["argv"][i + 1] for i, a in enumerate(c["argv"]) if a == "-e"]
        assert "VAARA_CAGE=gvisor" in envs
        assert "VAARA_CAGE_UPSTREAM=gvisor release-20261001.0" in envs
        assert c["argv"][-3:] == ["python:3.12", "python", "agent.py"]
        assert launch.state.confirmed and launch.state.basis == "engine_status"
        assert launch.state.detail["runtime"] == "runsc"
        events = list(d.events("g", since=1.0))
        assert events[0]["message"] == "hello from the sandbox"
        assert events[0]["ts"].startswith("2026-10-09T20:00:00")
        d.stop("g")
        assert _call(calls, "stop")["argv"] == ["stop", "g"]

    def test_wrong_runtime_is_not_confirmed(self, fake):
        bad = [dict(INSPECT[0], HostConfig={"Runtime": "runc"})]
        engine, _ = fake("docker", [[["inspect"], bad]])
        from vaara.cage.gvisor import GVisorDriver
        state = GVisorDriver(engine=engine, runsc="/nonexistent/runsc").enforcement_state("g")
        assert not state.confirmed and state.upstream == "gvisor"

    def test_kata_runtime_name_and_version(self, fake, tmp_path):
        engine, calls = fake("docker", [[["run"], "id\n"],
                                        [["inspect"], [dict(INSPECT[0], HostConfig={"Runtime": "kata"})]]])
        kata = tmp_path / "kata-runtime"
        kata.write_text("#!/bin/sh\necho 'kata-runtime  : 3.12.0'\necho '   commit   : abc'\n")
        kata.chmod(kata.stat().st_mode | stat.S_IXUSR)
        from vaara.cage.kata import KataDriver
        d = KataDriver(engine=engine, runtime="kata", kata_runtime=str(kata))
        launch = d.start(["agent"], None, name="k", image="img", security_opt=["no-new-privileges"])
        c = _call(calls, "run")
        assert "--runtime=kata" in c["argv"]
        assert c["argv"][c["argv"].index("--security-opt") + 1] == "no-new-privileges"
        assert launch.state.upstream == "kata 3.12.0" and launch.state.confirmed

    def test_policy_file_is_refused(self, fake, tmp_path):
        engine, _ = fake("docker", [])
        from vaara.cage.gvisor import GVisorDriver
        with pytest.raises(CageError, match="--image"):
            GVisorDriver(engine=engine).start(["agent"], tmp_path / "p.json", image="i")


# ── microsandbox ───────────────────────────────────────────────────────

class TestMicrosandbox:
    def test_run_status_logs(self, fake, tmp_path):
        binary, calls = fake("msb", [
            [["run"], "app\n"],
            [["status"], {"name": "app", "status": "Running", "image": "python:3.12",
                          "command": "agent", "cpus": 1, "memory_mib": 512}],
            [["logs"], '{"timestamp": "2026-10-09T20:00:00Z", "line": "booted"}\n'],
            [["stop"], ""], [["rm"], ""],
        ], version="msb 0.3.1")
        from vaara.cage.microsandbox import MicrosandboxDriver
        conf = tmp_path / "sandbox.yaml"
        conf.write_text("image: python:3.12\nnetwork:\n  allow: []\n")
        d = MicrosandboxDriver(binary=binary)
        launch = d.start(["agent", "run"], conf, name="app")
        c = _call(calls, "run")
        assert c["argv"][:5] == ["run", "--name", "app", "--detach", "--no-tty"]
        assert c["argv"][5:7] == ["--conf", str(conf)]
        envs = [c["argv"][i + 1] for i, a in enumerate(c["argv"]) if a == "-e"]
        assert f"VAARA_CAGE_DIGEST={_sha(conf.read_bytes())}" in envs
        assert "VAARA_CAGE_UPSTREAM=microsandbox 0.3.1" in envs
        assert c["argv"][-3:] == ["--", "agent", "run"]
        assert launch.state.confirmed and launch.state.detail["image"] == "python:3.12"
        assert list(d.events("app"))[0]["message"] == "booted"
        d.stop("app"); d.remove("app")
        assert _call(calls, "rm")["argv"] == ["rm", "app"]

    def test_needs_config_or_image(self, fake):
        binary, _ = fake("msb", [])
        from vaara.cage.microsandbox import MicrosandboxDriver
        with pytest.raises(CageError, match="configuration"):
            MicrosandboxDriver(binary=binary).start(["agent"], None)


# ── apple/container ────────────────────────────────────────────────────

APPLE_VERSION = [{"appName": "container", "buildType": "release", "commit": "abc1234",
                  "version": "0.6.0"},
                 {"appName": "container-apiserver", "buildType": "release",
                  "commit": "abc1234", "version": "container-apiserver version 0.6.0"}]


def _apple_inspect(status):
    return [{"id": "rev",
             "configuration": {"id": "rev", "image": {"reference": "python:3.12"},
                               "runtimeHandler": "container-runtime-linux",
                               "readOnly": True, "capDrop": ["CAP_NET_RAW"],
                               "initProcess": {"arguments": ["agent"]}},
             "status": status}]


class TestAppleContainer:
    def test_run_inspect_logs(self, fake):
        binary, calls = fake("container", [
            [["system", "version"], APPLE_VERSION],
            [["run"], "rev\n"],
            [["inspect"], _apple_inspect({"state": "running", "networks": []})],
            [["logs"], "booted\nworking\n"],
            [["stop"], ""], [["delete"], ""],
        ])
        from vaara.cage.apple_container import AppleContainerDriver
        d = AppleContainerDriver(binary=binary)
        launch = d.start(["agent", "go"], None, name="rev", image="python:3.12",
                         read_only=True, cap_drop=["CAP_NET_RAW"])
        c = _call(calls, "run")
        assert c["argv"][:6] == ["run", "-d", "--name", "rev", "--read-only", "--cap-drop"]
        envs = [c["argv"][i + 1] for i, a in enumerate(c["argv"]) if a == "-e"]
        assert "VAARA_CAGE=apple-container" in envs
        assert "VAARA_CAGE_UPSTREAM=apple-container 0.6.0" in envs
        assert "VAARA_CAGE_NAME=rev" in envs
        digest = next(e for e in envs if e.startswith("VAARA_CAGE_DIGEST="))
        assert digest.split("=", 1)[1] == launch.detail["declared"]["config_digest"]
        assert c["argv"][-3:] == ["python:3.12", "agent", "go"]
        assert launch.state.confirmed and launch.state.basis == "control_plane"
        assert launch.state.detail["image"] == "python:3.12"
        assert [e["message"] for e in d.events("rev")] == ["booted", "working"]
        d.stop("rev"); d.remove("rev")
        assert _call(calls, "delete")["argv"] == ["delete", "rev"]

    def test_released_versions_report_status_as_a_string(self, fake):
        binary, _ = fake("container", [[["system", "version"], APPLE_VERSION],
                                       [["inspect"], _apple_inspect("stopped")]])
        from vaara.cage.apple_container import AppleContainerDriver
        state = AppleContainerDriver(binary=binary).enforcement_state("rev")
        assert not state.confirmed and state.detail["status"] == "stopped"

    def test_needs_an_image_and_refuses_a_policy_file(self, fake, tmp_path):
        binary, _ = fake("container", [])
        from vaara.cage.apple_container import AppleContainerDriver
        d = AppleContainerDriver(binary=binary)
        with pytest.raises(CageError, match="image"):
            d.start(["agent"], None)
        with pytest.raises(CageError, match="no policy file"):
            d.start(["agent"], tmp_path / "p.yaml", image="python:3.12")

    def test_inside_check_is_the_hypervisor(self, monkeypatch):
        monkeypatch.setitem(cage._CONFIRM, "apple-container",
                            ((lambda: True, cage.BASIS_VM),))
        state = cage.observe({cage.CAGE_ENV: "apple-container"})
        assert state.confirmed and state.basis == "hypervisor_present"


# ── Firecracker ────────────────────────────────────────────────────────

class TestFirecracker:
    def test_boot_args_carry_the_declaration(self, fake, tmp_path, monkeypatch):
        binary, calls = fake("firecracker", [], version="Firecracker v1.10.1", sleep_on="--api-sock")
        from vaara.cage.firecracker import FirecrackerDriver
        config = tmp_path / "vm.json"
        config.write_text(json.dumps({
            "boot-source": {"kernel_image_path": "vmlinux", "boot_args": "console=ttyS0 reboot=k"},
            "drives": [{"drive_id": "rootfs", "path_on_host": "rootfs.ext4", "is_root_device": True}],
            "machine-config": {"vcpu_count": 1, "mem_size_mib": 256},
        }))
        d = FirecrackerDriver(binary=binary, run_dir=tmp_path / "run")
        monkeypatch.setattr("vaara.cage.firecracker.time.sleep", lambda s: None)
        launch = d.start(["agent", "--go"], config, name="vm1")
        try:
            c = _call(calls, "--api-sock")
            assert c["argv"][:4] == ["--api-sock", str(tmp_path / "run" / "vm1.firecracker.sock"),
                                     "--id", "vm1"]
            effective = json.loads(Path(c["argv"][5]).read_text())
            boot_args = effective["boot-source"]["boot_args"]
            assert boot_args.startswith("console=ttyS0 reboot=k vaara.cage=firecracker ")
            assert f"vaara.cage.digest={_sha(config.read_bytes())}" in boot_args
            assert "vaara.cage.upstream=firecracker_1.10.1" in boot_args
            assert 'vaara.agent=["agent","--go"]' in boot_args
            # No API answers in the fake: the state falls back to the declaration
            assert launch.state.basis == "declared" and not launch.state.confirmed
        finally:
            d.stop("vm1")

    def test_cmdline_declaration_is_read(self, tmp_path):
        cmdline = tmp_path / "cmdline"
        cmdline.write_text("console=ttyS0 vaara.cage=firecracker vaara.cage.digest=sha256:ab "
                           "vaara.cage.upstream=firecracker_v1.10.1 vaara.cage.name=vm1 quiet\n")
        env = cage.cmdline_declaration(str(cmdline))
        assert env == {cage.CAGE_ENV: "firecracker", cage.DIGEST_ENV: "sha256:ab",
                       cage.UPSTREAM_ENV: "firecracker_v1.10.1", cage.NAME_ENV: "vm1"}
        assert cage.declared(env).driver == "firecracker"

    def test_api_status(self, tmp_path):
        """A fake Firecracker API on a unix socket."""
        import socket as _socket
        from vaara.cage.firecracker import FirecrackerDriver
        run = tmp_path / "run"; run.mkdir()
        sock_path = run / "vm2.firecracker.sock"
        server = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        server.bind(str(sock_path)); server.listen(2)
        stop = threading.Event()

        def serve():
            server.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                except _socket.timeout:
                    continue
                with conn:
                    req = conn.recv(65536).decode()
                    line = req.split("\r\n", 1)[0]
                    if line.startswith("GET / "):
                        body = json.dumps({"id": "vm2", "state": "Running",
                                           "vmm_version": "1.10.1", "app_name": "Firecracker"})
                    elif line.startswith("GET /vm/config"):
                        body = json.dumps({"boot-source": {"boot_args": "x"}, "logger": None})
                    else:
                        body = ""
                    conn.sendall((f"HTTP/1.1 {'204 No Content' if not body else '200 OK'}\r\n"
                                  f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                                  f"Connection: close\r\n\r\n{body}").encode())

        t = threading.Thread(target=serve, daemon=True); t.start()
        try:
            d = FirecrackerDriver(binary="/nonexistent/firecracker", run_dir=run)
            state = d.enforcement_state("vm2")
            assert state.confirmed and state.basis == "api_status"
            assert state.upstream == "firecracker 1.10.1"
            assert state.config_digest.startswith("sha256:")
        finally:
            stop.set(); t.join(timeout=2); server.close()


# ── agent-sandbox ──────────────────────────────────────────────────────

SANDBOX_OBJ = {"apiVersion": "agents.x-k8s.io/v1beta1", "kind": "Sandbox",
               "metadata": {"name": "rev", "resourceVersion": "7", "generation": 1},
               "spec": {"operatingMode": "Running",
                        "podTemplate": {"spec": {"runtimeClassName": "gvisor",
                                                 "containers": [{"name": "agent", "image": "img"}]}}},
               "status": {"conditions": [{"type": "Ready", "status": "True", "reason": "PodReady"}],
                          "podIPs": ["10.0.0.5"]}}


class TestAgentSandbox:
    def test_apply_and_status(self, fake, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_READ_STDIN", "1")
        kubectl, calls = fake("kubectl", [
            [["get", "crd"], {"metadata": {"labels": {"app.kubernetes.io/version": "0.3.0"}},
                              "spec": {"versions": [{"name": "v1beta1"}]}}],
            [["apply"], "sandbox.agents.x-k8s.io/rev created\n"],
            [["get", "sandboxes.agents.x-k8s.io"], SANDBOX_OBJ],
            [["get", "-n", "agents", "sandboxes.agents.x-k8s.io"], SANDBOX_OBJ],
            [["get", "-n", "agents", "crd"], {"metadata": {"labels": {"app.kubernetes.io/version": "0.3.0"}}}],
            [["get", "-n", "agents", "events"], {"items": [{"reason": "Scheduled", "type": "Normal",
                                                             "message": "assigned", "lastTimestamp": "2026-10-09T20:00:00Z"}]}],
            [["delete"], ""],
        ])
        from vaara.cage.agent_sandbox import AgentSandboxDriver
        manifest = tmp_path / "sandbox.json"
        manifest.write_text(json.dumps({"kind": "Sandbox", "spec": {"podTemplate": {"spec": {
            "runtimeClassName": "gvisor",
            "containers": [{"name": "agent", "image": "img", "env": [{"name": "A", "value": "1"}]}]}}}}))
        d = AgentSandboxDriver(kubectl=kubectl, namespace="agents")
        launch = d.start(["python", "agent.py"], manifest, name="rev")
        applied = json.loads(_call(calls, "apply")["stdin"])
        assert _call(calls, "apply")["argv"] == ["apply", "-n", "agents", "-f", "-"]
        assert applied["metadata"] == {"name": "rev", "namespace": "agents"}
        assert applied["apiVersion"] == "agents.x-k8s.io/v1beta1"
        first = applied["spec"]["podTemplate"]["spec"]["containers"][0]
        assert first["command"] == ["python"] and first["args"] == ["agent.py"]
        env = {e["name"]: e["value"] for e in first["env"]}
        assert env["A"] == "1" and env["VAARA_CAGE"] == "agent-sandbox"
        assert env["VAARA_CAGE_DIGEST"] == _sha(manifest.read_bytes())
        assert env["VAARA_CAGE_UPSTREAM"] == "agent-sandbox 0.3.0"
        assert launch.state.confirmed and launch.state.detail["runtime_class"] == "gvisor"
        assert list(d.events("rev"))[0]["reason"] == "Scheduled"
        d.stop("rev")
        assert _call(calls, "delete")["argv"] == ["delete", "-n", "agents", "sandboxes.agents.x-k8s.io", "rev", "--wait=false"]

    def test_not_ready_is_not_confirmed(self, fake):
        obj = dict(SANDBOX_OBJ, status={"conditions": [{"type": "Ready", "status": "False", "reason": "PodPending"}]})
        kubectl, _ = fake("kubectl", [[["get", "sandboxes.agents.x-k8s.io"], obj],
                                      [["get", "crd"], {}]])
        from vaara.cage.agent_sandbox import AgentSandboxDriver
        state = AgentSandboxDriver(kubectl=kubectl).enforcement_state("rev")
        assert not state.confirmed and state.detail["ready_reason"] == "PodPending"

    def test_wrong_kind_is_refused(self, fake, tmp_path):
        kubectl, _ = fake("kubectl", [])
        from vaara.cage.agent_sandbox import AgentSandboxDriver
        manifest = tmp_path / "pod.json"
        manifest.write_text(json.dumps({"kind": "Pod"}))
        with pytest.raises(CageError, match="kind"):
            AgentSandboxDriver(kubectl=kubectl).start(["agent"], manifest)

    def test_inside_check_takes_gvisor_then_vm(self, monkeypatch):
        monkeypatch.setitem(cage._CONFIRM, "agent-sandbox", ((lambda: False, cage.BASIS_GVISOR),
                                                            (lambda: True, cage.BASIS_VM)))
        assert cage.observe({cage.CAGE_ENV: "agent-sandbox"}).basis == "hypervisor_present"


# ── E2B over a fake API and a fake envd ────────────────────────────────

class _E2BHandler(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):  # noqa: D401 - quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        _E2BHandler.seen.append({"method": "POST", "path": self.path, "raw": raw,
                                 "headers": dict(self.headers)})
        if self.path == "/sandboxes":
            assert self.headers.get("X-API-Key") == "key1"
            self._send(201, json.dumps({"sandboxID": "sbx1", "clientID": "c1", "templateID": "base",
                                        "envdAccessToken": "tok", "domain": "example.test"}).encode())
        elif self.path == "/process.Process/Start":
            msg = json.dumps({"event": {"start": {"pid": 42}}}).encode()
            end = json.dumps({}).encode()
            body = struct.pack(">BI", 0, len(msg)) + msg + struct.pack(">BI", 2, len(end)) + end
            self._send(200, body, "application/connect+json")
        else:
            self._send(404, b"{}")

    def do_GET(self):
        _E2BHandler.seen.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
        if self.path == "/sandboxes/sbx1":
            self._send(200, json.dumps({"sandboxID": "sbx1", "state": "running", "templateID": "base",
                                        "envdVersion": "0.2.9", "allowInternetAccess": False,
                                        "startedAt": "2026-10-09T20:00:00Z"}).encode())
        elif self.path.startswith("/sandboxes/sbx1/logs"):
            self._send(200, json.dumps({"logs": [{"timestamp": "2026-10-09T20:00:01Z", "line": "envd up"}]}).encode())
        else:
            self._send(404, b"{}")

    def do_DELETE(self):
        _E2BHandler.seen.append({"method": "DELETE", "path": self.path})
        self._send(204, b"")


@pytest.fixture
def e2b_server():
    _E2BHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _E2BHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}", _E2BHandler.seen
    server.shutdown()
    server.server_close()


class TestE2B:
    def test_create_start_status_logs_delete(self, e2b_server, tmp_path):
        url, seen = e2b_server
        from vaara.cage.e2b import E2BDriver
        request = tmp_path / "request.json"
        request.write_text(json.dumps({"templateID": "base", "timeout": 600,
                                       "allow_internet_access": False}))
        d = E2BDriver(api_url=url, api_key="key1", envd_url=url)
        launch = d.start(["python", "agent.py"], request, name="rev")
        create = next(s for s in seen if s["path"] == "/sandboxes")
        body = json.loads(create["raw"])
        assert body["templateID"] == "base" and body["timeout"] == 600
        assert body["envVars"]["VAARA_CAGE"] == "e2b"
        assert body["envVars"]["VAARA_CAGE_DIGEST"] == _sha(request.read_bytes())
        assert body["metadata"]["vaara.agent"] == '["python","agent.py"]'
        start = next(s for s in seen if s["path"] == "/process.Process/Start")
        assert start["headers"]["X-Access-Token"] == "tok"
        assert start["headers"]["Content-Type"] == "application/connect+json"
        flags, length = struct.unpack(">BI", start["raw"][:5])
        msg = json.loads(start["raw"][5:5 + length])
        assert flags == 0 and msg["process"]["cmd"] == "python"
        assert msg["process"]["args"] == ["agent.py"]
        assert msg["process"]["envs"]["VAARA_CAGE"] == "e2b"
        assert launch.detail["process"] == {"event": {"start": {"pid": 42}}}
        assert launch.state.confirmed and launch.state.basis == "api_status"
        assert launch.state.upstream == "e2b-infra envd 0.2.9"
        assert list(d.events("rev"))[0]["message"] == "envd up"
        d.stop("rev")
        assert any(s["method"] == "DELETE" and s["path"] == "/sandboxes/sbx1" for s in seen)

    def test_missing_configuration(self, tmp_path):
        from vaara.cage.e2b import E2BDriver
        with pytest.raises(CageError, match="templateID"):
            E2BDriver(api_url="http://x", api_key="k").start(["agent"], None)
        with pytest.raises(CageError, match="E2B_API_URL"):
            E2BDriver(api_url="", api_key="k").status("sbx")

    def test_first_frame_reports_an_envd_error(self):
        from vaara.cage.e2b import _first_frame
        err = json.dumps({"error": {"code": "permission_denied", "message": "no"}}).encode()
        with pytest.raises(CageError, match="permission_denied"):
            _first_frame(struct.pack(">BI", 2, len(err)) + err)


# ── Every driver is loadable and the inside map covers all of them ─────

def test_every_driver_has_an_inside_check_and_loads():
    for name in cage.DRIVERS:
        assert name in cage._CONFIRM, name
        assert cage.load_driver(name).name == name
