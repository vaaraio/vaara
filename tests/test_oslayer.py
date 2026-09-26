"""The Linux OS layer, the parts that run without root.

What needs root and a kernel (loading the floor, fanotify, cgroups, a real
agent under ``vaara run``) runs in the os-layer-e2e CI job. Here: the
profile text and whether AppArmor's parser accepts it, the operator's
selection, the kernel's refusal records, how ``vaara run`` finds the program
to start, and how the guard decides once an agent's open has reached it.
"""
from __future__ import annotations

import json
import os
import pwd
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import pytest

from vaara.approvals import write_decision
from vaara.oslayer import cgroup, denials, fanotify, floor, guard, manage, run, selection


# ── The floor ─────────────────────────────────────────────────────


def _render(**kw) -> str:
    kw.setdefault("install_paths", ["/opt/vaara/lib/"])
    kw.setdefault("abi", "abi <abi/4.0>,")
    return floor.render(["/home/op", "/root"], **kw)


def test_block_folder_is_sealed_in_both_profiles():
    text = _render(block_folders=["/home/op/secret"])
    rule = "deny /home/op/secret/** mrwlkx,"
    assert text.count(rule) == 2


def test_watched_folders_keep_hard_links_from_pointing_in():
    text = _render(watched_folders=["/home/op/clients"])
    assert text.count("deny link /** -> /home/op/clients/**,") == 2


def test_guard_socket_and_trail_are_sealed():
    text = _render()
    for path in ("/run/vaara/", "/var/lib/vaara/", "/home/op/.vaara/"):
        assert f"deny {path}** mrwlkx," in text


def test_apps_attach_by_exact_path():
    assert floor.attachment([]) == ""
    assert floor.attachment(["/usr/local/bin/copilot"]) == "/usr/local/bin/copilot"
    assert floor.attachment(["/usr/local/bin/copilot", "/opt/x/claude"]) == \
        "/{opt/x/claude,usr/local/bin/copilot}"
    text = _render(apps=["/usr/local/bin/copilot"])
    assert "profile vaara-agent /usr/local/bin/copilot flags=(attach_disconnected) {" in text


def test_profile_without_apps_has_no_attachment():
    assert "profile vaara-agent flags=(attach_disconnected) {" in _render()


def test_parser_accepts_the_profile(tmp_path):
    parser = guard.parser_path()
    if parser is None:
        pytest.skip("apparmor_parser not installed")
    abi = floor.abi_line()
    text = floor.render(
        ["/home/op", "/root"],
        block_folders=["/home/op/secret", "/home/op/with space"],
        watched_folders=["/home/op/clients"],
        apps=["/usr/local/bin/copilot", "/opt/x/claude"],
        install_paths=floor.vaara_install_paths(),
        abi=abi)
    profile = tmp_path / "vaara-agent"
    profile.write_text(text)
    done = subprocess.run([parser, "-Q", "-K", str(profile)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# ── The operator's selection ──────────────────────────────────────


def test_selection_round_trip(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    sel = selection.set_folder(selection.Selection(), str(work), "ask")
    selection.save(str(tmp_path), sel)
    loaded = selection.load(str(tmp_path))
    assert loaded.folders == [selection.Folder(str(work.resolve()), "ask")]
    assert selection.set_folder(loaded, str(work), None).folders == []


def test_deepest_folder_decides(tmp_path):
    outer, inner = tmp_path / "a", tmp_path / "a" / "b"
    inner.mkdir(parents=True)
    sel = selection.Selection(folders=[selection.Folder(str(outer), "record"),
                                       selection.Folder(str(inner), "ask")])
    assert sel.match(f"{inner}/f.txt").mode == "ask"
    assert sel.match(f"{outer}/f.txt").mode == "record"
    assert sel.match(str(outer)).mode == "record"
    assert sel.match(f"{outer}x/f.txt") is None


@pytest.mark.parametrize("bad", ["relative/path", "/", "/proc/self", "/run/vaara",
                                 "/var/lib/vaara/os-layer", "/tmp/star*dir"])
def test_folders_the_layer_owns_cannot_be_picked(bad):
    with pytest.raises(selection.SelectionError):
        selection.check_folder(bad)


def test_unknown_modes_and_missing_folders_are_dropped(tmp_path):
    ok = tmp_path / "ok"
    ok.mkdir()
    sel = selection.parse({"folders": [
        {"path": str(ok), "mode": "record"},
        {"path": str(tmp_path / "gone"), "mode": "ask"},
        {"path": str(ok), "mode": "shrug"},
    ], "ask_timeout": 0})
    assert [f.mode for f in sel.folders] == ["record"]
    assert sel.ask_timeout == selection.DEFAULT_ASK_TIMEOUT


# ── The kernel's refusal records ──────────────────────────────────

KMSG_LINE = ('audit: type=1400 audit(1758850000.123:456): apparmor="DENIED" '
             'operation="open" class="file" profile="vaara-agent//tool" '
             'name="/home/op/.vaara/trail/audit.db" pid=4321 comm="rm" '
             'requested_mask="wd" denied_mask="wd" fsuid=1000 ouid=1000')


def test_kernel_refusal_line_parses():
    d = denials.parse(KMSG_LINE)
    assert d is not None
    assert (d.stamp, d.operation, d.profile, d.target, d.pid, d.comm, d.denied) == (
        "1758850000.123:456", "open", "vaara-agent//tool",
        "/home/op/.vaara/trail/audit.db", 4321, "rm", "wd")


def test_auditd_line_parses_the_same():
    line = KMSG_LINE.replace("audit: type=1400 audit(", "type=AVC msg=audit(")
    assert denials.parse(line).stamp == "1758850000.123:456"


def test_hex_encoded_name_is_decoded():
    hexname = "/home/op/my notes".encode().hex().upper()
    line = KMSG_LINE.replace('name="/home/op/.vaara/trail/audit.db"', f"name={hexname}")
    assert denials.parse(line).target == "/home/op/my notes"


def test_capability_refusal_names_the_capability():
    line = ('audit: type=1400 audit(1.0:7): apparmor="DENIED" operation="capable" '
            'class="cap" profile="vaara-agent" pid=9 comm="insmod" capability=16 '
            'capname="sys_module"')
    assert denials.parse(line).target == "sys_module"


def test_other_profiles_and_allowed_records_are_ignored():
    assert denials.parse(KMSG_LINE.replace("vaara-agent//tool", "snap.firefox")) is None
    assert denials.parse(KMSG_LINE.replace('"DENIED"', '"ALLOWED"')) is None
    assert denials.parse("usb 1-1: new high-speed USB device") is None


def test_same_record_from_both_sources_counts_once():
    seen = []
    f = denials.Follower(seen.append, sources=())
    d = denials.parse(KMSG_LINE)
    assert f._fresh(d) and not f._fresh(d)


# ── fanotify, cgroups ─────────────────────────────────────────────


def test_fanotify_events_parse_from_a_buffer():
    meta = struct.Struct("IBBHQii")
    buf = (meta.pack(meta.size, 3, 0, meta.size, fanotify.FAN_OPEN_PERM, 7, 100)
           + meta.pack(meta.size, 3, 0, meta.size,
                       fanotify.FAN_OPEN_PERM | fanotify.FAN_OPEN_EXEC_PERM, 8, 200))
    events = list(fanotify.parse(buf))
    assert [(e.fd, e.pid, e.is_exec) for e in events] == [(7, 100, False), (8, 200, True)]


def test_mount_point_is_the_longest_prefix():
    points = ["/", "/home", "/home/op/mnt", "/homes"]
    assert fanotify.mount_point("/home/op/work", points) == "/home"
    assert fanotify.mount_point("/home/op/mnt/x", points) == "/home/op/mnt"
    assert fanotify.mount_point("/homesick", points) == "/"


def test_launch_id_comes_from_the_cgroup_path():
    assert cgroup.launch_in("0::/vaara/launch-ab12cd\n") == "ab12cd"
    assert cgroup.launch_in("0::/vaara/launch-ab12cd/sub\n") == "ab12cd"
    assert cgroup.launch_in("0::/user.slice/session-2.scope\n") is None


# ── vaara run: arguments and the program it starts ────────────────


def test_run_options_stop_at_the_agent():
    assert run.parse_args(["copilot", "--name", "x", "-p", "hi"]) == \
        (None, ["copilot", "--name", "x", "-p", "hi"])
    assert run.parse_args(["--name", "rev", "claude"]) == ("rev", ["claude"])
    assert run.parse_args(["--name=rev", "--", "-weird"]) == ("rev", ["-weird"])
    with pytest.raises(run.RunError):
        run.parse_args([])
    with pytest.raises(run.RunError):
        run.parse_args(["--bogus", "claude"])


def _script(tmp_path: Path, name: str, shebang: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!{shebang}\necho hi\n")
    path.chmod(0o755)
    return path


def test_env_script_starts_its_interpreter_directly(tmp_path):
    sh = os.path.realpath("/bin/sh")
    script = _script(tmp_path, "agent", "/usr/bin/env sh")
    prefix, binary = run.resolve(str(script), path_env=os.path.dirname(sh))
    assert prefix == [sh, str(script)]
    assert binary == str(script.resolve())


def test_env_split_passes_the_interpreter_its_flags(tmp_path):
    sh = os.path.realpath("/bin/sh")
    script = _script(tmp_path, "agent", "/usr/bin/env -S sh -e")
    prefix, _ = run.resolve(str(script), path_env=os.path.dirname(sh))
    assert prefix == [sh, "-e", str(script)]


def test_direct_interpreter_gets_the_rest_of_the_line_as_one_argument(tmp_path):
    script = _script(tmp_path, "agent", "/bin/sh -e -u")
    prefix, _ = run.resolve(str(script))
    assert prefix == ["/bin/sh", "-e -u", str(script)]


def test_binary_starts_as_itself():
    real = os.path.realpath("/bin/sh")
    assert run.resolve("/bin/sh") == ([real], real)


def test_missing_agent_is_refused(tmp_path):
    with pytest.raises(run.RunError, match="not found"):
        run.resolve("no-such-agent-here", path_env=str(tmp_path))


def test_run_refuses_without_apparmor(monkeypatch, capsys):
    monkeypatch.setattr(run.sys, "platform", "linux")
    monkeypatch.setattr(floor, "apparmor_enabled", lambda: False)
    assert run.main(["sh"]) == run.EXIT_REFUSED
    assert "unconfined" in capsys.readouterr().err


# ── The guard's decisions ─────────────────────────────────────────


class _Result:
    def __init__(self, action_id: str) -> None:
        self.action_id = action_id


class _Pipeline:
    """Stands in for InterceptionPipeline: records what the guard asks of it."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.resolved: list[tuple] = []
        self._n = 0

    def intercept(self, **kw):
        self._n += 1
        self.calls.append(kw)
        return _Result(f"act-{self._n}")

    def resolve_escalation(self, action_id, resolution, **kw):
        self.resolved.append((action_id, resolution, kw.get("approver"),
                              kw.get("human_disposed")))


@pytest.fixture
def g(tmp_path):
    me = pwd.getpwuid(os.getuid())
    instance = guard.Guard(me, trail_path=tmp_path / "trail" / "audit.db",
                           socket_path=tmp_path / "run" / "os-guard.sock")
    instance.home = str(tmp_path)
    instance.approvals_dir = tmp_path / ".vaara" / "approvals"
    instance._pipeline = _Pipeline()
    instance._selection = selection.Selection(ask_timeout=5)
    yield instance
    instance._pool.shutdown(wait=False)


def _event(exec_: bool = False) -> fanotify.Event:
    mask = fanotify.FAN_OPEN_PERM | (fanotify.FAN_OPEN_EXEC_PERM if exec_ else 0)
    return fanotify.Event(mask=mask, fd=-1, pid=os.getpid())


def test_block_folder_open_is_refused_and_recorded(g):
    folder = selection.Folder("/data/secret", "block")
    assert g.verdict(_event(), "/data/secret/a", folder, "vaara-agent//tool") is False
    call = g._pipeline.calls[-1]
    assert (call["tool_name"], call["policy_decision"], call["policy_id"]) == \
        ("os.open", "deny", "os-layer:block")


def test_record_folder_records_the_first_open_of_each_file_per_session(g):
    folder = selection.Folder("/data/notes", "record")
    for _ in range(3):
        assert g.verdict(_event(), "/data/notes/a", folder, "vaara-agent") is True
    assert g.verdict(_event(exec_=True), "/data/notes/a", folder, "vaara-agent") is True
    assert [(c["tool_name"], c["policy_decision"]) for c in g._pipeline.calls] == \
        [("os.open", "allow"), ("os.exec", "allow")]
    assert g._pipeline.calls[0]["parameters"]["path"] == "/data/notes/a"


def test_an_agent_outside_a_launch_is_named_by_its_program(g):
    agent, session = g.who(os.getpid())
    assert agent.startswith("app:") and session == f"pid-{os.getpid()}"


def _answer_requests(approvals_dir: Path, decision: str, seen: list) -> threading.Thread:
    def responder() -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            for req in approvals_dir.glob("*.request.json"):
                action_id = req.name.removesuffix(".request.json")
                if action_id in seen:
                    continue
                seen.append(action_id)
                seen.append(json.loads(req.read_text()))
                write_decision(action_id, decision, approvals_dir=approvals_dir)
                return
            time.sleep(0.02)

    t = threading.Thread(target=responder, daemon=True)
    t.start()
    return t


@pytest.mark.parametrize("decision,allowed", [("approve", True), ("deny", False)])
def test_ask_folder_waits_for_the_signed_answer(g, decision, allowed):
    g._prepare_approvals()
    seen: list = []
    responder = _answer_requests(g.approvals_dir, decision, seen)
    folder = selection.Folder("/data/clients", "ask")
    assert g.verdict(_event(), "/data/clients/x.pdf", folder, "vaara-agent//tool") is allowed
    responder.join(5)
    request = seen[1]
    assert request["tool_name"] == "os.open"
    assert "wants to open /data/clients/x.pdf" in request["reason"]
    assert g._pipeline.calls[0]["policy_decision"] == "escalate"
    assert g._pipeline.resolved == [("act-1", "allow" if allowed else "deny", "human", True)]
    # Answered once per file per session: the next open does not ask again.
    assert g.verdict(_event(), "/data/clients/x.pdf", folder, "vaara-agent//tool") is allowed
    assert len(g._pipeline.calls) == 1


def test_unanswered_ask_is_refused(g):
    g._prepare_approvals()
    g._selection = selection.Selection(ask_timeout=0.3)
    folder = selection.Folder("/data/clients", "ask")
    assert g.verdict(_event(), "/data/clients/x.pdf", folder, "vaara-agent") is False
    assert g._pipeline.resolved == []
    assert list(g.approvals_dir.glob("*.json")) == []


def test_concurrent_opens_of_one_file_ask_once(g):
    g._prepare_approvals()
    seen: list = []
    _answer_requests(g.approvals_dir, "approve", seen)
    folder = selection.Folder("/data/clients", "ask")
    results: list = []
    threads = [threading.Thread(target=lambda: results.append(
        g.verdict(_event(), "/data/clients/x.pdf", folder, "vaara-agent")))
        for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results == [True] * 4
    assert len(g._pipeline.calls) == 1


def test_approvals_dir_and_key_belong_to_the_operator(g):
    from vaara.approvals import approval_key_path

    g._prepare_approvals()
    key = approval_key_path(g.approvals_dir)
    assert key.stat().st_uid == g.uid
    assert key.stat().st_mode & 0o777 == 0o600
    assert g.approvals_dir.stat().st_uid == g.uid


def test_undecidable_open_is_refused(g, monkeypatch):
    held = []
    monkeypatch.setattr(g, "_respond", lambda event, allow: held.append(allow))
    monkeypatch.setattr(g, "verdict", lambda *a: 1 / 0)
    event = _event()
    g._held[1] = (event, time.monotonic() + 30)
    g._decide(1, event, "/data/x", selection.Folder("/data", "record"), "vaara-agent")
    assert held == [False]


def test_late_answers_are_refused_and_only_once(g, monkeypatch):
    answered = []
    monkeypatch.setattr(g, "_respond", lambda event, allow: answered.append(allow))
    g._held[1] = (_event(), time.monotonic() - 1)
    g._held[2] = (_event(), time.monotonic() + 60)
    g._sweep_held()
    assert answered == [False] and list(g._held) == [2]
    g._answer(1, True)
    assert answered == [False]


def test_floor_refusal_is_recorded_as_a_deny(g):
    g._on_denial(denials.parse(KMSG_LINE))
    call = g._pipeline.calls[-1]
    assert call["tool_name"] == "os.floor"
    assert call["policy_decision"] == "deny"
    assert call["parameters"]["target"] == "/home/op/.vaara/trail/audit.db"


# ── The guard's socket ────────────────────────────────────────────


def _ask(g, payload: dict) -> dict:
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with ours, theirs:
        theirs.sendall((json.dumps(payload) + "\n").encode())
        reply, launch = g.handle_request(ours)
    assert launch is None
    return reply


def test_status_names_the_folders(g, tmp_path):
    g._selection = selection.Selection(folders=[selection.Folder(str(tmp_path), "ask")])
    reply = _ask(g, {"op": "status"})
    assert reply["ok"] and reply["folders"] == [{"path": str(tmp_path), "mode": "ask"}]


def test_other_users_are_not_served(g):
    g.uid = g.uid + 12345
    # Root is always served; anyone else only when they are the operator.
    assert _ask(g, {"op": "status"})["ok"] is (os.getuid() == 0)


def test_launch_of_a_process_that_is_not_the_callers_child_is_refused(g):
    reply = _ask(g, {"op": "launch", "pid": 1})
    assert reply == {"ok": False, "error": "pid 1 is not a child of the caller"}


def test_unknown_op_is_refused(g):
    assert _ask(g, {"op": "format-disk"})["ok"] is False


# ── vaara os-layer ────────────────────────────────────────────────


def test_pending_lists_only_the_guards_requests(tmp_path):
    (tmp_path / "a.request.json").write_text(json.dumps(
        {"action_id": "a", "tool_name": "os.open", "reason": "r", "requested_at": 2}))
    (tmp_path / "b.request.json").write_text(json.dumps(
        {"action_id": "b", "tool_name": "Bash", "reason": "r", "requested_at": 1}))
    (tmp_path / "c.request.json").write_text("{half")
    assert [r["action_id"] for r in manage.pending(tmp_path)] == ["a"]


# ── The dashboard panel ───────────────────────────────────────────


@pytest.fixture
def operator_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(manage, "_guard", lambda op: None)
    monkeypatch.setattr("vaara.approvals.APPROVALS_DIR", tmp_path / ".vaara" / "approvals")
    return tmp_path


def test_panel_sets_a_folder_mode(operator_home):
    from vaara.dashboard import _oslayer_change

    work = operator_home / "work"
    work.mkdir()
    body, code = _oslayer_change({"action": "folder", "path": str(work), "mode": "ask"})
    assert code == 200 and body["guard_told"] is False
    assert selection.load(str(operator_home)).folders == [
        selection.Folder(str(work.resolve()), "ask")]
    body, code = _oslayer_change({"action": "folder", "path": str(work), "mode": "off"})
    assert code == 200 and selection.load(str(operator_home)).folders == []


@pytest.mark.parametrize("payload,code", [
    ({"action": "folder", "path": "/tmp", "mode": "watch"}, 400),
    ({"action": "folder", "path": "/proc", "mode": "ask"}, 400),
    ({"action": "decide", "action_id": "gone", "decision": "approve"}, 409),
    ({"action": "decide", "action_id": "gone", "decision": "maybe"}, 400),
    ({"action": "rm"}, 400),
])
def test_panel_refuses_what_it_does_not_offer(operator_home, payload, code):
    from vaara.dashboard import _oslayer_change

    assert _oslayer_change(payload)[1] == code


def test_panel_state_lists_the_guards_questions(operator_home, monkeypatch):
    from vaara.dashboard import _oslayer_state

    monkeypatch.setattr(run.sys, "platform", "linux")
    approvals = operator_home / ".vaara" / "approvals"
    approvals.mkdir(parents=True)
    (approvals / "a.request.json").write_text(json.dumps(
        {"action_id": "a", "tool_name": "os.exec", "reason": "r", "requested_at": 1}))
    state = _oslayer_state()
    assert state["available"] and state["guard"] is None
    assert [r["action_id"] for r in state["pending"]] == ["a"]
    assert state["modes"] == ["record", "ask", "block"]


# ── The client, against the guard's own socket server ─────────────


def test_client_asks_the_guard_over_its_socket(g):
    from vaara.oslayer import client

    g._serve()
    try:
        reply = client.request({"op": "status"}, socket_path=g.socket_path)
        assert reply["ok"] and reply["user"] == g.user
        with pytest.raises(client.GuardRefused, match="unknown op"):
            client.request({"op": "nope"}, socket_path=g.socket_path)
        with pytest.raises(client.GuardRefused, match="not a child"):
            client.open_launch({"op": "launch", "pid": 1}, socket_path=g.socket_path)
    finally:
        g._stop.set()
        g._server.close()


def test_client_says_plainly_when_no_guard_runs(tmp_path):
    from vaara.oslayer import client

    with pytest.raises(client.GuardUnavailable, match="sudo vaara os-guard"):
        client.request({"op": "status"}, socket_path=tmp_path / "none.sock")


def test_run_refuses_without_a_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(run.sys, "platform", "linux")
    monkeypatch.setattr(floor, "apparmor_enabled", lambda: True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(run.RunError, match="sudo vaara os-guard"):
        run.run(None, ["sh", "-c", "touch ran"], socket_path=tmp_path / "none.sock")
    assert not (tmp_path / "ran").exists()
