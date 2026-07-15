"""Tests for the proxy-as-a-managed-service layer (P2 of one-install governance).

Everything runs against a temp HOME and a recording fake runner; no real
launchctl/systemctl is ever invoked.
"""
import plistlib

from vaara.integrations.proxy_service import (
    LAUNCHD_LABEL,
    SYSTEMD_UNIT,
    install_proxy_service,
    render_launchd_plist,
    render_systemd_unit,
    uninstall_proxy_service,
    unit_path,
)


class RecordingRunner:
    """Stands in for subprocess.run; records commands, always succeeds."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if self.fail:
            raise OSError("no such binary")

        class R:
            returncode = 0
        return R()


# ---------------------------------------------------------------------------
# Unit rendering


def test_launchd_plist_runs_vaara_proxy_and_survives_logout():
    text = render_launchd_plist(
        vaara_bin="/opt/venv/bin/vaara",
        listen="127.0.0.1:8788",
        upstream="http://127.0.0.1:11434",
        trail_db="/home/op/.vaara/trail/audit.db",
        log_dir="/home/op/.vaara/logs",
    )
    plist = plistlib.loads(text.encode())
    assert plist["Label"] == LAUNCHD_LABEL
    assert plist["ProgramArguments"] == [
        "/opt/venv/bin/vaara", "proxy",
        "--listen", "127.0.0.1:8788",
        "--upstream", "http://127.0.0.1:11434",
        "--trail", "/home/op/.vaara/trail/audit.db",
    ]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert plist["StandardOutPath"].startswith("/home/op/.vaara/logs/")
    assert plist["StandardErrorPath"].startswith("/home/op/.vaara/logs/")


def test_systemd_unit_runs_vaara_proxy_and_restarts():
    text = render_systemd_unit(
        vaara_bin="/opt/venv/bin/vaara",
        listen="127.0.0.1:8788",
        upstream="http://127.0.0.1:11434",
        trail_db="/home/op/.vaara/trail/audit.db",
    )
    assert ("ExecStart=/opt/venv/bin/vaara proxy "
            "--listen 127.0.0.1:8788 "
            "--upstream http://127.0.0.1:11434 "
            "--trail /home/op/.vaara/trail/audit.db") in text
    assert "Restart=on-failure" in text
    assert "WantedBy=default.target" in text


def test_unit_path_per_platform(tmp_path):
    assert unit_path("darwin", tmp_path) == (
        tmp_path / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist")
    assert unit_path("linux", tmp_path) == (
        tmp_path / ".config" / "systemd" / "user" / SYSTEMD_UNIT)


def test_unit_path_unsupported_platform(tmp_path):
    assert unit_path("win32", tmp_path) is None


# ---------------------------------------------------------------------------
# Install


def test_install_darwin_writes_plist_and_bootstraps(tmp_path):
    runner = RecordingRunner()
    report = install_proxy_service(
        vaara_bin="/opt/venv/bin/vaara",
        home=tmp_path,
        system="darwin",
        runner=runner,
    )
    path = unit_path("darwin", tmp_path)
    assert report.installed
    assert report.path == path
    assert path.exists()
    plist = plistlib.loads(path.read_bytes())
    assert plist["Label"] == LAUNCHD_LABEL
    # unload-if-present then load: idempotent activation
    assert ["launchctl", "unload", str(path)] in runner.calls
    assert ["launchctl", "load", "-w", str(path)] in runner.calls
    assert report.warnings == []


def test_install_linux_writes_unit_and_enables(tmp_path):
    runner = RecordingRunner()
    report = install_proxy_service(
        vaara_bin="/opt/venv/bin/vaara",
        home=tmp_path,
        system="linux",
        runner=runner,
    )
    path = unit_path("linux", tmp_path)
    assert report.installed
    assert path.exists()
    assert "ExecStart=/opt/venv/bin/vaara proxy" in path.read_text()
    assert ["systemctl", "--user", "daemon-reload"] in runner.calls
    assert ["systemctl", "--user", "enable", "--now",
            SYSTEMD_UNIT] in runner.calls


def test_install_activation_failure_keeps_unit_and_warns(tmp_path):
    """No systemctl/launchctl (container, minimal host): unit file stays on
    disk so a later boot picks it up, and the report carries a warning."""
    report = install_proxy_service(
        vaara_bin="/opt/venv/bin/vaara",
        home=tmp_path,
        system="linux",
        runner=RecordingRunner(fail=True),
    )
    assert report.installed
    assert unit_path("linux", tmp_path).exists()
    assert report.warnings and "systemctl" in report.warnings[0]


def test_install_unsupported_platform_warns_and_writes_nothing(tmp_path):
    runner = RecordingRunner()
    report = install_proxy_service(
        vaara_bin="/opt/venv/bin/vaara",
        home=tmp_path,
        system="win32",
        runner=runner,
    )
    assert not report.installed
    assert report.warnings
    assert runner.calls == []


def test_install_is_idempotent(tmp_path):
    runner = RecordingRunner()
    kwargs = dict(vaara_bin="/opt/venv/bin/vaara", home=tmp_path,
                  system="linux", runner=runner)
    first = install_proxy_service(**kwargs)
    second = install_proxy_service(**kwargs)
    assert first.installed and second.installed
    assert unit_path("linux", tmp_path).read_text().count("[Service]") == 1


# ---------------------------------------------------------------------------
# Uninstall


def test_uninstall_darwin_unloads_and_removes(tmp_path):
    runner = RecordingRunner()
    install_proxy_service(vaara_bin="/x/vaara", home=tmp_path,
                          system="darwin", runner=runner)
    path = unit_path("darwin", tmp_path)
    removed = uninstall_proxy_service(home=tmp_path, system="darwin",
                                      runner=runner)
    assert removed
    assert not path.exists()
    assert runner.calls[-1] == ["launchctl", "unload", str(path)]


def test_uninstall_linux_disables_and_removes(tmp_path):
    runner = RecordingRunner()
    install_proxy_service(vaara_bin="/x/vaara", home=tmp_path,
                          system="linux", runner=runner)
    removed = uninstall_proxy_service(home=tmp_path, system="linux",
                                      runner=runner)
    assert removed
    assert not unit_path("linux", tmp_path).exists()
    assert ["systemctl", "--user", "disable", "--now",
            SYSTEMD_UNIT] in runner.calls


def test_uninstall_when_absent_is_a_noop(tmp_path):
    runner = RecordingRunner()
    assert uninstall_proxy_service(home=tmp_path, system="linux",
                                   runner=runner) is False
    assert runner.calls == []


def test_uninstall_survives_missing_service_manager(tmp_path):
    ok_runner = RecordingRunner()
    install_proxy_service(vaara_bin="/x/vaara", home=tmp_path,
                          system="linux", runner=ok_runner)
    removed = uninstall_proxy_service(home=tmp_path, system="linux",
                                      runner=RecordingRunner(fail=True))
    assert removed
    assert not unit_path("linux", tmp_path).exists()


# ---------------------------------------------------------------------------
# Enforce mode in the installed service (the wall, not just the window)


def test_launchd_plist_enforce_carries_gate_flags():
    text = render_launchd_plist(
        vaara_bin="/opt/venv/bin/vaara",
        listen="127.0.0.1:8788",
        upstream="http://127.0.0.1:11434",
        trail_db="/home/op/.vaara/trail/audit.db",
        log_dir="/home/op/.vaara/logs",
        enforce=True,
        allow=["mcp__github__*", "mcp__fs__read*"],
        approvals_dir="/home/op/.vaara/approvals",
    )
    argv = plistlib.loads(text.encode())["ProgramArguments"]
    assert "--enforce" in argv
    assert argv[argv.index("--approvals-dir") + 1] == "/home/op/.vaara/approvals"
    allow_values = [argv[i + 1] for i, a in enumerate(argv) if a == "--allow"]
    assert allow_values == ["mcp__github__*", "mcp__fs__read*"]


def test_systemd_unit_enforce_carries_gate_flags():
    text = render_systemd_unit(
        vaara_bin="/opt/venv/bin/vaara",
        listen="127.0.0.1:8788",
        upstream="http://127.0.0.1:11434",
        trail_db="/home/op/.vaara/trail/audit.db",
        enforce=True,
        allow=["mcp__github__*"],
        approvals_dir="/home/op/.vaara/approvals",
    )
    exec_line = next(
        line for line in text.splitlines() if line.startswith("ExecStart=")
    )
    assert "--enforce" in exec_line
    assert "--allow mcp__github__*" in exec_line
    assert "--approvals-dir /home/op/.vaara/approvals" in exec_line


def test_observe_mode_units_carry_no_gate_flags():
    text = render_launchd_plist(
        vaara_bin="/opt/venv/bin/vaara",
        listen="127.0.0.1:8788",
        upstream="http://127.0.0.1:11434",
        trail_db="/t.db",
        log_dir="/logs",
    )
    argv = plistlib.loads(text.encode())["ProgramArguments"]
    for flag in ("--enforce", "--allow", "--approvals-dir"):
        assert flag not in argv


def test_install_enforce_defaults_approvals_dir_to_the_approval_surface(tmp_path):
    """--enforce with no allow list must still satisfy the proxy's own
    refuse-to-start guard: the service defaults the approvals dir to
    ~/.vaara/approvals, the directory approval watchers poll."""
    runner = RecordingRunner()
    report = install_proxy_service(
        vaara_bin="/opt/venv/bin/vaara",
        home=tmp_path,
        system="darwin",
        runner=runner,
        enforce=True,
    )
    assert report.installed
    argv = plistlib.loads(report.path.read_bytes())["ProgramArguments"]
    assert "--enforce" in argv
    assert argv[argv.index("--approvals-dir") + 1] == str(
        tmp_path / ".vaara" / "approvals")


def test_install_enforce_respects_explicit_approvals_dir(tmp_path):
    runner = RecordingRunner()
    report = install_proxy_service(
        vaara_bin="/opt/venv/bin/vaara",
        home=tmp_path,
        system="linux",
        runner=runner,
        enforce=True,
        allow=["mcp__github__*"],
        approvals_dir="/custom/approvals",
    )
    text = report.path.read_text()
    assert "--approvals-dir /custom/approvals" in text
    assert "--allow mcp__github__*" in text
