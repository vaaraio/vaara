"""The pre-tool-use gate fails closed when the hook itself cannot answer.

Codex, Gemini CLI and Claude Code let a tool call run when its hook
crashes, is missing, or outlives the host's timeout. The gate is the
command ``vaara init`` writes in front of ``vaara hook pre-tool-use``: it
passes Vaara's own verdicts (exit 0 and 2) through and turns every other
ending into exit 2. These tests run the written command under ``sh`` with
stand-in engines for each way a hook can fail to answer.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from vaara.integrations import _hook_gate

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX sh gate")


def _engine(tmp: Path, body: str) -> Path:
    exe = tmp / "bin" / "vaara"
    exe.parent.mkdir(exist_ok=True)
    exe.write_text("#!/bin/sh\n" + body + "\n")
    exe.chmod(0o755)
    return exe


def _run(command: str, tmp: Path, stdin: str = "{}", **env: str):
    home = tmp / "home"
    home.mkdir(exist_ok=True)
    full = {"PATH": os.environ["PATH"], "HOME": str(home), **env}
    start = time.monotonic()
    proc = subprocess.run(["sh", "-c", command], input=stdin, env=full,
                          capture_output=True, text=True, timeout=60)
    return proc, time.monotonic() - start


def test_verdicts_pass_through_with_stdin_and_stdout(tmp_path):
    exe = _engine(tmp_path, 'cat; echo \'{"decision":"allow"}\'; exit 0')
    proc, _ = _run(_hook_gate.pre_command(str(exe), "codex"), tmp_path, stdin="EVENT")
    assert proc.returncode == 0
    assert proc.stdout == 'EVENT{"decision":"allow"}\n'

    exe = _engine(tmp_path, 'echo "denied: rm_rf_root" >&2; exit 2')
    proc, _ = _run(_hook_gate.pre_command(str(exe), "codex"), tmp_path)
    assert proc.returncode == 2
    assert "rm_rf_root" in proc.stderr


def test_the_command_keeps_the_marker_and_client(tmp_path):
    cmd = _hook_gate.pre_command("/opt/v/bin/vaara", "gemini")
    assert "/opt/v/bin/vaara hook pre-tool-use --client gemini" in cmd
    assert "vaara hook " in cmd
    assert "--client" not in _hook_gate.pre_command("/opt/v/bin/vaara")


@pytest.mark.parametrize("body", [
    "exit 1",                                    # a traceback exits 1
    'kill -9 $$',                                 # killed by a signal
    "exit 3",                                     # anything but 0 and 2
])
def test_an_engine_that_crashes_blocks(tmp_path, body):
    exe = _engine(tmp_path, body)
    proc, _ = _run(_hook_gate.pre_command(str(exe), "codex"), tmp_path)
    assert proc.returncode == 2
    assert "fail-closed" in proc.stderr


def test_a_missing_engine_blocks(tmp_path):
    proc, _ = _run(_hook_gate.pre_command(str(tmp_path / "gone" / "vaara"), "codex"),
                   tmp_path)
    assert proc.returncode == 2
    assert "fail-closed" in proc.stderr


def test_an_engine_that_hangs_blocks_at_the_deadline(tmp_path):
    exe = _engine(tmp_path, "sleep 30")
    proc, took = _run(_hook_gate.pre_command(str(exe), "codex"), tmp_path,
                      VAARA_HOOK_DEADLINE="2")
    assert proc.returncode == 2
    assert "fail-closed" in proc.stderr and "2 s" in proc.stderr
    assert took < 10


def test_the_deadline_cannot_be_raised_past_the_default(tmp_path):
    cmd = _hook_gate.pre_command("/x/vaara", "codex")
    assert f"-gt {_hook_gate.DEADLINE}" in cmd
    assert _hook_gate.DEADLINE < _hook_gate.HOST_TIMEOUT


def test_fail_open_is_the_documented_opt_out(tmp_path):
    exe = _engine(tmp_path, "exit 1")
    cmd = _hook_gate.pre_command(str(exe), "codex")
    proc, _ = _run(cmd, tmp_path, VAARA_PLUGIN_FAIL_OPEN="1")
    assert proc.returncode == 1

    cfg = tmp_path / "home" / ".vaara" / "claude-code" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"fail_open": True}))
    proc, _ = _run(cmd, tmp_path)
    assert proc.returncode == 1

    cfg.write_text(json.dumps({"fail_open": False}))
    proc, _ = _run(cmd, tmp_path)
    assert proc.returncode == 2


def test_the_watchdog_does_not_outlive_a_quick_verdict(tmp_path):
    if not Path("/proc/self/cmdline").exists():
        pytest.skip("reads /proc")
    exe = _engine(tmp_path, "exit 0")
    _run(_hook_gate.pre_command(str(exe), "codex"), tmp_path)
    time.sleep(1.5)
    left = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            left.append((p / "cmdline").read_bytes())
        except OSError:
            continue
    assert not any(str(exe).encode() in c for c in left)


# ---------------------------------------------------------------------------
# The Claude Code plugin's run.sh carries the same gate in shell, since the
# plugin runs before any Python is known to work.

RUN_SH = (Path(__file__).parent.parent / "plugins" / "claude-code-vaara-governance"
          / "hooks" / "run.sh")
_UTILS = ("sh", "cat", "grep", "mktemp", "rm", "sleep", "dirname", "kill")


def _plugin_path(tmp: Path, engine: str | None) -> str:
    """A PATH with the shell utilities, and a stand-in vaara if given."""
    import shutil

    bindir = tmp / "pbin"
    bindir.mkdir(exist_ok=True)
    for tool in _UTILS:
        real = shutil.which(tool)
        if real and not (bindir / tool).exists():
            (bindir / tool).symlink_to(real)
    exe = bindir / "vaara"
    exe.unlink(missing_ok=True)
    if engine is not None:
        exe.write_text('#!/bin/sh\n[ "$2" = --help ] && exit 0\n' + engine + "\n")
        exe.chmod(0o755)
    return str(bindir)


def _plugin(tmp: Path, engine: str | None, **env: str):
    return _run(f"sh {RUN_SH} pre-tool-use", tmp,
                PATH=_plugin_path(tmp, engine), **env)


def test_plugin_passes_verdicts_through(tmp_path):
    proc, _ = _plugin(tmp_path, "cat; exit 0")
    assert (proc.returncode, proc.stdout) == (0, "{}")
    proc, _ = _plugin(tmp_path, 'echo "rule=rm_rf_root" >&2; exit 2')
    assert proc.returncode == 2 and "rm_rf_root" in proc.stderr


def test_plugin_blocks_a_crashed_engine(tmp_path):
    proc, _ = _plugin(tmp_path, "exit 1")
    assert proc.returncode == 2 and "fail-closed" in proc.stderr


def test_plugin_blocks_a_hung_engine_at_the_deadline(tmp_path):
    proc, took = _plugin(tmp_path, "sleep 30", VAARA_HOOK_DEADLINE="2")
    assert proc.returncode == 2 and "within 2 s" in proc.stderr
    assert took < 10


def test_plugin_blocks_when_no_engine_can_run(tmp_path):
    proc, _ = _plugin(tmp_path, None)
    assert proc.returncode == 2
    assert "every tool call is blocked" in proc.stderr


def test_plugin_fail_open_still_opts_out(tmp_path):
    proc, _ = _plugin(tmp_path, "exit 1", VAARA_PLUGIN_FAIL_OPEN="1")
    assert proc.returncode == 1
    proc, _ = _plugin(tmp_path, None, VAARA_PLUGIN_FAIL_OPEN="1")
    assert proc.returncode == 0 and "unchecked" in proc.stderr


def test_plugin_and_installer_share_deadline_and_timeout():
    manifest = json.loads((RUN_SH.parent / "hooks.json").read_text())
    pre = manifest["hooks"]["PreToolUse"][0]["hooks"][0]
    assert pre["timeout"] == _hook_gate.HOST_TIMEOUT
    assert f"deadline={_hook_gate.DEADLINE}\n" in RUN_SH.read_text()


def test_an_approval_wait_ends_inside_the_deadline(monkeypatch):
    from vaara.integrations.claude_code_hooks import approvals_timeout

    monkeypatch.delenv("VAARA_PLUGIN_APPROVALS_TIMEOUT", raising=False)
    monkeypatch.delenv("VAARA_HOOK_DEADLINE", raising=False)
    assert approvals_timeout({}) == 60.0
    assert approvals_timeout({"approvals_timeout": 600}) == _hook_gate.DEADLINE - 5
    monkeypatch.setenv("VAARA_HOOK_DEADLINE", "20")
    assert approvals_timeout({}) == 15.0
