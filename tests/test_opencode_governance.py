"""OpenCode governance: the mapping, the hook runner, the plugin and init.

v1.53.0 said `vaara init` detects and governs OpenCode. It pointed at a
path in the maintainer's checkout, read a config key OpenCode does not use,
and governed nothing. These tests hold each part of the replacement: the
translation of OpenCode's tool calls, the hook's verdict on them, the plugin
turning that verdict into a stopped call, and init putting the plugin where
OpenCode loads it.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vaara.integrations import init_governance as ig
from vaara.integrations import opencode

pytest.importorskip("cryptography")

PLUGIN = Path(opencode.__file__).with_name("opencode_plugin.js")


# --- mapping ----------------------------------------------------------------


def test_bash_maps_to_bash_command():
    [event] = opencode.to_hook_events(
        {"tool": "bash", "args": {"command": "ls", "description": "list"},
         "sessionID": "s1"})
    assert event["tool_name"] == "Bash"
    assert event["tool_input"]["command"] == "ls"
    assert event["session_id"] == "s1"


def test_write_and_edit_carry_claude_code_field_names():
    [write] = opencode.to_hook_events(
        {"tool": "write", "args": {"filePath": "/a/b.py", "content": "x"}})
    assert write["tool_name"] == "Write"
    assert write["tool_input"]["file_path"] == "/a/b.py"
    assert write["tool_input"]["content"] == "x"
    [edit] = opencode.to_hook_events(
        {"tool": "edit", "args": {"filePath": "/a/b.py", "oldString": "o",
                                  "newString": "n"}})
    assert edit["tool_name"] == "Edit"
    assert edit["tool_input"]["file_path"] == "/a/b.py"
    assert edit["tool_input"]["old_string"] == "o"
    assert edit["tool_input"]["new_string"] == "n"
    # OpenCode's own names stay beside the translated ones.
    assert edit["tool_input"]["filePath"] == "/a/b.py"


def test_patch_gives_one_event_per_file():
    patch = textwrap.dedent("""\
        *** Begin Patch
        *** Update File: src/app.py
        @@
        -a
        +b
        *** Add File: /home/u/.ssh/authorized_keys
        +ssh-ed25519 AAAA
        *** End Patch""")
    events = opencode.to_hook_events(
        {"tool": "apply_patch", "args": {"patchText": patch}})
    assert [e["tool_input"]["file_path"] for e in events] == [
        "src/app.py", "/home/u/.ssh/authorized_keys"]
    assert all(e["tool_name"] == "Edit" for e in events)
    assert events[0]["tool_input"]["new_string"] == patch


def test_unknown_tool_goes_to_the_classifier_as_mcp():
    [event] = opencode.to_hook_events(
        {"tool": "github_create_issue", "args": {"title": "t"}})
    assert event["tool_name"] == "mcp__opencode__github_create_issue"
    assert event["tool_input"] == {"title": "t"}


def test_non_object_args_are_kept_not_dropped():
    [event] = opencode.to_hook_events({"tool": "bash", "args": "rm -rf /"})
    assert event["tool_input"]["_raw"] == "rm -rf /"


# --- the hook runner ---------------------------------------------------------


def _run_hook(args, event: dict, home: Path, extra_env: dict | None = None):
    (home / ".vaara").mkdir(parents=True, exist_ok=True)
    (home / ".vaara" / "config.json").write_text("{}")
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "VAARA_PLUGIN_SHADOW": "0",
        "PYTHONPATH": os.pathsep.join(sys.path),
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))",
         *args],
        input=json.dumps(event), capture_output=True, text=True,
        env=env, timeout=120,
    )


def _trail(home: Path) -> Path:
    return home / ".vaara" / "claude-code" / "audit.db"


def test_hook_blocks_opencode_bash_on_deny_rule(tmp_path):
    proc = _run_hook(
        ["hook", "pre-tool-use", "--client", "opencode"],
        {"tool": "bash", "args": {"command": "cat /etc/shadow"}}, tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "BLOCKED Bash" in proc.stderr


def test_hook_allows_benign_opencode_bash_and_records_it_as_opencode(tmp_path):
    proc = _run_hook(
        ["hook", "pre-tool-use", "--client", "opencode"],
        {"tool": "bash", "args": {"command": "ls -la"}, "sessionID": "s"},
        tmp_path)
    assert proc.returncode == 0, proc.stderr
    rows = sqlite3.connect(_trail(tmp_path)).execute(
        "SELECT agent_id, tool_name FROM audit_records").fetchall()
    assert ("opencode", "Bash") in rows


def test_hook_blocks_a_patch_whose_second_file_is_denied(tmp_path):
    patch = ("*** Begin Patch\n*** Update File: notes.md\n+x\n"
             "*** Add File: /home/u/.ssh/authorized_keys\n+k\n*** End Patch")
    proc = _run_hook(
        ["hook", "pre-tool-use", "--client", "opencode"],
        {"tool": "apply_patch", "args": {"patchText": patch}}, tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "ssh_authorized_keys" in proc.stderr


def test_hook_blocks_edits_to_the_opencode_plugin(tmp_path):
    target = str(tmp_path / ".config" / "opencode" / "plugin" / "vaara.js")
    proc = _run_hook(
        ["hook", "pre-tool-use", "--client", "opencode"],
        {"tool": "write", "args": {"filePath": target, "content": ""}},
        tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "harness_config_write" in proc.stderr
    proc = _run_hook(
        ["hook", "pre-tool-use", "--client", "opencode"],
        {"tool": "bash", "args": {"command": f"rm {target}"}}, tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "harness_config_shell_write" in proc.stderr


def test_hook_post_records_the_outcome_against_the_opencode_call(tmp_path):
    call = {"tool": "bash", "args": {"command": "false"}, "sessionID": "s"}
    assert _run_hook(["hook", "pre-tool-use", "--client", "opencode"],
                     call, tmp_path).returncode == 0
    proc = _run_hook(
        ["hook", "post-tool-use", "--client", "opencode"],
        {**call, "output": {"title": "false", "metadata": {"exit": 1}}},
        tmp_path)
    assert proc.returncode == 0, proc.stderr
    events = [r[0] for r in sqlite3.connect(_trail(tmp_path)).execute(
        "SELECT event_type FROM audit_records WHERE agent_id = 'opencode'")]
    assert any("outcome" in e.lower() for e in events), events


# --- the plugin ----------------------------------------------------------------


def _fake_vaara(tmp_path: Path, code: int, message: str = "") -> Path:
    fake = tmp_path / "vaara"
    fake.write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        f"echo '{message}' >&2\nexit {code}\n")
    fake.chmod(0o755)
    return fake


def _call_plugin(tmp_path: Path, vaara_bin: str, home: Path | None = None) -> dict:
    """Load the plugin under Node and run one tool call through its hook."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    installed = tmp_path / "plugin.mjs"
    installed.write_text(PLUGIN.read_text().replace("__VAARA_BIN__", vaara_bin))
    driver = tmp_path / "driver.mjs"
    driver.write_text(textwrap.dedent(f"""\
        import {{ VaaraGovernance }} from {json.dumps(str(installed))};
        const hooks = await VaaraGovernance({{}});
        try {{
          await hooks["tool.execute.before"](
            {{ tool: "bash", sessionID: "s", callID: "c" }},
            {{ args: {{ command: "ls" }} }});
          console.log(JSON.stringify({{ ran: true }}));
        }} catch (err) {{
          console.log(JSON.stringify({{ ran: false, error: String(err.message) }}));
        }}
        """))
    env = {"PATH": os.environ.get("PATH", ""),
           "HOME": str(home or tmp_path)}
    out = subprocess.run([node, str(driver)], capture_output=True, text=True,
                         env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_plugin_lets_the_call_run_on_exit_0(tmp_path):
    assert _call_plugin(tmp_path, str(_fake_vaara(tmp_path, 0))) == {"ran": True}


def test_plugin_stops_the_call_on_exit_2_with_vaaras_reason(tmp_path):
    fake = _fake_vaara(tmp_path, 2, "vaara-governance: BLOCKED Bash (rule=x). no")
    result = _call_plugin(tmp_path, str(fake))
    assert result["ran"] is False
    assert result["error"] == "vaara-governance: BLOCKED Bash (rule=x). no"


def test_plugin_fails_closed_when_the_engine_is_missing(tmp_path):
    result = _call_plugin(tmp_path, str(tmp_path / "no-such-vaara"))
    assert result["ran"] is False
    assert "fail-closed" in result["error"]


def test_plugin_fails_closed_when_the_engine_crashes(tmp_path):
    result = _call_plugin(tmp_path, str(_fake_vaara(tmp_path, 1, "Traceback")))
    assert result["ran"] is False
    assert "fail-closed" in result["error"]


def test_plugin_fail_open_is_the_operators_choice(tmp_path):
    home = tmp_path / "home"
    cfg = home / ".vaara" / "claude-code"
    cfg.mkdir(parents=True)
    (cfg / "config.json").write_text(json.dumps({"fail_open": True}))
    result = _call_plugin(tmp_path, str(tmp_path / "no-such-vaara"), home=home)
    assert result == {"ran": True}


def test_plugin_through_the_real_engine_blocks_a_denied_call(tmp_path):
    """The installed plugin, the real `vaara hook`, a real deny rule."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    home = tmp_path / "home"
    (home / ".vaara").mkdir(parents=True)
    (home / ".vaara" / "config.json").write_text("{}")
    shim = tmp_path / "vaara"
    shim.write_text(
        f"#!/bin/sh\nexec {sys.executable} -c "
        "'import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))' \"$@\"\n")
    shim.chmod(0o755)
    installed = tmp_path / "plugin.mjs"
    installed.write_text(PLUGIN.read_text().replace("__VAARA_BIN__", str(shim)))
    driver = tmp_path / "driver.mjs"
    driver.write_text(textwrap.dedent(f"""\
        import {{ VaaraGovernance }} from {json.dumps(str(installed))};
        const hooks = await VaaraGovernance({{}});
        const out = [];
        for (const command of ["ls -la", "cat /etc/shadow"]) {{
          try {{
            await hooks["tool.execute.before"](
              {{ tool: "bash", sessionID: "s", callID: command }},
              {{ args: {{ command }} }});
            out.push("ran");
          }} catch (err) {{
            out.push(String(err.message));
          }}
        }}
        console.log(JSON.stringify(out));
        """))
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home),
           "VAARA_PLUGIN_SHADOW": "0",
           "PYTHONPATH": os.pathsep.join(sys.path)}
    proc = subprocess.run([node, str(driver)], capture_output=True, text=True,
                          env=env, timeout=240)
    assert proc.returncode == 0, proc.stderr
    allowed, denied = json.loads(proc.stdout.strip().splitlines()[-1])
    assert allowed == "ran"
    assert denied.startswith("vaara-governance: BLOCKED Bash")


# --- init and ungovern --------------------------------------------------------


def test_init_installs_the_plugin_pinned_to_the_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    oc_dir = tmp_path / "opencode"
    oc_dir.mkdir()
    report = ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        govern_mcp=False, opencode_dir=oc_dir)
    plugin = oc_dir / "plugin" / "vaara.js"
    assert report.opencode_plugin == plugin
    assert report.opencode_changed is True
    text = plugin.read_text()
    assert '"/opt/bin/vaara"' in text
    assert "__VAARA_BIN__" not in text
    again = ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        govern_mcp=False, opencode_dir=oc_dir)
    assert again.opencode_changed is False


def test_init_skips_opencode_when_it_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    monkeypatch.setattr(opencode.shutil, "which", lambda name: None)
    report = ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        govern_mcp=False, opencode_dir=tmp_path / "absent")
    assert report.opencode_plugin is None
    assert not (tmp_path / "absent").exists()


def test_ungovern_removes_the_plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    oc_dir = tmp_path / "opencode"
    oc_dir.mkdir()
    opencode.install_plugin("/opt/bin/vaara", oc_dir)
    report = ig.run_ungovern(
        settings_path=tmp_path / "s.json", service_home=tmp_path,
        service_system="linux", service_runner=lambda cmd, **kw: None,
        opencode_dir=oc_dir)
    assert report.opencode_removed is True
    assert not (oc_dir / "plugin" / "vaara.js").exists()


def test_opencode_is_not_rewritten_as_an_mcp_client():
    # The plugin gates OpenCode's MCP calls; a proxy rewrite as well would
    # record each call twice. The old entry pointed at a maintainer path.
    assert all(name != "OpenCode" for name, _ in ig.KNOWN_MCP_CLIENTS)
