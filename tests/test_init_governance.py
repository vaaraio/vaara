"""Tests for ``vaara init`` / ``vaara ungovern`` local governance setup."""
from __future__ import annotations

import json
from pathlib import Path

from vaara.integrations import init_governance as ig


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


# --- Claude Code hooks -----------------------------------------------------


def test_write_claude_hooks_creates_all_three_events(tmp_path):
    settings = tmp_path / "settings.json"
    changed = ig.write_claude_hooks(settings, "/usr/bin/vaara")
    assert changed is True
    hooks = _read(settings)["hooks"]
    assert set(hooks) == {"SessionStart", "PreToolUse", "PostToolUse"}
    cmd = hooks["PreToolUse"][0]["hooks"][0]["command"]
    assert cmd == "/usr/bin/vaara hook pre-tool-use"
    assert hooks["PreToolUse"][0]["matcher"] == ig.HOOK_MATCHER
    # SessionStart has no matcher.
    assert "matcher" not in hooks["SessionStart"][0]


def test_write_claude_hooks_is_idempotent(tmp_path):
    settings = tmp_path / "settings.json"
    assert ig.write_claude_hooks(settings, "/usr/bin/vaara") is True
    # Second run makes no change and does not duplicate entries.
    assert ig.write_claude_hooks(settings, "/usr/bin/vaara") is False
    hooks = _read(settings)["hooks"]
    assert len(hooks["PreToolUse"]) == 1
    assert len(hooks["PreToolUse"][0]["hooks"]) == 1


def test_write_claude_hooks_self_heals_truncated_file(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("{}")  # the 27-byte-reset failure mode
    assert ig.write_claude_hooks(settings, "/usr/bin/vaara") is True
    assert "PreToolUse" in _read(settings)["hooks"]


def test_write_claude_hooks_preserves_foreign_hooks_and_settings(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "model": "opus",
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [
                    {"type": "command", "command": "/opt/other-tool run"}
                ]}
            ]
        },
    }))
    ig.write_claude_hooks(settings, "/usr/bin/vaara")
    data = _read(settings)
    assert data["model"] == "opus"
    commands = [
        h["command"]
        for group in data["hooks"]["PreToolUse"]
        for h in group["hooks"]
    ]
    assert "/opt/other-tool run" in commands
    assert "/usr/bin/vaara hook pre-tool-use" in commands


def test_rewriting_with_new_binary_path_does_not_duplicate(tmp_path):
    settings = tmp_path / "settings.json"
    ig.write_claude_hooks(settings, "/old/vaara")
    ig.write_claude_hooks(settings, "/new/vaara")
    hooks = _read(settings)["hooks"]
    commands = [h["command"] for h in hooks["PreToolUse"][0]["hooks"]]
    # Old entry stripped, only the new one remains.
    assert commands == ["/new/vaara hook pre-tool-use"]


def test_remove_claude_hooks_leaves_foreign_hooks(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [
                    {"type": "command", "command": "/opt/other run"}
                ]}
            ]
        },
    }))
    ig.write_claude_hooks(settings, "/usr/bin/vaara")
    assert ig.remove_claude_hooks(settings) is True
    hooks = _read(settings)["hooks"]
    commands = [
        h["command"]
        for group in hooks.get("PreToolUse", [])
        for h in group["hooks"]
    ]
    assert commands == ["/opt/other run"]
    assert "SessionStart" not in hooks


def test_remove_claude_hooks_drops_empty_hooks_block(tmp_path):
    settings = tmp_path / "settings.json"
    ig.write_claude_hooks(settings, "/usr/bin/vaara")
    assert ig.remove_claude_hooks(settings) is True
    assert "hooks" not in _read(settings)


def test_remove_claude_hooks_missing_file(tmp_path):
    assert ig.remove_claude_hooks(tmp_path / "nope.json") is False


# --- hook config -----------------------------------------------------------


def test_write_hook_config_merges_audit_db(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mode": "watch"}))
    ig.write_hook_config(config, tmp_path / "trail" / "audit.db")
    data = _read(config)
    assert data["mode"] == "watch"  # preserved
    assert data["audit_db"].endswith("trail/audit.db")


# --- MCP govern / restore --------------------------------------------------


def _mcp_config(tmp_path, servers) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


def test_govern_mcp_rewrites_naked_server(tmp_path):
    config = _mcp_config(tmp_path, {
        "fs": {"command": "npx", "args": ["-y", "server-fs"]},
    })
    trail = tmp_path / "trail" / "audit.db"
    n = ig.govern_mcp_config(config, "/usr/bin/vaara-mcp-proxy", trail)
    assert n == 1
    fs = _read(config)["mcpServers"]["fs"]
    assert fs["command"] == "/usr/bin/vaara-mcp-proxy"
    assert fs["args"][:4] == ["--upstream", "npx", "--upstream-arg", "-y"]
    assert "--db" in fs["args"] and "--agent-id" in fs["args"]
    assert "mcp:fs" in fs["args"]


def test_govern_mcp_writes_backup_once(tmp_path):
    config = _mcp_config(tmp_path, {"fs": {"command": "npx"}})
    original = config.read_text()
    trail = tmp_path / "audit.db"
    ig.govern_mcp_config(config, "/usr/bin/vaara-mcp-proxy", trail)
    backup = config.with_name(config.name + ".vaara-backup")
    assert backup.exists()
    assert backup.read_text() == original
    # Second govern must not overwrite the pristine backup.
    ig.govern_mcp_config(config, "/usr/bin/vaara-mcp-proxy", trail)
    assert backup.read_text() == original


def test_govern_mcp_skips_already_governed(tmp_path):
    config = _mcp_config(tmp_path, {
        "fs": {"command": "/usr/bin/vaara-mcp-proxy", "args": ["--upstream", "npx"]},
    })
    n = ig.govern_mcp_config(config, "/usr/bin/vaara-mcp-proxy", tmp_path / "a.db")
    assert n == 0


def test_restore_mcp_config_recovers_original(tmp_path):
    config = _mcp_config(tmp_path, {"fs": {"command": "npx"}})
    original = config.read_text()
    ig.govern_mcp_config(config, "/usr/bin/vaara-mcp-proxy", tmp_path / "a.db")
    assert config.read_text() != original
    assert ig.restore_mcp_config(config) is True
    assert config.read_text() == original


def test_restore_mcp_config_no_backup(tmp_path):
    config = _mcp_config(tmp_path, {"fs": {"command": "npx"}})
    assert ig.restore_mcp_config(config) is False


def test_scan_mcp_client_counts(tmp_path):
    config = _mcp_config(tmp_path, {
        "a": {"command": "npx"},
        "b": {"command": "/usr/bin/vaara-mcp-proxy", "args": []},
        "c": {"url": "https://remote"},  # remote, ignored
    })
    status = ig.scan_mcp_client("t", str(config), "/usr/bin/vaara-mcp-proxy")
    assert status.exists and status.governed == 1 and status.ungoverned == 1


def test_scan_mcp_client_missing(tmp_path):
    status = ig.scan_mcp_client("t", str(tmp_path / "nope.json"),
                                "/usr/bin/vaara-mcp-proxy")
    assert status.exists is False


# --- orchestration ---------------------------------------------------------


def test_run_init_and_ungovern_roundtrip(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    config = tmp_path / "config.json"
    trail = tmp_path / "trail" / "audit.db"

    mcp = _mcp_config(tmp_path, {"fs": {"command": "npx"}})
    original_mcp = mcp.read_text()
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [("Test", str(mcp))])
    # Pretend the proxy is installed so MCP rewrite runs.
    monkeypatch.setattr(ig.shutil, "which",
                        lambda name: "/usr/bin/" + name)

    report = ig.run_init(
        trail_db=trail, settings_path=settings, config_path=config,
        vaara_bin="/usr/bin/vaara", proxy_bin="/usr/bin/vaara-mcp-proxy",
    )
    assert report.hooks_changed is True
    assert report.mcp_rewritten == {"Test": 1}
    assert "PreToolUse" in _read(settings)["hooks"]
    assert _read(config)["audit_db"] == str(trail)

    ung = ig.run_ungovern(settings_path=settings,
                          proxy_bin="/usr/bin/vaara-mcp-proxy")
    assert ung.hooks_changed is True
    assert ung.mcp_restored == ["Test"]
    assert "hooks" not in _read(settings)
    assert mcp.read_text() == original_mcp


def test_run_init_warns_when_proxy_missing(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    config = tmp_path / "config.json"
    mcp = _mcp_config(tmp_path, {"fs": {"command": "npx"}})
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [("Test", str(mcp))])
    monkeypatch.setattr(ig.shutil, "which", lambda name: None)

    report = ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=settings,
        config_path=config, vaara_bin="/usr/bin/vaara",
    )
    assert any("vaara-mcp-proxy not found" in w for w in report.warnings)
    assert report.mcp_rewritten == {}
    # MCP config left untouched.
    assert "vaara-mcp-proxy" not in mcp.read_text()


def test_run_init_installs_proxy_service_when_asked(tmp_path, monkeypatch):
    from vaara.integrations import proxy_service as ps
    settings = tmp_path / "settings.json"
    config = tmp_path / "config.json"
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    monkeypatch.setattr(ig.shutil, "which", lambda name: "/usr/bin/" + name)

    calls = []
    report = ig.run_init(
        trail_db=tmp_path / "trail" / "audit.db",
        settings_path=settings, config_path=config,
        vaara_bin="/usr/bin/vaara",
        proxy_service=True, service_home=tmp_path, service_system="linux",
        service_runner=lambda cmd, **kw: calls.append(list(cmd)),
    )
    unit = ps.unit_path("linux", tmp_path)
    assert report.service_path == unit
    assert unit.exists()
    assert ["systemctl", "--user", "enable", "--now", ps.SYSTEMD_UNIT] in calls
    # trail path is threaded into the unit
    assert str(tmp_path / "trail" / "audit.db") in unit.read_text()


def test_run_init_without_flag_installs_no_service(tmp_path, monkeypatch):
    from vaara.integrations import proxy_service as ps
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    monkeypatch.setattr(ig.shutil, "which", lambda name: None)
    report = ig.run_init(
        trail_db=tmp_path / "a.db",
        settings_path=tmp_path / "settings.json",
        config_path=tmp_path / "config.json",
        vaara_bin="/usr/bin/vaara",
        service_home=tmp_path, service_system="linux",
    )
    assert report.service_path is None
    assert ps.unit_path("linux", tmp_path).exists() is False


def test_run_ungovern_removes_proxy_service(tmp_path, monkeypatch):
    from vaara.integrations import proxy_service as ps
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    ps.install_proxy_service(
        vaara_bin="/usr/bin/vaara", home=tmp_path, system="linux",
        runner=lambda cmd, **kw: None,
    )
    report = ig.run_ungovern(
        settings_path=tmp_path / "settings.json",
        service_home=tmp_path, service_system="linux",
        service_runner=lambda cmd, **kw: None,
    )
    assert report.service_removed is True
    assert ps.unit_path("linux", tmp_path).exists() is False


# --- CLI wiring --------------------------------------------------------------


def test_cli_init_proxy_service_flag(monkeypatch, capsys):
    from vaara import cli

    captured = {}

    def fake_run_init(**kwargs):
        captured.update(kwargs)
        report = ig.InitReport()
        report.service_path = Path("/tmp/unit")
        return report

    monkeypatch.setattr(ig, "run_init", fake_run_init)
    assert cli.main(["init", "--proxy-service"]) == 0
    assert captured["proxy_service"] is True
    assert "proxy service installed" in capsys.readouterr().out.lower()


def test_cli_init_default_no_service(monkeypatch, capsys):
    from vaara import cli

    captured = {}
    monkeypatch.setattr(
        ig, "run_init",
        lambda **kw: (captured.update(kw), ig.InitReport())[1])
    assert cli.main(["init"]) == 0
    assert captured["proxy_service"] is False


def test_cli_ungovern_reports_service_removed(monkeypatch, capsys):
    from vaara import cli

    report = ig.InitReport()
    report.service_removed = True
    monkeypatch.setattr(ig, "run_ungovern", lambda **kw: report)
    assert cli.main(["ungovern"]) == 0
    assert "proxy service removed" in capsys.readouterr().out.lower()


# --- Enforce mode threaded through init to the installed service ------------


def test_run_init_proxy_enforce_threads_gate_flags_into_unit(tmp_path, monkeypatch):
    from vaara.integrations import proxy_service as ps
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    monkeypatch.setattr(ig.shutil, "which", lambda name: "/usr/bin/" + name)

    report = ig.run_init(
        trail_db=tmp_path / "trail" / "audit.db",
        settings_path=tmp_path / "settings.json",
        config_path=tmp_path / "config.json",
        vaara_bin="/usr/bin/vaara",
        proxy_service=True,
        proxy_enforce=True,
        proxy_allow=["mcp__github__*"],
        service_home=tmp_path, service_system="linux",
        service_runner=lambda cmd, **kw: None,
    )
    text = ps.unit_path("linux", tmp_path).read_text()
    assert report.service_path is not None
    assert "--enforce" in text
    assert "--allow mcp__github__*" in text
    # no explicit approvals dir: defaults to the app's watch directory
    assert f"--approvals-dir {tmp_path / '.vaara' / 'approvals'}" in text


def test_cli_init_proxy_enforce_flags(monkeypatch):
    from vaara import cli

    captured = {}

    def fake_run_init(**kwargs):
        captured.update(kwargs)
        return ig.InitReport()

    monkeypatch.setattr(ig, "run_init", fake_run_init)
    assert cli.main([
        "init", "--proxy-service", "--proxy-enforce",
        "--proxy-allow", "mcp__github__*",
        "--proxy-allow", "mcp__fs__read*",
    ]) == 0
    assert captured["proxy_enforce"] is True
    assert captured["proxy_allow"] == ["mcp__github__*", "mcp__fs__read*"]


def test_cli_init_proxy_enforce_requires_proxy_service(monkeypatch, capsys):
    from vaara import cli

    monkeypatch.setattr(
        ig, "run_init", lambda **kw: (_ for _ in ()).throw(AssertionError(
            "run_init must not be called when flags are inconsistent")))
    assert cli.main(["init", "--proxy-enforce"]) == 2
    assert "--proxy-service" in capsys.readouterr().err
