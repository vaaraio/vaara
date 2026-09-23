"""`vaara init --shadow` and `--auto` put the hook in watch mode.

`--shadow` is documented as "govern in watch-only (shadow) mode: record
without blocking", and `--auto` as generating "a default shadow-mode policy".
The flag reached only the MCP proxy rewrite, and `--auto` wrote its shadow
setting to a policy file and a discovery config that nothing reads. The hook
reads `~/.vaara/claude-code/config.json` and no other file, so after either
command the hooks went on blocking.
"""
from __future__ import annotations

import json

from vaara.integrations import claude_code_hooks as hooks
from vaara.integrations.init_governance import run_init, write_hook_config


def _cfg(path):
    return json.loads(path.read_text())


def test_shadow_sets_watch(tmp_path):
    cfg = tmp_path / "config.json"
    write_hook_config(cfg, tmp_path / "audit.db", shadow=True)
    assert _cfg(cfg)["mode"] == "watch"
    assert hooks.shadow_mode(_cfg(cfg))


def test_shadow_overrides_an_earlier_protect(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"mode": "protect"}))
    write_hook_config(cfg, tmp_path / "audit.db", shadow=True)
    assert _cfg(cfg)["mode"] == "watch"


def test_plain_init_leaves_mode_alone(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"mode": "watch", "protection": "strict"}))
    write_hook_config(cfg, tmp_path / "audit.db")
    assert _cfg(cfg)["mode"] == "watch"
    assert _cfg(cfg)["protection"] == "strict"


def test_auto_starts_in_watch_at_its_preset_when_unset(tmp_path):
    cfg = tmp_path / "config.json"
    write_hook_config(cfg, tmp_path / "audit.db", auto=True, auto_preset="eco")
    assert _cfg(cfg)["mode"] == "watch"
    assert _cfg(cfg)["protection"] == "eco"


def test_auto_keeps_an_operator_choice(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"mode": "protect", "protection": "strict"}))
    write_hook_config(cfg, tmp_path / "audit.db", auto=True, auto_preset="eco")
    assert _cfg(cfg)["mode"] == "protect"
    assert _cfg(cfg)["protection"] == "strict"


def test_run_init_shadow_reaches_the_file_the_hook_reads(tmp_path):
    settings = tmp_path / "settings.json"
    cfg = tmp_path / "config.json"
    run_init(trail_db=tmp_path / "audit.db", settings_path=settings,
             config_path=cfg, vaara_bin="vaara", shadow=True, govern_mcp=False)
    # The predicate the hook itself uses to decide it must not block.
    assert hooks.shadow_mode(_cfg(cfg))


def test_silent_first_run_setup_leaves_the_hooks_blocking(tmp_path):
    """The first use of any command runs setup with shadow and auto. It must
    not switch the hooks to watch: that would stop them blocking unasked."""
    cfg = tmp_path / "config.json"
    run_init(trail_db=tmp_path / "audit.db", settings_path=tmp_path / "s.json",
             config_path=cfg, vaara_bin="vaara", shadow=True, auto=False,
             govern_mcp=False, set_hook_mode=False)
    assert not hooks.shadow_mode(_cfg(cfg))


def test_the_cli_first_run_passes_set_hook_mode_false(monkeypatch):
    from vaara import cli
    from vaara.integrations import init_governance as ig

    seen = {}
    monkeypatch.setattr(ig, "run_init", lambda **kw: seen.update(kw))
    cli._run_first_time_setup()
    assert seen.get("set_hook_mode") is False
