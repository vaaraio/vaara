"""``vaara init`` names every agent it finds and says whether it is governed.

It used to print what it wrote and then "Vaara is governing", whatever it
found. An agent with no Vaara adapter (Windsurf) got no line at
all, an untrusted Codex hook was a note above the verdict, and an MCP config
Vaara could not read was skipped without a word.
"""
from __future__ import annotations

import json
from pathlib import Path

from vaara.integrations import init_governance as ig


def _none(_name):
    return None


def _report(**kw) -> ig.InitReport:
    return ig.InitReport(**kw)


def _client(name, path, **kw) -> ig.MCPClientStatus:
    return ig.MCPClientStatus(name=name, path=Path(path), exists=True, **kw)


def _rows(report, which=_none):
    return {r.name: (r.state, r.detail) for r in ig.coverage(report, which=which)}


def test_hook_adapters_are_governed():
    rows = _rows(_report(cursor_hooks=Path("h"), opencode_plugin=Path("p"),
                         codex_hooks=Path("c"), codex_trust="trusted",
                         gemini_settings=Path("g"), gemini_status="active"),
                 which=lambda b: "/bin/claude" if b == "claude" else None)
    assert {n: s for n, (s, _) in rows.items()} == {
        "Claude Code": "governed", "Cursor": "governed",
        "OpenCode": "governed", "Codex": "governed", "Gemini CLI": "governed"}


def test_an_untrusted_codex_hook_is_not_governed():
    state, detail = _rows(_report(codex_hooks=Path("c"), codex_trust="untrusted"))["Codex"]
    assert state == "NOT governed"
    assert "trust" in detail


def test_an_agent_with_no_adapter_is_named_as_not_governed():
    rows = _rows(_report(), which=lambda b: "/usr/bin/windsurf" if b == "windsurf" else None)
    state, detail = rows["Windsurf"]
    assert state == "NOT governed"
    assert "no Vaara adapter" in detail


def test_windsurf_with_routed_mcp_is_mcp_only():
    report = _report(clients=[_client("Windsurf", "/w/mcp_config.json", ungoverned=2)],
                     mcp_rewritten={"Windsurf": 2})
    state, detail = _rows(report)["Windsurf"]
    assert state == "MCP only"
    assert "2 MCP server(s)" in detail and "own tool calls run unchecked" in detail


def test_an_unreadable_mcp_config_is_said():
    report = _report(clients=[_client("Claude Desktop", "/d/cfg.json", readable=False)])
    state, detail = _rows(report)["Claude Desktop"]
    assert state == "NOT governed"
    assert "could not read" in detail


def test_scan_marks_a_config_it_cannot_read(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mcpServers": [{"command": "node"}]}))
    assert ig.scan_mcp_client("X", str(cfg), "vaara-mcp-proxy").readable is False
    cfg.write_text("{ not json")
    assert ig.scan_mcp_client("X", str(cfg), "vaara-mcp-proxy").readable is False
    cfg.write_text(json.dumps({"mcpServers": {}}))
    assert ig.scan_mcp_client("X", str(cfg), "vaara-mcp-proxy").readable is True


def test_cli_init_counts_what_it_governs(tmp_path, monkeypatch, capsys):
    from vaara import cli

    monkeypatch.setattr(ig, "coverage", lambda report, **_: [
        ig.Coverage("Claude Code", "governed", "every tool call, through its hooks"),
        ig.Coverage("Gemini CLI", "NOT governed", "no Vaara adapter yet"),
    ])
    monkeypatch.setattr(ig, "run_init", lambda **_: ig.InitReport(
        hooks_path=tmp_path / "settings.json", trail_db=tmp_path / "audit.db"))
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert "Gemini CLI     NOT governed: no Vaara adapter yet" in out
    assert "Vaara is governing 1 of 2 agents found." in out
    assert "Not fully governed: Gemini CLI" in out
    assert "Vaara is governing. " not in out
