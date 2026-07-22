"""Tests for `vaara menu`: the tiered interactive menu."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("cryptography")

from vaara import menu


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(menu, "CONFIG_PATH", path)
    return path


def _feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a: next(it))


def test_levels_gate_items(cfg):
    basic = [label for label, _ in menu.visible_items("basic")]
    pro = [label for label, _ in menu.visible_items("professional")]
    ent = [label for label, _ in menu.visible_items("enterprise")]
    assert any("Article 50 disclosure" in l for l in basic)
    assert not any("Verify" in l for l in basic)
    assert any("Verify" in l for l in pro)
    assert not any("Article 12" in l for l in pro)
    assert any("Article 12" in l for l in ent)
    assert len(basic) < len(pro) < len(ent)


def test_user_level_defaults_and_reads_config(cfg):
    assert menu.user_level() == "basic"
    cfg.write_text(json.dumps({"user_level": "enterprise"}))
    assert menu.user_level() == "enterprise"
    cfg.write_text(json.dumps({"user_level": "bogus"}))
    assert menu.user_level() == "basic"


def test_menu_renders_and_quits(cfg, monkeypatch, capsys):
    _feed(monkeypatch, ["q"])
    assert menu.run_menu() == 0
    out = capsys.readouterr().out
    assert "settings depth: basic" in out
    assert "Record an Article 50 disclosure" in out
    assert "Verify" not in out


def test_settings_change_level_persists(cfg, monkeypatch, capsys):
    _feed(monkeypatch, ["1", "professional"])
    menu._settings()
    assert json.loads(cfg.read_text())["user_level"] == "professional"
    assert "Saved" in capsys.readouterr().out


def test_settings_sets_article50_principal(cfg, monkeypatch, capsys):
    _feed(monkeypatch, ["4", "Example Oy"])
    menu._settings()
    assert json.loads(cfg.read_text())["article50_on_behalf_of"] == "Example Oy"


def test_record_disclosure_via_menu(cfg, monkeypatch, tmp_path, capsys):
    db = tmp_path / "audit.db"
    _feed(monkeypatch, [
        str(db), "I am an AI agent acting for Example Oy.",
        "Example Oy", "first_interaction",
    ])
    menu._record_disclosure()
    out = capsys.readouterr().out
    assert "agent profile" in out

    from vaara.audit.article50 import find_disclosures
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    events = find_disclosures(SQLiteAuditBackend(db).load_trail()._records)
    assert events[0]["on_behalf_of"] == "Example Oy"


def test_status_without_trail(cfg, monkeypatch, tmp_path, capsys):
    _feed(monkeypatch, [str(tmp_path / "missing.db")])
    menu._status()
    assert "No trail yet" in capsys.readouterr().out


def test_cli_wires_menu(cfg, monkeypatch, capsys):
    from vaara.cli import main

    _feed(monkeypatch, ["q"])
    assert main(["menu"]) == 0
    assert "settings depth" in capsys.readouterr().out
