"""Tests for `vaara menu`: the interactive menu."""
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


def test_menu_has_expected_items():
    labels = [label for label, _ in menu.ITEMS]
    assert any("Status" in label for label in labels)
    assert any("Shadow" in label for label in labels)
    assert any("Shadow" in label for label in labels)
    assert any("Export" in label for label in labels)
    assert any("Verify" in label for label in labels)
    assert any("Settings" in label for label in labels)
    assert any("Check" in label for label in labels)


def test_menu_renders_and_quits(cfg, monkeypatch, capsys):
    _feed(monkeypatch, ["q"])
    assert menu.run_menu() == 0
    out = capsys.readouterr().out
    assert "vaara" in out
    assert "Status" in out


def test_settings_gate_mode(cfg, monkeypatch, capsys):
    _feed(monkeypatch, ["1", "watch"])
    menu._settings()
    assert json.loads(cfg.read_text())["mode"] == "watch"


def test_settings_protection_preset(cfg, monkeypatch, capsys):
    _feed(monkeypatch, ["2", "strict"])
    menu._settings()
    assert json.loads(cfg.read_text())["protection"] == "strict"


def test_status_without_trail(cfg, monkeypatch, tmp_path, capsys):
    _feed(monkeypatch, [str(tmp_path / "missing.db")])
    menu._status()
    assert "No trail yet" in capsys.readouterr().out


def test_cli_wires_menu(cfg, monkeypatch, capsys):
    from vaara.cli import main

    _feed(monkeypatch, ["q"])
    assert main(["menu"]) == 0
    assert "vaara" in capsys.readouterr().out
