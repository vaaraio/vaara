"""Tests for `vaara proxy`: the model-endpoint front door, observe mode.

Zero required flags: defaults bind 127.0.0.1:8788, front a local ollama,
and record every tool call the model requests into ~/.vaara/proxy/audit.db
(overridable). uvicorn is faked so no socket is bound.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest

for _mod in ("cryptography", "httpx", "fastapi"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip("proxy deps not installed", allow_module_level=True)

from vaara.cli import main


@pytest.fixture
def fake_uvicorn(monkeypatch):
    calls: list[dict] = []
    fake = types.ModuleType("uvicorn")

    def run(app, **kwargs):
        calls.append({"app": app, **kwargs})

    fake.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    return calls


def test_proxy_defaults_build_app_and_trail(tmp_path, fake_uvicorn):
    trail_db = tmp_path / "proxy" / "audit.db"
    rc = main(["proxy", "--trail", str(trail_db)])
    assert rc == 0
    assert len(fake_uvicorn) == 1
    assert fake_uvicorn[0]["host"] == "127.0.0.1"
    assert fake_uvicorn[0]["port"] == 8788
    assert fake_uvicorn[0]["app"] is not None
    assert trail_db.is_file()  # trail is armed before the server starts


def test_proxy_listen_and_upstream_flags(tmp_path, fake_uvicorn):
    rc = main([
        "proxy", "--trail", str(tmp_path / "t.db"),
        "--listen", "0.0.0.0:9900", "--upstream", "http://127.0.0.1:8080",
    ])
    assert rc == 0
    assert fake_uvicorn[0]["host"] == "0.0.0.0"
    assert fake_uvicorn[0]["port"] == 9900


def test_proxy_bad_listen_errors(tmp_path, fake_uvicorn, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["proxy", "--trail", str(tmp_path / "t.db"),
              "--listen", "nonsense"])
    assert exc.value.code == 2
    assert fake_uvicorn == []


def test_proxy_enforce_flag_gates(tmp_path, fake_uvicorn):
    rc = main([
        "proxy", "--trail", str(tmp_path / "t.db"), "--enforce",
        "--approvals-dir", str(tmp_path / "approvals"),
    ])
    assert rc == 0
    assert len(fake_uvicorn) == 1


def test_proxy_default_is_observe_mode(tmp_path, fake_uvicorn):
    rc = main(["proxy", "--trail", str(tmp_path / "t.db")])
    assert rc == 0


# --- enforce allow-list UX (the 2026-07-14 vanished-tools incident) ---------


def test_enforce_without_allow_or_approvals_refuses_to_start(capsys):
    from vaara import cli
    rc = cli.main(["proxy", "--enforce"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--allow" in err and "--approvals-dir" in err


def test_enforce_with_allow_patterns_starts(monkeypatch):
    from vaara import cli

    captured = {}

    def fake_build_app(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "vaara.integrations._infer_proxy_app.build_app", fake_build_app)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    rc = cli.main(["proxy", "--enforce", "--allow", "read_*",
                   "--allow", "mcp__github__*", "--trail", "/tmp/t-allow.db"])
    assert rc == 0
    assert captured["allow_patterns"] == ["read_*", "mcp__github__*"]


def test_observe_mode_needs_no_allow(monkeypatch):
    from vaara import cli
    monkeypatch.setattr(
        "vaara.integrations._infer_proxy_app.build_app",
        lambda **k: object())
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    assert cli.main(["proxy", "--trail", "/tmp/t-obs.db"]) == 0
