# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara cage drivers``: which cages are ready here, and what is missing."""

from __future__ import annotations

import json
import stat
import sys

import pytest

from vaara import cage
from vaara.cage import cli as cage_cli
from vaara.cage import readiness


def _tool(tmp_path, name, version_line):
    if sys.platform == "win32":
        pytest.skip("the fake tool is an executable script, which Windows does not run")
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\necho '{version_line}'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_every_driver_states_where_it_runs():
    assert set(readiness.PLATFORMS) == set(cage.DRIVERS)
    for platforms in readiness.PLATFORMS.values():
        assert platforms and set(platforms) <= {"linux", "darwin", "win32", "any"}


def test_more_than_linux_is_covered():
    darwin = [d for d, p in readiness.PLATFORMS.items() if "darwin" in p or "any" in p]
    win = [d for d, p in readiness.PLATFORMS.items() if "win32" in p or "any" in p]
    assert {"openshell", "sandbox-runtime", "codex", "nono", "microsandbox",
            "apple-container"} <= set(darwin)
    assert "microsandbox" in win


def test_a_cage_for_another_os_is_not_ready_and_says_why():
    r = readiness.check("apple-container", platform="linux")
    assert not r.on_this_os and not r.ready
    assert r.missing == ["runs on macOS, not here"]


def test_a_missing_tool_names_its_install_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("NONO_BIN", str(tmp_path / "absent"))
    r = readiness.check("nono", platform="linux")
    assert r.on_this_os and not r.ready
    assert "NONO_BIN" in r.missing[0] and "install nono" in r.missing[0]


def test_a_found_tool_is_ready_with_its_version(monkeypatch, tmp_path):
    monkeypatch.setenv("MSB_BIN", _tool(tmp_path, "msb", "msb 0.3.1"))
    r = readiness.check("microsandbox", platform="linux")
    assert r.ready and r.missing == [] and r.version == "microsandbox 0.3.1"


def test_e2b_needs_its_endpoint_and_key(monkeypatch):
    monkeypatch.delenv("E2B_API_URL", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    r = readiness.check("e2b", platform="darwin")
    assert r.on_this_os and r.missing == ["E2B_API_URL is not set", "E2B_API_KEY is not set"]


def test_the_vaara_cage_needs_apparmor(monkeypatch):
    from vaara.oslayer import floor
    monkeypatch.setattr(floor, "apparmor_enabled", lambda: False)
    r = readiness.check("vaara-cage", platform="linux")
    assert not r.ready and "AppArmor" in r.missing[0]
    assert readiness.check("vaara-cage", platform="darwin").missing == [
        "runs on Linux, not here"]


def test_firecracker_needs_kvm(monkeypatch, tmp_path):
    monkeypatch.setenv("FIRECRACKER_BIN", _tool(tmp_path, "firecracker", "Firecracker v1.10.1"))
    real_exists = readiness.os.path.exists
    monkeypatch.setattr(readiness.os.path, "exists",
                        lambda p: False if p == "/dev/kvm" else real_exists(p))
    r = readiness.check("firecracker", platform="linux")
    assert r.missing == ["/dev/kvm is not there; Firecracker needs KVM"]


def test_drivers_command_lists_all_twelve(capsys):
    assert cage_cli.main(["drivers", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["driver"] for r in rows] == list(cage.DRIVERS)
    assert len(rows) == 12
    for r in rows:
        assert r["ready"] or r["missing"]


def test_drivers_table_says_where_each_runs(capsys):
    assert cage_cli.main(["drivers"]) == 0
    out = capsys.readouterr().out
    assert "apple-container" in out and "runs on macOS" in out
    assert "runs on Linux, macOS, Windows" in out


def test_the_menu_has_the_cage_entry():
    from vaara import menu
    labels = [label for label, _ in menu.ITEMS]
    assert any(label.startswith("Cages") for label in labels)
