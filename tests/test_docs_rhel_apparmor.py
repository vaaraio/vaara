# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The supported-platforms page says what the Linux OS layer does on RHEL 9.

RHEL confines with SELinux and ships no AppArmor. ``vaara os-guard`` and
``vaara run`` refuse to start without AppArmor rather than run an agent
unconfined, so the page listing RHEL 9 as supported must say that layer
does not run there, and what does.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from vaara.oslayer import floor, run

ROOT = Path(__file__).resolve().parents[1]


def _rhel_section() -> str:
    text = (ROOT / "docs" / "supported-platforms.md").read_text(encoding="utf-8")
    section = text.split("## Red Hat Enterprise Linux", 1)[1].split("\n## ", 1)[0]
    return re.sub(r"\s+", " ", section)


def test_the_rhel_section_names_the_apparmor_limit():
    section = _rhel_section()
    assert "AppArmor" in section
    assert "`vaara run`" in section
    assert "vaara os-guard`" in section
    assert "refuse" in section


def test_the_rhel_section_names_what_still_runs():
    section = _rhel_section()
    for surface in ("MCP proxy", "LLM API proxy", "shell proxy", "hooks", "trail"):
        assert surface in section, surface


def test_the_rhel_section_does_not_claim_a_cgroup_floor():
    # The cgroup the guard can end exists only under the guard, which
    # does not start without AppArmor.
    assert "cgroup" not in _rhel_section()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="vaara run is Linux only")
def test_vaara_run_refuses_without_apparmor(monkeypatch):
    monkeypatch.setattr(floor, "apparmor_enabled", lambda: False)
    with pytest.raises(run.RunError, match="AppArmor is not enabled"):
        run.run(None, ["true"])
