"""Values copied by hand into another language or manifest stay equal.

The macOS app restates the four protection presets in Swift, and six
manifests restate the version. Nothing compared either copy to its source,
so a threshold change in ``vaara.policy.modes`` would have left the app
describing an operating point the engine no longer uses, with every test
green.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from vaara.policy.modes import available_modes, get_mode

ROOT = Path(__file__).resolve().parents[1]
SWIFT = ROOT / "clients" / "macos" / "Sources" / "VaaraMenuBar" / "Model.swift"


def _swift_presets() -> dict[str, tuple[float, float]]:
    text = SWIFT.read_text()
    pat = re.compile(
        r'Preset\(id:\s*"(\w+)".*?escalate:\s*([0-9.]+),\s*deny:\s*([0-9.]+)\)',
        re.S)
    return {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in pat.finditer(text)}


def test_swift_presets_match_the_mode_table():
    engine = {n: (get_mode(n).escalate, get_mode(n).deny)
              for n in available_modes()}
    assert _swift_presets() == engine


def _pyproject_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"',
                  (ROOT / "pyproject.toml").read_text(), re.M)
    assert m
    return m.group(1)


def test_every_version_manifest_matches_pyproject():
    v = _pyproject_version()
    import vaara
    assert vaara.__version__ == v
    manifests = {
        "clients/ts/package.json": lambda d: d["version"],
        "plugins/claude-code-vaara-governance/.claude-plugin/plugin.json":
            lambda d: d["version"],
        "server.json": lambda d: d["version"],
        "server-vaara-server.json": lambda d: d["version"],
    }
    for rel, get in manifests.items():
        assert get(json.loads((ROOT / rel).read_text())) == v, rel
