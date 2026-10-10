# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The release script bumps every file tests/test_version.py holds to the version.

test_version.py fails the build when pyproject.toml, src/vaara/__init__.py or
either macOS bundle's Info.plist disagree on the version. The release script
runs that suite right after its bumps, so a file the script does not bump
stops every release at the test step. The plists were tied to the version in
2.2.0 and the script had not heard.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "release_prepare.sh").read_text(encoding="utf-8")
PLISTS = sorted((ROOT / "clients" / "macos" / "Sources").glob("*/Info.plist"))


def test_release_script_bumps_every_version_file():
    assert len(PLISTS) == 2
    for plist in PLISTS:
        rel = plist.relative_to(ROOT).as_posix()
        assert rel in SCRIPT, f"release_prepare.sh never bumps {rel}"
        # Named once where the bump and the read-back loop over the plists,
        # and once more where the explicit paths are staged.
        assert SCRIPT.count(rel) >= 2, f"{rel} is not both bumped and staged"
        assert rel in SCRIPT[SCRIPT.index("git add CHANGELOG.md"):], f"{rel} is not staged"
    assert "CFBundleShortVersionString" in SCRIPT
    for known in ("pyproject.toml", "src/vaara/__init__.py", "clients/ts/package.json",
                  "server.json", "server-vaara-server.json"):
        assert known in SCRIPT


def test_plist_sed_pattern_matches_the_real_files():
    """The sed the script will run has to hit the line as the plists write it."""
    m = re.search(r'sed "\$\{SED_I\[@\]\}" (.*CFBundleShortVersionString.*)', SCRIPT)
    assert m, "no sed for CFBundleShortVersionString in release_prepare.sh"
    for plist in PLISTS:
        text = plist.read_text(encoding="utf-8")
        assert re.search(
            r"<key>CFBundleShortVersionString</key>\n\s*<string>\d+\.\d+\.\d+</string>", text,
        ), f"{plist} does not carry the version on the line after the key"
