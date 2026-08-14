# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The site serves its own copy of the wordmark, so the copies must match.

verify.html states that it works with the network off. It cannot make that
claim while fetching its own logo from a third party, so webpage/ carries the
marks and loads them from the same origin. Two copies of a file drift, and a
site quietly showing last year's brand is the kind of thing nobody notices for
months, so this pins them together.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBPAGE = ROOT / "webpage"
PAGES = ("index.html", "verify.html")


def test_the_site_carries_the_wordmark_and_it_matches_docs():
    for name in ("vaara-wordmark-light.png", "vaara-wordmark-dark.png"):
        served, canonical = WEBPAGE / name, ROOT / "docs" / name
        assert served.exists(), f"webpage/{name} is missing"
        assert served.read_bytes() == canonical.read_bytes(), (
            f"webpage/{name} has drifted from docs/{name}"
        )


def test_no_page_asset_is_fetched_from_another_origin():
    """A page that promises to work offline cannot depend on someone else."""
    offenders = {}
    for page in PAGES:
        external = re.findall(r'src="(https?://[^"]+)"', (WEBPAGE / page).read_text())
        if external:
            offenders[page] = sorted(set(external))
    assert offenders == {}, f"off-origin assets: {offenders}"
