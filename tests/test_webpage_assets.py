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

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEBPAGE = ROOT / "webpage"
PAGES = ("index.html", "verify.html")
#: Every page that ships structured data. A crawler or a model arriving at any
#: of these reads the block instead of guessing from the layout, so a block
#: that does not parse is worse than none: it is silently dropped.
PAGES_WITH_STRUCTURED_DATA = ("index.html", "verify.html", "conformance.html")

#: The nodes the whole site shares. Declared with the same @id on every page so
#: the graphs join into one Vaara instead of describing three unrelated ones.
SHARED_IDS = {
    "https://vaara.io/#henri",
    "https://vaara.io/#organization",
    "https://vaara.io/#website",
}


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


def blocks(page: str) -> list[str]:
    return re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        (WEBPAGE / page).read_text(encoding="utf-8"),
        re.S,
    )


@pytest.mark.parametrize("page", PAGES_WITH_STRUCTURED_DATA)
def test_the_structured_data_on_every_page_parses(page):
    found = blocks(page)
    assert found, f"{page} ships no structured data, so it is read as layout"
    for blob in found:
        json.loads(blob)


@pytest.mark.parametrize("page", PAGES_WITH_STRUCTURED_DATA)
def test_every_page_declares_the_same_vaara(page):
    """Three pages describing three unrelated Vaaras is worse than one page.

    Nodes are joined by @id, so the identifiers have to be identical strings
    across the site. They are easy to get almost right and nothing complains.
    """
    ids = set()
    for blob in blocks(page):
        for node in json.loads(blob).get("@graph", []):
            if "@id" in node:
                ids.add(node["@id"])
    missing = SHARED_IDS - ids
    assert not missing, f"{page} does not carry {sorted(missing)}"


@pytest.mark.parametrize("page", PAGES_WITH_STRUCTURED_DATA)
def test_every_page_says_where_it_lives(page):
    """A canonical URL, so the same page under two hosts is not two pages."""
    text = (WEBPAGE / page).read_text(encoding="utf-8")
    assert 'rel="canonical"' in text, f"{page} has no canonical URL"
    assert 'name="description"' in text, f"{page} has no description to quote"
    assert 'property="og:title"' in text, f"{page} has no link preview"
