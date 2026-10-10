# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every vaara.io page wears the same bar and footer, from scripts/site_shell.py.

The site was rebuilt with a new bar while three pages kept an older one, and a
reader moving from the home page to the results table landed on what looked
like another site. One module now holds the frame. These tests fail when a page
carries any other bar, when the conformance renderer stops using the module, or
when a menu link points at a page that is not there.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEBPAGE = ROOT / "webpage"
sys.path.insert(0, str(ROOT / "scripts"))
import site_shell as shell  # noqa: E402

BAR = re.compile(r'<header class="bar">.*?</header>', re.S)
PAGES = sorted(p for p in WEBPAGE.rglob("*.html") if "badge" not in p.parts)


def _menu_links(bar_html: str) -> list[str]:
    """The menu hrefs, without the language switch, which differs per page."""
    nav = re.sub(r'<span class="lang">.*?</span>', "", bar_html, flags=re.S)
    return re.findall(r'href="([^"]+)"', nav)


@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(WEBPAGE)) for p in PAGES])
def test_every_page_carries_the_shared_bar(page: Path) -> None:
    html = page.read_text(encoding="utf-8")
    bars = BAR.findall(html)
    assert len(bars) == 1, f"{page.name}: {len(bars)} site bars"
    lang = "fi" if "fi" in page.relative_to(WEBPAGE).parts[:1] else "en"
    expected = _menu_links(shell.bar(lang, "/"))
    assert _menu_links(bars[0]) == expected, f"{page.name} carries a different menu"
    assert 'class="topbar"' not in html, f"{page.name} still carries the old bar"
    assert '<footer class="site">' in html, f"{page.name} has no site footer"
    assert shell.THEME_KEY in html, f"{page.name} does not use the shared theme key"


def test_every_menu_link_resolves() -> None:
    for lang in ("en", "fi"):
        for href in _menu_links(shell.bar(lang, "/")):
            if href.startswith("http"):
                continue
            target = WEBPAGE / href.lstrip("/")
            if href.endswith("/"):
                target = target / "index.html"
            assert target.is_file(), f"{lang} menu links {href}, which is not in webpage/"


def test_the_conformance_renderer_uses_the_shared_bar() -> None:
    spec = importlib.util.spec_from_file_location(
        "render_conformance_page", ROOT / "scripts" / "render_conformance_page.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {"generated_at": "2026-10-10T00:00:00Z", "totals": {
        "suites": 0, "passed": 0, "failed": 0, "skipped": 0, "cases_passed": 0}, "suites": []}
    html = module.render(report, {"reproductions": []})
    assert shell.bar("en", "/conformance.html", "test") in html
    assert shell.footer("en") in html
    assert 'class="topbar"' not in html
    # the Finnish page has the same frame in Finnish, and its switch leads back to the English page
    fi = module.render(report, {"reproductions": []}, "fi")
    assert shell.bar("fi", "/conformance.html", "test") in fi
    assert shell.footer("fi") in fi
    assert '<html lang="fi">' in fi
