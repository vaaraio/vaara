# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A malformed media query is dropped silently, so nothing here can be silent.

verify.html shipped with three media queries written `@media (max-width:640px{`,
missing the closing parenthesis. CSS treats an invalid media prelude as never
matching, so the browser discards the whole block without a console error and
without a visible failure anywhere a server can see it. The page served
byte-identical bytes, every asset returned 200, the CSS braces balanced and the
HTML parsed clean. Three of the four responsive blocks were dead and the only
symptom was that the layout collapsed on a narrow window.

That is the failure mode worth a test: valid enough to serve, invalid enough to
ignore, and invisible to every check that does not parse the prelude itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEBPAGE = Path(__file__).resolve().parent.parent / "webpage"
PAGES = sorted(p.name for p in WEBPAGE.glob("*.html"))


def _style_blocks(html: str) -> list[str]:
    return re.findall(r"<style[^>]*>(.*?)</style>", html, re.S)


@pytest.mark.parametrize("page", PAGES)
def test_every_media_query_prelude_is_balanced(page: str) -> None:
    """`@media (max-width:640px{` parses as a query that can never match."""
    html = (WEBPAGE / page).read_text(encoding="utf-8")
    broken = []
    for match in re.finditer(r"@media([^{]*)\{", html):
        condition = match.group(1)
        if condition.count("(") != condition.count(")"):
            line = html[: match.start()].count("\n") + 1
            broken.append(f"{page}:{line} @media{condition.strip()}")
    assert not broken, (
        "media query prelude has unbalanced parentheses, so the browser drops "
        "the whole block and the page loses those rules with no error: "
        + "; ".join(broken)
    )


@pytest.mark.parametrize("page", PAGES)
def test_inline_css_braces_balance(page: str) -> None:
    """An unclosed rule swallows every rule after it."""
    html = (WEBPAGE / page).read_text(encoding="utf-8")
    for index, css in enumerate(_style_blocks(html)):
        opens, closes = css.count("{"), css.count("}")
        assert opens == closes, (
            f"{page} style block {index}: {opens} '{{' against {closes} '}}'. "
            "An unbalanced block silently discards the rules after it."
        )


@pytest.mark.parametrize("page", PAGES)
def test_no_custom_property_is_used_without_being_defined(page: str) -> None:
    """A var() with no definition and no fallback resolves to nothing."""
    html = (WEBPAGE / page).read_text(encoding="utf-8")
    for index, css in enumerate(_style_blocks(html)):
        defined = set(re.findall(r"(--[A-Za-z0-9_-]+)\s*:", css))
        # A var() carrying a fallback survives a missing definition, so only
        # the bare form is a defect.
        used = set(re.findall(r"var\(\s*(--[A-Za-z0-9_-]+)\s*\)", css))
        missing = sorted(used - defined)
        assert not missing, (
            f"{page} style block {index} reads {missing} with no definition and "
            "no fallback, which resolves to an empty value."
        )
