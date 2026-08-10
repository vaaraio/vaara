# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A command line printed in the docs has to be one the CLI accepts.

COMPLIANCE.md gave the Article 12(2) retention recipe as::

    vaara trail purge --db PATH --retention-days N

which exits 2 without touching the database: the parser requires a
tenant selector (``--tenant ID`` or ``--all-tenants``) that the doc never
mentioned. A deployer following the retention section would have
concluded the purge was broken.

Two rules, both narrow enough to stay quiet on prose:

* every ``vaara ...`` line inside a fenced shell block has to parse;
* every inline ``vaara ...`` span that prints two or more flags has to
  parse, because at that point the doc is showing a command to run, not
  naming one.

Prose that names a command (``vaara compliance report``) or illustrates a
single flag (``vaara keygen --dev``) is deliberately left alone.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

import pytest

from vaara.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent

#: Only files that are actually in the checkout. CAPABILITIES.md, for one,
#: is untracked in some working trees and absent on CI.
DOCS = [
    path
    for path in sorted(ROOT.glob("docs/*.md"))
    + [ROOT / name for name in ("README.md", "SPEC.md", "CAPABILITIES.md", "PRIVACY.md")]
    if path.exists()
]

_FENCE = re.compile(r"```(?:bash|sh|console|shell)\n(.*?)```", re.S)
_INLINE = re.compile(r"`(vaara [^`]+)`")

# Placeholders the docs use for values a deployer substitutes.
_PLACEHOLDERS = {
    "PATH": "/tmp/x", "DB": "/tmp/x.db", "N": "30", "ID": "t1",
    "POLICY_PATH": "/tmp/p.yaml", "CASES_PATH": "/tmp/c.yaml",
    "SECONDS": "60", "URL": "https://example.invalid",
}


def _runnable(line: str) -> bool:
    """Lines the docs present as something to paste, not something to name."""
    if not line.startswith("vaara ") or line.startswith("vaara-"):
        return False
    # Shell plumbing, help output and elisions are not recipes to replay.
    return not any(token in line for token in ("--help", "|", ">", "...", "<", "$("))


def _substitute(parts: list[str]) -> list[str]:
    return [_PLACEHOLDERS.get(p, p) for p in parts]


def _parse(line: str) -> str | None:
    """None if the CLI accepts the line, else the parser's complaint."""
    try:
        parts = _substitute(shlex.split(line))
    except ValueError as exc:  # unbalanced quotes in the doc itself
        return f"unparseable shell syntax: {exc}"
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            build_parser().parse_args(parts[1:])
    except SystemExit:
        complaint = captured.getvalue().strip().splitlines()
        return complaint[-1] if complaint else "rejected by the parser"
    return None


def _fenced_commands(doc: Path) -> list[str]:
    found = []
    for block in _FENCE.findall(doc.read_text(encoding="utf-8")):
        for raw in block.replace("\\\n", " ").splitlines():
            line = re.sub(r"\s+#.*$", "", raw.strip().lstrip("$ ").strip())
            if _runnable(line):
                found.append(" ".join(line.split()))
    return found


def _inline_recipes(doc: Path) -> list[str]:
    """Inline spans that print two or more flags."""
    found = []
    for span in _INLINE.findall(doc.read_text(encoding="utf-8")):
        line = " ".join(span.split())
        if _runnable(line) and len(re.findall(r"(?<![\w-])--[a-z][\w-]*", line)) >= 2:
            found.append(line)
    return found


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_fenced_command_blocks_parse(doc):
    failures = [(line, _parse(line)) for line in _fenced_commands(doc)]
    failures = [f for f in failures if f[1]]
    assert not failures, "\n".join(
        f"{doc.name}: `{line}` -> {why}" for line, why in failures
    )


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_inline_multi_flag_recipes_parse(doc):
    failures = [(line, _parse(line)) for line in _inline_recipes(doc)]
    failures = [f for f in failures if f[1]]
    assert not failures, "\n".join(
        f"{doc.name}: `{line}` -> {why}" for line, why in failures
    )


def test_the_scan_actually_finds_commands():
    """A regex that silently matches nothing would pass every test above."""
    fenced = sum(len(_fenced_commands(d)) for d in DOCS if d.exists())
    inline = sum(len(_inline_recipes(d)) for d in DOCS if d.exists())
    assert fenced >= 20, f"only {fenced} fenced command lines found"
    assert inline >= 1, f"only {inline} inline recipes found"


def test_the_retention_recipe_names_the_tenant_selector():
    """Regression: the purge recipe that could not run as printed."""
    text = (ROOT / "docs" / "COMPLIANCE.md").read_text(encoding="utf-8")
    recipe = re.search(r"`(vaara trail purge [^`]+)`", text)
    assert recipe, "COMPLIANCE.md no longer gives a retention recipe"
    assert _parse(" ".join(recipe.group(1).split())) is None
