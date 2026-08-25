#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stamp the three site files that go stale on their own.

Conformance rows land on the unprotected ``vcr`` branch, never on main, and the
Pages job overlays them at publish time. Three files on main describe those rows
and none of them were told when a row arrived:

``llms.txt``
    The file a model reads first, and it did not mention the results register at
    all. Five named third parties reproducing the corpus is the strongest thing
    the project has, and it was absent from the one file written for machines.
    The section is generated here rather than typed, for the same reason the
    results page is: a hand-maintained list of parties is wrong the day after
    the next row lands, and a wrong list of other people's names is worse than
    no list.

``sitemap.xml``
    Carried ``lastmod`` dates written by hand. The conformance entry said
    2026-08-15 with ``changefreq weekly`` while three rows had landed since,
    which tells a crawler the page is quiet when it is the one that moves most.
    Dates come from git here, or from the vcr branch for the page that is
    published from it.

``conformance.json``
    Did not exist. The register was only reachable one row at a time under
    ``/badge/<slug>.json``, so anyone wanting the whole table had to scrape the
    page they were being asked not to trust. This publishes the same bytes the
    desk wrote, as one file.

Nothing here invents a value. Every date comes from git or from a row, and every
row comes from the register.

Usage:
    python scripts/stamp_site.py                       # from the checkout
    python scripts/stamp_site.py --rows /tmp/vcr/vcr/reproductions.json \\
        --conformance-date 2026-08-25                  # at publish time
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEBPAGE = REPO / "webpage"
DEFAULT_ROWS = REPO / "conformance" / "reproductions.json"
LLMS = WEBPAGE / "llms.txt"
SITEMAP = WEBPAGE / "sitemap.xml"
REGISTER = WEBPAGE / "conformance.json"

#: The generated block in llms.txt sits between these. Everything outside them
#: is written by hand and is never touched.
BEGIN = "<!-- generated: conformance register, by scripts/stamp_site.py -->"
END = "<!-- end generated -->"

#: Which page each sitemap entry is, so a date can be found for it. A URL that
#: is not here keeps whatever date it already had rather than being guessed at.
SITEMAP_PAGES = {
    "https://vaara.io/": "index.html",
    "https://vaara.io/verify.html": "verify.html",
    "https://vaara.io/conformance.html": "conformance.html",
}


def renderer():
    """The page renderer, loaded by path since ``scripts/`` is not a package.

    Borrowed for one thing: the prose for a row's kind. That wording is a
    commitment about what a run does and does not establish, it is already
    written once in the renderer, and a second copy of it here would be a third
    place for the same sentence to drift.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "render_conformance_page", REPO / "scripts" / "render_conformance_page.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_date(path: str) -> str | None:
    """The commit date of a file, YYYY-MM-DD, or None if git cannot say."""
    run = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", path],
        cwd=REPO, capture_output=True, text=True,
    )
    date = run.stdout.strip()
    return date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else None


def register_block(rows: dict) -> str:
    """The conformance section of llms.txt, from the rows themselves.

    Every party listed here asked to be listed and knew the entry is permanent,
    so naming them is the same publication the page already is. What is not
    restated is anything the rows do not say: no party is described as an
    authority, a certifier or a customer, because none of them is.
    """
    entries = rows.get("reproductions", [])
    lines = [
        "## Conformance results",
        "",
        "Vaara publishes a results register: every conformance suite, its verdict,",
        "and every party other than the maintainer who ran the checkers and reported",
        "the outcome in public. Each suite ships an independent checker that imports",
        "no Vaara code and recomputes its verdicts from the bytes of its own case",
        "files, so a stranger can disagree with an expected result and show their work.",
        "",
        "- [Conformance results](https://vaara.io/conformance.html): the register as a page,"
        " generated from the runner's own report",
        "- [conformance.json](https://vaara.io/conformance.json): the same register as one"
        " machine-readable file",
        "- [Row request form](https://github.com/vaaraio/vaara/issues/new?template=conformance-row.yml):"
        " open to anyone who ran the checkers, free, no approval step",
        "- Concept DOI for the corpus: [10.5281/zenodo.22027975](https://doi.org/10.5281/zenodo.22027975)",
        "",
        "What a row is, stated exactly. It records that a named party ran the checkers",
        "at a commit on a date and said so somewhere public. It is not a certification,",
        "it does not say Vaara is compliant with anything, and it does not say the party",
        "endorses Vaara. The vectors are Vaara's own and no ratification body stands",
        "behind them, which is a real limit on any neutrality claim. Rows are permanent",
        "and chained, and each published head is recorded in a public transparency log",
        "the maintainer does not operate.",
        "",
        "Every row names what kind of run it was, because what a run establishes is a",
        "property of who wrote the verifier and who wrote the vectors rather than of how",
        "well the run went. A reproduction is the author's checkers over the author's",
        "vectors. An independent implementation from the text, run against the author's",
        "vectors, establishes something about the text. An independent implementation run",
        "against independently constructed vectors establishes something about both. A row",
        "that does not name its kind reads as the first, the weakest of the three.",
        "",
    ]
    if not entries:
        lines += [
            "No independent reproduction is recorded yet. A row appears when someone",
            "other than the maintainer runs the checkers, reports the result somewhere",
            "public, and asks to be listed.",
            "",
        ]
        return "\n".join(lines)

    was = "run is" if len(entries) == 1 else "runs are"
    lines += [
        f"Listed reproductions, {len(entries)} to date. Every claim below is the",
        f"party's own, quoted as they scoped it, and the {was} recorded somewhere",
        "public by someone other than Vaara:",
        "",
    ]
    kind_text = renderer().kind_text
    for row in entries:
        who = row.get("party", "")
        lines.append(f"### Row {row['id']}: {who}")
        # An affiliation that is already inside the party name is not a second
        # fact. Row 1 is "babyblueviper1 (invinoveritas)" affiliated to
        # "invinoveritas", which printed the same word twice.
        affiliation = row.get("affiliation", "")
        if affiliation and affiliation.lower() not in who.lower():
            lines.append(f"- Affiliation, as the party stated it: {affiliation}")
        lines += [
            f"- Ran on {row.get('date', '')}, at commit {row.get('at_commit', '')}",
            f"- Kind of run: {kind_text(row)}",
            f"- Result they reported: {row.get('result', '')}",
            f"- Their own public record: {row.get('record', '')}",
            f"- This row: https://vaara.io/conformance.html#row-{row['id']}",
            f"- The row as bytes: https://vaara.io/badge/{row['slug']}.json",
            "",
        ]
    return "\n".join(lines)


def stamp_llms(rows: dict, path: Path = LLMS) -> bool:
    """Replace the generated block in llms.txt. Returns whether anything moved."""
    text = path.read_text(encoding="utf-8")
    block = f"{BEGIN}\n{register_block(rows)}{END}\n"
    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"{path} carries no generated block. Add the {BEGIN} and {END} markers "
            "back before running this, rather than letting it guess where to write."
        )
    start = text.index(BEGIN)
    stop = text.index(END) + len(END) + 1
    updated = text[:start] + block + text[stop:]
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def stamp_sitemap(
    path: Path = SITEMAP, conformance_date: str | None = None
) -> dict[str, str]:
    """Set each ``lastmod`` from git, or from the vcr branch where one applies.

    Returns the dates it wrote, so a caller can print them rather than trust
    that this did something.
    """
    text = path.read_text(encoding="utf-8")
    written: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        url = match.group("url")
        page = SITEMAP_PAGES.get(url)
        if page is None:
            return match.group(0)
        if page == "conformance.html" and conformance_date:
            date = conformance_date
        else:
            date = git_date(f"webpage/{page}")
        if not date:
            return match.group(0)
        written[url] = date
        return (
            f"{match.group('head')}<lastmod>{date}</lastmod>{match.group('tail')}"
        )

    updated = re.sub(
        r"(?P<head><loc>(?P<url>[^<]+)</loc>\s*)"
        r"<lastmod>[^<]*</lastmod>"
        r"(?P<tail>)",
        replace,
        text,
    )
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return written


def publish_register(rows: dict, path: Path = REGISTER) -> int:
    """Serve the whole register as one file, formatted to be read.

    Deliberately not the canonical bytes. Per-row canonical bytes are what the
    digests are taken over and they are served unchanged at
    ``/badge/<slug>.json``; this is the table, for someone who wants all of it
    at once. Indented so a person opening it in a browser can read it, and it
    carries no digest of its own so nothing here can be mistaken for the thing
    a badge commits to.
    """
    blob = json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    path.write_text(blob, encoding="utf-8")
    return len(rows.get("reproductions", []))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rows", type=Path, default=DEFAULT_ROWS,
        help="the register to publish from. At publish time this is the copy "
             "taken off the vcr branch, not the empty one on main.",
    )
    ap.add_argument(
        "--conformance-date",
        help="date the published conformance page last changed, YYYY-MM-DD. "
             "The page is published from the vcr branch, so its own history is "
             "there rather than in this checkout.",
    )
    args = ap.parse_args(argv)

    # The workflow always passes the flag and leaves it empty when there is no
    # vcr branch, so empty means "fall back to this checkout". Anything else
    # that is not a date is a caller bug, and writing it into the sitemap would
    # publish a broken date rather than a stale one.
    if args.conformance_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.conformance_date):
        raise SystemExit(f"--conformance-date is not YYYY-MM-DD: {args.conformance_date!r}")

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    listed = publish_register(rows)
    moved = stamp_llms(rows)
    dates = stamp_sitemap(conformance_date=args.conformance_date)

    print(f"conformance.json: {listed} rows")
    print(f"llms.txt: {'rewritten' if moved else 'already current'}")
    for url, date in dates.items():
        print(f"sitemap: {url} -> {date}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
