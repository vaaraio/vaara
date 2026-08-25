# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Three site files describe rows that live on another branch.

``llms.txt`` is the file a model reads first and it did not mention the results
register at all. ``sitemap.xml`` carried hand-typed dates that told a crawler
the conformance page was quiet on a week when three rows landed on it. The
register itself was only reachable one row at a time, so anyone wanting the
table had to scrape the page they were being asked not to trust.

``scripts/stamp_site.py`` fills all three from the rows being published. These
tests hold it to the property that makes it worth having: it restates the rows
and never adds to them, and it can be run twice without the file moving.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAMPER = ROOT / "scripts" / "stamp_site.py"
LLMS = ROOT / "webpage" / "llms.txt"
SITEMAP = ROOT / "webpage" / "sitemap.xml"


def stamper():
    spec = importlib.util.spec_from_file_location("stamp_site", STAMPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROWS = {
    "terms_version": "2026-08-21",
    "reproductions": [
        {
            "id": 1,
            "slug": "handle-thing",
            "date": "2026-08-19",
            "party": "handle (thing)",
            "affiliation": "thing",
            "suites": ["agent_decision_v0"],
            "result": "1 passed, 0 failed, 0 skipped, 1 case",
            "at_commit": "c" * 40,
            "their_scoping": "Ran it.",
            "record": "https://example.org/one",
        },
        {
            "id": 2,
            "slug": "named-party",
            "date": "2026-08-24",
            "party": "Named Party",
            "affiliation": "Some Company",
            "suites": ["agent_decision_v0", "record_set_v0"],
            "result": "2 passed, 0 failed, 0 skipped, 2 cases",
            "at_commit": "d" * 40,
            "kind": "independent_implementation",
            "their_scoping": "Wrote our own from the text.",
            "record": "https://example.org/two",
        },
    ],
}


def test_every_listed_party_reaches_the_file_a_model_reads_first():
    block = stamper().register_block(ROWS)
    for row in ROWS["reproductions"]:
        assert row["party"] in block, f"{row['party']} is missing from llms.txt"
        assert row["record"] in block, "a row without its public record is an assertion"
        assert row["at_commit"] in block
        assert f"#row-{row['id']}" in block, "no way to cite the individual row"


def test_the_kind_of_run_comes_from_the_renderer_not_a_second_copy():
    """One sentence about what a run establishes, written in one place."""
    module = stamper()
    block = module.register_block(ROWS)
    kinds = module.renderer()
    assert kinds.KIND_TEXT["independent_implementation"] in block
    # Row 1 states no kind, so the rule that covers it has to appear instead.
    assert kinds.KIND_UNSTATED in block


def test_an_affiliation_already_inside_the_party_name_is_not_printed_twice():
    """Row 1 is "handle (thing)" affiliated to "thing"."""
    block = stamper().register_block(ROWS)
    row = block[block.index("### Row 1"):block.index("### Row 2")]
    assert "Affiliation" not in row, f"the same word twice:\n{row}"
    # The one that genuinely adds a fact still appears.
    assert "Some Company" in block


def test_no_reproduction_is_claimed_when_there_are_none():
    block = stamper().register_block({"reproductions": []})
    assert "No independent reproduction is recorded yet" in block
    assert "Listed reproductions" not in block


def test_the_register_never_gains_a_claim_the_rows_do_not_carry(tmp_path):
    """The published register is the rows, unchanged."""
    module = stamper()
    out = tmp_path / "conformance.json"
    module.publish_register(ROWS, path=out)
    assert json.loads(out.read_text(encoding="utf-8")) == ROWS


def test_stamping_llms_twice_does_not_move_the_file(tmp_path):
    module = stamper()
    scratch = tmp_path / "llms.txt"
    scratch.write_text(
        f"# Head\n\n{module.BEGIN}\nold\n{module.END}\n\n## Tail\n", encoding="utf-8"
    )
    assert module.stamp_llms(ROWS, path=scratch) is True
    first = scratch.read_text(encoding="utf-8")
    assert module.stamp_llms(ROWS, path=scratch) is False
    assert scratch.read_text(encoding="utf-8") == first
    # Hand-written text either side is left alone.
    assert first.startswith("# Head\n")
    assert first.endswith("## Tail\n")


def test_a_missing_marker_refuses_rather_than_guessing_where_to_write(tmp_path):
    module = stamper()
    scratch = tmp_path / "llms.txt"
    scratch.write_text("# Head\n\nno markers here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        module.stamp_llms(ROWS, path=scratch)
    assert scratch.read_text(encoding="utf-8") == "# Head\n\nno markers here\n"


def test_the_committed_llms_file_still_carries_its_markers():
    """Without these the publish step fails the build rather than silently skipping."""
    module = stamper()
    text = LLMS.read_text(encoding="utf-8")
    assert module.BEGIN in text and module.END in text
    assert text.index(module.BEGIN) < text.index(module.END)


def test_the_committed_block_matches_what_the_stamper_would_write():
    """Same rule the conformance page runs under: generated, never hand-edited.

    The published copy is regenerated from the vcr rows at publish time, so a
    stale committed block never reaches the site. It still gets caught here,
    because a generated file that people edit by hand stops being generated.
    """
    module = stamper()
    rows = json.loads(
        (ROOT / "conformance" / "reproductions.json").read_text(encoding="utf-8")
    )
    text = LLMS.read_text(encoding="utf-8")
    committed = text[text.index(module.BEGIN) + len(module.BEGIN) + 1:text.index(module.END)]
    assert committed == module.register_block(rows), (
        "webpage/llms.txt is stale. Regenerate:\n"
        "  python scripts/stamp_site.py"
    )


def test_the_committed_llms_file_points_at_the_register():
    text = LLMS.read_text(encoding="utf-8")
    assert "https://vaara.io/conformance.html" in text
    assert "https://vaara.io/conformance.json" in text
    assert "10.5281/zenodo.22027975" in text


def test_every_sitemap_date_comes_from_git(tmp_path):
    """A hand-typed lastmod was ten days stale on the page that moves most."""
    module = stamper()
    scratch = tmp_path / "sitemap.xml"
    scratch.write_text(SITEMAP.read_text(encoding="utf-8"), encoding="utf-8")
    written = module.stamp_sitemap(path=scratch)
    assert set(written) == set(module.SITEMAP_PAGES), (
        f"no date found for {set(module.SITEMAP_PAGES) - set(written)}"
    )
    for url, date in written.items():
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), f"{url} got {date!r}"
        assert f"<lastmod>{date}</lastmod>" in scratch.read_text(encoding="utf-8")


def test_the_published_page_takes_its_date_from_the_branch_it_publishes_from(tmp_path):
    """conformance.html ships from `vcr`, so its history is not in this checkout."""
    module = stamper()
    scratch = tmp_path / "sitemap.xml"
    scratch.write_text(SITEMAP.read_text(encoding="utf-8"), encoding="utf-8")
    written = module.stamp_sitemap(path=scratch, conformance_date="2026-08-25")
    assert written["https://vaara.io/conformance.html"] == "2026-08-25"
    # The override reaches that one page and no other.
    assert set(written) == set(module.SITEMAP_PAGES)


def test_a_url_the_stamper_does_not_know_keeps_the_date_it_had(tmp_path):
    module = stamper()
    scratch = tmp_path / "sitemap.xml"
    scratch.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        "  <url>\n"
        "    <loc>https://vaara.io/somewhere-else.html</loc>\n"
        "    <lastmod>2020-01-01</lastmod>\n"
        "  </url>\n"
        "</urlset>\n",
        encoding="utf-8",
    )
    assert module.stamp_sitemap(path=scratch) == {}
    assert "<lastmod>2020-01-01</lastmod>" in scratch.read_text(encoding="utf-8")


def test_a_date_that_is_not_a_date_is_refused(tmp_path):
    module = stamper()
    with pytest.raises(SystemExit):
        module.main(["--conformance-date", "yesterday"])
