# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vaara Conformance Results has to say what the checkers actually did.

The page exists so a stranger can see every suite, its verdict, and every
outside party who reproduced the vectors. That is worth nothing if the numbers
on it are from a run three releases ago, and a stale conformance claim is worse
than no page: it invites someone to check and find the table wrong.

So the page is generated from the runner's own report and never edited by hand,
and this test regenerates it and fails if what is committed differs. It also
holds the two properties the page would quietly lose first: the honest limits
paragraph, and no off-origin resource loads on a site whose whole claim is that
it works with the network off.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "webpage" / "conformance.html"
RENDERER = ROOT / "scripts" / "render_conformance_page.py"
RUNNER = ROOT / "scripts" / "conformance_runner.py"
REPRODUCTIONS = ROOT / "conformance" / "reproductions.json"


def fresh_report(tmp_path) -> dict:
    """Run the corpus and hand back the report, refusing to guess on failure."""
    report = tmp_path / "report.json"
    run = subprocess.run(
        [sys.executable, str(RUNNER), "--json", str(report)],
        capture_output=True, text=True, timeout=900, cwd=ROOT,
    )
    assert report.exists(), f"runner produced no report:\n{run.stdout}\n{run.stderr}"
    return json.loads(report.read_text(encoding="utf-8"))


def page_count(page: str, key: str) -> int:
    found = re.search(rf"<b>(\d+)</b><span>{key}</span>", page)
    assert found, f"the page carries no {key} count at all"
    return int(found.group(1))


def require_the_same_corpus(report: dict) -> None:
    """Skip where the environment cannot reach every suite the page was built from.

    The committed page is generated with the optional extras installed, so a
    checkout without them skips suites the page counts as passing. Comparing
    the two reports a missing wheel as a stale page, which sends whoever reads
    the failure looking for a drift that is not there. A real edit to the page
    leaves the skip count alone, so this stays sharp against the thing it
    guards.
    """
    committed_skips = page_count(PAGE.read_text(encoding="utf-8"), "skipped")
    if report["totals"]["skipped"] > committed_skips:
        missing = [
            s["suite"] for s in report["suites"]
            if s["status"] == "SKIP"
        ]
        pytest.skip(
            f"environment skips {report['totals']['skipped']} suites against "
            f"{committed_skips} on the page, so it cannot reproduce it. "
            f"Install the extras to run this. Skipped here: {', '.join(missing)}"
        )


def test_page_exists():
    assert PAGE.is_file(), "webpage/conformance.html is missing"


def test_page_matches_a_fresh_run(tmp_path):
    """Regenerate from a live run and compare. Catches a stale committed page."""
    require_the_same_corpus(fresh_report(tmp_path))
    check = subprocess.run(
        [sys.executable, str(RENDERER), str(tmp_path / "report.json"), str(PAGE), "--check"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert check.returncode == 0, check.stderr or check.stdout


def test_totals_on_the_page_are_the_real_totals(tmp_path):
    report = fresh_report(tmp_path)
    require_the_same_corpus(report)
    page = PAGE.read_text(encoding="utf-8")
    for key in ("suites", "passed", "failed", "skipped"):
        assert f"<b>{report['totals'][key]}</b><span>{key}</span>" in page, (
            f"the page does not carry the real {key} count ({report['totals'][key]})"
        )


def test_every_suite_is_listed(tmp_path):
    """No suite is quietly dropped, including the skips.

    This one holds in a reduced environment too: a suite that skips for a
    missing dependency still has to appear on the page, so the comparison is
    over names rather than verdicts.
    """
    report = fresh_report(tmp_path)
    page = PAGE.read_text(encoding="utf-8")
    missing = [s["suite"] for s in report["suites"] if s["suite"] not in page]
    assert not missing, f"suites run but not shown on the page: {missing}"


def test_the_honest_limits_paragraph_survives():
    """The first thing a marketing edit would delete."""
    # Whitespace-normalised: the sentences are line-wrapped in the source, and
    # a test that breaks on rewrapping gets deleted rather than fixed.
    page = re.sub(r"\s+", " ", PAGE.read_text(encoding="utf-8"))
    assert "there is no ratification process behind them" in page
    assert "Recompute is checkable by strangers. Authorship is not." in page


def test_no_off_origin_resource_loads():
    """vaara.io claims to work with the network off. A link is fine, a fetch is not."""
    page = PAGE.read_text(encoding="utf-8")
    loads = re.findall(r'(?:src|srcset)\s*=\s*"([^"]+)"', page)
    offsite = [u for u in loads if u.startswith(("http://", "https://", "//"))]
    assert not offsite, f"off-origin resource loads on the conformance page: {offsite}"


def test_reproductions_file_is_valid_and_every_record_is_public():
    """A reproduction row without a public record is an assertion, not evidence."""
    data = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    for row in data["reproductions"]:
        for field in ("date", "party", "result", "record", "record_held_by", "their_scoping"):
            assert row.get(field), f"reproduction row missing {field}: {row}"
        assert row["record"].startswith("https://"), (
            f"the record for {row['party']} is not a public link"
        )
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]), row["date"]


def test_reproductions_are_not_the_maintainer():
    """The page is for outside reproductions. Vaara grading itself proves nothing."""
    data = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    for row in data["reproductions"]:
        assert "sirkkavaara" not in row["party"].lower(), (
            "the maintainer is not an independent reproduction"
        )


def test_page_is_linked_from_the_site():
    index = (ROOT / "webpage" / "index.html").read_text(encoding="utf-8")
    sitemap = (ROOT / "webpage" / "sitemap.xml").read_text(encoding="utf-8")
    assert "/conformance.html" in index, "nothing on the front page links to it"
    assert "conformance.html" in sitemap, "not in the sitemap, so it will not be indexed"


# ---------------------------------------------------------------------------
# Structured data.
#
# Five named third parties reproducing the corpus is the strongest thing the
# project has, and it only compounds if a machine can tell that is what it is
# looking at. A rendered table is not that. These hold the machine-readable
# copy to the same rule the visible page is held to: it says what the checkers
# and the rows say, and it never says more.
# ---------------------------------------------------------------------------


def structured_data(page: str) -> dict:
    """The one ld+json block on the page, parsed. Fails loudly if it is not."""
    blocks = re.findall(
        r'<script type="application/ld\+json">\n(.*?)\n</script>', page, re.S
    )
    assert len(blocks) == 1, f"expected one ld+json block, found {len(blocks)}"
    return json.loads(blocks[0])


def by_type(graph: list, wanted: str) -> list:
    return [n for n in graph if wanted in (n["@type"] if isinstance(n["@type"], list) else [n["@type"]])]


SYNTHETIC_ROWS = {
    "terms_version": "2026-08-21",
    "reproductions": [
        {
            "id": 1,
            "slug": "someone-else",
            "date": "2026-08-19",
            "terms_version": "2026-08-21",
            "prev": "sha256:" + "0" * 64,
            "party": "Someone Else",
            "affiliation": "Some Lab",
            "suites": ["agent_decision_v0", "record_set_v0"],
            "result": "2 passed, 0 failed, 0 skipped, 4 cases",
            "at_commit": "a" * 40,
            "kind": "reproduction",
            "their_scoping": "Ran the checkers, nothing more.",
            "record": "https://example.org/their-post",
        },
        {
            "id": 2,
            "slug": "no-kind-stated",
            "date": "2026-08-20",
            "terms_version": "2026-08-15",
            "prev": "sha256:" + "1" * 64,
            "party": "Older Row",
            "affiliation": "",
            "suites": ["agent_decision_v0"],
            "result": "1 passed, 0 failed, 0 skipped, 1 case",
            "at_commit": "b" * 40,
            "their_scoping": "Listed before the kind field existed.",
            "record": "https://example.org/older",
        },
    ],
}

FAKE_REPORT = {
    "generated_at": "2026-08-25T09:14:04Z",
    "totals": {"suites": 46, "passed": 45, "failed": 0, "skipped": 1, "cases_passed": 75},
    "suites": [{"suite": "agent_decision_v0", "status": "PASS", "reason": ""}],
}


def test_the_committed_page_carries_parseable_structured_data():
    structured_data(PAGE.read_text(encoding="utf-8"))


def test_the_corpus_is_described_with_the_real_totals():
    graph = structured_data(PAGE.read_text(encoding="utf-8"))["@graph"]
    dataset = by_type(graph, "Dataset")
    assert len(dataset) == 1, "the corpus is not described as a Dataset"
    measured = {v["name"]: v["value"] for v in dataset[0]["variableMeasured"]}
    page = PAGE.read_text(encoding="utf-8")
    for key in ("suites", "passed", "failed", "skipped"):
        assert measured[key] == page_count(page, key), (
            f"structured data says {key}={measured[key]} while the page shows "
            f"{page_count(page, key)}. The two are generated from one report and "
            "must not be able to disagree."
        )


def test_the_corpus_carries_its_doi_and_licence():
    """The citation path a reader needs is in the data, not only in the prose."""
    dataset = by_type(structured_data(PAGE.read_text(encoding="utf-8"))["@graph"], "Dataset")[0]
    assert "10.5281/zenodo.22027975" in dataset["identifier"]
    assert dataset["license"].endswith("agpl-3.0.html")


def test_every_row_gets_a_node_an_anchor_and_a_citable_digest():
    pytest.importorskip("rfc8785")
    module = renderer()
    page = module.render(FAKE_REPORT, SYNTHETIC_ROWS)
    graph = structured_data(page)["@graph"]
    reports = by_type(graph, "Report")
    assert len(reports) == len(SYNTHETIC_ROWS["reproductions"])
    for row, node in zip(SYNTHETIC_ROWS["reproductions"], reports):
        anchor = f'id="row-{row["id"]}"'
        assert anchor in page, f"row {row['id']} has no anchor, so it cannot be cited"
        assert node["@id"].endswith(f"#row-{row['id']}")
        assert node["citation"]["url"] == row["record"]
        assert node["identifier"]["value"] == module.row_digest(row).removeprefix("sha256:")
        assert node["identifier"]["url"].endswith(f"/badge/{row['slug']}.json")


def test_a_row_is_never_typed_as_a_review_a_rating_or_a_certification():
    """The page spends three paragraphs saying it is none of those.

    The structured copy is what a machine reads instead of those paragraphs, so
    a `Review` or an `aggregateRating` here would quietly assert exactly what
    the visible page refuses to.
    """
    pytest.importorskip("rfc8785")
    data = structured_data(renderer().render(FAKE_REPORT, SYNTHETIC_ROWS))
    blob = json.dumps(data)
    for banned in ("Review", "aggregateRating", "ratingValue", "Certification"):
        assert banned not in blob, f"structured data claims {banned!r}"
    # And it says so outright, where a reader of the data alone will find it.
    listed = by_type(data["@graph"], "ItemList")[0]["description"]
    assert "not a certification" in listed
    assert "does not say the party endorses Vaara" in listed


def test_a_party_is_never_given_a_type_the_register_does_not_record():
    """The rows record a party name, not whether it is a person or a company.

    The listed parties are a mix of both plus a pseudonym. Typing them would
    write a guess into the machine-readable copy of a page whose argument is
    that nothing on it is guessed.
    """
    pytest.importorskip("rfc8785")
    graph = structured_data(renderer().render(FAKE_REPORT, SYNTHETIC_ROWS))["@graph"]
    for node in by_type(graph, "Report"):
        assert isinstance(node["author"], str), (
            f"row {node['reportNumber']} types its party as {node['author']}"
        )


def test_an_unstated_kind_is_carried_as_the_weakest_claim():
    """Row 2 has no kind. The rule the page states has to reach the data too."""
    pytest.importorskip("rfc8785")
    graph = structured_data(renderer().render(FAKE_REPORT, SYNTHETIC_ROWS))["@graph"]
    older = [n for n in by_type(graph, "Report") if n["reportNumber"] == "2"][0]
    assert "weakest of the three" in older["disambiguatingDescription"]


def test_the_description_counts_the_rows_rather_than_claiming_them():
    module = renderer()
    assert "5 named independent parties" in module.page_description(
        {"reproductions": [{"id": i} for i in range(5)]}
    )
    assert "1 named independent party" in module.page_description(
        {"reproductions": [{"id": 1}]}
    )
    none_yet = module.page_description({"reproductions": []})
    assert "every independent party" in none_yet and "0" not in none_yet


def test_nothing_can_close_the_script_element_early():
    """Party names and scoping come off a submitted issue form."""
    pytest.importorskip("rfc8785")
    hostile = json.loads(json.dumps(SYNTHETIC_ROWS))
    hostile["reproductions"][0]["party"] = "</script><script>alert(1)</script>"
    page = renderer().render(FAKE_REPORT, hostile)
    block = re.search(
        r'<script type="application/ld\+json">\n(.*?)\n</script>', page, re.S
    ).group(1)
    assert "</script" not in block
    assert json.loads(block), "escaping broke the JSON"


# ---------------------------------------------------------------------------
# Badge geometry.
#
# The badge is the only part of this work that travels: it gets pasted into
# other people's READMEs, next to CI, licence and Scorecard shields that all
# share one geometry. A badge that renders differently from its neighbours
# reads as homemade, whatever it says.
#
# The faults these lock down were all live on the published row #1 badge:
# glyphs squeezed by a forced textLength, a mark sitting low in the plate, and
# a corpus shield that stayed green no matter how many suites failed.
# ---------------------------------------------------------------------------


def renderer():
    """Import the renderer by path, since scripts/ is not an importable package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("render_conformance_page", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# shields.io states the rendered width of its own text in `textLength`, at ten
# times scale. These two came off live badges and are what the table is
# calibrated against, so a drift here means our plates stopped matching every
# other shield in the row.
SHIELDS_WIDTHS = {"license": 37.0, "AGPL-3.0": 51.0}


@pytest.mark.parametrize("text,expected", sorted(SHIELDS_WIDTHS.items()))
def test_text_width_matches_shields_own_metrics(text, expected):
    got = renderer().text_width(text)
    assert abs(got - expected) / expected < 0.02, (
        f"{text!r} measures {got:.1f}px against shields.io's {expected}px. "
        "The plate padding will not match neighbouring badges."
    )


def test_nothing_forces_the_glyph_width():
    """textLength plus lengthAdjust is what squeezed the label out of shape.

    The width table is an estimate, so pinning text to it distorts the type
    whenever the estimate is off. Sizing the plate from the estimate and
    letting the text render naturally costs a little uneven padding instead.
    """
    module = renderer()
    svg = module.corpus_badge_svg({"totals": {"suites": 43, "failed": 0}})
    assert "textLength" not in svg
    assert "lengthAdjust" not in svg


def test_corpus_shield_turns_red_when_a_suite_fails():
    module = renderer()
    clean = module.corpus_badge_svg({"totals": {"suites": 43, "failed": 0}})
    broken = module.corpus_badge_svg({"totals": {"suites": 43, "failed": 3}})
    assert module._OK in clean and module._FAIL not in clean
    assert module._FAIL in broken and module._OK not in broken, (
        "a failing corpus rendered in the same green as a clean one"
    )


def test_the_mark_is_centred_in_the_plate():
    """A 20px badge centres its logo slot on y=10. The mark sat at 5.5 to 15."""
    svg = renderer().corpus_badge_svg({"totals": {"suites": 43, "failed": 0}})
    points = re.search(r'<polygon points="([^"]+)"', svg).group(1)
    ys = [float(p.split(",")[1]) for p in points.split()]
    assert abs((min(ys) + max(ys)) / 2 - 10) < 0.01, f"mark is off centre: {points}"


def test_text_centres_sit_inside_their_own_plates():
    svg = renderer().corpus_badge_svg({"totals": {"suites": 43, "failed": 0}})
    module = renderer()
    svg = module.corpus_badge_svg({"totals": {"suites": 43, "failed": 0}})
    plate = float(re.search(
        rf'<rect width="([0-9.]+)" height="20" fill="{module._LABEL}"', svg).group(1))
    # x is at ten times scale, to match the scale(.1) the text group carries.
    label_cx, message_cx = (int(x) / 10 for x in re.findall(r'<text x="(\d+)" y="140"', svg))
    assert 0 < label_cx < plate, "the label is centred outside its own plate"
    assert message_cx > plate, "the message is centred over the label plate"


def test_rendering_does_not_delete_the_badges_it_does_not_own(tmp_path):
    """The writer and the staleness check have to agree on what is managed.

    ``badge_drift`` exempts ``UNMANAGED_BADGES`` and the corpus shield, but the
    writer swept the whole directory: rendering the page deleted the DOI badge
    the README links to, and ``--check`` then reported the page current, so the
    removal travelled to a broken image on the site with nothing failing.
    """
    module = renderer()
    badge_dir = tmp_path / "badge"
    badge_dir.mkdir()
    for name in (*module.UNMANAGED_BADGES, module.CORPUS_BADGE):
        (badge_dir / name).write_text("<svg/>", encoding="utf-8")
    (badge_dir / "withdrawn-row.svg").write_text("<svg/>", encoding="utf-8")

    module.write_badges({"reproductions": []}, badge_dir=badge_dir)

    survivors = {p.name for p in badge_dir.iterdir()}
    assert module.UNMANAGED_BADGES <= survivors
    assert module.CORPUS_BADGE in survivors
    # A row that came down still takes its badge with it.
    assert "withdrawn-row.svg" not in survivors
