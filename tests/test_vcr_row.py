# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The desk decides rows with no human in the loop, so the checks are the test.

Every path here is one a stranger can drive from a public issue form. What
matters is that a row cannot reach the page without consent, without a commit
that exists, or without a link a reader can open, and that a number once issued
is never handed to anybody else.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

# Chaining a row computes a digest over its JCS-canonical bytes, so the whole
# desk needs rfc8785 from the attestation extra. A checkout without it cannot
# exercise any of this, and the extras job in CI covers it properly.
pytest.importorskip("rfc8785", reason="the desk needs the attestation extra")

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("vcr_row", ROOT / "scripts" / "vcr_row.py")
vcr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vcr)

render_spec = importlib.util.spec_from_file_location(
    "render_conformance_page", ROOT / "scripts" / "render_conformance_page.py"
)
render = importlib.util.module_from_spec(render_spec)
render_spec.loader.exec_module(render)

HEAD = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.strip()

FORM = """### Name to publish

{party}

### Affiliation

EMILIA Protocol

### Which suites you ran

{suites}

### Commit you ran at

{commit}

### Result

9 of 9 and 8 of 8, including the tampered-proof and forked-root negatives

### Your own scoping

Validates the pinned independent vector lanes.

### Link to where you reported it

{record}

### Consent

- [{c1}] I read the section above and I agree to be listed.
- [{c2}] I understand the row is permanent.
- [{c3}] The link I gave is my own public record of this run.
"""


def form(**over) -> str:
    values = {
        "party": "Iman Schrock",
        "suites": "transparency_consistency_v0, evidence_bundle_v0",
        "commit": HEAD,
        "record": "https://mailarchive.ietf.org/arch/msg/scitt/fE4RYzpmR440PVrZaknTvVKjUoI/",
        "c1": "X",
        "c2": "X",
        "c3": "X",
    }
    values.update(over)
    return FORM.format(**values)


def row_from(body: str, author: str = "someone") -> dict:
    return vcr.validate(vcr.parse_issue_form(body), author)


def test_a_complete_form_becomes_a_row():
    row = row_from(form())
    assert row["party"] == "Iman Schrock"
    assert row["suites"] == ["transparency_consistency_v0", "evidence_bundle_v0"]
    assert row["at_commit"] == HEAD
    assert row["submitted_by"] == "someone"


@pytest.mark.parametrize(
    "c1,c2,c3",
    [(" ", "X", "X"), ("X", " ", "X"), ("X", "X", " "), (" ", " ", " ")],
)
def test_an_unticked_consent_box_stops_the_row(c1, c2, c3):
    """A permanent row is only fair if they knew before asking. No box, no row."""
    with pytest.raises(vcr.Rejected, match="consent"):
        row_from(form(c1=c1, c2=c2, c3=c3))


def test_the_record_has_to_be_reachable_by_a_reader():
    for bad in ["http://example.com/x", "mailto:a@b.c", "screenshot.png", ""]:
        with pytest.raises(vcr.Rejected, match="https"):
            row_from(form(record=bad))


def test_a_suite_that_does_not_exist_is_refused():
    with pytest.raises(vcr.Rejected, match="not suites"):
        row_from(form(suites="totally_made_up_v0"))


def test_the_runners_own_whole_corpus_value_is_accepted():
    """The desk has to accept what our own tool tells people to submit.

    scripts/conformance_runner.py ends every full run by printing a prefilled
    issue link carrying `suites=all <N> suites`, deliberately, so a whole-corpus
    run says so instead of listing forty-three names. The validator only knew
    directory names, so the one path we actively tell people to walk was refused
    by our own gate. Found 2026-08-19 on the first real application, issue #587.
    """
    n = len(vcr.known_suites())
    row = row_from(form(suites=f"all {n} suites"))
    assert row["suites"] == sorted(vcr.known_suites())


def test_a_whole_corpus_claim_has_to_match_the_corpus():
    """`all 9 suites` against a 43-suite corpus is a claim about a different run."""
    n = len(vcr.known_suites())
    with pytest.raises(vcr.Rejected, match="suites"):
        row_from(form(suites=f"all {n - 1} suites"))


def test_prose_around_the_suites_is_still_refused():
    """Liberal enough for the tool's output, not for a hand-written paragraph."""
    n = len(vcr.known_suites())
    with pytest.raises(vcr.Rejected, match="not suites"):
        row_from(form(suites=f"all {n} suites (full default run, my own venv)"))


def test_the_commit_has_to_be_in_this_repository():
    """A row nobody can rerun is an assertion. The SHA is what makes it checkable."""
    with pytest.raises(vcr.Rejected, match="not in this repository"):
        row_from(form(commit="a" * 40))
    with pytest.raises(vcr.Rejected, match="40-character"):
        row_from(form(commit="12ff466"))


def test_the_maintainer_cannot_list_the_maintainer():
    with pytest.raises(vcr.Rejected, match="not an independent"):
        row_from(form(party="Henri Sirkkavaara"))
    with pytest.raises(vcr.Rejected, match="not an independent"):
        row_from(form(), author="vaaraio")


def test_an_overlong_field_is_refused():
    with pytest.raises(vcr.Rejected, match="longer than"):
        row_from(form(party="x" * 200))


def test_the_same_record_cannot_be_listed_twice():
    data = {"reproductions": []}
    vcr.add_row(row_from(form()), "2026-08-14", data)
    with pytest.raises(vcr.Rejected, match="already listed"):
        vcr.add_row(row_from(form(party="Someone Else")), "2026-08-15", data)


def test_numbers_run_forward_and_are_never_reissued():
    """Position in the order is the one thing a row holds that cannot be given."""
    data = {"reproductions": []}
    first = vcr.add_row(row_from(form()), "2026-08-14", data)
    second = vcr.add_row(
        row_from(form(party="Someone Else", record="https://example.com/run")),
        "2026-08-15",
        data,
    )
    assert (first["id"], second["id"]) == (1, 2)


def test_two_parties_with_the_same_name_get_different_badges():
    data = {"reproductions": []}
    a = vcr.add_row(row_from(form()), "2026-08-14", data)
    b = vcr.add_row(
        row_from(form(record="https://example.com/other")), "2026-08-15", data
    )
    assert a["slug"] != b["slug"]


def test_a_badge_dates_itself_without_phoning_home():
    """Baked into the pixels so the badge dates itself without phoning home."""
    svg = render.badge_svg(
        {"id": 7, "slug": "x", "date": "2026-08-14", "at_commit": HEAD, "party": "X"}
    )
    assert "2026-08-14" in svg
    assert HEAD[:7] in svg
    assert "http://" not in svg.replace('xmlns="http://www.w3.org/2000/svg"', "")


def test_the_badge_face_does_not_rank_the_holder():
    """The row number is a chain index, never a placing.

    Rows are chained, so an order exists and the number has to live in the row,
    the served bytes and the metadata a reader recomputes against. Putting it on
    the face turns that index into a standing: row 1 and row 250 are the
    identical claim, and only one of them reads as an achievement. The table
    pays off on volume and on strangers walking in years later, so a badge that
    quietly tells the fiftieth party their sticker is worth less is working
    against the thing it was built for.
    """
    import xml.etree.ElementTree as ET

    svg = render.badge_svg(
        {"id": 7, "slug": "x", "date": "2026-08-14", "at_commit": HEAD, "party": "X"}
    )
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    face = " ".join((el.text or "") for el in root.iter(f"{ns}text"))
    assert "#7" not in face
    assert "7" not in face.replace("2026-08-14", "").replace(HEAD[:7], "")
    assert "reproduced" in face

    # It still has to be recoverable, or the badge stops pointing at a row.
    assert root.find(".//{https://vaara.io/ns/vcr/v1}row").text == "7"


def test_a_badge_commits_to_its_own_row(tmp_path):
    """The badge carries a digest a stranger can recompute from published bytes.

    This is the corpus property turned on the badge itself. A reader who trusts
    neither the listed party nor Vaara downloads the row, hashes it, and
    compares. Nothing in that path needs a key or a Vaara install.
    """
    import hashlib

    data = {"terms_version": "2026-08-15", "reproductions": []}
    row = vcr.add_row(row_from(form()), "2026-08-14", data)
    render.write_badges(data, tmp_path)

    served = (tmp_path / f"{row['slug']}.json").read_bytes()
    svg = (tmp_path / f"{row['slug']}.svg").read_text(encoding="utf-8")
    assert f"sha256:{hashlib.sha256(served).hexdigest()}" in svg
    assert json.loads(served) == row


def test_the_printable_sheet_carries_the_scoping_and_the_limit(tmp_path):
    """The wall version has to state what it does not claim, same as the page.

    A framed sheet outlives the conversation around it, so it travels with the
    limit written on it rather than a link to where the limit lives.
    """
    data = {"terms_version": "2026-08-15", "reproductions": []}
    row = vcr.add_row(row_from(form()), "2026-08-14", data)
    sheet = render.certificate_html(row)

    assert "Validates the pinned independent vector lanes." in sheet
    assert "not a certification" in sheet
    # the template wraps, so match on words rather than a phrase spanning a line
    assert "ratification" in sheet
    assert "recomputes from committed bytes" in " ".join(sheet.split())
    assert render.row_digest(row) in sheet
    assert row["at_commit"] in sheet
    # The party's own record URL is printed as text and has to be, since a
    # framed sheet that cannot be traced back to the public record is decor.
    # What must not appear is a fetched resource, so the sheet still renders
    # with the network off.
    loads = re.findall(r'(?:src|href)\s*=\s*"(https?://[^"]+)"', sheet)
    assert not loads, f"the sheet fetches something off-origin: {loads}"
    assert row["record"] in sheet


def test_the_served_row_is_canonical_bytes_and_nothing_else():
    """One command has to answer it, so the file cannot carry stray formatting."""
    row = {"b": 2, "a": 1}
    assert render.row_bytes(row) == b'{"a":1,"b":2}'


def test_editing_a_row_changes_its_digest(tmp_path):
    data = {"terms_version": "2026-08-15", "reproductions": []}
    row = vcr.add_row(row_from(form()), "2026-08-14", data)
    before = render.row_digest(row)
    row["result"] = "43 of 43, actually"
    assert render.row_digest(row) != before
    render.write_badges(data, tmp_path)
    assert render.badge_drift(data, tmp_path) == []


def test_a_row_records_which_terms_it_agreed_to():
    """Terms can change for later rows and never reach back to an earlier yes."""
    data = {"terms_version": "2026-08-15", "reproductions": []}
    first = vcr.add_row(row_from(form()), "2026-08-14", data)
    assert first["terms_version"] == "2026-08-15"

    data["terms_version"] = "2027-01-01"
    second = vcr.add_row(
        row_from(form(party="Someone Else", record="https://example.com/run")),
        "2027-01-02",
        data,
    )
    assert second["terms_version"] == "2027-01-01"
    assert first["terms_version"] == "2026-08-15"


def test_the_committed_table_ships_with_no_rows_and_no_badges():
    """Nobody appears on the page until they have asked to be there."""
    data = json.loads((ROOT / "conformance" / "reproductions.json").read_text())
    assert data["reproductions"] == []
    assert render.badge_drift(data) == []


def test_the_chain_catches_a_removed_row():
    """The promise that nothing comes down has to be checkable, not just made.

    The person who would quietly drop a row is the person publishing the
    promise, so the promise is worth nothing on its own. Removing an entry
    breaks every link after it, and the break names the row.
    """
    data = {"genesis": "sha256:" + "0" * 64, "reproductions": []}
    for i, party in enumerate(["A Party", "B Party", "C Party"]):
        vcr.add_row(
            row_from(form(party=party, record=f"https://example.com/{i}")),
            f"2026-08-{14 + i}",
            data,
        )
    assert vcr.verify_chain(data) == []

    del data["reproductions"][1]
    problems = vcr.verify_chain(data)
    assert problems, "dropping the middle row went unnoticed"
    assert "C Party" in problems[0] or "3" in problems[0]


def test_the_chain_catches_an_edited_row():
    """A published row is not rewritten. A correction is a new row."""
    data = {"genesis": "sha256:" + "0" * 64, "reproductions": []}
    vcr.add_row(row_from(form()), "2026-08-14", data)
    vcr.add_row(
        row_from(form(party="Second", record="https://example.com/second")),
        "2026-08-15",
        data,
    )
    assert vcr.verify_chain(data) == []

    data["reproductions"][0]["result"] = "43 of 43, actually"
    assert vcr.verify_chain(data), "an edited row still verified"


def test_the_chain_catches_a_reordering():
    data = {"genesis": "sha256:" + "0" * 64, "reproductions": []}
    vcr.add_row(row_from(form()), "2026-08-14", data)
    vcr.add_row(
        row_from(form(party="Second", record="https://example.com/second")),
        "2026-08-15",
        data,
    )
    data["reproductions"].reverse()
    assert vcr.verify_chain(data), "a swapped pair still verified"


def test_the_first_row_links_to_the_declared_genesis():
    data = {"genesis": "sha256:" + "0" * 64, "reproductions": []}
    first = vcr.add_row(row_from(form()), "2026-08-14", data)
    assert first["prev"] == data["genesis"]


def test_the_committed_file_declares_a_genesis_and_verifies():
    """Whatever ships has to pass its own checker, empty or not."""
    data = json.loads((ROOT / "conformance" / "reproductions.json").read_text())
    assert data.get("genesis", "").startswith("sha256:")
    assert vcr.verify_chain(data) == []
