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
import subprocess
from pathlib import Path

import pytest

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
- [{c2}] The link I gave is my own public record of this run.
"""


def form(**over) -> str:
    values = {
        "party": "Iman Schrock",
        "suites": "transparency_consistency_v0, evidence_bundle_v0",
        "commit": HEAD,
        "record": "https://mailarchive.ietf.org/arch/msg/scitt/fE4RYzpmR440PVrZaknTvVKjUoI/",
        "c1": "X",
        "c2": "X",
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


@pytest.mark.parametrize("c1,c2", [(" ", "X"), ("X", " "), (" ", " ")])
def test_an_unticked_consent_box_stops_the_row(c1, c2):
    """Consent is the whole basis for publishing a name. No box, no row."""
    with pytest.raises(vcr.Rejected, match="consent"):
        row_from(form(c1=c1, c2=c2))


def test_the_record_has_to_be_reachable_by_a_reader():
    for bad in ["http://example.com/x", "mailto:a@b.c", "screenshot.png", ""]:
        with pytest.raises(vcr.Rejected, match="https"):
            row_from(form(record=bad))


def test_a_suite_that_does_not_exist_is_refused():
    with pytest.raises(vcr.Rejected, match="not suites"):
        row_from(form(suites="totally_made_up_v0"))


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


def test_a_number_is_never_reissued_after_a_withdrawal():
    """The number is the one thing a listed party holds that cannot be given away.

    A withdrawn row leaves a gap on purpose. Reusing the gap would hand a
    newcomer somebody else's position in the order.
    """
    data = {"reproductions": []}
    first = vcr.add_row(row_from(form()), "2026-08-14", data)
    assert first["id"] == 1
    data["reproductions"].clear()  # they asked to come down
    second = vcr.add_row(
        row_from(form(party="Someone Else", record="https://example.com/run")),
        "2026-08-15",
        data,
    )
    assert second["id"] == 2


def test_two_parties_with_the_same_name_get_different_badges():
    data = {"reproductions": []}
    a = vcr.add_row(row_from(form()), "2026-08-14", data)
    b = vcr.add_row(
        row_from(form(record="https://example.com/other")), "2026-08-15", data
    )
    assert a["slug"] != b["slug"]


def test_a_badge_carries_the_number_the_date_and_the_commit():
    """Baked into the pixels so the badge dates itself without phoning home."""
    svg = render.badge_svg(
        {"id": 7, "slug": "x", "date": "2026-08-14", "at_commit": HEAD, "party": "X"}
    )
    assert "VCR #7" in svg
    assert "2026-08-14" in svg
    assert HEAD[:7] in svg
    assert "http://" not in svg.replace('xmlns="http://www.w3.org/2000/svg"', "")


def test_withdrawing_a_row_takes_its_badge_down(tmp_path):
    """Removal from the page is cosmetic if the badge URL still resolves."""
    pytest.importorskip("rfc8785", reason="ships in the attestation extra")
    data = {"reproductions": []}
    row = vcr.add_row(row_from(form()), "2026-08-14", data)
    render.write_badges(data, tmp_path)
    assert (tmp_path / f"{row['slug']}.svg").exists()
    assert render.badge_drift(data, tmp_path) == []

    data["reproductions"].clear()
    assert render.badge_drift(data, tmp_path) == sorted(
        [f"{row['slug']}.svg", f"{row['slug']}.json"]
    )
    render.write_badges(data, tmp_path)
    assert not list(tmp_path.iterdir())


def test_a_badge_commits_to_its_own_row(tmp_path):
    """The badge carries a digest a stranger can recompute from published bytes.

    This is the corpus property turned on the badge itself. A reader who trusts
    neither the listed party nor Vaara downloads the row, hashes it, and
    compares. Nothing in that path needs a key or a Vaara install.
    """
    import hashlib

    pytest.importorskip("rfc8785", reason="ships in the attestation extra")
    data = {"terms_version": "2026-08-15", "reproductions": []}
    row = vcr.add_row(row_from(form()), "2026-08-14", data)
    render.write_badges(data, tmp_path)

    served = (tmp_path / f"{row['slug']}.json").read_bytes()
    svg = (tmp_path / f"{row['slug']}.svg").read_text(encoding="utf-8")
    assert f"sha256:{hashlib.sha256(served).hexdigest()}" in svg
    assert json.loads(served) == row


def test_the_served_row_is_canonical_bytes_and_nothing_else():
    """One command has to answer it, so the file cannot carry stray formatting."""
    pytest.importorskip("rfc8785", reason="ships in the attestation extra")
    row = {"b": 2, "a": 1}
    assert render.row_bytes(row) == b'{"a":1,"b":2}'


def test_editing_a_row_changes_its_digest(tmp_path):
    data = {"terms_version": "2026-08-15", "reproductions": []}
    row = vcr.add_row(row_from(form()), "2026-08-14", data)
    before = render.row_digest(row)
    pytest.importorskip("rfc8785", reason="ships in the attestation extra")
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
