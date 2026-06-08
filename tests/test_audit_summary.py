"""The one-page audit summary and the `vaara audit-summary` command.

The human-readable face of `verify-records`: the same keyless set check,
rendered as a page of Markdown a regulator reads. Covers the renderer against
the committed golden pages (byte-exact, deterministic), its structural
guarantees, and the CLI end to end. Keyless, like the record check, so the
suite runs in the base install.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation._audit_summary import SUMMARY_SCHEMA
from vaara.attestation.receipt import check_record_set, render_record_set_summary
from vaara.cli import main

VECTORS = Path(__file__).resolve().parent / "vectors" / "audit_summary_v0"
RECORD_SETS = Path(__file__).resolve().parent / "vectors" / "record_set_v0" / "sets"
PAGES = VECTORS / "pages"


def _cases():
    return sorted(p.stem for p in PAGES.glob("*.md"))


def _load_set(name: str):
    files = sorted((RECORD_SETS / name).glob("*.json"))
    return [(p.name, json.loads(p.read_text())) for p in files]


# ── Golden pages ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _cases())
def test_render_matches_golden_page(name):
    page = render_record_set_summary(check_record_set(_load_set(name)))
    assert page == (PAGES / f"{name}.md").read_text(encoding="utf-8")


def test_at_least_four_golden_pages_present():
    assert len(_cases()) >= 4


def test_independent_checker_confirms_pages_match_verdict():
    # The Vaara-free checker re-derives each verdict from record_set_v0 and
    # asserts the rendered page states it faithfully.
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_public_reexport_is_wired():
    from vaara.attestation.receipt import render_record_set_summary as public
    assert public is render_record_set_summary
    assert SUMMARY_SCHEMA == "vaara-record-set-summary/1"


# ── Deterministic and faithful ────────────────────────────────────────────────


def test_render_is_deterministic():
    records = _load_set("clean")
    a = render_record_set_summary(check_record_set(records))
    b = render_record_set_summary(check_record_set(records))
    assert a == b  # no clock, no key: same set, same bytes


def test_conforming_set_says_conforms_and_has_no_findings_section_body():
    page = render_record_set_summary(check_record_set(_load_set("clean")))
    assert "**Verdict: CONFORMS**" in page
    assert "No cross-record findings." in page


def test_nonconforming_set_names_the_required_finding():
    page = render_record_set_summary(check_record_set(_load_set("duplicate_call")))
    assert "**Verdict: NON-CONFORMING**" in page
    assert "Required (these gate conformance):" in page
    assert "duplicate_call" in page


def test_advisory_gap_is_under_the_advisory_heading():
    page = render_record_set_summary(check_record_set(_load_set("executed_gap")))
    assert "Advisory (gaps that do not gate conformance):" in page
    assert "executed_without_result_commitment" in page


def test_nonconforming_record_is_listed_with_its_reason():
    page = render_record_set_summary(check_record_set(_load_set("mixed_nonconforming")))
    assert "## Non-conforming records" in page
    assert "`bad.json`" in page


def test_empty_set_renders_a_clean_page():
    page = render_record_set_summary(check_record_set([]))
    assert "No records were supplied." in page
    assert "**Verdict: CONFORMS**" in page


def test_custom_title_is_used():
    page = render_record_set_summary(check_record_set([]), title="Quarterly review")
    assert page.startswith("# Quarterly review\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_clean_set_to_stdout(capsys):
    rc = main(["audit-summary", str(RECORD_SETS / "clean")])
    out = capsys.readouterr().out
    assert rc == 0
    assert out == (PAGES / "clean.md").read_text(encoding="utf-8")


def test_cli_nonconforming_set_exit_1(capsys):
    rc = main(["audit-summary", str(RECORD_SETS / "duplicate_call")])
    assert rc == 1
    assert "NON-CONFORMING" in capsys.readouterr().out


def test_cli_writes_to_out_file(tmp_path, capsys):
    target = tmp_path / "summary.md"
    rc = main(["audit-summary", str(RECORD_SETS / "clean"), "--out", str(target)])
    assert rc == 0
    assert target.read_text(encoding="utf-8") == (PAGES / "clean.md").read_text("utf-8")
    assert "wrote audit summary" in capsys.readouterr().err


def test_cli_not_a_directory(tmp_path, capsys):
    rc = main(["audit-summary", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_no_matching_files(tmp_path, capsys):
    rc = main(["audit-summary", str(tmp_path)])
    assert rc == 2
    assert "no files matched" in capsys.readouterr().err


def test_cli_unreadable_file_notes_and_gates(tmp_path, capsys):
    (tmp_path / "good.json").write_text(
        (RECORD_SETS / "clean" / "r1.json").read_text())
    (tmp_path / "bad.json").write_text("{ not json")
    rc = main(["audit-summary", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "could not be read" in out
    assert "bad.json" in out
