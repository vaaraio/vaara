"""The conformance statement and the `vaara conformance-statement` command.

B2 of the widening plan: the self-test an emitter runs to prove SEP-2828
conformance against the published corpus rather than ask to be trusted. Covers
the builder and renderer against the committed goldens (deterministic, keyless),
the real corpus self-test, the Vaara-free independent checker, and the CLI end
to end. Keyless, so the suite runs in the base install.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation.receipt import (
    STATEMENT_SCHEMA,
    ConformanceCorpusError,
    build_conformance_statement,
    render_conformance_statement,
)
from vaara.cli import main

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "conformance" / "sep2828"
VECTORS = Path(__file__).resolve().parent / "vectors" / "conformance_statement_v0"
EMITTER = VECTORS / "emitter_records"
PAGES = VECTORS / "pages"
EXPECTED = json.loads((VECTORS / "expected.json").read_text(encoding="utf-8"))

SCENARIO_RECORDS = {
    "selftest_only": None, "clean": "clean", "flawed": "flawed", "duplicate": "duplicate",
}


def _records(scenario: str):
    sub = SCENARIO_RECORDS[scenario]
    if sub is None:
        return None
    return [(p.name, json.loads(p.read_text())) for p in sorted((EMITTER / sub).glob("*.json"))]


def _build(scenario: str):
    return build_conformance_statement(CORPUS, records=_records(scenario))


# ── Goldens ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scenario", sorted(SCENARIO_RECORDS))
def test_statement_matches_golden_json(scenario):
    assert _build(scenario).to_dict() == EXPECTED[scenario]


@pytest.mark.parametrize("scenario", sorted(SCENARIO_RECORDS))
def test_render_matches_golden_page(scenario):
    page = render_conformance_statement(_build(scenario))
    assert page == (PAGES / f"{scenario}.md").read_text(encoding="utf-8")


def test_all_scenarios_present():
    assert sorted(p.stem for p in PAGES.glob("*.md")) == [
        "clean", "duplicate", "flawed", "selftest_only"
    ]


def test_independent_checker_confirms_statements():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_public_reexport_is_wired():
    from vaara.attestation.receipt import build_conformance_statement as public
    assert public is build_conformance_statement
    assert STATEMENT_SCHEMA == "sep2828-conformance-statement"


# ── Real corpus self-test ─────────────────────────────────────────────────────


def test_real_corpus_self_test_reproduces_every_verdict():
    statement = build_conformance_statement(CORPUS)
    assert statement.corpus.verified
    assert statement.self_test.conforms
    assert statement.self_test.reproduced == statement.self_test.cases
    assert statement.conforms  # no records supplied, so corpus + self-test decide it


def test_self_test_covers_both_published_suites():
    statement = build_conformance_statement(CORPUS)
    names = {s.name for s in statement.self_test.suites}
    assert names == {"record_conformance_v0", "record_set_v0"}
    assert all(s.runnable and not s.mismatches for s in statement.self_test.suites)


def test_clean_records_conform_and_flawed_do_not():
    assert _build("clean").records.conforms
    flawed = _build("flawed").records
    assert not flawed.conforms
    assert flawed.conforming < flawed.total
    assert flawed.nonconforming  # the non-conforming record is named


def test_duplicate_set_gate_each_record_conforms_but_set_does_not():
    # Every record is individually well-formed, so the verdict turns on a
    # required cross-record property: two outcomes pinning one call.
    statement = _build("duplicate")
    records = statement.records
    assert records.conforming == records.total  # each record conforms on its own
    assert not records.conforms  # but the set does not
    assert any(f.id == "duplicate_call" and f.severity == "required" for f in records.findings)
    assert not statement.conforms


# ── Deterministic and faithful ────────────────────────────────────────────────


def test_render_is_deterministic():
    a = render_conformance_statement(build_conformance_statement(CORPUS))
    b = render_conformance_statement(build_conformance_statement(CORPUS))
    assert a == b  # no clock, no key: same inputs, same bytes


def test_as_of_is_echoed_verbatim_not_from_a_clock():
    statement = build_conformance_statement(CORPUS, as_of="2026-06-08")
    page = render_conformance_statement(statement)
    assert statement.as_of == "2026-06-08"
    assert "As of 2026-06-08." in page


def test_as_of_default_is_omitted():
    page = render_conformance_statement(build_conformance_statement(CORPUS))
    assert "As of " not in page


def test_statement_names_the_exact_corpus_byte_set():
    page = render_conformance_statement(build_conformance_statement(CORPUS))
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text())
    assert manifest["corpusDigest"] in page
    assert manifest["version"] in page


# ── Tampered corpus ───────────────────────────────────────────────────────────


def _copy_corpus(dst: Path) -> Path:
    shutil.copytree(CORPUS, dst)
    return dst


def test_tampered_fixture_breaks_integrity_and_gates(tmp_path):
    corpus = _copy_corpus(tmp_path / "corpus")
    victim = corpus / "record_conformance_v0" / "records" / "conforming_refused_no_commitment.json"
    victim.write_text(victim.read_text() + "\n", encoding="utf-8")  # one byte changes the digest
    statement = build_conformance_statement(corpus)
    assert not statement.corpus.verified
    assert statement.corpus.problems
    assert not statement.conforms  # a tampered corpus cannot yield a conforming statement


def test_extra_file_in_corpus_is_an_integrity_problem(tmp_path):
    corpus = _copy_corpus(tmp_path / "corpus")
    (corpus / "record_set_v0" / "sets" / "clean" / "rogue.json").write_text("{}", "utf-8")
    statement = build_conformance_statement(corpus)
    assert not statement.corpus.verified
    assert any("unexpected file" in p for p in statement.corpus.problems)


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_default_corpus_to_stdout(capsys):
    rc = main(["conformance-statement", "--corpus", str(CORPUS)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out == (PAGES / "selftest_only.md").read_text(encoding="utf-8")


def test_cli_clean_records_conform(capsys):
    rc = main(["conformance-statement", "--corpus", str(CORPUS),
               "--records", str(EMITTER / "clean")])
    out = capsys.readouterr().out
    assert rc == 0
    assert out == (PAGES / "clean.md").read_text(encoding="utf-8")


def test_cli_flawed_records_exit_1(capsys):
    rc = main(["conformance-statement", "--corpus", str(CORPUS),
               "--records", str(EMITTER / "flawed")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NON-CONFORMING" in out


def test_cli_json_output(capsys):
    rc = main(["conformance-statement", "--corpus", str(CORPUS), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload == EXPECTED["selftest_only"]


def test_cli_writes_to_out_file(tmp_path, capsys):
    target = tmp_path / "statement.md"
    rc = main(["conformance-statement", "--corpus", str(CORPUS), "--out", str(target)])
    assert rc == 0
    assert target.read_text(encoding="utf-8") == (PAGES / "selftest_only.md").read_text("utf-8")
    assert "wrote conformance statement" in capsys.readouterr().err


def test_cli_corpus_not_a_directory(tmp_path, capsys):
    rc = main(["conformance-statement", "--corpus", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a corpus directory" in capsys.readouterr().err


def test_cli_corpus_without_manifest(tmp_path, capsys):
    rc = main(["conformance-statement", "--corpus", str(tmp_path)])
    assert rc == 2
    assert "no MANIFEST.json" in capsys.readouterr().err


def test_cli_records_not_a_directory(tmp_path, capsys):
    rc = main(["conformance-statement", "--corpus", str(CORPUS),
               "--records", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_records_no_matching_files(tmp_path, capsys):
    rc = main(["conformance-statement", "--corpus", str(CORPUS), "--records", str(tmp_path)])
    assert rc == 2
    assert "no files matched" in capsys.readouterr().err


def test_cli_unreadable_record_gates(tmp_path, capsys):
    (tmp_path / "good.json").write_text(
        (EMITTER / "clean" / "decision.json").read_text(), "utf-8")
    (tmp_path / "bad.json").write_text("{ not json", "utf-8")
    rc = main(["conformance-statement", "--corpus", str(CORPUS), "--records", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "could not be read" in out
    assert "bad.json" in out


def test_missing_manifest_raises_corpus_error(tmp_path):
    with pytest.raises(ConformanceCorpusError):
        build_conformance_statement(tmp_path)
