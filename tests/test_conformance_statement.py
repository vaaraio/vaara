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
    "unproved": "unproved",
}


def _records(scenario: str):
    """Mirror how the CLI reads a real directory: unparseable files stay visible."""
    sub = SCENARIO_RECORDS[scenario]
    if sub is None:
        return None, []
    records, unreadable = [], []
    for p in sorted((EMITTER / sub).glob("*.json")):
        try:
            records.append((p.name, json.loads(p.read_text())))
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append((p.name, type(exc).__name__))
    return records, unreadable


def _build(scenario: str):
    records, unreadable = _records(scenario)
    return build_conformance_statement(CORPUS, records=records, unreadable=unreadable)


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
        "clean", "duplicate", "flawed", "selftest_only", "unproved"
    ]


# ── The three states ──────────────────────────────────────────────────────────


def test_grades_separate_a_failed_check_from_an_unreached_one():
    """The distinction a boolean cannot carry, pinned end to end."""
    assert _build("clean").grade == "proved"
    assert _build("flawed").grade == "false"
    assert _build("unproved").grade == "unproved"


def test_unproved_run_does_not_claim_the_records_failed():
    statement = _build("unproved")
    assert statement.records is not None
    # Nothing that was read disagreed with the spec.
    assert statement.records.nonconforming == ()
    assert statement.records.conforming == statement.records.total
    # The unread file is why, and it is named rather than folded into a failure.
    assert [n for n, _ in statement.records.unreadable] == ["broken.json"]
    assert statement.records.grade == "unproved"
    assert statement.conforms is False


def test_unproved_page_never_says_non_conforming():
    page = render_conformance_statement(_build("unproved"))
    assert "**Statement: UNPROVED**" in page
    assert "NON-CONFORMING" not in page
    assert "do NOT conform" not in page


def test_a_real_failure_outranks_an_unreached_check():
    from vaara.attestation.receipt import combine_grades

    assert combine_grades(["proved", "unproved", "false"]) == "false"
    assert combine_grades(["proved", "unproved"]) == "unproved"
    assert combine_grades(["proved", "proved"]) == "proved"
    # Nothing checked establishes nothing.
    assert combine_grades([]) == "unproved"


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


# ── Windows checkout (CRLF) ───────────────────────────────────────────────────
#
# Git for Windows installs with core.autocrlf=true, so a plain clone rewrites
# every fixture's line endings. The corpus is pinned byte for byte, so that
# breaks all 32 digests without changing a byte of content. Before the fix the
# statement called that NON-CONFORMING, which reads as "this emitter disagrees
# with SEP-2828" and is the kind of thing that gets filed as a permanent public
# conformance row. It is not a disagreement. It is a checkout artefact.


def _to_crlf(corpus: Path) -> int:
    """Do to the corpus what a core.autocrlf=true checkout does."""
    n = 0
    for path in sorted(corpus.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        raw = path.read_bytes()
        if b"\r" in raw or b"\n" not in raw or b"\0" in raw:
            continue
        path.write_bytes(raw.replace(b"\n", b"\r\n"))
        n += 1
    return n


def test_windows_checkout_is_unproved_not_a_disagreement(tmp_path):
    corpus = _copy_corpus(tmp_path / "corpus")
    assert _to_crlf(corpus) > 0
    statement = build_conformance_statement(corpus)

    # The implementation still agrees with the spec on every recorded case:
    # CRLF changes the bytes on disk, not what the records mean.
    assert statement.self_test.conforms
    assert statement.self_test.reproduced == statement.self_test.cases

    # So the run establishes nothing, rather than contradicting the spec.
    assert statement.corpus.grade == "unproved"
    assert statement.grade == "unproved"
    assert statement.corpus.line_endings_only

    page = render_conformance_statement(statement)
    assert "**Statement: UNPROVED**" in page
    assert "NON-CONFORMING" not in page  # the false-disagreement regression
    assert "core.autocrlf" in page  # and it names the actual cause


def test_mangled_corpus_still_cannot_yield_a_pass(tmp_path):
    """UNPROVED gates exactly as hard as FALSE did. It is honest, not lenient."""
    corpus = _copy_corpus(tmp_path / "corpus")
    _to_crlf(corpus)
    statement = build_conformance_statement(corpus)
    assert not statement.conforms
    assert not statement.corpus.verified
    rc = main(["conformance-statement", "--corpus", str(corpus)])
    assert rc == 1


def test_line_ending_detection_is_exact_not_a_guess(tmp_path):
    """A real content edit is never excused as a line-ending artefact.

    The check re-hashes with the translation undone, so only a file whose
    content is byte for byte the published content can qualify.
    """
    corpus = _copy_corpus(tmp_path / "corpus")
    victim = corpus / "record_set_v0" / "sets" / "clean" / "r1.json"
    doc = json.loads(victim.read_text())
    doc["alg"] = "HS256-tampered"
    victim.write_bytes(json.dumps(doc, indent=2).encode() + b"\r\n")  # CRLF *and* edited
    statement = build_conformance_statement(corpus)
    assert not statement.corpus.line_ending_mismatches
    assert any("digest mismatch" in p for p in statement.corpus.problems)
    assert not statement.corpus.line_endings_only


def test_mangling_plus_tampering_does_not_hide_the_tampering(tmp_path):
    corpus = _copy_corpus(tmp_path / "corpus")
    _to_crlf(corpus)
    victim = corpus / "record_set_v0" / "sets" / "clean" / "r2.json"
    victim.write_bytes(b'{"not": "the published record"}')
    statement = build_conformance_statement(corpus)
    assert not statement.corpus.line_endings_only  # one file is genuinely changed
    page = render_conformance_statement(statement)
    assert "digest mismatch: record_set_v0/sets/clean/r2.json" in page


def test_standalone_runner_names_line_endings(tmp_path):
    """The Vaara-free path a third party runs must say the same thing."""
    corpus = _copy_corpus(tmp_path / "corpus")
    _to_crlf(corpus)
    proc = subprocess.run([sys.executable, "run.py", "--verify-manifest"],
                          cwd=corpus, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "line endings, not content" in proc.stdout
    assert "core.autocrlf" in proc.stdout
    assert "says nothing about SEP-2828 conformance" in proc.stdout


def test_vectors_checker_skips_a_mangled_clone_instead_of_failing(tmp_path):
    """The symptom a stranger actually reports, and the one that becomes a row.

    This suite compares committed goldens against a fresh derivation from the
    corpus. On a CRLF clone the derivation runs over bytes that are not the
    published corpus, so every scenario mismatched and the suite reported 0/5.
    That reads as Vaara disagreeing with its own vectors and is exactly the kind
    of thing that gets filed as a permanent public conformance row.

    It must skip with a reason instead. Exit 77 is the profile runner's SKIP
    contract, and the runner takes the last stderr line as the reason.
    """
    shutil.copytree(CORPUS, tmp_path / "conformance" / "sep2828")
    dst = tmp_path / "tests" / "vectors" / "conformance_statement_v0"
    shutil.copytree(VECTORS, dst)
    assert _to_crlf(tmp_path / "conformance") > 0

    proc = subprocess.run([sys.executable, str(dst / "_check_independent.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 77, proc.stdout + proc.stderr
    assert proc.stderr.strip().splitlines()[-1].startswith("SKIP: ")
    assert "core.autocrlf" in proc.stderr
    assert "Not a conformance result" in proc.stderr
    # It must not read as a disagreement.
    assert "statements match the independent derivation" not in proc.stdout


def test_vectors_checker_still_fails_on_real_corpus_tampering(tmp_path):
    """Only a provably newlines-only difference goes quiet. Tampering stays loud."""
    shutil.copytree(CORPUS, tmp_path / "conformance" / "sep2828")
    dst = tmp_path / "tests" / "vectors" / "conformance_statement_v0"
    shutil.copytree(VECTORS, dst)
    victim = (tmp_path / "conformance" / "sep2828" / "record_set_v0" / "sets"
              / "clean" / "r1.json")
    victim.write_bytes(b'{"tampered": true}')

    proc = subprocess.run([sys.executable, str(dst / "_check_independent.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "SKIP" not in proc.stderr


def test_gitattributes_checks_the_corpus_out_verbatim():
    """The root fix: git must never translate the byte-pinned corpus.

    Asserts the effective attribute rather than the file's text, so the guard
    survives any reshuffle of .gitattributes that keeps the promise.
    """
    if not (REPO / ".gitattributes").is_file():
        pytest.fail(".gitattributes is missing; the corpus is unpinned on Windows")
    probe = "conformance/sep2828/record_set_v0/sets/clean/r1.json"
    proc = subprocess.run(["git", "check-attr", "text", "--", probe],
                          cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("git unavailable")
    assert proc.stdout.strip().endswith("text: unset"), proc.stdout


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
