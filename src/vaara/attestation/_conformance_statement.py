# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conformance self-test and statement for the published SEP-2828 corpus.

An emitter that claims SEP-2828 conformance answers "trust us" with "prove it
against the neutral suite". This builds that proof. It runs this
implementation's keyless conformance check over the published conformance
corpus, confirms the bytes match their manifest, optionally runs the emitter's
own records through the same set-level check, and produces one statement that
names the exact corpus byte set it was checked against.

The statement has three parts:

* **Corpus integrity** - every fixture file's SHA-256 matches ``MANIFEST.json``
  and the single ``corpusDigest`` recomputes, so the statement pins the
  published bytes rather than a moving target.
* **Self-test** - the running implementation's conformance check reproduces
  every verdict the corpus records in each suite's ``expected.json``. This is
  the "prove it": the tool agrees with the neutral suite, case for case.
* **Records** (optional) - the emitter's own records run through the same
  keyless set check, with the verdict reported beside the self-test.

Every part grades to one of three states rather than a boolean:

* ``proved`` - the check ran and the property holds.
* ``unproved`` - the check could not be reached, so nothing is asserted.
* ``false`` - the check ran and the property does not hold.

A boolean cannot tell a reader whether a check failed or was never reached, and
those call for different repairs. ``unproved`` is only ever produced by an
explicit execution state the runner recorded (a suite it could not place, a
record file it could not read, a corpus whose bytes are not the published set),
never inferred from a primary check that failed. Where a run mixes states, the
worst one wins: a check that ran and failed outranks one that never ran.

Corpus integrity is a precondition and grades ``unproved`` when it fails, never
``false``. It asks whether the published byte set is present, not whether any
implementation agrees with the spec, so a corpus that does not verify means the
self-test measured something other than the published corpus and the run
established nothing. A common cause is entirely benign and local: a Windows
clone with Git's ``core.autocrlf`` on rewrites every fixture's line endings,
which breaks every digest without changing a single byte of content. The
statement detects that case exactly, by re-hashing with the translation undone,
and names it rather than reporting a disagreement that no check observed.

Deterministic and keyless. There is no clock: an ``as_of`` date is echoed
verbatim when the caller supplies one and is never read from the system, so the
same inputs render the same statement byte for byte. The whole check needs no
signing key, which is what lets a third party reproduce it.

Pure standard library; importable without the ``attestation`` extra.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from vaara.attestation._receipt_conformance import check_record_conformance
from vaara.attestation._record_set_conformance import check_record_set
from vaara.attestation._record_set_findings import SetFinding

STATEMENT_SCHEMA = "sep2828-conformance-statement"
#: v3 adds ``corpus.lineEndingMismatches`` / ``corpus.lineEndingsOnly`` and
#: grades an unverified corpus ``unproved`` rather than ``false``. Additive for
#: readers; the verdict for a byte-exact corpus is unchanged.
STATEMENT_SCHEMA_VERSION = 3

#: The check ran and the property holds.
PROVED = "proved"
#: The check could not be reached, so the statement asserts nothing either way.
UNPROVED = "unproved"
#: The check ran and the property does not hold.
FALSE = "false"

#: Worst-first. A check that ran and failed outranks one that never ran, because
#: a known failure is a stronger claim than an absence of evidence.
_GRADE_ORDER = (FALSE, UNPROVED, PROVED)


def combine_grades(grades: Sequence[str]) -> str:
    """Fold sub-grades into one, worst first.

    ``FALSE`` dominates ``UNPROVED`` dominates ``PROVED``. An empty sequence is
    ``UNPROVED``: nothing was checked, so nothing is asserted.
    """
    for grade in _GRADE_ORDER:
        if grade in grades:
            return grade
    return UNPROVED


class ConformanceCorpusError(ValueError):
    """The corpus directory is missing the manifest the statement is built from."""


# ── Result records ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorpusIntegrity:
    """Does the corpus on disk match the byte set its manifest pins?"""

    name: str
    spec: str
    version: str
    corpus_digest: str
    file_count: int
    verified: bool
    problems: tuple[str, ...]
    #: Files that differ from the manifest only by line endings. Their content
    #: is the published content; the checkout rewrote the newlines.
    line_ending_mismatches: tuple[str, ...] = ()

    @property
    def grade(self) -> str:
        """``PROVED`` when the bytes match, ``UNPROVED`` when they do not.

        Corpus integrity is a precondition, not a conformance claim. It answers
        "am I holding the published byte set", and nothing about whether an
        implementation agrees with SEP-2828. When the bytes on disk are not the
        published set, the self-test ran against something else, so the run
        established nothing about conformance to the published corpus.

        That is ``UNPROVED``, not ``FALSE``. Grading it ``FALSE`` would fold a
        local checkout problem into a verdict that reads as a disagreement with
        the spec, which is a claim no part of the run observed. The most common
        cause is exactly that benign: a Windows clone with Git's
        ``core.autocrlf`` on, which rewrites every fixture's line endings and so
        breaks every digest while changing no content at all.

        ``UNPROVED`` still gates. A statement conforms only when every in-scope
        check grades ``PROVED``, so a tampered or mangled corpus can never yield
        a pass; it yields "cannot tell", which is the honest answer.
        """
        return PROVED if self.verified else UNPROVED

    @property
    def line_endings_only(self) -> bool:
        """True when line endings explain every problem found."""
        return bool(self.line_ending_mismatches) and len(self.line_ending_mismatches) == len(
            [p for p in self.problems if not p.startswith("corpusDigest mismatch")]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "version": self.version,
            "corpusDigest": self.corpus_digest,
            "fileCount": self.file_count,
            "verified": self.verified,
            "grade": self.grade,
            "problems": list(self.problems),
            "lineEndingMismatches": list(self.line_ending_mismatches),
            "lineEndingsOnly": self.line_endings_only,
        }


@dataclass(frozen=True)
class SuiteResult:
    """How many of one suite's recorded verdicts this implementation reproduced."""

    name: str
    runnable: bool
    cases: int
    reproduced: int
    mismatches: tuple[str, ...]

    @property
    def grade(self) -> str:
        """``UNPROVED`` when the suite could not be run at all.

        ``runnable`` is an execution state recorded by the runner, not an
        inference drawn from a failed check. A suite it cannot place never
        reaches a verdict, so the statement asserts nothing about it rather than
        reporting it as a disagreement.
        """
        if not self.runnable:
            return UNPROVED
        return FALSE if self.mismatches else PROVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runnable": self.runnable,
            "grade": self.grade,
            "cases": self.cases,
            "reproduced": self.reproduced,
            "mismatches": list(self.mismatches),
        }


@dataclass(frozen=True)
class SelfTest:
    """Did this implementation reproduce every verdict the corpus records?"""

    conforms: bool
    cases: int
    reproduced: int
    suites: tuple[SuiteResult, ...]

    @property
    def grade(self) -> str:
        """Worst suite grade wins; no suites at all is ``UNPROVED``."""
        return combine_grades([s.grade for s in self.suites])

    def to_dict(self) -> dict[str, Any]:
        return {
            "conforms": self.conforms,
            "grade": self.grade,
            "cases": self.cases,
            "reproduced": self.reproduced,
            "suites": [s.to_dict() for s in self.suites],
        }


@dataclass(frozen=True)
class RecordsResult:
    """The keyless set verdict over the emitter's own records."""

    conforms: bool
    total: int
    conforming: int
    findings: tuple[SetFinding, ...]
    nonconforming: tuple[tuple[str, tuple[str, ...]], ...]
    unreadable: tuple[tuple[str, str], ...]

    @property
    def grade(self) -> str:
        """A file that could not be read is ``UNPROVED``, never ``FALSE``.

        Being unreadable is an execution state: the set check never saw the
        record, so nothing is established about it. Grading it as a failure
        would claim more than the run supports. A record that was read and did
        not conform, or a required finding across the set, is ``FALSE``.
        """
        if self.nonconforming or any(f.severity == "required" for f in self.findings):
            return FALSE
        if self.unreadable:
            return UNPROVED
        if self.total == 0:
            return UNPROVED
        return PROVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "conforms": self.conforms,
            "grade": self.grade,
            "total": self.total,
            "conforming": self.conforming,
            "findings": [
                {"id": f.id, "severity": f.severity, "records": list(f.records)}
                for f in self.findings
            ],
            "nonconforming": [
                {"name": n, "requiredFailed": list(rf)} for n, rf in self.nonconforming
            ],
            "unreadable": [{"name": n, "error": e} for n, e in self.unreadable],
        }


@dataclass(frozen=True)
class ConformanceStatement:
    """A reproducible claim of SEP-2828 conformance against a named corpus."""

    corpus: CorpusIntegrity
    self_test: SelfTest
    records: Optional[RecordsResult]
    conforms: bool
    as_of: Optional[str]

    @property
    def grade(self) -> str:
        """Worst grade across the checks that were in scope for this run.

        Records are optional. When none were supplied the section is absent
        entirely and contributes no grade, which is different from supplying
        records that could not be read. Not requested and not reachable are not
        the same claim.
        """
        grades = [self.corpus.grade, self.self_test.grade]
        if self.records is not None:
            grades.append(self.records.grade)
        return combine_grades(grades)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATEMENT_SCHEMA,
            "schemaVersion": STATEMENT_SCHEMA_VERSION,
            "conforms": self.conforms,
            "grade": self.grade,
            "asOf": self.as_of,
            "corpus": self.corpus.to_dict(),
            "selfTest": self.self_test.to_dict(),
            "records": self.records.to_dict() if self.records is not None else None,
        }


# ── Corpus integrity ──────────────────────────────────────────────────────────


def _load_manifest(corpus_dir: Path) -> dict[str, Any]:
    manifest_path = corpus_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ConformanceCorpusError(
            f"no MANIFEST.json in {corpus_dir}; point --corpus at a published "
            "SEP-2828 conformance corpus directory"
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConformanceCorpusError(f"cannot read MANIFEST.json: {exc}") from exc
    if not isinstance(data, dict) or "suites" not in data or "files" not in data:
        raise ConformanceCorpusError("MANIFEST.json is not a corpus manifest")
    return data


def _file_digests(corpus_dir: Path, suites: Sequence[str]) -> dict[str, str]:
    """SHA-256 of every fixture under each suite, keyed by POSIX relpath.

    Mirrors the corpus builder: ``__pycache__`` and ``.pyc`` are skipped so the
    digest set is the source fixtures only, never a local test artefact.
    """
    out: dict[str, str] = {}
    for suite in suites:
        for path in sorted((corpus_dir / suite).rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            rel = path.relative_to(corpus_dir).as_posix()
            out[rel] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _corpus_digest(files: dict[str, str]) -> str:
    """One digest over the whole set: SHA-256 of the sorted ``<hex>  <path>`` list."""
    lines = [f"{files[k].split(':', 1)[1]}  {k}" for k in sorted(files)]
    blob = ("\n".join(lines) + "\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _is_line_ending_only(path: Path, want: str) -> bool:
    """Does this file match the manifest once its line endings are undone?

    Exact, not a guess: undo the CRLF (and lone-CR) translation a checkout can
    apply and re-hash. A match means the content is byte for byte the published
    content and only the newlines were rewritten, which is a property of the
    working tree rather than of the corpus.

    Never used to pass the integrity check. The digest stays byte-exact; this
    only lets the statement say which of the two very different things happened.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    if b"\r" not in raw:
        return False
    normalised = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return "sha256:" + hashlib.sha256(normalised).hexdigest() == want


def verify_corpus_integrity(corpus_dir: Path, manifest: dict[str, Any]) -> CorpusIntegrity:
    """Confirm the corpus bytes on disk match what its manifest pins.

    Recomputes every file digest and the single ``corpusDigest`` and compares
    them to the manifest. ``verified`` is true iff the file set, every digest,
    and the corpus digest all match. Needs no key: anyone holding the files can
    reproduce this.
    """
    suites: list[str] = list(manifest["suites"])
    want: dict[str, str] = dict(manifest["files"])
    got = _file_digests(corpus_dir, suites)

    problems: list[str] = []
    line_endings: list[str] = []
    for rel in sorted(set(want) | set(got)):
        if rel not in got:
            problems.append(f"missing file: {rel}")
        elif rel not in want:
            problems.append(f"unexpected file: {rel}")
        elif want[rel] != got[rel]:
            # Separate "this file was changed" from "this checkout rewrote the
            # newlines". They look identical to a digest and call for opposite
            # responses, so the statement must not report them with one word.
            if _is_line_ending_only(corpus_dir / rel, want[rel]):
                line_endings.append(rel)
                problems.append(f"line endings changed, content intact: {rel}")
            else:
                problems.append(f"digest mismatch: {rel}")

    want_corpus = str(manifest.get("corpusDigest", ""))
    got_corpus = _corpus_digest(got)
    if want_corpus != got_corpus:
        problems.append(f"corpusDigest mismatch: manifest {want_corpus} != computed {got_corpus}")

    return CorpusIntegrity(
        name=str(manifest.get("corpus", "")),
        spec=str(manifest.get("spec", "")),
        version=str(manifest.get("version", "")),
        corpus_digest=want_corpus,
        file_count=len(got),
        verified=not problems,
        problems=tuple(problems),
        line_ending_mismatches=tuple(line_endings),
    )


# ── Self-test ─────────────────────────────────────────────────────────────────


def _record_suite_result(name: str, suite_dir: Path, expected: dict[str, Any]) -> SuiteResult:
    """Reproduce each per-record verdict and compare to the suite's expected.json."""
    mismatches: list[str] = []
    for case in sorted(expected):
        doc = json.loads((suite_dir / "records" / f"{case}.json").read_text(encoding="utf-8"))
        report = check_record_conformance(doc)
        got = {
            "conforms": report.conforms,
            "requiredFailed": sorted(report.required_failed),
            "advisories": sorted(report.advisories),
        }
        if got != _normalise_record_expected(expected[case]):
            mismatches.append(case)
    cases = len(expected)
    return SuiteResult(name, True, cases, cases - len(mismatches), tuple(sorted(mismatches)))


def _set_suite_result(name: str, suite_dir: Path, expected: dict[str, Any]) -> SuiteResult:
    """Reproduce each set verdict and compare to the suite's expected.json."""
    mismatches: list[str] = []
    for case in sorted(expected):
        files = sorted((suite_dir / "sets" / case).glob("*.json"))
        records = [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in files]
        report = check_record_set(records)
        got = {
            "conforms": report.conforms,
            "total": report.total,
            "conforming": report.conforming,
            "statusCounts": dict(report.status_counts),
            "verdictCounts": dict(report.verdict_counts),
            "findings": [
                {"id": f.id, "severity": f.severity, "records": list(f.records)}
                for f in report.findings
            ],
        }
        if got != _normalise_set_expected(expected[case]):
            mismatches.append(case)
    cases = len(expected)
    return SuiteResult(name, True, cases, cases - len(mismatches), tuple(sorted(mismatches)))


def _normalise_record_expected(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "conforms": case["conforms"],
        "requiredFailed": sorted(case.get("requiredFailed", [])),
        "advisories": sorted(case.get("advisories", [])),
    }


def _normalise_set_expected(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "conforms": case["conforms"],
        "total": case["total"],
        "conforming": case["conforming"],
        "statusCounts": dict(case.get("statusCounts", {})),
        "verdictCounts": dict(case.get("verdictCounts", {})),
        "findings": [
            {"id": f["id"], "severity": f["severity"], "records": sorted(f["records"])}
            for f in sorted(
                case.get("findings", []), key=lambda f: (f["id"], sorted(f["records"]))
            )
        ],
    }


def run_self_test(corpus_dir: Path, manifest: dict[str, Any]) -> SelfTest:
    """Reproduce every recorded verdict the corpus carries, suite by suite.

    A suite is run by its shape: a ``records/`` directory means a per-record
    suite, a ``sets/`` directory means a set-level suite. A suite the runner
    cannot place is reported as not runnable and fails the self-test honestly,
    rather than being silently skipped.
    """
    results: list[SuiteResult] = []
    for name in manifest["suites"]:
        suite_dir = corpus_dir / name
        expected_path = suite_dir / "expected.json"
        if not expected_path.is_file():
            results.append(SuiteResult(name, False, 0, 0, ("no expected.json",)))
            continue
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        if (suite_dir / "records").is_dir():
            results.append(_record_suite_result(name, suite_dir, expected))
        elif (suite_dir / "sets").is_dir():
            results.append(_set_suite_result(name, suite_dir, expected))
        else:
            results.append(SuiteResult(name, False, len(expected), 0, ("unknown suite shape",)))

    cases = sum(s.cases for s in results)
    reproduced = sum(s.reproduced for s in results)
    conforms = bool(results) and all(s.runnable and not s.mismatches for s in results)
    return SelfTest(conforms, cases, reproduced, tuple(results))


# ── Records ───────────────────────────────────────────────────────────────────


def _records_result(
    records: Sequence[tuple[str, Any]], unreadable: Sequence[tuple[str, str]]
) -> RecordsResult:
    report = check_record_set(records)
    nonconforming = tuple(
        (e.name, e.required_failed) for e in report.entries if not e.conforms
    )
    return RecordsResult(
        conforms=report.conforms and not unreadable,
        total=report.total,
        conforming=report.conforming,
        findings=report.findings,
        nonconforming=nonconforming,
        unreadable=tuple(unreadable),
    )


# ── Build ─────────────────────────────────────────────────────────────────────


def build_conformance_statement(
    corpus_dir: Path,
    *,
    records: Optional[Sequence[tuple[str, Any]]] = None,
    unreadable: Sequence[tuple[str, str]] = (),
    as_of: Optional[str] = None,
) -> ConformanceStatement:
    """Build a conformance statement for ``corpus_dir``.

    Verifies the corpus integrity, runs the self-test, and (when ``records`` is
    given) the keyless set check over the emitter's own records. The statement
    conforms iff the corpus bytes verify, the self-test reproduced every verdict,
    and any supplied records conform. Raises :class:`ConformanceCorpusError`
    when the directory holds no readable corpus manifest.
    """
    manifest = _load_manifest(corpus_dir)
    corpus = verify_corpus_integrity(corpus_dir, manifest)
    self_test = run_self_test(corpus_dir, manifest)

    records_result = (
        _records_result(records, unreadable) if records is not None else None
    )

    # One source of truth: the statement conforms exactly when every in-scope
    # check reached a verdict and that verdict held. An unreachable check leaves
    # the statement not-conforming without claiming the property is false.
    provisional = ConformanceStatement(corpus, self_test, records_result, False, as_of)
    return ConformanceStatement(
        corpus, self_test, records_result, provisional.grade == PROVED, as_of
    )


# ── Render ────────────────────────────────────────────────────────────────────

_VERDICT_WORD = {
    PROVED: "CONFORMS",
    UNPROVED: "UNPROVED",
    FALSE: "NON-CONFORMING",
}

_HOW = (
    "This statement is keyless and reproducible. Anyone holding the same corpus "
    "version can re-run `vaara conformance-statement` and reach the same verdict. "
    "It covers the wire schema, the record's self-proving digest, and the "
    "cross-record set properties; it is not signature verification, issuer trust, "
    "or time-anchor verification, which need external material and are checked "
    "separately."
)


def render_conformance_statement(statement: ConformanceStatement) -> str:
    """Render a conformance statement as a one-page Markdown document.

    Deterministic: the page depends only on ``statement`` (no clock unless the
    caller passed an ``as_of`` date, which is echoed verbatim), so the same
    inputs render byte-identical every time.
    """
    c = statement.corpus
    verdict = _VERDICT_WORD[statement.grade]
    lines: list[str] = ["# SEP-2828 conformance statement", ""]
    lines.append(f"**Statement: {verdict}**")
    lines.append("")
    if statement.grade == UNPROVED:
        lines.append(
            "At least one check could not be reached, so this statement does not "
            "establish the property either way. That is a different result from a "
            "check that ran and failed, and the sections below name which."
        )
        lines.append("")
    lines.append(
        f"Checked against corpus `{_safe(c.name)}` version {_safe(c.version)} "
        f"(corpusDigest `{_safe(c.corpus_digest)}`)."
    )
    if statement.as_of is not None:
        lines.append(f"As of {_safe(statement.as_of)}.")
    lines.append("")

    lines.append("## Corpus integrity")
    lines.append("")
    if c.verified:
        lines.append(
            f"Verified: all {c.file_count} fixture files match `MANIFEST.json` "
            "and the corpusDigest recomputes."
        )
    else:
        if c.line_endings_only:
            _render_line_ending_diagnosis(c, lines)
        else:
            lines.append(f"NOT verified: {len(c.problems)} problem(s) with the corpus bytes.")
            for p in c.problems:
                lines.append(f"- {_safe(p)}")
            if c.line_ending_mismatches:
                lines.append("")
                lines.append(
                    f"Of these, {len(c.line_ending_mismatches)} differ by line endings "
                    "only and their content is intact; see the note on "
                    "`core.autocrlf` in the corpus README."
                )
    lines.append("")

    st = statement.self_test
    lines.append("## Self-test")
    lines.append("")
    state = "reproduced" if st.conforms else "did NOT reproduce"
    lines.append(
        f"This implementation's keyless conformance check {state} "
        f"{st.reproduced} of {st.cases} recorded verdicts."
    )
    if not c.verified:
        # Cuts the mirror-image misreading: a clean self-test over bytes that
        # were never pinned is not a pass against the published corpus either.
        lines.append("")
        lines.append(
            "Read this against the corpus integrity section above: these cases "
            "came from the files on disk, which did not match the published "
            "manifest, so the count is not a claim about the published corpus."
        )
    lines.append("")
    for s in st.suites:
        if not s.runnable:
            lines.append(
                f"- `{s.name}`: unproved, the suite could not be run "
                f"({_names(s.mismatches)}). No verdict is claimed for it."
            )
        else:
            tail = "" if not s.mismatches else f"; mismatched {_names(s.mismatches)}"
            lines.append(f"- `{s.name}`: {s.reproduced}/{s.cases} reproduced{tail}")
    lines.append("")

    if statement.records is not None:
        _render_records(statement.records, lines)

    lines.append("---")
    lines.append("")
    lines.append(_HOW)
    lines.append("")
    return "\n".join(lines)


def _render_line_ending_diagnosis(c: CorpusIntegrity, lines: list[str]) -> None:
    """Say plainly that the checkout rewrote the newlines and nothing else.

    Every one of these files carries the published content. Listing them one by
    one would repeat a single fact many times and bury it; the count and the
    cause are what a reader needs, and both are stated exactly.
    """
    n = len(c.line_ending_mismatches)
    lines.append(
        f"NOT verified: all {n} fixture file{_s(n)} differ from `MANIFEST.json` "
        "by line endings only, so the corpusDigest does not recompute either. "
        "The content is the published content, byte for byte, once the newlines "
        "are undone."
    )
    lines.append("")
    lines.append(
        "This is a checkout artefact, not a problem with the corpus or with any "
        "emitter. Git for Windows installs with `core.autocrlf=true`, which "
        "rewrites LF to CRLF on checkout. The corpus is pinned byte for byte, so "
        "that translation breaks every digest while changing no content."
    )
    lines.append("")
    lines.append("To check out the corpus verbatim:")
    lines.append("")
    lines.append("```")
    lines.append("git config core.autocrlf false")
    lines.append("git rm --cached -r .")
    lines.append("git reset --hard")
    lines.append("```")
    lines.append("")
    lines.append(
        "Nothing above is a disagreement with SEP-2828. This statement is "
        "UNPROVED because the published bytes were not present to check "
        "against, not because a check ran and failed."
    )


def _render_records(r: RecordsResult, lines: list[str]) -> None:
    lines.append("## Your records")
    lines.append("")
    rv = {
        PROVED: "CONFORM",
        FALSE: "do NOT conform",
        UNPROVED: "are unproved: what was read conformed, but the set was not complete",
    }[r.grade]
    if r.total == 0:
        lines.append(
            "No readable records were supplied to this run, so nothing is established."
        )
    else:
        lines.append(
            f"{r.total} record{_s(r.total)} checked, {r.conforming} "
            f"conform{'s' if r.conforming == 1 else ''}; your records {rv}."
        )
    lines.append("")

    if r.findings:
        required = [f for f in r.findings if f.severity == "required"]
        advisory = [f for f in r.findings if f.severity == "advisory"]
        if required:
            lines.append("Required (these gate conformance):")
            lines.append("")
            for f in required:
                lines.append(f"- **{f.id}**: {_safe(f.detail)} ({_names(f.records)})")
            lines.append("")
        if advisory:
            lines.append("Advisory (gaps that do not gate conformance):")
            lines.append("")
            for f in advisory:
                lines.append(f"- **{f.id}**: {_safe(f.detail)} ({_names(f.records)})")
            lines.append("")

    if r.nonconforming:
        lines.append("Non-conforming records:")
        lines.append("")
        for name, rf in r.nonconforming:
            why = ", ".join(rf) if rf else "did not conform"
            lines.append(f"- `{_safe(name)}`: {why}")
        lines.append("")

    if r.unreadable:
        names = ", ".join(_safe(n) for n, _ in r.unreadable)
        lines.append(
            f"> Unproved: {len(r.unreadable)} file(s) could not be read, so the set "
            f"check never saw them and claims nothing about them: {names}"
        )
        lines.append("")


def _s(n: int) -> str:
    return "" if n == 1 else "s"


def _names(records: Sequence[str]) -> str:
    # Record names can be foreign (a filename under --records, a corpus case
    # name); escape control characters so a crafted name cannot forge a line.
    return ", ".join(_safe(n) for n in records)


def _safe(value: str) -> str:
    """Escape C0 control characters so a foreign value cannot forge a line.

    Record names and an ``as_of`` value can come from outside the corpus; this
    keeps a crafted newline from injecting extra Markdown lines into the page.
    """
    return "".join(ch if ch.isprintable() else f"\\x{ord(ch):02x}" for ch in value)
