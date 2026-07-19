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
STATEMENT_SCHEMA_VERSION = 1


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "version": self.version,
            "corpusDigest": self.corpus_digest,
            "fileCount": self.file_count,
            "verified": self.verified,
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class SuiteResult:
    """How many of one suite's recorded verdicts this implementation reproduced."""

    name: str
    runnable: bool
    cases: int
    reproduced: int
    mismatches: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runnable": self.runnable,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "conforms": self.conforms,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "conforms": self.conforms,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATEMENT_SCHEMA,
            "schemaVersion": STATEMENT_SCHEMA_VERSION,
            "conforms": self.conforms,
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
    for rel in sorted(set(want) | set(got)):
        if rel not in got:
            problems.append(f"missing file: {rel}")
        elif rel not in want:
            problems.append(f"unexpected file: {rel}")
        elif want[rel] != got[rel]:
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

    conforms = (
        corpus.verified
        and self_test.conforms
        and (records_result is None or records_result.conforms)
    )
    return ConformanceStatement(corpus, self_test, records_result, conforms, as_of)


# ── Render ────────────────────────────────────────────────────────────────────

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
    verdict = "CONFORMS" if statement.conforms else "NON-CONFORMING"
    lines: list[str] = ["# SEP-2828 conformance statement", ""]
    lines.append(f"**Statement: {verdict}**")
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
        lines.append(f"NOT verified: {len(c.problems)} problem(s) with the corpus bytes.")
        for p in c.problems:
            lines.append(f"- {_safe(p)}")
    lines.append("")

    st = statement.self_test
    lines.append("## Self-test")
    lines.append("")
    state = "reproduced" if st.conforms else "did NOT reproduce"
    lines.append(
        f"This implementation's keyless conformance check {state} "
        f"{st.reproduced} of {st.cases} recorded verdicts."
    )
    lines.append("")
    for s in st.suites:
        if not s.runnable:
            lines.append(f"- `{s.name}`: not runnable ({_names(s.mismatches)})")
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


def _render_records(r: RecordsResult, lines: list[str]) -> None:
    lines.append("## Your records")
    lines.append("")
    rv = "CONFORM" if r.conforms else "do NOT conform"
    if r.total == 0:
        lines.append("No records were supplied to this run.")
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
        lines.append(f"> Note: {len(r.unreadable)} file(s) could not be read: {names}")
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
