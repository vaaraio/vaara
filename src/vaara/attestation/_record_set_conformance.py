"""Set-level conformance for a pile of SEP-2828 execution records.

``check_record_conformance`` answers a question about one record. An
auditor who receives a directory of records, possibly from more than one
emitter, asks a different question: over the whole set, how many conform,
and where are the gaps in the chain? This is the receiving side of the
evidence, and it needs no signing key either.

Two things stack here. First, every record is conformance-checked on its
own. Second, the set is checked for properties that only show up across
records:

* **Unique calls.** Each record pins the attestation it answers through
  ``backLink``. Two records pinning the same ``(attestationDigest,
  attestationNonce)`` mean the same call was recorded twice, a replay or
  a double-write. Required: a set that double-counts a call is not a
  faithful record of what happened.
* **Outcome coverage.** A record whose status is ``executed`` but that
  carries no ``resultCommitment`` is a gap: the action ran and left no
  committed evidence of its result. Advisory, not malformed, but exactly
  the hole a regulator looks for under EU AI Act Article 12.

Set-level checks run only over records that individually conform, since
the linkage fields of a malformed record cannot be trusted. The set
conforms iff every record conforms and no required finding fired.

Pure standard library; importable without the ``attestation`` extra.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from vaara.attestation._receipt_conformance import (
    ADVISORY,
    REQUIRED,
    check_record_conformance,
)

SET_SCHEMA_NAME = "sep2828-execution-record-set"
SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RecordSetEntry:
    """Per-record verdict inside a set."""

    name: str
    conforms: bool
    required_failed: tuple[str, ...]
    advisories: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "conforms": self.conforms,
            "requiredFailed": list(self.required_failed),
            "advisories": list(self.advisories),
        }


@dataclass(frozen=True)
class SetFinding:
    """A property that holds (or fails) across more than one record.

    ``required`` findings gate set conformance; ``advisory`` findings are
    reported as gaps but do not. ``records`` is sorted for a stable report.
    """

    id: str
    severity: str
    detail: str
    records: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "detail": self.detail,
            "records": list(self.records),
        }


@dataclass(frozen=True)
class RecordSetReport:
    """Outcome of checking a set of candidate records together."""

    conforms: bool
    total: int
    conforming: int
    entries: tuple[RecordSetEntry, ...]
    findings: tuple[SetFinding, ...]
    status_counts: dict[str, int]

    @property
    def required_findings(self) -> tuple[SetFinding, ...]:
        return tuple(f for f in self.findings if f.severity == REQUIRED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SET_SCHEMA_NAME,
            "schemaVersion": SET_SCHEMA_VERSION,
            "conforms": self.conforms,
            "total": self.total,
            "conforming": self.conforming,
            "statusCounts": dict(self.status_counts),
            "findings": [f.to_dict() for f in self.findings],
            "entries": [e.to_dict() for e in self.entries],
        }


def check_record_set(records: Sequence[tuple[str, Any]]) -> RecordSetReport:
    """Check a set of ``(name, parsed_json)`` candidates together.

    Each record is conformance-checked on its own; the set is then checked
    for cross-record properties over the records that conform. Never raises
    on a malformed record. The set conforms iff every record conforms and
    no required-severity finding fired.
    """
    entries: list[RecordSetEntry] = []
    conforming: list[tuple[str, Any]] = []

    for name, doc in records:
        report = check_record_conformance(doc)
        entries.append(
            RecordSetEntry(
                name=name,
                conforms=report.conforms,
                required_failed=report.required_failed,
                advisories=report.advisories,
            )
        )
        if report.conforms:
            conforming.append((name, doc))

    findings = list(_unique_call_findings(conforming))
    findings.extend(_outcome_coverage_findings(conforming))
    findings.sort(key=lambda f: (f.id, f.records))

    all_conform = all(e.conforms for e in entries)
    no_required = not any(f.severity == REQUIRED for f in findings)

    return RecordSetReport(
        conforms=all_conform and no_required,
        total=len(entries),
        conforming=len(conforming),
        entries=tuple(entries),
        findings=tuple(findings),
        status_counts=_status_counts(conforming),
    )


def _call_key(doc: Any) -> tuple[str, str]:
    """The ``(attestationDigest, attestationNonce)`` a conforming record pins.

    Safe to read only for records that already passed conformance, where
    ``backLink`` is guaranteed to be an object with both string fields.
    """
    bl = doc["backLink"]
    return bl["attestationDigest"], bl["attestationNonce"]


def _unique_call_findings(conforming: Sequence[tuple[str, Any]]) -> list[SetFinding]:
    by_call: dict[tuple[str, str], list[str]] = {}
    for name, doc in conforming:
        by_call.setdefault(_call_key(doc), []).append(name)
    findings: list[SetFinding] = []
    for (digest, nonce), names in by_call.items():
        if len(names) > 1:
            findings.append(
                SetFinding(
                    id="duplicate_call",
                    severity=REQUIRED,
                    detail=(
                        f"{len(names)} records pin the same call "
                        f"(attestationDigest {digest}, nonce {nonce}); "
                        "a call MUST be recorded once"
                    ),
                    records=tuple(sorted(names)),
                )
            )
    return findings


def _outcome_coverage_findings(
    conforming: Sequence[tuple[str, Any]],
) -> list[SetFinding]:
    uncovered = sorted(
        name
        for name, doc in conforming
        if doc["outcomeDerived"].get("status") == "executed"
        and doc["outcomeDerived"].get("resultCommitment") is None
    )
    if not uncovered:
        return []
    return [
        SetFinding(
            id="executed_without_result_commitment",
            severity=ADVISORY,
            detail=(
                f"{len(uncovered)} executed records carry no resultCommitment; "
                "an executed action SHOULD commit to its result"
            ),
            records=tuple(uncovered),
        )
    ]


def _status_counts(conforming: Sequence[tuple[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for _name, doc in conforming:
        counter[doc["outcomeDerived"]["status"]] += 1
    return dict(sorted(counter.items()))
