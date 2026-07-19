# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Set-level conformance for a pile of SEP-2828 records.

``check_record_conformance`` answers a question about one record. An
auditor who receives a directory of records, possibly from more than one
emitter, asks a different question: over the whole set, how many conform,
and where are the gaps in the chain? This is the receiving side of the
evidence, and it needs no signing key either.

A set is not all one shape. SEP-2828 carries two record types that pair:
a **decision** record (``decisionDerived``: allow / block / escalate,
before the act) and an **outcome** record (``outcomeDerived``: executed /
refused / errored, after). Each is classified by which derived block it
carries and checked against its own schema; a record carrying neither (or
both) is an unknown shape and fails. The shared ``backLink`` pairs the two
with no signing key.

Cross-record properties stack on top of the per-record checks:

* **Unique calls** (required). Two *outcome* records pinning the same
  call recorded it twice, a replay or double-write. A decision and an
  outcome on one call are the expected *pair*, so this looks at outcomes.
* **Pairing coverage** (advisory). In a set holding both kinds, an
  allow/escalate decision with no matching outcome, or an outcome with no
  matching decision: was every authorised action recorded, and was every
  recorded action authorised?
* **Outcome coverage** (advisory). An ``executed`` outcome with no
  ``resultCommitment``: the Article 12 hole.

Cross-record checks run only over records that individually conform. The
set conforms iff every record conforms and no required finding fired; the
advisory gaps are reported but do not gate.

Pure standard library; importable without the ``attestation`` extra.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from vaara.attestation._decision_conformance import check_decision_conformance
from vaara.attestation._receipt_conformance import (
    REQUIRED,
    check_record_conformance,
)
from vaara.attestation._record_set_findings import (
    SetFinding,
    duplicate_outcome_findings,
    outcome_coverage_findings,
    pairing_findings,
)

SET_SCHEMA_NAME = "sep2828-execution-record-set"
SET_SCHEMA_VERSION = 1

RecordKind = Literal["decision", "outcome", "unknown"]


@dataclass(frozen=True)
class RecordSetEntry:
    """Per-record verdict inside a set."""

    name: str
    kind: RecordKind
    conforms: bool
    required_failed: tuple[str, ...]
    advisories: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "conforms": self.conforms,
            "requiredFailed": list(self.required_failed),
            "advisories": list(self.advisories),
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
    verdict_counts: dict[str, int]

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
            "verdictCounts": dict(self.verdict_counts),
            "findings": [f.to_dict() for f in self.findings],
            "entries": [e.to_dict() for e in self.entries],
        }


def classify_record(doc: Any) -> RecordKind:
    """Sort a candidate into ``decision`` / ``outcome`` / ``unknown``.

    A SEP-2828 record carries exactly one derived block: ``decisionDerived``
    for the verdict before the act, ``outcomeDerived`` for what happened
    after. A record carrying neither, or both, is an unknown shape that no
    per-type schema can judge, so it cannot conform.
    """
    if not isinstance(doc, dict):
        return "unknown"
    has_decision = isinstance(doc.get("decisionDerived"), dict)
    has_outcome = isinstance(doc.get("outcomeDerived"), dict)
    if has_decision and not has_outcome:
        return "decision"
    if has_outcome and not has_decision:
        return "outcome"
    return "unknown"


def check_record_set(records: Sequence[tuple[str, Any]]) -> RecordSetReport:
    """Check a set of ``(name, parsed_json)`` candidates together.

    Each record is classified and conformance-checked against its own
    type's schema; the set is then checked for cross-record properties over
    the records that conform. Never raises on a malformed record. The set
    conforms iff every record conforms and no required-severity finding
    fired (the advisory pairing gaps are reported, not gating).
    """
    entries: list[RecordSetEntry] = []
    decisions: list[tuple[str, Any]] = []
    outcomes: list[tuple[str, Any]] = []

    for name, doc in records:
        kind = classify_record(doc)
        if kind == "decision":
            report = check_decision_conformance(doc)
        elif kind == "outcome":
            report = check_record_conformance(doc)
        else:
            entries.append(RecordSetEntry(name, "unknown", False, ("record_type",), ()))
            continue

        entries.append(
            RecordSetEntry(
                name=name,
                kind=kind,
                conforms=report.conforms,
                required_failed=report.required_failed,
                advisories=report.advisories,
            )
        )
        if report.conforms:
            (decisions if kind == "decision" else outcomes).append((name, doc))

    findings: list[SetFinding] = []
    findings.extend(duplicate_outcome_findings(outcomes))
    # Pairing is meaningful only when the set holds both kinds; a set of
    # outcomes alone (or decisions alone) has nothing to pair against, and
    # flagging every record as unpaired would be noise, not a gap.
    if decisions and outcomes:
        findings.extend(pairing_findings(decisions, outcomes))
    findings.extend(outcome_coverage_findings(outcomes))
    findings.sort(key=lambda f: (f.id, f.records))

    all_conform = all(e.conforms for e in entries)
    no_required = not any(f.severity == REQUIRED for f in findings)

    return RecordSetReport(
        conforms=all_conform and no_required,
        total=len(entries),
        conforming=len(decisions) + len(outcomes),
        entries=tuple(entries),
        findings=tuple(findings),
        status_counts=_status_counts(outcomes),
        verdict_counts=_verdict_counts(decisions),
    )


def _status_counts(outcomes: Sequence[tuple[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for _name, doc in outcomes:
        counter[doc["outcomeDerived"]["status"]] += 1
    return dict(sorted(counter.items()))


def _verdict_counts(decisions: Sequence[tuple[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for _name, doc in decisions:
        counter[doc["decisionDerived"]["decision"]] += 1
    return dict(sorted(counter.items()))
