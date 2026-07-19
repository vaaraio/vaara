# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-record findings for a set of SEP-2828 records.

The properties that only show up across records, factored out of
``_record_set_conformance`` so each stays under one screen. Every
function takes already-conforming records (the linkage fields of a
malformed record cannot be trusted) and returns ``SetFinding`` rows.

Two record types pair here. A decision record (``decisionDerived``:
allow / block / escalate, before the act) and an outcome record
(``outcomeDerived``: executed / refused / errored, after) for the same
call share the same ``backLink``. That shared key is what lets an auditor
ask the completeness question with no signing key: was every authorised
action recorded, and was every recorded action authorised?

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from vaara.attestation._receipt_conformance import ADVISORY, REQUIRED

# Verdicts that authorise the call to proceed, so a missing outcome is a
# real gap. A ``block`` decision legitimately has no outcome.
_ACTING_VERDICTS = frozenset({"allow", "escalate"})


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


def call_key(doc: Any) -> tuple[str, str]:
    """The ``(attestationDigest, attestationNonce)`` a conforming record pins.

    Safe to read only for records that already passed conformance, where
    ``backLink`` is an object with both string fields. Both decision and
    outcome records carry it, which is what makes them pairable.
    """
    bl = doc["backLink"]
    return bl["attestationDigest"], bl["attestationNonce"]


def _by_call(records: Sequence[tuple[str, Any]]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for name, doc in records:
        out.setdefault(call_key(doc), []).append(name)
    return out


def duplicate_outcome_findings(outcomes: Sequence[tuple[str, Any]]) -> list[SetFinding]:
    """An executed/refused outcome recorded twice for one call: a replay or double-write."""
    findings: list[SetFinding] = []
    for (digest, nonce), names in _by_call(outcomes).items():
        if len(names) > 1:
            findings.append(SetFinding(
                "duplicate_call", REQUIRED,
                f"{len(names)} outcome records pin the same call "
                f"(attestationDigest {digest}, nonce {nonce}); "
                "a call MUST be recorded once",
                tuple(sorted(names)),
            ))
    return findings


def pairing_findings(
    decisions: Sequence[tuple[str, Any]],
    outcomes: Sequence[tuple[str, Any]],
) -> list[SetFinding]:
    """The completeness gaps: an authorised act with no outcome, an act with no decision."""
    decision_calls = _by_call(decisions)
    outcome_calls = _by_call(outcomes)

    acting = {
        key: names for key, names in decision_calls.items()
        if any(_verdict(d) in _ACTING_VERDICTS for n, d in decisions if call_key(d) == key)
    }
    no_outcome = sorted(
        n for key, names in acting.items() if key not in outcome_calls for n in names
    )
    no_decision = sorted(
        n for key, names in outcome_calls.items() if key not in decision_calls for n in names
    )

    findings: list[SetFinding] = []
    if no_outcome:
        findings.append(SetFinding(
            "decision_without_outcome", ADVISORY,
            f"{len(no_outcome)} allow/escalate decisions have no matching outcome "
            "record; an authorised action SHOULD leave a recorded outcome",
            tuple(no_outcome),
        ))
    if no_decision:
        findings.append(SetFinding(
            "outcome_without_decision", ADVISORY,
            f"{len(no_decision)} outcome records have no matching decision record; "
            "a recorded action SHOULD trace to the decision that authorised it",
            tuple(no_decision),
        ))
    return findings


def outcome_coverage_findings(outcomes: Sequence[tuple[str, Any]]) -> list[SetFinding]:
    """An executed action that committed no result: the Article 12 hole."""
    uncovered = sorted(
        name for name, doc in outcomes
        if doc["outcomeDerived"].get("status") == "executed"
        and doc["outcomeDerived"].get("resultCommitment") is None
    )
    if not uncovered:
        return []
    return [SetFinding(
        "executed_without_result_commitment", ADVISORY,
        f"{len(uncovered)} executed records carry no resultCommitment; "
        "an executed action SHOULD commit to its result",
        tuple(uncovered),
    )]


def _verdict(doc: Any) -> Any:
    return doc["decisionDerived"].get("decision")
