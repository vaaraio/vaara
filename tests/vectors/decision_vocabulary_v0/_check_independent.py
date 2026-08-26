#!/usr/bin/env python3
"""Independent checker for the v0 decision-vocabulary vectors.

Standard library only. It does not import Vaara. For each committed case it
reads the exported audit records, recomputes every record hash and every chain
link from the bytes, and re-derives the vocabulary verdict, then compares
against ``expected.json``.

What is being checked, and why a stranger should care:

A gate that offers five decisions has to answer a question the marketing does
not: when the policy said something finer than allow, deny or escalate, what
did the permanent record end up saying? Two failures are possible and both are
invisible from the outside without vectors. The record can carry a word no
consumer knows, which quietly breaks every checker holding the closed verdict
set. Or the record can say ``allow`` against arguments that were never the ones
that ran, which is worse, because it reads as clean evidence of something that
did not happen.

So this checker enforces three things from the bytes alone:

1. every decision record's verdict is one of ``allow`` / ``deny`` / ``escalate``,
2. a refinement, where present, is one of ``modify`` / ``step_up`` / ``defer``
   and agrees with the verdict recorded beside it, and
3. where a modify proposed different arguments and the caller retried, the
   retry's arguments are exactly the proposed ones, digest for digest.

Rule 3 is the one that matters. A modify blocks what it was asked about and the
retry is decided on its own, so no record is ever bound to arguments other than
the ones it decided.

Run: ``python tests/vectors/decision_vocabulary_v0/_check_independent.py``.
Exit 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The verdicts a decision record may carry. Closed, and closed on purpose: this
# is the same set the SEP-2828 record checkers hold.
VALID_VERDICTS = {"allow", "deny", "escalate"}

# The refinements a policy layer may name, and the verdict each must record.
REFINEMENT_TO_VERDICT = {
    "modify": "deny",
    "step_up": "escalate",
    "defer": "escalate",
}

DECISION_EVENTS = {"decision_made", "action_blocked"}


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def recompute_hash(record) -> str:
    """Reproduce the record hash from the record's own fields.

    Mirrors the documented hashing surface: the identity and content fields
    plus the regulatory provenance and the previous link, with the tenant and
    chain version bound in from chain version 2. The four transparency
    annotations are deliberately outside the hash and stay outside here.
    """
    content = {
        "record_id": record.get("record_id"),
        "action_id": record.get("action_id"),
        "event_type": record.get("event_type"),
        "timestamp": record.get("timestamp"),
        "agent_id": record.get("agent_id"),
        "tool_name": record.get("tool_name"),
        "data": record.get("data"),
        "regulatory_articles": record.get("regulatory_articles"),
        "previous_hash": record.get("previous_hash"),
    }
    if record.get("chain_version", 1) >= 2:
        content["tenant_id"] = record.get("tenant_id")
        content["chain_version"] = record.get("chain_version")
    return hashlib.sha256(_canonical(content)).hexdigest()


def _chain_findings(records) -> list:
    findings = []
    broken = []
    previous = ""
    for record in records:
        rid = record.get("record_id")
        if recompute_hash(record) != record.get("record_hash"):
            broken.append(rid)
        elif record.get("previous_hash") != previous:
            broken.append(rid)
        previous = record.get("record_hash")
    if broken:
        findings.append({"id": "chain_break", "severity": "required",
                         "records": sorted(broken)})
    return findings


def _vocabulary_findings(decisions) -> list:
    findings = []
    unknown_verdict, unknown_refinement = [], []
    mismatched, orphan_params = [], []

    for record in decisions:
        rid = record.get("record_id")
        data = record.get("data") or {}
        verdict = data.get("decision")
        detail = data.get("decision_detail")

        if verdict not in VALID_VERDICTS:
            unknown_verdict.append(rid)
        if detail is not None:
            if detail not in REFINEMENT_TO_VERDICT:
                unknown_refinement.append(rid)
            elif REFINEMENT_TO_VERDICT[detail] != verdict:
                mismatched.append(rid)
        if "modified_parameters" in data and detail != "modify":
            orphan_params.append(rid)

    for finding_id, records in (
        ("unknown_verdict", unknown_verdict),
        ("unknown_refinement", unknown_refinement),
        ("refinement_contradicts_verdict", mismatched),
        ("proposed_arguments_without_modify", orphan_params),
    ):
        if records:
            findings.append({"id": finding_id, "severity": "required",
                             "records": sorted(records)})
    return findings


def _rebinding_findings(records, decisions) -> list:
    """A retry must carry exactly the arguments the modify proposed."""
    findings = []
    unbound, no_retry = [], []

    requests_by_parent = {}
    for record in records:
        if record.get("event_type") != "action_requested":
            continue
        parent = (record.get("data") or {}).get("parent_action_id")
        if parent:
            requests_by_parent.setdefault(parent, []).append(record)

    for record in decisions:
        data = record.get("data") or {}
        if data.get("decision_detail") != "modify":
            continue
        proposed = data.get("modified_parameters")
        retries = requests_by_parent.get(record.get("action_id"), [])
        if not retries:
            no_retry.append(record.get("record_id"))
            continue
        want = _digest(proposed)
        for retry in retries:
            got = _digest((retry.get("data") or {}).get("parameters"))
            if got != want:
                unbound.append(retry.get("record_id"))

    if unbound:
        findings.append({"id": "retry_not_bound_to_proposed_arguments",
                         "severity": "required", "records": sorted(unbound)})
    if no_retry:
        findings.append({"id": "modify_without_retry", "severity": "advisory",
                         "records": sorted(no_retry)})
    return findings


def check_case(records) -> dict:
    """Reproduce the case verdict from the exported records."""
    decisions = [r for r in records
                 if r.get("event_type") in DECISION_EVENTS]

    findings = (_chain_findings(records)
                + _vocabulary_findings(decisions)
                + _rebinding_findings(records, decisions))
    findings.sort(key=lambda f: (f["id"], f["records"]))

    verdicts: dict = {}
    refinements: dict = {}
    for record in decisions:
        data = record.get("data") or {}
        verdict = data.get("decision")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        detail = data.get("decision_detail")
        if detail is not None:
            refinements[detail] = refinements.get(detail, 0) + 1

    return {
        "conforms": not any(f["severity"] == "required" for f in findings),
        "records": len(records),
        "decisions": len(decisions),
        "verdicts": dict(sorted(verdicts.items())),
        "refinements": dict(sorted(refinements.items())),
        "findings": findings,
    }


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text())
    failures = 0
    for name in sorted(expected):
        path = HERE / "cases" / f"{name}.json"
        got = check_case(json.loads(path.read_text()))
        ok = got == expected[name]
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: "
              f"{got['decisions']} decision(s) in {got['records']} records")
        if not ok:
            print("  want:", json.dumps(expected[name], sort_keys=True))
            print("  got :", json.dumps(got, sort_keys=True))
    print(f"\n{len(expected) - failures}/{len(expected)} cases matched expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
