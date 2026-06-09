#!/usr/bin/env python3
"""Generate the handoff_set_v0 vectors from the single-verb corpus.

A set is a list of ``cross_org_handoff_v0`` case names plus the mode the batch
applies uniformly: ``strict`` and an optional set-level trusted DID document
(named by a case whose ``trustedDidDocument`` is reused). This script reuses the
single suite's committed packages, groups them into sets, and records the
roll-up ``check_handoff_set`` produces. Each package's anchor time is taken from
its case's pre-verified ``anchoredTime`` (the timeanchor token itself is not
re-verified here, exactly as the single suite does). ``expected.json`` is the
committed truth the in-process test (Vaara) and ``_check_independent.py``
(Vaara-free) both reproduce.

Run: ``python tests/vectors/handoff_set_v0/_generate.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from vaara.attestation.receipt import check_handoff_set

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "cross_org_handoff_v0"

# Each set names cases from the single-verb suite, the set-level strict flag, and
# (optionally) the case whose trustedDidDocument the batch pins every package
# against. A single out-of-band identity reference applies to the whole set, so
# only packages whose bound key appears in it pin.
SETS: dict[str, dict] = {
    # Two clean default-mode records, one anchor-corroborated. Passes; no
    # producer is pinned (no trusted document), so the pinning gap fires.
    "all_verifiable": {
        "cases": ["clean_no_anchor", "corroborated"],
        "strict": False,
        "trusted_from_case": None,
    },
    # A clean record beside one signed after the key retired: the set fails.
    "with_failure": {
        "cases": ["clean_no_anchor", "signed_after_retirement"],
        "strict": False,
        "trusted_from_case": None,
    },
    # Strict, with the producer pinned against a trusted archive: a corroborated,
    # pinned record passes the regulator-grade bar.
    "strict_pinned": {
        "cases": ["pinned_corroborated"],
        "strict": True,
        "trusted_from_case": "pinned_corroborated",
    },
    # Strict without a trusted document: a corroborated record still fails strict
    # because its producer identity is self-asserted.
    "strict_unmet": {
        "cases": ["corroborated"],
        "strict": True,
        "trusted_from_case": None,
    },
}

SUMMARY_KEYS = (
    "ok", "strict", "total", "loaded", "passed", "verifiable",
    "corroborated", "pinned", "pinningGap",
)


def main() -> int:
    cases = {c["name"]: c
             for c in json.loads((SIBLING / "cases.json").read_text())["cases"]}
    expected: dict[str, dict] = {}
    for set_name, spec in SETS.items():
        trusted = None
        if spec["trusted_from_case"] is not None:
            trusted = cases[spec["trusted_from_case"]]["trustedDidDocument"]
        packages = []
        for case_name in spec["cases"]:
            case = cases[case_name]
            package = case["package"]
            anchor_present = package.get("evidence", {}).get("anchor") is not None
            anchor_time = case.get("anchoredTime") if anchor_present else None
            packages.append((case_name, package, anchor_time))
        report = check_handoff_set(
            packages, trusted_did_document=trusted, strict=spec["strict"])
        d = report.to_dict()
        expected[set_name] = {k: d[k] for k in SUMMARY_KEYS}

    sets_doc = {
        "sets": {
            name: {
                "cases": spec["cases"],
                "strict": spec["strict"],
                "trusted_from_case": spec["trusted_from_case"],
            }
            for name, spec in SETS.items()
        }
    }
    (HERE / "sets.json").write_text(json.dumps(sets_doc, indent=2, sort_keys=True) + "\n")
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(SETS)} sets to sets.json and expected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
