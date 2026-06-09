#!/usr/bin/env python3
"""Independent checker for the enforcement_set_v0 vectors.

Imports only the standard library plus the single-verb suite's own Vaara-free
evaluator. It does not import Vaara. A set is a list of
``enforcement_attestation_v0`` case names plus the mode the batch applies
uniformly. For each set this checker:

1. Runs every referenced case through the single suite's ``_evaluate`` (the
   Vaara-free reproduction of one ``verify_enforcement`` verdict), re-applying
   the set-level ``expected_measurement`` and ``strict`` so each verdict is
   judged under the batch's mode, not the single case's.
2. Rolls the per-case verdicts up the same way ``check_enforcement_set`` does:
   the pass count, the per-tier tally, the bound and measurement-pinned counts,
   and the pinning-coverage gap (no case pinned a measurement).

The summary is compared against ``expected.json``. The single suite already
proves each verdict Vaara-free; this proves the roll-up Vaara-free, so the chain
from report bytes to set verdict never touches Vaara. Run:
``python tests/vectors/enforcement_set_v0/_check_independent.py``.
Exit code 0 means every set matched its expected summary.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "enforcement_attestation_v0"

TIER_NAMES = ("unverified", "bound", "measurement_pinned")
SUMMARY_KEYS = (
    "ok", "strict", "total", "loaded", "passed", "bound",
    "measurementPinned", "tierCounts", "pinningGap",
)


def _load_single_evaluator():
    spec = importlib.util.spec_from_file_location(
        "_enforcement_single_checker", SIBLING / "_check_independent.py")
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError("cannot load the single-verb checker")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._evaluate


def _rollup(verdicts: list[dict], *, strict: bool) -> dict:
    """Reproduce check_enforcement_set's summary from per-case verdicts.

    Every referenced case is well-formed here, so each verdict loaded; an
    unloadable triple (a CLI-only concern: a missing companion file) is not
    reachable from committed cases and is exercised by the in-process test.
    """
    total = len(verdicts)
    loaded = total
    passed = sum(1 for v in verdicts if v["ok"])
    bound = sum(1 for v in verdicts if v["bound"])
    measurement_pinned = sum(
        1 for v in verdicts if v["measurement_basis"] == "pinned")
    tier_counts = {name: 0 for name in TIER_NAMES}
    for v in verdicts:
        tier_counts[v["tier"]] = tier_counts.get(v["tier"], 0) + 1
    return {
        "ok": passed == total,
        "strict": strict,
        "total": total,
        "loaded": loaded,
        "passed": passed,
        "bound": bound,
        "measurementPinned": measurement_pinned,
        "tierCounts": tier_counts,
        "pinningGap": loaded > 0 and measurement_pinned == 0,
    }


def main() -> int:
    evaluate = _load_single_evaluator()
    cases = {c["name"]: c
             for c in json.loads((SIBLING / "cases.json").read_text())["cases"]}
    sets = json.loads((HERE / "sets.json").read_text())["sets"]
    expected = json.loads((HERE / "expected.json").read_text())

    failures = []
    for set_name, spec in sets.items():
        verdicts = []
        for case_name in spec["cases"]:
            case = dict(cases[case_name])
            # Re-apply the set-level mode: the batch judges every case under one
            # expected_measurement and one strict flag, not the single case's.
            case["expected_measurement"] = spec["expected_measurement"]
            case["strict"] = spec["strict"]
            verdicts.append(evaluate(case))
        got = _rollup(verdicts, strict=spec["strict"])
        want = {k: expected[set_name][k] for k in SUMMARY_KEYS}
        if {k: got[k] for k in SUMMARY_KEYS} != want:
            failures.append(f"{set_name}:\n    expected {want}\n    got      {got}")
        else:
            print(f"{set_name}: OK ok={got['ok']} passed={got['passed']}/{got['total']} "
                  f"pinned={got['measurementPinned']} gap={got['pinningGap']}")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall enforcement_set_v0 sets matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
