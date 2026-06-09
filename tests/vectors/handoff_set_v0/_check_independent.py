#!/usr/bin/env python3
"""Independent checker for the handoff_set_v0 vectors.

Imports only the standard library plus the single-verb suite's own Vaara-free
evaluator. It does not import Vaara. A set is a list of
``cross_org_handoff_v0`` case names plus the mode the batch applies uniformly:
``strict`` and an optional set-level trusted DID document. For each set this
checker:

1. Runs every referenced case through the single suite's ``_evaluate`` (the
   Vaara-free reproduction of one ``verify_handoff`` verdict), re-applying the
   set-level ``strict`` and the one trusted document the batch pins every
   package against, while keeping each package's own pre-verified anchor time.
2. Rolls the per-case verdicts up the same way ``check_handoff_set`` does: the
   pass count, the verifiable and corroborated tiers, the producer-pin count,
   and the pinning-coverage gap (no package pinned its producer).

The summary is compared against ``expected.json``. The single suite already
proves each verdict Vaara-free; this proves the roll-up Vaara-free, so the chain
from package bytes to set verdict never touches Vaara. Run:
``python tests/vectors/handoff_set_v0/_check_independent.py``.
Exit code 0 means every set matched its expected summary.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "cross_org_handoff_v0"

SUMMARY_KEYS = (
    "ok", "strict", "total", "loaded", "passed", "verifiable",
    "corroborated", "pinned", "pinningGap",
)


def _load_single_evaluator():
    spec = importlib.util.spec_from_file_location(
        "_handoff_single_checker", SIBLING / "_check_independent.py")
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError("cannot load the single-verb checker")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._evaluate


def _rollup(verdicts: list[dict], *, strict: bool) -> dict:
    total = len(verdicts)
    loaded = total
    passed = sum(1 for v in verdicts if v["ok"])
    verifiable = sum(1 for v in verdicts if v["verifiable"])
    corroborated = sum(1 for v in verdicts if v["corroborated"])
    pinned = sum(1 for v in verdicts
                 if v["producer_identity_basis"] == "pinned")
    return {
        "ok": passed == total,
        "strict": strict,
        "total": total,
        "loaded": loaded,
        "passed": passed,
        "verifiable": verifiable,
        "corroborated": corroborated,
        "pinned": pinned,
        "pinningGap": loaded > 0 and pinned == 0,
    }


def main() -> int:
    evaluate = _load_single_evaluator()
    cases = {c["name"]: c
             for c in json.loads((SIBLING / "cases.json").read_text())["cases"]}
    sets = json.loads((HERE / "sets.json").read_text())["sets"]
    expected = json.loads((HERE / "expected.json").read_text())

    failures = []
    for set_name, spec in sets.items():
        trusted = None
        if spec["trusted_from_case"] is not None:
            trusted = cases[spec["trusted_from_case"]]["trustedDidDocument"]
        verdicts = []
        for case_name in spec["cases"]:
            case = dict(cases[case_name])
            # Re-apply the set-level mode: one strict flag and one trusted
            # document across the whole set; keep the package's own anchor time.
            case["strict"] = spec["strict"]
            if trusted is not None:
                case["trustedDidDocument"] = trusted
            else:
                case.pop("trustedDidDocument", None)
            verdicts.append(evaluate(case))
        got = _rollup(verdicts, strict=spec["strict"])
        want = {k: expected[set_name][k] for k in SUMMARY_KEYS}
        if {k: got[k] for k in SUMMARY_KEYS} != want:
            failures.append(f"{set_name}:\n    expected {want}\n    got      {got}")
        else:
            print(f"{set_name}: OK ok={got['ok']} passed={got['passed']}/{got['total']} "
                  f"corroborated={got['corroborated']} pinned={got['pinned']}")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall handoff_set_v0 sets matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
