#!/usr/bin/env python3
"""Independent batch full-lens checker for the bundle_set_v0 vectors.

A second implementation of the set-level roll-up that ``vaara verify-bundles``
produces: load each evidence-bundle document in a set, run the full lens stack,
and aggregate. It does not import Vaara. The per-bundle verdict logic is the
same one the ``bundle_doc_v0`` checker uses, so rather than copy it this file
reuses ``_evaluate`` from the ``evidence_bundle_v0`` sibling checker by path.
That module is itself Vaara-free (standard library plus ``rfc8785`` and
``cryptography``), so the whole chain stays independent of Vaara.

The batch question a single-file check cannot answer: over a pile of bundles,
how many verify, how many had their signature established at all, and what does
the evidence cover. The coverage gap names the lenses no bundle in the set
exercised, the evidence the whole set never carried. This checker reproduces
that roll-up from the bundles alone and compares it to ``expected.json``; exit
0 means every set matched.

Run: ``python tests/vectors/bundle_set_v0/_check_independent.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = HERE / "sets"
SIBLING = HERE.parent / "evidence_bundle_v0" / "_check_independent.py"

# The six lenses, in the order the verdict reports them.
LENS_NAMES = (
    "identity",
    "signature",
    "back_link",
    "inclusion",
    "consistency",
    "revocation",
)


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("evidence_bundle_v0_check", SIBLING)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling checker at {SIBLING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._evaluate


def check_set(evaluate, bundles):
    """Reproduce the set roll-up from (name, doc) pairs.

    Every vector document is a well-formed bundle, so no load-failure path is
    needed here; ``loaded`` always equals the bundle count. The Vaara module
    carries the malformed-document path, exercised by its unit tests.
    """
    lens_applicable = {name: 0 for name in LENS_NAMES}
    lens_passed = {name: 0 for name in LENS_NAMES}
    passed = 0
    authenticated = 0

    for _name, doc in bundles:
        verdict = evaluate(doc)
        for lens in LENS_NAMES:
            res = verdict["lenses"][lens]
            if res["applicable"]:
                lens_applicable[lens] += 1
                if res["ok"]:
                    lens_passed[lens] += 1
        if verdict["authenticity_established"]:
            authenticated += 1
        if verdict["ok"]:
            passed += 1

    total = len(bundles)
    lens_gaps = [name for name in LENS_NAMES if lens_applicable[name] == 0]
    return {
        "ok": passed == total,
        "total": total,
        "loaded": total,
        "passed": passed,
        "authenticated": authenticated,
        "lensApplicable": lens_applicable,
        "lensPassed": lens_passed,
        "lensGaps": lens_gaps,
    }


def main() -> int:
    evaluate = _load_evaluate()
    expected = json.loads((HERE / "expected.json").read_text())
    failures = 0
    for name in sorted(expected):
        files = sorted((SETS / name).glob("*.json"))
        bundles = [(p.name, json.loads(p.read_text())) for p in files]
        got = check_set(evaluate, bundles)
        ok = got == expected[name]
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {got['passed']}/{got['total']} verify")
        if not ok:
            print("  want:", expected[name])
            print("  got :", got)
    print(f"\n{len(expected) - failures}/{len(expected)} sets matched expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
