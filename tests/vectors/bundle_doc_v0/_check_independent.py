#!/usr/bin/env python3
"""Independent conformance checker for the bundle_doc_v0 vectors.

Imports only the standard library plus ``rfc8785`` and ``cryptography`` (the
latter two transitively, through the sibling ``evidence_bundle_v0`` checker it
reuses). It does not import Vaara.

Each committed file in ``bundles/`` is one self-contained evidence-bundle
document, the exact on-disk shape ``vaara verify-bundle`` reads. This checker
reads each file on its own and reproduces the single ``verify_evidence_bundle``
verdict, then compares it to ``expected.json``. It is the file-level twin of
``evidence_bundle_v0/_check_independent.py``: that one parses the aggregate
``cases.json``, this one parses the individual files a verifier is actually
handed.

The verdict logic is identical, so rather than copy it this checker imports
``_evaluate`` from the sibling checker by path. That module is itself
Vaara-free (standard library plus ``rfc8785`` and ``cryptography``), so the
whole chain stays independent of Vaara. ``expected.json`` carries the
reference verdict per file name; exit 0 means every file matched it.

Run: ``python tests/vectors/bundle_doc_v0/_check_independent.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLES = HERE / "bundles"
SIBLING = HERE.parent / "evidence_bundle_v0" / "_check_independent.py"


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("evidence_bundle_v0_check", SIBLING)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling checker at {SIBLING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._evaluate


def main() -> int:
    evaluate = _load_evaluate()
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for name in sorted(expected):
        bundle = json.loads((BUNDLES / f"{name}.json").read_text())
        got = evaluate(bundle)
        want = expected[name]
        if got != want:
            failures.append(f"{name}: expected {want}, got {got}")
        else:
            print(f"{name}: OK ok={got['ok']}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall bundle_doc_v0 files matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
