#!/usr/bin/env python3
"""Generate the bundle_doc_v0 conformance vectors.

These vectors cover the *on-disk* evidence-bundle document: the single JSON
file a holder hands to a verifier and that ``vaara verify-bundle`` reads. The
document shape is identical to the ``bundle`` object the ``evidence_bundle_v0``
vectors already commit, so this generator derives the standalone files from
those cases rather than re-signing anything: each case's ``bundle`` becomes
``bundles/<name>.json`` and its ``expected`` verdict goes into
``expected.json`` keyed by the same name.

The point under test is that one self-contained file on disk is verifiable on
its own. The independent checker (``_check_independent.py``) reads each file
and reproduces its verdict with no Vaara import.

Run from the repo root: ``python tests/vectors/bundle_doc_v0/_generate.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "evidence_bundle_v0" / "cases.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    bundles_dir = HERE / "bundles"
    bundles_dir.mkdir(exist_ok=True)

    expected: dict[str, dict] = {}
    for case in source["cases"]:
        name = case["name"]
        (bundles_dir / f"{name}.json").write_text(
            json.dumps(case["bundle"], indent=2, sort_keys=True) + "\n"
        )
        expected[name] = case["expected"]

    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(expected)} standalone bundle documents to bundles/")


if __name__ == "__main__":
    main()
