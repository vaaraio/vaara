#!/usr/bin/env python3
"""Independent conformance checker for the build_bundle_v0 vectors.

Imports only the standard library plus ``rfc8785`` and ``cryptography`` (the
latter two transitively, through the ``evidence_bundle_v0`` checker it reuses
for the verdict). It does not import Vaara.

Each ``pieces/<name>/`` directory holds an issuer's separate artifacts: the
receipt and whatever identity, signature, back-link, inclusion, consistency,
and revocation material the issuer has, each in the conventional file ``vaara
build-bundle --from-dir`` reads. This checker reproduces what ``build-bundle``
does, with no Vaara import:

1. assemble the pieces into one document, rendered canonically (sorted keys,
   two-space indent), and assert it is byte-for-byte the committed
   ``documents/<name>.json`` (which is itself the ``bundle_doc_v0`` document a
   verifier reads). This is the round-trip the format promises: the issuer
   assembles exactly the file the verifier checks.
2. reproduce the single ``verify_evidence_bundle`` verdict over that assembled
   document and compare it to ``expected.json``.

Step 2 reuses ``_evaluate`` from the sibling ``evidence_bundle_v0`` checker by
path; that module is itself Vaara-free, so the whole chain stays independent.

Run: ``python tests/vectors/build_bundle_v0/_check_independent.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIECES = HERE / "pieces"
DOCUMENTS = HERE / "documents"
SIBLING = HERE.parent / "evidence_bundle_v0" / "_check_independent.py"

# Conventional piece file names, mirroring
# vaara.attestation._bundle_io.BUNDLE_PIECE_FILES / BUNDLE_PIECE_SCALARS. Held
# here as a literal so this checker needs no Vaara import.
OBJECT_PIECES = {
    "receipt.json": "receipt",
    "did_document.json": "did_document",
    "verifying_jwk.json": "verifying_jwk",
    "attestation.json": "attestation",
    "inclusion.json": "inclusion",
    "consistency.json": "consistency",
    "registry.json": "registry",
}
SCALAR_PIECES = {
    "expected_keyid.txt": "expected_keyid",
    "inclusion_leaf_hex.txt": "inclusion_leaf_hex",
}


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("evidence_bundle_v0_check", SIBLING)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling checker at {SIBLING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._evaluate


def _assemble(case_dir: Path) -> dict:
    document: dict = {}
    for filename, key in OBJECT_PIECES.items():
        path = case_dir / filename
        if path.is_file():
            document[key] = json.loads(path.read_text())
    for filename, key in SCALAR_PIECES.items():
        path = case_dir / filename
        if path.is_file():
            document[key] = path.read_text().strip()
    return document


def _render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> int:
    evaluate = _load_evaluate()
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for name in sorted(expected):
        assembled = _assemble(PIECES / name)
        rendered = _render(assembled)

        document_text = (DOCUMENTS / f"{name}.json").read_text()
        if rendered != document_text:
            failures.append(f"{name}: assembled bytes do not match documents/{name}.json")
            continue

        got = evaluate(assembled)
        want = expected[name]
        if got != want:
            failures.append(f"{name}: expected {want}, got {got}")
        else:
            print(f"{name}: OK ok={got['ok']} (assembled == document)")

    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall build_bundle_v0 piece sets assembled to the verifier's document")
    return 0


if __name__ == "__main__":
    sys.exit(main())
