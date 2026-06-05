#!/usr/bin/env python3
"""Generate the build_bundle_v0 conformance vectors.

These vectors cover the *issuer side* of the evidence bundle: assembling the
single on-disk document a verifier reads from the separate pieces an issuer
holds. They are the mirror of ``bundle_doc_v0`` (which covers the verifier
reading that document) and, like it, derive from the committed evidence rather
than re-signing anything.

For each ``bundle_doc_v0`` document, this generator splits the one document
back into the issuer's separate artifacts and writes them under
``pieces/<name>/`` by the same conventional file names ``vaara build-bundle
--from-dir`` discovers (``receipt.json``, ``attestation.json``, and so on;
scalar fields as ``expected_keyid.txt`` / ``inclusion_leaf_hex.txt``). The
expected assembled output goes to ``documents/<name>.json``, byte-for-byte the
``bundle_doc_v0`` document it came from. ``expected.json`` carries the
reference verdict per name.

The point under test: re-assembling the pieces reproduces exactly the document
the verifier reads. The generator asserts that round-trip as it writes, and the
independent checker (``_check_independent.py``) reproduces it with no Vaara
import.

Run from the repo root: ``python tests/vectors/build_bundle_v0/_generate.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "bundle_doc_v0"
SOURCE_BUNDLES = SOURCE / "bundles"
SOURCE_EXPECTED = SOURCE / "expected.json"

# Bundle-document key -> the file an issuer keeps it in. Mirrors
# vaara.attestation._bundle_io.BUNDLE_PIECE_FILES / BUNDLE_PIECE_SCALARS, kept
# here as a literal so this generator (and the checker) need no Vaara import.
OBJECT_PIECES = {
    "receipt": "receipt.json",
    "did_document": "did_document.json",
    "verifying_jwk": "verifying_jwk.json",
    "attestation": "attestation.json",
    "inclusion": "inclusion.json",
    "consistency": "consistency.json",
    "registry": "registry.json",
}
SCALAR_PIECES = {
    "expected_keyid": "expected_keyid.txt",
    "inclusion_leaf_hex": "inclusion_leaf_hex.txt",
}


def _render(document: dict) -> str:
    """The canonical on-disk rendering: sorted keys, two-space indent."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _assemble(pieces_dir: Path) -> dict:
    """Re-assemble a document from an issuer's pieces directory (Vaara-free)."""
    document: dict = {}
    for key, filename in OBJECT_PIECES.items():
        path = pieces_dir / filename
        if path.is_file():
            document[key] = json.loads(path.read_text())
    for key, filename in SCALAR_PIECES.items():
        path = pieces_dir / filename
        if path.is_file():
            document[key] = path.read_text().strip()
    return document


def main() -> None:
    expected = json.loads(SOURCE_EXPECTED.read_text())

    pieces_root = HERE / "pieces"
    documents_dir = HERE / "documents"
    for stale in (pieces_root, documents_dir):
        if stale.exists():
            shutil.rmtree(stale)
    pieces_root.mkdir()
    documents_dir.mkdir()

    for name in sorted(expected):
        document_text = (SOURCE_BUNDLES / f"{name}.json").read_text()
        document = json.loads(document_text)

        case_dir = pieces_root / name
        case_dir.mkdir()
        for key, value in document.items():
            if key in OBJECT_PIECES:
                (case_dir / OBJECT_PIECES[key]).write_text(
                    json.dumps(value, indent=2, sort_keys=True) + "\n"
                )
            elif key in SCALAR_PIECES:
                (case_dir / SCALAR_PIECES[key]).write_text(str(value) + "\n")
            else:
                raise ValueError(f"{name}: unknown bundle key {key!r}")

        # The expected assembled document is the bundle_doc_v0 document verbatim.
        (documents_dir / f"{name}.json").write_text(document_text)

        # Round-trip assertion at generation time: re-assembling the pieces we
        # just wrote reproduces the exact document bytes the verifier reads.
        rebuilt = _render(_assemble(case_dir))
        if rebuilt != document_text:
            raise AssertionError(f"{name}: re-assembled pieces do not match document")

    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(expected)} piece sets to pieces/ and documents/")


if __name__ == "__main__":
    main()
