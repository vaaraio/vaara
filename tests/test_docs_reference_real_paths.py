# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A path a public document names has to be a path a reader can open.

Three documents claimed the MIT AI Risk Repository v4 database and its
mitigations sheet were "tracked under `research/external/` for
reproducibility". `research/` has never been in the repository: `git
ls-files research` returns nothing. A reproducibility claim pointing at a
directory only we can see is the worst direction for that claim to be
wrong in, and eidas-qel-profile.md went further and cited an internal
research file in its Sources list.

PRIOR_ART.md named `src/vaara/audit/ots_anchor.py`, removed in v1.53.0
along with the OpenTimestamps anchor method it implemented.

The check resolves the shorthand the docs legitimately use (`scorer/
adaptive.py` and `vaara/compliance/engine.py` both mean the file under
`src/`), then fails on anything left that does not exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    path
    for path in sorted(ROOT.glob("docs/*.md"))
    + [ROOT / name for name in ("README.md", "SPEC.md", "PRIVACY.md", "LICENSING.md")]
    if path.exists()
]

#: Backticked file paths: `tests/vectors/x402_settlement_v0/expected.json`.
_FILE = re.compile(
    r"`([\w./\-]+/[\w./\-]+\.(?:py|json|yaml|yml|md|jsonl|cbor|toml|sh|txt|pem))`"
)
#: Backticked directories under a known top-level tree.
_DIR = re.compile(
    r"`((?:tests|src|docs|scripts|examples|conformance|bench|clients|ietf|plugins)"
    r"/[\w./\-]*/)`"
)

#: Roots a document may leave off. `scorer/adaptive.py` means
#: `src/vaara/scorer/adaptive.py`; `vaara/compliance/engine.py` means the same
#: file under `src/`.
_PREFIXES = ("", "src/vaara/", "src/", "docs/")


def _exists(reference: str) -> bool:
    # removeprefix, not lstrip: lstrip("./") also eats the leading dot of
    # `.github/workflows/release.yml`, which then never resolves.
    candidate = reference.removeprefix("./").removeprefix("../")
    return any((ROOT / (prefix + candidate)).exists() for prefix in _PREFIXES)


def _references(doc: Path) -> set[str]:
    text = doc.read_text(encoding="utf-8")
    found = set(_FILE.findall(text)) | set(_DIR.findall(text))
    return {
        reference for reference in found
        if not reference.startswith(("http", "<", "~", "$"))
    }


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_paths_exist(doc):
    missing = sorted(r for r in _references(doc) if not _exists(r))
    assert not missing, (
        f"{doc.relative_to(ROOT)} names paths that are not in the repository: "
        f"{missing}. Either the file moved, or the document is pointing a "
        f"reader at something only we can see."
    )


def test_no_public_document_cites_the_private_research_tree():
    """`research/` is internal and untracked; nothing public may lean on it."""
    # Path references only. A URL that happens to contain the word is not a
    # citation of our tree.
    citing = [
        doc.relative_to(ROOT)
        for doc in DOCS
        if any(
            reference.removeprefix("../").startswith("research/")
            for reference in _references(doc)
        )
    ]
    assert not citing, (
        f"these public documents cite the untracked research tree: {citing}. "
        f"A reader cannot open it, and its filenames are internal."
    )


def test_the_scan_finds_something():
    """A regex matching nothing would pass every case above."""
    total = sum(len(_references(doc)) for doc in DOCS)
    assert total >= 50, f"only {total} path references found across {len(DOCS)} docs"
