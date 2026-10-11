# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every import a public document shows a reader has to resolve.

docs/design/evidence-ref-mapping-spec.md told a reader to run ``from
vaara.attestation._sep2787_canonical import canonical_json`` for two years
after the module was renamed; the CHANGELOG documented
``vaara.attestation.build_decision_basis`` and the package never exported
it. A path guard does not see an import statement, and the reachability
guard treats an import it cannot resolve as nothing. This one imports what
the documents show, with the extras installed or not, and fails on a module
that is missing or a name the module does not have.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DOCS = (
    sorted(ROOT.glob("docs/**/*.md"))
    + [ROOT / "README.md", ROOT / "SPEC.md"]
    + sorted(ROOT.glob("examples/**/*.md"))
    + sorted(ROOT.glob("webpage/**/*.html"))
)

_FROM = re.compile(r"^\s*from\s+(vaara(?:\.\w+)*)\s+import\s+\(?([\w\s,]+?)\)?\s*(?:#.*)?$", re.M)
_IMPORT = re.compile(r"^\s*import\s+(vaara(?:\.\w+)*)(?:\s+as\s+\w+)?\s*(?:#.*)?$", re.M)


def _shown(doc: Path) -> list[tuple[str, str | None]]:
    """(module, name) pairs a document shows; name is None for `import x`."""
    text = doc.read_text(encoding="utf-8")
    if doc.suffix == ".html":
        text = re.sub(r"<[^>]+>", "", text).replace("&lt;", "<").replace("&gt;", ">")
    pairs: list[tuple[str, str | None]] = []
    for module, names in _FROM.findall(text):
        for clause in names.split(","):
            name = clause.strip().split()[0] if clause.strip() else ""
            if name:
                pairs.append((module, name))
    for module in _IMPORT.findall(text):
        pairs.append((module, None))
    return pairs


def _cases():
    seen = set()
    for doc in DOCS:
        for module, name in _shown(doc):
            key = (module, name)
            if key not in seen:
                seen.add(key)
                yield pytest.param(module, name, str(doc.relative_to(ROOT)),
                                   id=f"{module}:{name or '*'}")


@pytest.mark.parametrize("module,name,doc", list(_cases()))
def test_a_documented_import_resolves(module, name, doc):
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        if exc.name and exc.name.split(".")[0] != "vaara":
            pytest.skip(f"{module} needs {exc.name}, an extra not installed here")
        pytest.fail(f"{doc} shows `from {module} import ...`; the module does not import: {exc}")
    if name is not None and not hasattr(imported, name):
        try:
            importlib.import_module(f"{module}.{name}")
        except ImportError:
            pytest.fail(f"{doc} shows `from {module} import {name}`; {module} has no {name}")


def test_the_scan_finds_something():
    assert len(list(_cases())) >= 20
