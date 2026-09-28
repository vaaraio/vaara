# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The IDENT-1 row in the OVERT mapping says what the code does.

docs/COMPLIANCE.md and docs/OVERT_CONTROLS.md graded IDENT-1 as partly met
on the strength of ``vaara.auth`` accepting an authenticated caller identity
into the audit record. Nothing in the tree calls the key store or the role
check, so no identity reaches a record that way. The row may claim it again
only once a caller exists.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "docs" / "COMPLIANCE.md", ROOT / "docs" / "OVERT_CONTROLS.md")
SRC = ROOT / "src" / "vaara"


def _callers(symbol: str) -> list[Path]:
    hits = []
    for path in SRC.rglob("*.py"):
        if path.name in ("auth.py", "sqlite_backend.py"):
            continue
        if re.search(rf"\b{symbol}\b", path.read_text()):
            hits.append(path)
    return hits


def test_ident1_row_matches_whether_anything_calls_vaara_auth():
    wired = bool(_callers("authenticate_api_key") or _callers("require_role"))
    for doc in DOCS:
        text = doc.read_text()
        row = re.search(r"\*\*IDENT-1\*\*.*?(?=\n- \*\*|\n#)", text, re.S)
        assert row, f"{doc.name} has no IDENT-1 row"
        claims_identity = "accepts authenticated caller identity" in row.group(0)
        if wired:
            assert claims_identity, f"{doc.name}: vaara.auth is wired, say so"
        else:
            assert not claims_identity, (
                f"{doc.name}: nothing calls vaara.auth, so no caller identity "
                "reaches a record through it"
            )
            assert "◯" in row.group(0), f"{doc.name}: IDENT-1 is not met"
