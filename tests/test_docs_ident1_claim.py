# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The IDENT-1 row in the OVERT mapping says what the code does.

docs/COMPLIANCE.md and docs/OVERT_CONTROLS.md once graded IDENT-1 as partly
met on the strength of a ``vaara.auth`` key store accepting an authenticated
caller identity into the audit record. Nothing ever called that store, and
it is gone. The row may claim a caller identity again only once a per-caller
credential reaches a record, and a module of that name may come back only
with a caller.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "docs" / "COMPLIANCE.md", ROOT / "docs" / "OVERT_CONTROLS.md")
SRC = ROOT / "src" / "vaara"


def test_ident1_row_says_no_caller_identity_reaches_a_record():
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        row = re.search(r"\*\*IDENT-1\*\*.*?(?=\n- \*\*|\n#)", text, re.S)
        assert row, f"{doc.name} has no IDENT-1 row"
        assert "accepts authenticated caller identity" not in row.group(0), (
            f"{doc.name}: nothing authenticates a caller identity, so the row "
            "may not claim one"
        )
        assert "◯" in row.group(0), f"{doc.name}: IDENT-1 is not met"
        assert "vaara.auth" not in row.group(0), (
            f"{doc.name}: the row names a module that no longer ships"
        )


def test_no_key_store_module_ships_without_a_caller():
    assert not (SRC / "auth.py").exists(), (
        "src/vaara/auth.py is back; wire it to a surface and update the "
        "IDENT-1 rows, or remove it again"
    )
