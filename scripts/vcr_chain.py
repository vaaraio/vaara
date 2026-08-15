#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check that nothing has been removed from Vaara Conformance Results.

The table records occurrences. A run happened, on a date, at a commit, and that
does not stop being true afterwards, so rows are never taken down or edited.
Saying so is worth little on its own, because the person who would quietly edit
the table is the person publishing the promise. So each row carries ``prev``,
the digest of the row before it, and this script recomputes the whole chain.

Drop a row, reorder two, or change a single character of a published one, and
every digest after the change stops matching. The break names the row.

This imports no Vaara code, so it checks the maintainer as readily as anyone
else. It needs ``rfc8785`` and nothing more.

Usage:
    python scripts/vcr_chain.py                       # check the committed file
    python scripts/vcr_chain.py path/to/rows.json     # or one you downloaded
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "conformance" / "reproductions.json"
ZERO = "sha256:" + "0" * 64


def digest(row: dict) -> str:
    import rfc8785

    return "sha256:" + hashlib.sha256(rfc8785.dumps(row)).hexdigest()


def check(data: dict) -> list[str]:
    problems = []
    expected = data.get("genesis", ZERO)
    for position, row in enumerate(data.get("reproductions", [])):
        if row.get("prev") != expected:
            problems.append(
                f"row {row.get('id', '?')} ({row.get('party', 'unknown')}) at "
                f"position {position} links to {row.get('prev')}, but the chain "
                f"reaches it at {expected}. Something before it was removed, "
                f"reordered or rewritten."
            )
        expected = digest(row)
    return problems


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else DEFAULT
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("reproductions", [])
    problems = check(data)

    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} break(s) in the chain.", file=sys.stderr)
        return 1

    head = digest(rows[-1]) if rows else data.get("genesis", ZERO)
    print(f"{len(rows)} row(s), chain intact.")
    print(f"head: {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
