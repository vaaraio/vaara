"""Fail when pytest skipped a test for a reason nobody reviewed.

Usage: python scripts/check_skips.py PYTEST_OUTPUT ALLOWLIST

PYTEST_OUTPUT is the text of a `pytest -rs` run. ALLOWLIST holds one regular
expression per line; blank lines and `#` comments are ignored. Every
`SKIPPED` reason must match one of them. Exit 1 lists the ones that do not.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_SKIP = re.compile(r"^SKIPPED \[\d+\] [^:]+:\d+: (.*)$")


def unexpected(output: str, allowlist: str) -> list[str]:
    patterns = [re.compile(line) for line in allowlist.splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    found: list[str] = []
    for line in output.splitlines():
        m = _SKIP.match(line.strip())
        if m and not any(p.search(m.group(1)) for p in patterns):
            found.append(line.strip())
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    bad = unexpected(Path(argv[1]).read_text(), Path(argv[2]).read_text())
    for line in bad:
        print(f"::error::unreviewed skip: {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
