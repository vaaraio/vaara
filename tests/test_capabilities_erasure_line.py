"""The capability index names the code that does GDPR Article 17 erasure.

The line once pointed at a read-time substitution in the SQLite backend that
was removed because a presented record that differs from the stored one
cannot be recomputed. Erasure is ``vaara trail erase``: it keeps the record's
hash, appends a ``record_erased`` record, and both verifiers report the slot
as erased. The line says that, and every path it names exists.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPABILITIES = ROOT / "docs" / "capabilities.md"


def _line() -> str:
    for line in CAPABILITIES.read_text(encoding="utf-8").splitlines():
        if line.startswith("- GDPR Article 17"):
            return line
    raise AssertionError("docs/capabilities.md has no GDPR Article 17 line")


def test_the_line_names_the_erase_module_and_the_commands():
    line = _line()
    assert "`audit/erase.py`" in line
    assert "`vaara trail erase`" in line
    assert "--erasure-key" in line


def test_the_line_does_not_name_the_removed_redaction():
    line = _line().lower()
    assert "read-time" not in line
    assert "redaction" not in line
    assert "sqlite_backend" not in line


def test_the_line_says_what_erasure_reaches():
    line = _line()
    assert "trail" in line and "hash" in line


def test_every_source_path_the_line_names_exists():
    paths = re.findall(r"`([a-z_/]+\.py)`", _line())
    assert paths
    for rel in paths:
        assert (ROOT / "src" / "vaara" / rel).is_file(), rel
