"""The matrix job's skip gate catches a skip nobody reviewed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_skips", ROOT / "scripts" / "check_skips.py")
check_skips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_skips)  # type: ignore[union-attr]

ALLOW = (ROOT / "tests" / "ci_allowed_skips.txt").read_text()


def test_a_reviewed_skip_passes():
    out = "SKIPPED [3] tests/test_x.py:10: could not import 'yaml': No module named 'yaml'\n"
    assert check_skips.unexpected(out, ALLOW) == []


def test_an_unreviewed_skip_fails():
    out = ("SKIPPED [1] tests/test_x.py:10: could not import 'fastapi': "
           "No module named 'fastapi'\n")
    assert check_skips.unexpected(out, ALLOW) == [out.strip()]


def test_the_allowlist_has_no_catch_all():
    for line in ALLOW.splitlines():
        if line.strip() and not line.startswith("#"):
            assert line.strip() not in (".*", ".+", "^.*$"), line
