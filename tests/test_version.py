"""Release hygiene: the package must report the version being released.

v1.23.0 through v1.25.0 shipped wheels whose ``vaara.__version__`` still said
"1.22.0" because the string in ``src/vaara/__init__.py`` is bumped by hand and
the release workflow never checks it. These tests make the drift fail the
build instead of shipping.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import vaara

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject_version() -> str:
    # Regex instead of tomllib: tomllib is 3.11+, and this is the one line
    # we need. Anchored to the [project] table's version key.
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match, "could not find version in pyproject.toml"
    return match.group(1)


@pytest.mark.skipif(not _PYPROJECT.exists(), reason="pyproject.toml not present (installed wheel)")
def test_package_version_matches_pyproject():
    assert vaara.__version__ == _pyproject_version()


def test_py_typed_marker_ships_with_package():
    # The "Typing :: Typed" classifier promises PEP 561 type information.
    # Without the marker file, type checkers ignore every annotation in the
    # installed package.
    marker = Path(vaara.__file__).with_name("py.typed")
    assert marker.is_file(), "src/vaara/py.typed is missing"
