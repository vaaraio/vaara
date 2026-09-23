"""Every command a parser names itself after is an installed script.

`vaara-infer-proxy` and `vaara-console` printed `usage: vaara-infer-proxy` and
`usage: vaara-console` in their help while pyproject installed neither, so
the only way to start them was `python -m`. This reads every ``prog="vaara..."``
in the source and requires a matching `[project.scripts]` entry, or a
`vaara <sub>` form that the main CLI owns.
"""
from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]


def _scripts() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return data["project"]["scripts"]


def _prog_names() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "src" / "vaara").rglob("*.py"):
        for m in re.finditer(r'prog\s*=\s*"(vaara[\w-]*)"', path.read_text()):
            found.add(m.group(1))
    return found


def test_every_standalone_prog_is_a_script():
    scripts = _scripts()
    missing = sorted(p for p in _prog_names() if p not in scripts)
    assert missing == [], f"prog names with no [project.scripts] entry: {missing}"


def test_every_script_target_exists():
    # Checked in the source, not by import: several targets need optional
    # extras (fastapi, httpx) that the base CI environment does not install.
    for name, target in _scripts().items():
        module, _, attr = target.partition(":")
        path = ROOT / "src" / Path(*module.split("."))
        src = path.with_suffix(".py") if path.with_suffix(".py").exists() \
            else path / "__init__.py"
        assert src.exists(), f"{name}: no module {module}"
        assert re.search(rf"^def {attr}\(", src.read_text(), re.M), \
            f"{name}: {module} defines no {attr}()"
