"""Extras that pull in a dependency with a known poisoned release exclude it.

litellm 1.107.0 was published to PyPI from an attacker-controlled repository on
2026-10-21 and carried credential-stealing code (docs.litellm.ai/blog/
security-incident-oct-21-2026). PyPI pulled it, but mirrors and wheel caches can
still hand it out. guardrails-ai depends on litellm, so the extra names the bad
version directly and no resolver can pick it through us.
"""
from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]

POISONED = {"litellm": ["1.107.0"]}
EXTRAS_PULLING = {"litellm": ["guardrails-ai"]}


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_extras_exclude_poisoned_releases():
    extras = _extras()
    for package, extra_names in EXTRAS_PULLING.items():
        for extra in extra_names:
            reqs = [Requirement(r) for r in extras[extra]]
            pins = [r for r in reqs if r.name == package]
            assert pins, f"extra {extra!r} does not constrain {package}"
            for bad in POISONED[package]:
                assert all(not r.specifier.contains(bad, prereleases=True) for r in pins), (
                    f"extra {extra!r} still allows {package}=={bad}"
                )
