"""Guard: the committed v0 receipt vectors verify under the independent walker.

Runs ``_check_independent.py`` (stdlib + cryptography + rfc8785, no Vaara
imports) over the committed fixtures. This keeps the published vectors
honest: if the format or a fixture drifts, the second-implementation
checker fails here in CI rather than silently shipping broken vectors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

VECTORS = Path(__file__).parent / "vectors" / "execution_receipt_v0"
CHECKER = VECTORS / "_check_independent.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_receipt_vector_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_walker_passes_all_cases():
    assert _load_checker().main() == 0


def test_at_least_five_cases_present():
    cases = [p for p in (VECTORS / "normative").iterdir() if p.is_dir()]
    assert len(cases) >= 5
