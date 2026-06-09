"""Guard: the committed SEP-2828 decision/outcome vectors verify independently.

Runs ``_check_independent.py`` (stdlib + cryptography + rfc8785, no Vaara
imports) over the committed fixtures. If the format or a fixture drifts,
the second-implementation checker fails here in CI rather than silently
shipping broken vectors.
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

VECTORS = Path(__file__).parent / "vectors" / "decision_pairing_v0"
CHECKER = VECTORS / "_check_independent.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "_decision_pairing_vector_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_walker_passes_all_cases():
    assert _load_checker().main() == 0


def test_at_least_six_cases_present():
    cases = [p for p in (VECTORS / "normative").iterdir() if p.is_dir()]
    assert len(cases) >= 6


def test_vaara_verifier_reproduces_fallback_binding():
    """The Vaara reference verifier recomputes the no-2787 fallback binding
    from the committed request envelope, and rejects a replayed envelope
    whose arguments differ. This mirrors the independent checker's
    fallback_binding_ok / replayed_binding_ok verdicts."""
    import json

    from vaara.attestation.decision import (
        parse_decision_record,
        verify_decision_fallback_binding,
    )

    case = VECTORS / "normative" / "fallback_envelope_binding"
    decision = parse_decision_record(
        json.loads((case / "decision.json").read_text()))
    env = json.loads((case / "request_envelope.json").read_text())
    env_replayed = json.loads(
        (case / "request_envelope_replayed.json").read_text())

    assert verify_decision_fallback_binding(
        decision, request_envelope=env).ok is True
    assert verify_decision_fallback_binding(
        decision, request_envelope=env_replayed).ok is False
