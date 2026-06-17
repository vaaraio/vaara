"""Guard: the committed evidenceRef vectors verify independently.

Runs ``_check_independent.py`` (stdlib + cryptography + rfc8785, no Vaara
imports) over the committed fixtures. If the format or a fixture drifts,
the second-implementation checker fails here in CI rather than silently
shipping broken vectors.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

VECTORS = Path(__file__).parent / "vectors" / "evidence_ref_v0"
CHECKER = VECTORS / "_check_independent.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "_evidence_ref_vector_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_checker_passes_all_cases():
    assert _load_checker().main() == 0


def test_at_least_five_cases_present():
    cases = [p for p in (VECTORS / "normative").iterdir() if p.is_dir()]
    assert len(cases) >= 5


def test_valid_digest_matches_published_worked_example():
    """The valid fixtures cite the address from the mapping spec's worked
    example, recomputed here from the committed drift record bytes through a
    second canonicalizer (the checker's), not the generator's."""
    checker = _load_checker()
    case = VECTORS / "normative" / "valid_evidence_ref_resolves"
    drift = json.loads((case / "drift_record.json").read_text())
    decision = json.loads((case / "decision.json").read_text())
    recomputed = checker._sha256_hex(checker._jcs(drift))
    cited = decision["decisionDerived"]["evidenceRef"]["digest"]
    assert recomputed == cited
    assert recomputed == (
        "sha256:8e22e733c3526ca8e7987ab2355f18e66752f"
        "29ac629dbd41c9b80650822a56b")


def test_signature_and_resolution_are_independent():
    """tampered_drift_record keeps a valid signature while resolution fails:
    a verifier that only checked the signature would miss the evidence
    substitution. Confirms the two verdicts are not collapsible into one."""
    checker = _load_checker()
    case = VECTORS / "normative" / "tampered_drift_record"
    decision = json.loads((case / "decision.json").read_text())
    drift = json.loads((case / "drift_record.json").read_text())
    assert checker.verify_signature(decision, checker._DECISION_BLOCKS) is True
    assert checker.evidence_ref_resolves(decision, drift) is False


def test_vaara_round_trips_the_valid_evidence_ref():
    """The Vaara reference parser reads the committed valid decision and the
    EvidenceRef survives the round trip, so the published bytes and the
    in-repo type agree on the field shape."""
    from vaara.attestation.decision import parse_decision_record

    case = VECTORS / "normative" / "valid_evidence_ref_resolves"
    decision = parse_decision_record(
        json.loads((case / "decision.json").read_text()))
    ref = decision.decision_derived.evidence_ref
    assert ref is not None
    assert ref.canonicalization == "JCS"
    assert ref.schema == "interlock.drift-record/v0"
    assert ref.digest == decision.to_dict()["decisionDerived"][
        "evidenceRef"]["digest"]
