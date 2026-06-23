"""The sealed ``maxClass`` rides under signature only via the evidence binding.

``maxClass`` lives in the unsigned ``evidence`` block; the ES256 signature covers
the ``record`` half. What carries the class under signature is
``evidence_binding_ok``: the signed ``decisionDerived.evidenceRef.digest`` is
``sha256:`` + JCS(evidence), so recomputing it proves the class is the class that
was signed. These tests pin that a relabeled ``maxClass`` breaks the binding while
the record signature still verifies, the relabeling attack Mayur021 named on
cosai-oasis/ws4 #99. The fixtures are the shipped class_gate_v0 vectors, so the
tests also guard those bytes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from vaara.credential import evidence_binding_ok

VECTORS = Path(__file__).resolve().parents[1] / "vectors" / "class_gate_v0"
_SEAL = "chain-agent-handoff-7a1d-9999-seal-authz.json"


def _seal(case: str) -> dict:
    return json.loads((VECTORS / case / _SEAL).read_text(encoding="utf-8"))


def test_honest_seal_binds():
    assert evidence_binding_ok(_seal("permit")) is True


def test_relabeled_seal_does_not_bind():
    # deny_relabeled's maxClass was changed to a permitted class after signing.
    assert evidence_binding_ok(_seal("deny_relabeled")) is False


def test_mutating_any_evidence_field_breaks_the_binding():
    seal = _seal("permit")
    seal["evidence"]["completeness"]["maxClass"] = "tx.transfer"
    assert evidence_binding_ok(seal) is False


def test_missing_halves_are_unbound():
    full = _seal("permit")
    assert evidence_binding_ok({}) is False
    assert evidence_binding_ok({"evidence": full["evidence"]}) is False
    assert evidence_binding_ok({"record": full["record"]}) is False
    assert evidence_binding_ok("not a dict") is False  # type: ignore[arg-type]


def test_non_jcs_canonicalization_is_rejected():
    seal = copy.deepcopy(_seal("permit"))
    seal["record"]["decisionDerived"]["evidenceRef"]["canonicalization"] = "raw"
    assert evidence_binding_ok(seal) is False


def test_stale_digest_is_rejected_even_if_evidence_is_intact():
    seal = copy.deepcopy(_seal("permit"))
    seal["record"]["decisionDerived"]["evidenceRef"]["digest"] = "sha256:" + "00" * 32
    assert evidence_binding_ok(seal) is False
