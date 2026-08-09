"""Guard: the committed x402<->Vaara accountability vectors verify independently.

Runs ``_check_independent.py`` (stdlib + cryptography + rfc8785, no Vaara
imports) over the committed fixtures, then confirms the Vaara reference
implementation reproduces the same binding and signature. If the format or a
fixture drifts, the second-implementation checker fails here in CI rather than
silently shipping a broken worked example.

The vectors bind an x402 settlement to a SEP-2828 decision record across an
action lifecycle (an in-progress step0 and a terminal step1) on two rails: a
generic ``paymentHash`` and a Sui exact-scheme settlement result.
"""

from __future__ import annotations

import hashlib
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

VECTORS = Path(__file__).parent / "vectors" / "x402_settlement_v0"
CHECKER = VECTORS / "_check_independent.py"
_RAILS = ("generic", "sui")
_STEPS = ("step0", "step1")


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "_x402_settlement_vector_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(rail: str, step: str, name: str) -> dict:
    return json.loads((VECTORS / rail / step / name).read_text())


def test_independent_checker_passes_all_cases():
    assert _load_checker().main() == 0


def test_both_rails_present_with_lifecycle_steps():
    for rail in _RAILS:
        for step in _STEPS:
            assert (VECTORS / rail / step / "settlement.json").is_file()
            assert (VECTORS / rail / step / "receipt.json").is_file()


def test_vaara_reproduces_binding_and_signature():
    """The Vaara reference verifier reproduces, on its own implementation, the
    two properties the independent checker asserts: the receipt signature
    verifies under the committed public key, and the receipt's evidenceRef
    digest is the JCS-canonical SHA-256 of the settlement it binds."""
    import rfc8785
    from cryptography.hazmat.primitives import serialization

    from vaara.attestation.decision import (
        parse_decision_record,
        verify_decision_signature,
    )

    pub = serialization.load_pem_public_key(
        (VECTORS / "keys" / "es256_public.pem").read_bytes())

    for rail in _RAILS:
        for step in _STEPS:
            settlement = _load(rail, step, "settlement.json")
            receipt_dict = _load(rail, step, "receipt.json")

            record = parse_decision_record(receipt_dict)
            assert verify_decision_signature(
                record, verifying_material=pub) is True, f"{rail}/{step} sig"

            ref = record.decision_derived.evidence_ref
            assert ref is not None and ref.canonicalization == "JCS"
            digest = "sha256:" + hashlib.sha256(
                rfc8785.dumps(settlement)).hexdigest()
            assert ref.digest == digest, f"{rail}/{step} binding"


def test_lifecycle_steps_have_distinct_action_refs():
    """A mid-task (terminal=false) receipt cannot be passed off as the final
    (terminal=true) one: the two lifecycle points carry distinct action_refs
    and opposite terminal flags on every rail."""
    for rail in _RAILS:
        s0 = _load(rail, "step0", "settlement.json")
        s1 = _load(rail, "step1", "settlement.json")
        assert s0["terminal"] is False
        assert s1["terminal"] is True
        assert s0["actionRef"] != s1["actionRef"]


def test_committed_vectors_keep_the_JCS_label():
    """The committed label is pinned deliberately and must never be rewritten.

    Five outside parties (agent-guard, nobulex, elara, payperbyte, and a live
    Sui gasless run) reproduce these fixtures byte for byte against a
    content-pinned copy of the checker. Relabelling a committed vector, even to
    the SPEC-preferred ``jcs-rfc8785``, changes those bytes and invalidates
    every digest they published. New labels belong in new vectors.
    """
    for rail in _RAILS:
        for step in _STEPS:
            ref = _load(rail, step, "receipt.json")["decisionDerived"][
                "evidenceRef"]
            assert ref["canonicalization"] == "JCS"


def test_checker_accepts_every_spec_alias():
    """SPEC.md section 1: producers SHOULD emit ``jcs-rfc8785``; consumers MUST
    accept ``jcs-rfc8785``, ``JCS`` and ``jcs-json-v1``.

    The checker was widened to all three, but nothing exercised the widening,
    so it could regress to a bare ``== "JCS"`` and the suite would stay green.
    That regression is not hypothetical: an issuer following our own spec emits
    ``jcs-rfc8785`` and would fail a gate our spec is cited as defining.

    Mutates in-memory copies only. The committed fixtures keep their ``JCS``
    label (see ``test_committed_vectors_keep_the_JCS_label``).
    """
    checker = _load_checker()
    settlement = _load("sui", "step1", "settlement.json")
    receipt = _load("sui", "step1", "receipt.json")

    for alias in ("jcs-rfc8785", "JCS", "jcs-json-v1"):
        probe = json.loads(json.dumps(receipt))
        probe["decisionDerived"]["evidenceRef"]["canonicalization"] = alias
        assert checker.settlement_binding_resolves(probe, settlement) is True, (
            f"consumer rejected the spec alias {alias!r}")


def test_checker_rejects_an_unknown_canonicalization():
    """Accepting all three aliases must not become accepting anything. A label
    the spec does not define means the digest was computed some other way, so
    the binding cannot be assumed to hold."""
    checker = _load_checker()
    settlement = _load("sui", "step1", "settlement.json")
    probe = _load("sui", "step1", "receipt.json")
    probe["decisionDerived"]["evidenceRef"]["canonicalization"] = "sorted-keys"
    assert checker.settlement_binding_resolves(probe, settlement) is False
