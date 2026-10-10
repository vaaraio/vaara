# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The envelope rules SPEC.md section 2 states, held by the library itself.

The issuer block names the algorithm a second time inside the signed bytes,
and a record whose two names disagree is refused. The decision record and
its decisionDerived block are closed, as the execution receipt's blocks
already were: a member the format does not define is an error, not a field
to skip.
"""

from __future__ import annotations

import dataclasses

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation._attest_types import AttestationError
from vaara.attestation._decision_emit import emit_decision_record, verify_decision_signature
from vaara.attestation._decision_types import (
    DecisionDerived,
    decision_record_from_dict,
)
from vaara.attestation._receipt_emit import emit_receipt, verify_receipt_signature
from vaara.attestation._receipt_types import BackLink, OutcomeDerived, receipt_from_dict

KEY = ec.generate_private_key(ec.SECP256R1())
LINK = BackLink(attestation_digest="sha256:" + "0" * 64, attestation_nonce="n-1")


def _decision():
    return emit_decision_record(
        back_link=LINK,
        decision_derived=DecisionDerived(decision="allow", decided_at="2026-10-11T00:00:00.000Z"),
        iss="vaara:test", sub="agent", secret_version="es256:test", alg="ES256",
        signing_material=KEY,
    )


def _receipt():
    return emit_receipt(
        back_link=LINK,
        outcome_derived=OutcomeDerived(status="refused", completed_at="2026-10-11T00:00:01Z"),
        iss="vaara:test", sub="agent", secret_version="es256:test", alg="ES256",
        signing_material=KEY,
    )


def test_a_decision_whose_issuer_block_names_another_alg_does_not_verify():
    record = _decision()
    assert verify_decision_signature(record, verifying_material=KEY.public_key())
    lying = dataclasses.replace(
        record, issuer_asserted=dataclasses.replace(record.issuer_asserted, alg="RS256"))
    assert not verify_decision_signature(lying, verifying_material=KEY.public_key())


def test_a_receipt_whose_issuer_block_names_another_alg_does_not_verify():
    receipt = _receipt()
    assert verify_receipt_signature(receipt, verifying_material=KEY.public_key())
    lying = dataclasses.replace(
        receipt, receipt_asserted=dataclasses.replace(receipt.receipt_asserted, alg="HS256"))
    assert not verify_receipt_signature(lying, verifying_material=KEY.public_key())


def test_the_decision_record_round_trips_with_anchors_beside_it():
    wire = _decision().to_dict()
    wire["timestampAnchors"] = []
    assert decision_record_from_dict(wire).decision_derived.decision == "allow"


@pytest.mark.parametrize("where", ["top", "decisionDerived"])
def test_a_member_the_format_does_not_define_is_refused(where):
    wire = _decision().to_dict()
    (wire if where == "top" else wire["decisionDerived"])["extra"] = "x"
    with pytest.raises(AttestationError, match="unrecognized"):
        decision_record_from_dict(wire)


def test_a_pq_signature_member_the_format_does_not_define_is_refused():
    wire = _receipt().to_dict()
    wire["pqSignature"] = {"alg": "ML-DSA-65", "keyid": "k", "sig": "ab", "extra": "x"}
    with pytest.raises(AttestationError, match="unrecognized"):
        receipt_from_dict(wire)


@pytest.mark.parametrize("version", [0, 2, "1", True])
def test_a_version_other_than_one_is_refused_on_both_kinds(version):
    decision = _decision().to_dict()
    decision["version"] = version
    with pytest.raises(AttestationError, match="version"):
        decision_record_from_dict(decision)
    receipt = _receipt().to_dict()
    receipt["version"] = version
    with pytest.raises(AttestationError, match="version"):
        receipt_from_dict(receipt)


def test_a_decision_receipt_carrying_a_hybrid_suite_is_refused():
    wire = _decision().to_dict()
    wire["issuerAsserted"]["sigSuite"] = "ES256+ML-DSA-65"
    with pytest.raises(AttestationError, match="sigSuite"):
        decision_record_from_dict(wire)
