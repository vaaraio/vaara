# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The attribute attestation, and the one field that decides whether it is worth anything.

A signed record can assert anything. What a relying party needs to know is where
the value came from, so every attribute carries a standing from a closed ordered
set and the caller states the floor it will accept.

The property under test above the individual cases: a value that is sound but
weaker than asked for must never be reported the same way as a forged one, and
must never be reported the same way as green.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.attestation.attribute import (  # noqa: E402
    REASON_STATE,
    SCHEMA,
    STANDING_RANK,
    Attribute,
    AttributeQuery,
    AttributeReason,
    AttributeState,
    SourceStanding,
    Subject,
    attestation_digest,
    emit_attribute_attestation,
    evaluate,
    verify_attestation_signature,
)
from vaara.audit.signer import Ed25519Signer, Ed25519Verifier  # noqa: E402

ISSUER = "kiinteistoautomaatio.example"
SUBJECT = Subject(kind="detection-event", id="alarm-2026-08-21-0413-b17")
NOW = "2026-08-21T06:00:00Z"
NOT_BEFORE = "2026-08-21T04:13:00Z"
NOT_AFTER = "2026-09-21T04:13:00Z"
LATER = "2026-10-01T00:00:00Z"
EARLIER = "2026-08-20T00:00:00Z"


def _key(seed=0):
    return ed25519.Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)


def _verifier(seed=0):
    return Ed25519Verifier(_key(seed).public_key().public_bytes_raw())


ATTRS = (
    Attribute("triggerClass", "animal", SourceStanding.MEASURED,
              "fauna-classifier v3.1.0"),
    Attribute("sensorId", "PIR-B17-04", SourceStanding.PROTOCOL_DEFINED),
    Attribute("confidence", "0.94", SourceStanding.MEASURED, "fauna-classifier v3.1.0"),
    Attribute("operatorNote", "toistuva", SourceStanding.OPERATOR_DECLARED),
)


def _attestation(attributes=ATTRS, *, seed=0, not_before=NOT_BEFORE, not_after=NOT_AFTER):
    return emit_attribute_attestation(
        signer=Ed25519Signer(_key(seed)),
        issuer=ISSUER,
        attestation_id="att-b17-0413",
        subject=SUBJECT,
        attributes=attributes,
        not_before=not_before,
        not_after=not_after,
    )


def _q(name="triggerClass", minimum=SourceStanding.MEASURED, **kw):
    return AttributeQuery(name=name, minimum_source=minimum, **kw)


# --- the ladder and the partition -------------------------------------------


def test_standing_is_a_total_order_with_undeclared_at_the_floor():
    ranks = [STANDING_RANK[s] for s in SourceStanding]
    assert len(set(ranks)) == len(ranks)
    assert STANDING_RANK[SourceStanding.UNDECLARED] == min(ranks)
    assert STANDING_RANK[SourceStanding.PROTOCOL_DEFINED] == max(ranks)
    assert (
        STANDING_RANK[SourceStanding.OPERATOR_DECLARED]
        < STANDING_RANK[SourceStanding.MEASURED]
    )


def test_every_reason_maps_to_exactly_one_state():
    assert set(REASON_STATE) == set(AttributeReason)
    assert set(REASON_STATE.values()) == set(AttributeState)


def test_insufficient_and_forged_never_share_a_state():
    withheld = {r for r, s in REASON_STATE.items() if s is AttributeState.WITHHELD}
    refused = {r for r, s in REASON_STATE.items() if s is AttributeState.REFUSED}
    assert not withheld & refused
    assert REASON_STATE[AttributeReason.SOURCE_BELOW_FLOOR] is AttributeState.WITHHELD
    assert REASON_STATE[AttributeReason.SIGNATURE_INVALID] is AttributeState.REFUSED


def test_only_one_reason_accepts():
    accepting = [r for r, s in REASON_STATE.items() if s is AttributeState.ACCEPTED]
    assert accepting == [AttributeReason.ATTRIBUTE_ATTESTED]


# --- the document -----------------------------------------------------------


def test_attestation_is_signed_and_content_addressed():
    a = _attestation()
    assert a["schema"] == SCHEMA
    assert verify_attestation_signature(a, verifier=_verifier())
    assert attestation_digest(a).startswith("sha256:")


def test_signature_covers_the_source_standing():
    # Downgrading a value from measured to operator-declared, or upgrading it,
    # is exactly the edit the format exists to catch.
    a = _attestation()
    tampered = json.loads(json.dumps(a))
    for attr in tampered["attributes"]:
        if attr["name"] == "operatorNote":
            attr["source"] = "measured"
    assert not verify_attestation_signature(tampered, verifier=_verifier())


def test_attributes_are_emitted_sorted_so_two_issuers_agree():
    names = [a["name"] for a in _attestation()["attributes"]]
    assert names == sorted(names)


def test_empty_attestation_is_refused_at_construction():
    with pytest.raises(AttestationError):
        _attestation(attributes=())


def test_duplicate_attribute_names_are_refused():
    with pytest.raises(AttestationError):
        _attestation(attributes=(
            Attribute("x", "1", SourceStanding.MEASURED),
            Attribute("x", "2", SourceStanding.MEASURED),
        ))


def test_backwards_window_is_refused():
    with pytest.raises(AttestationError):
        _attestation(not_before=NOT_AFTER, not_after=NOT_BEFORE)


def test_float_value_is_refused_at_construction():
    with pytest.raises(AttestationError):
        Attribute("confidence", 0.94, SourceStanding.MEASURED)  # type: ignore[arg-type]


# --- evaluation -------------------------------------------------------------


def test_measured_value_clears_a_measured_floor():
    d = evaluate(_attestation(), _q(), now=NOW, verifier=_verifier())
    assert d.state is AttributeState.ACCEPTED
    assert d.reason is AttributeReason.ATTRIBUTE_ATTESTED
    assert d.accepted is True
    assert d.value == "animal"
    assert d.source is SourceStanding.MEASURED


def test_operator_declared_value_is_withheld_against_a_measured_floor():
    # The supplier typed it in. It is sound, it verifies, and it is not evidence.
    d = evaluate(_attestation(), _q("operatorNote"), now=NOW, verifier=_verifier())
    assert d.state is AttributeState.WITHHELD
    assert d.reason is AttributeReason.SOURCE_BELOW_FLOOR
    assert d.value == "toistuva"
    assert d.source is SourceStanding.OPERATOR_DECLARED


def test_the_same_value_is_accepted_when_the_caller_accepts_that_floor():
    d = evaluate(
        _attestation(),
        _q("operatorNote", SourceStanding.OPERATOR_DECLARED),
        now=NOW, verifier=_verifier(),
    )
    assert d.state is AttributeState.ACCEPTED


def test_protocol_defined_clears_every_floor():
    for floor in SourceStanding:
        d = evaluate(_attestation(), _q("sensorId", floor), now=NOW,
                     verifier=_verifier())
        assert d.state is AttributeState.ACCEPTED, floor


def test_absent_attribute_is_withheld_and_names_absence():
    d = evaluate(_attestation(), _q("modelLicence"), now=NOW, verifier=_verifier())
    assert d.state is AttributeState.WITHHELD
    assert d.reason is AttributeReason.ATTRIBUTE_ABSENT
    assert d.value is None


def test_subject_mismatch_is_withheld():
    d = evaluate(_attestation(), _q(subject_id="alarm-somewhere-else"), now=NOW,
                 verifier=_verifier())
    assert d.state is AttributeState.WITHHELD
    assert d.reason is AttributeReason.SUBJECT_MISMATCH


def test_unaccepted_issuer_is_withheld():
    d = evaluate(_attestation(), _q(accepted_issuers=frozenset({"someone.else"})),
                 now=NOW, verifier=_verifier())
    assert d.state is AttributeState.WITHHELD
    assert d.reason is AttributeReason.ISSUER_NOT_ACCEPTED


@pytest.mark.parametrize("when", [EARLIER, LATER])
def test_outside_the_window_expires(when):
    d = evaluate(_attestation(), _q(), now=when, verifier=_verifier())
    assert d.state is AttributeState.EXPIRED
    assert d.reason is AttributeReason.OUTSIDE_VALIDITY_WINDOW


@pytest.mark.parametrize("when", [NOT_BEFORE, NOT_AFTER])
def test_the_window_is_inclusive_at_both_ends(when):
    d = evaluate(_attestation(), _q(), now=when, verifier=_verifier())
    assert d.state is AttributeState.ACCEPTED


def test_tampered_attestation_refuses():
    a = json.loads(json.dumps(_attestation()))
    a["attributes"][0]["value"] = "human"
    d = evaluate(a, _q(), now=NOW, verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is AttributeReason.SIGNATURE_INVALID


def test_attestation_from_another_key_refuses():
    d = evaluate(_attestation(seed=1), _q(), now=NOW, verifier=_verifier(0))
    assert d.state is AttributeState.REFUSED
    assert d.reason is AttributeReason.SIGNATURE_INVALID


def test_no_verifier_refuses_rather_than_accepting():
    d = evaluate(_attestation(), _q(), now=NOW, verifier=None)
    assert d.state is AttributeState.REFUSED
    assert d.reason is AttributeReason.KEY_ABSENT


def test_malformed_attestation_refuses():
    d = evaluate({"schema": "something.else/v0"}, _q(), now=NOW, verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is AttributeReason.ATTESTATION_MALFORMED


def test_unknown_standing_is_malformed_not_silently_floored():
    # A verifier that quietly downgrades a standing it does not recognise hands
    # a forger a way to introduce one.
    a = json.loads(json.dumps(_attestation()))
    a["attributes"][0]["source"] = "certified_by_us"
    d = evaluate(a, _q(), now=NOW, verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is AttributeReason.ATTESTATION_MALFORMED


# --- ordering ---------------------------------------------------------------


def test_a_broken_signature_outranks_an_expired_window():
    a = json.loads(json.dumps(_attestation()))
    a["attributes"][0]["value"] = "human"
    d = evaluate(a, _q(), now=LATER, verifier=_verifier())
    assert d.reason is AttributeReason.SIGNATURE_INVALID


def test_an_expired_window_outranks_a_missing_attribute():
    d = evaluate(_attestation(), _q("modelLicence"), now=LATER, verifier=_verifier())
    assert d.reason is AttributeReason.OUTSIDE_VALIDITY_WINDOW


# --- the decision is portable ------------------------------------------------


def test_decision_serialises_to_the_wire_shape():
    from vaara.attestation._attest_canonical import canonical_json

    d = evaluate(_attestation(), _q(), now=NOW, verifier=_verifier())
    wire = d.to_dict()
    assert wire["state"] == "accepted"
    assert wire["reason"] == "attribute_attested"
    assert wire["source"] == "measured"
    assert wire["evaluatedAt"] == NOW
    assert json.loads(canonical_json(wire).decode()) == wire
