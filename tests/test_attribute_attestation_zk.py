# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The attribute attestation with the value taken out of it.

A qualified attestation provider has to hold what it attests, and anything held
can be sold. This format commits to the value at issuance and hands the opening
to the holder, so the issuer has nothing left to sell. A relying party still
learns that a predicate holds and how strongly the value was sourced.

The property under test above the individual cases: the wire carries no value,
the standing stays in the clear and stays gradeable, a proof does not move to
another document, and a predicate that does not hold has no proof.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.attestation.attribute_zk import (  # noqa: E402
    MAX_VALUE,
    PROOF_SCHEMA,
    REASON_STATE,
    SCHEMA,
    AttributeState,
    AttributeValue,
    Opening,
    Predicate,
    PredicateKind,
    PredicateQuery,
    PredicateReason,
    SourceStanding,
    Subject,
    attestation_digest,
    evaluate,
    issue,
    open_predicate,
    verify_attestation_signature,
    verify_predicate,
)
from vaara.audit.signer import Ed25519Signer, Ed25519Verifier  # noqa: E402

ISSUER = "vakuutus.example"
SUBJECT = Subject(kind="person-pseudonym", id="holder-7f3c")
NOW = "2026-08-21T06:00:00Z"
NOT_BEFORE = "2026-08-21T04:13:00Z"
NOT_AFTER = "2026-09-21T04:13:00Z"
LATER = "2026-10-01T00:00:00Z"
EARLIER = "2026-08-20T00:00:00Z"

VALUES = (
    AttributeValue("age", 37, SourceStanding.PROTOCOL_DEFINED, "passport MRZ"),
    AttributeValue("residencyYears", 12, SourceStanding.MEASURED, "population register"),
    AttributeValue("selfReportedIncome", 48000, SourceStanding.OPERATOR_DECLARED),
)

AT_LEAST_18 = Predicate(PredicateKind.AT_LEAST, lower=18)


def _key(seed=0):
    return ed25519.Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)


def _verifier(seed=0):
    return Ed25519Verifier(_key(seed).public_key().public_bytes_raw())


def _issue(values=VALUES, *, seed=0, not_before=NOT_BEFORE, not_after=NOT_AFTER):
    return issue(
        signer=Ed25519Signer(_key(seed)),
        issuer=ISSUER,
        attestation_id="att-holder-7f3c",
        subject=SUBJECT,
        values=values,
        not_before=not_before,
        not_after=not_after,
    )


def _opening(issued, name="age") -> Opening:
    return next(o for o in issued.release_to_holder() if o.name == name)


def _q(name="age", predicate=AT_LEAST_18,
       minimum=SourceStanding.PROTOCOL_DEFINED, **kw):
    return PredicateQuery(
        name=name, predicate=predicate, minimum_source=minimum, **kw
    )


# --- the partition ----------------------------------------------------------


def test_every_reason_maps_to_exactly_one_state():
    assert set(REASON_STATE) == set(PredicateReason)
    assert set(REASON_STATE.values()) == set(AttributeState)


def test_insufficient_and_broken_never_share_a_state():
    withheld = {r for r, s in REASON_STATE.items() if s is AttributeState.WITHHELD}
    refused = {r for r, s in REASON_STATE.items() if s is AttributeState.REFUSED}
    assert not withheld & refused
    assert REASON_STATE[PredicateReason.SOURCE_BELOW_FLOOR] is AttributeState.WITHHELD
    assert REASON_STATE[PredicateReason.PROOF_ABSENT] is AttributeState.WITHHELD
    assert REASON_STATE[PredicateReason.PROOF_INVALID] is AttributeState.REFUSED


def test_only_one_reason_accepts():
    accepting = [r for r, s in REASON_STATE.items() if s is AttributeState.ACCEPTED]
    assert accepting == [PredicateReason.PREDICATE_PROVEN]


# --- the document carries no value ------------------------------------------


def test_no_attribute_value_reaches_the_wire():
    a = _issue().attestation
    for attr in a["attributes"]:
        assert "value" not in attr
        assert set(attr) <= {"name", "commitment", "source", "sourceDetail"}
        assert len(attr["commitment"]) == 66
        bytes.fromhex(attr["commitment"])

    # Blank the two opaque hex fields, which are allowed to contain any digits,
    # and no plaintext value is left anywhere in the document.
    scrubbed = json.loads(json.dumps(a))
    scrubbed["signature"] = ""
    for attr in scrubbed["attributes"]:
        attr["commitment"] = ""
    blob = json.dumps(scrubbed)
    for value in ("37", "12", "48000"):
        assert value not in blob


def test_source_and_detail_stay_in_the_clear():
    a = _issue().attestation
    age = next(x for x in a["attributes"] if x["name"] == "age")
    assert age["source"] == "protocol_defined"
    assert age["sourceDetail"] == "passport MRZ"


def test_attestation_is_signed_and_content_addressed():
    a = _issue().attestation
    assert a["schema"] == SCHEMA
    assert a["proofSystem"] == "vaara-p256-cap-v0"
    assert verify_attestation_signature(a, verifier=_verifier())
    assert attestation_digest(a).startswith("sha256:")


def test_attributes_are_emitted_sorted_so_two_issuers_agree():
    names = [x["name"] for x in _issue().attestation["attributes"]]
    assert names == sorted(names)


def test_the_same_value_commits_differently_every_time():
    # A commitment is perfectly hiding only if the blind is fresh. Two issuances
    # of the same value must not produce the same 33 bytes, or a relying party
    # holding both learns they are equal.
    first = _issue().attestation["attributes"]
    second = _issue().attestation["attributes"]
    assert [x["commitment"] for x in first] != [x["commitment"] for x in second]


# --- the issuance ritual ----------------------------------------------------


def test_releasing_to_the_holder_leaves_the_issuer_holding_nothing():
    issued = _issue()
    assert issued.retains_openings is True
    openings = issued.release_to_holder()
    assert {o.name for o in openings} == {v.name for v in VALUES}
    assert issued.retains_openings is False
    with pytest.raises(AttestationError):
        issued.release_to_holder()


def test_the_attestation_survives_the_release():
    issued = _issue()
    before = json.dumps(issued.attestation, sort_keys=True)
    issued.release_to_holder()
    assert json.dumps(issued.attestation, sort_keys=True) == before


def test_the_opening_actually_opens_the_commitment():
    from vaara.attestation.zk._commit import commit

    issued = _issue()
    a = issued.attestation
    for o in issued.release_to_holder():
        published = next(x for x in a["attributes"] if x["name"] == o.name)
        assert commit(o.value, o.blind).to_bytes().hex() == published["commitment"]


def test_opening_round_trips_through_its_wire_shape():
    o = _opening(_issue())
    assert Opening.from_dict(o.to_dict()) == o


# --- construction limits ----------------------------------------------------


def test_a_value_outside_the_committed_range_is_refused():
    with pytest.raises(AttestationError):
        _issue(values=(AttributeValue("big", MAX_VALUE, SourceStanding.MEASURED),))


def test_a_negative_value_is_refused():
    with pytest.raises(AttestationError):
        _issue(values=(AttributeValue("neg", -1, SourceStanding.MEASURED),))


def test_a_non_integer_value_is_refused():
    with pytest.raises(AttestationError):
        AttributeValue("age", "37", SourceStanding.MEASURED)  # type: ignore[arg-type]


def test_a_boolean_is_not_an_integer_here():
    with pytest.raises(AttestationError):
        AttributeValue("flag", True, SourceStanding.MEASURED)  # type: ignore[arg-type]


def test_empty_attestation_is_refused_at_construction():
    with pytest.raises(AttestationError):
        _issue(values=())


def test_duplicate_names_are_refused():
    with pytest.raises(AttestationError):
        _issue(values=(
            AttributeValue("x", 1, SourceStanding.MEASURED),
            AttributeValue("x", 2, SourceStanding.MEASURED),
        ))


def test_backwards_window_is_refused():
    with pytest.raises(AttestationError):
        _issue(not_before=NOT_AFTER, not_after=NOT_BEFORE)


@pytest.mark.parametrize("bad", [
    {"kind": PredicateKind.AT_LEAST, "upper": 18},
    {"kind": PredicateKind.AT_MOST, "lower": 18},
    {"kind": PredicateKind.IN_RANGE, "lower": 18},
    {"kind": PredicateKind.IN_RANGE, "lower": 40, "upper": 18},
    {"kind": PredicateKind.AT_LEAST, "lower": -1},
    {"kind": PredicateKind.AT_LEAST, "lower": MAX_VALUE},
])
def test_a_predicate_missing_or_reversing_its_bounds_is_refused(bad):
    with pytest.raises(AttestationError):
        Predicate(**bad)


# --- proving ----------------------------------------------------------------


@pytest.mark.parametrize("predicate", [
    Predicate(PredicateKind.AT_LEAST, lower=18),
    Predicate(PredicateKind.AT_LEAST, lower=37),
    Predicate(PredicateKind.AT_MOST, lower=None, upper=37),
    Predicate(PredicateKind.AT_MOST, upper=120),
    Predicate(PredicateKind.IN_RANGE, lower=18, upper=65),
])
def test_a_true_predicate_proves_and_verifies(predicate):
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), predicate)
    assert proof["schema"] == PROOF_SCHEMA
    assert verify_predicate(issued.attestation, proof) is True


@pytest.mark.parametrize("predicate", [
    Predicate(PredicateKind.AT_LEAST, lower=38),
    Predicate(PredicateKind.AT_MOST, upper=36),
    Predicate(PredicateKind.IN_RANGE, lower=18, upper=30),
])
def test_a_false_predicate_has_no_proof(predicate):
    # The prover refuses rather than emitting something that will not verify.
    # A lie has no witness, the same anchor the decision circuit uses.
    issued = _issue()
    with pytest.raises(AttestationError):
        open_predicate(issued.attestation, _opening(issued), predicate)


def test_an_opening_that_does_not_open_the_commitment_is_refused():
    issued = _issue()
    real = _opening(issued)
    with pytest.raises(AttestationError):
        open_predicate(
            issued.attestation,
            Opening(name=real.name, value=real.value + 1, blind=real.blind),
            AT_LEAST_18,
        )


def test_proving_over_an_absent_attribute_is_refused():
    issued = _issue()
    real = _opening(issued)
    with pytest.raises(AttestationError):
        open_predicate(
            issued.attestation,
            Opening(name="notThere", value=real.value, blind=real.blind),
            AT_LEAST_18,
        )


def test_the_proof_reveals_no_value():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    assert "value" not in json.dumps(proof)
    assert proof["predicate"] == {"kind": "at_least", "lower": 18}


# --- the proof does not travel ----------------------------------------------


def test_a_proof_does_not_verify_against_another_attestation():
    # Same subject, same values, a second issuance. The Fiat-Shamir prefix binds
    # to the attestation digest, so the proof is not portable.
    first, second = _issue(), _issue()
    proof = open_predicate(first.attestation, _opening(first), AT_LEAST_18)
    assert verify_predicate(second.attestation, proof) is False


def test_a_proof_does_not_verify_for_another_attribute():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    moved = dict(proof, name="residencyYears")
    assert verify_predicate(issued.attestation, moved) is False


def test_a_proof_does_not_verify_for_a_weaker_threshold():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    restated = dict(proof, predicate={"kind": "at_least", "lower": 65})
    assert verify_predicate(issued.attestation, restated) is False


def test_an_edited_commitment_breaks_the_proof():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    a = json.loads(json.dumps(issued.attestation))
    other = _issue()
    for x in a["attributes"]:
        if x["name"] == "age":
            x["commitment"] = next(
                y["commitment"] for y in other.attestation["attributes"]
                if y["name"] == "age"
            )
    assert verify_predicate(a, proof) is False


@pytest.mark.parametrize("mutation", [
    {"proof": "00" * 10},
    {"proof": "zz"},
    {"proofSystem": "something-else"},
    {"schema": "vaara.something/v0"},
])
def test_a_mangled_proof_envelope_does_not_verify(mutation):
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    assert verify_predicate(issued.attestation, dict(proof, **mutation)) is False


# --- evaluation -------------------------------------------------------------


def _decide(issued, proof, query=None, *, now=NOW, verifier_seed=0, keyed=True):
    return evaluate(
        issued.attestation,
        query or _q(),
        proof=proof,
        now=now,
        verifier=_verifier(verifier_seed) if keyed else None,
    )


def test_a_proven_predicate_over_a_strong_source_accepts():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    d = _decide(issued, proof)
    assert d.state is AttributeState.ACCEPTED
    assert d.reason is PredicateReason.PREDICATE_PROVEN
    assert d.accepted is True
    assert d.source is SourceStanding.PROTOCOL_DEFINED
    assert d.predicate == {"kind": "at_least", "lower": 18}


def test_the_decision_carries_no_value():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    wire = _decide(issued, proof).to_dict()
    assert "value" not in wire
    assert set(wire) == {"schema", "state", "reason", "attestationDigest",
                         "evaluatedAt", "source", "predicate"}
    assert "37" not in json.dumps(dict(wire, attestationDigest=""))


def test_no_proof_withholds_rather_than_refusing():
    # Nothing was proved. That is not the same fact as a forgery and must not
    # read as one, and it must not read as green either.
    issued = _issue()
    d = _decide(issued, None)
    assert d.state is AttributeState.WITHHELD
    assert d.reason is PredicateReason.PROOF_ABSENT


def test_a_proof_that_does_not_verify_refuses():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    broken = dict(proof)
    raw = bytearray(bytes.fromhex(broken["proof"]))
    raw[-1] ^= 0x01
    broken["proof"] = raw.hex()
    d = _decide(issued, broken)
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.PROOF_INVALID


def test_a_proof_from_another_attestation_refuses_as_unbound():
    first, second = _issue(), _issue()
    proof = open_predicate(first.attestation, _opening(first), AT_LEAST_18)
    d = evaluate(second.attestation, _q(), proof=proof, now=NOW,
                 verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.PROOF_NOT_BOUND


def test_a_proof_of_a_different_predicate_refuses_as_unbound():
    issued = _issue()
    proof = open_predicate(
        issued.attestation, _opening(issued),
        Predicate(PredicateKind.AT_LEAST, lower=1),
    )
    d = _decide(issued, proof, _q(predicate=AT_LEAST_18))
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.PROOF_NOT_BOUND


def test_a_structurally_broken_proof_envelope_refuses_as_malformed():
    issued = _issue()
    d = _decide(issued, {"schema": PROOF_SCHEMA})
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.PROOF_MALFORMED


def test_a_valid_proof_over_a_weak_source_withholds():
    # The proof is sound. The value was typed in by the party being judged, and
    # a relying party that asked for measured has not been answered.
    issued = _issue()
    opening = _opening(issued, "selfReportedIncome")
    predicate = Predicate(PredicateKind.AT_LEAST, lower=30000)
    proof = open_predicate(issued.attestation, opening, predicate)
    d = _decide(issued, proof, _q("selfReportedIncome", predicate,
                                  SourceStanding.MEASURED))
    assert d.state is AttributeState.WITHHELD
    assert d.reason is PredicateReason.SOURCE_BELOW_FLOOR
    assert d.source is SourceStanding.OPERATOR_DECLARED


def test_the_same_proof_accepts_when_the_caller_accepts_that_floor():
    issued = _issue()
    opening = _opening(issued, "selfReportedIncome")
    predicate = Predicate(PredicateKind.AT_LEAST, lower=30000)
    proof = open_predicate(issued.attestation, opening, predicate)
    d = _decide(issued, proof, _q("selfReportedIncome", predicate,
                                  SourceStanding.OPERATOR_DECLARED))
    assert d.state is AttributeState.ACCEPTED


def test_an_absent_attribute_withholds():
    issued = _issue()
    d = _decide(issued, None, _q("modelLicence"))
    assert d.state is AttributeState.WITHHELD
    assert d.reason is PredicateReason.ATTRIBUTE_ABSENT


def test_subject_mismatch_withholds():
    issued = _issue()
    d = _decide(issued, None, _q(subject_id="someone-else"))
    assert d.state is AttributeState.WITHHELD
    assert d.reason is PredicateReason.SUBJECT_MISMATCH


def test_unaccepted_issuer_withholds():
    issued = _issue()
    d = _decide(issued, None, _q(accepted_issuers=frozenset({"muu.example"})))
    assert d.state is AttributeState.WITHHELD
    assert d.reason is PredicateReason.ISSUER_NOT_ACCEPTED


@pytest.mark.parametrize("when", [EARLIER, LATER])
def test_outside_the_window_expires(when):
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    d = _decide(issued, proof, now=when)
    assert d.state is AttributeState.EXPIRED
    assert d.reason is PredicateReason.OUTSIDE_VALIDITY_WINDOW


@pytest.mark.parametrize("when", [NOT_BEFORE, NOT_AFTER])
def test_the_window_is_inclusive_at_both_ends(when):
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    assert _decide(issued, proof, now=when).state is AttributeState.ACCEPTED


def test_an_edited_standing_refuses_on_the_signature():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    a = json.loads(json.dumps(issued.attestation))
    for x in a["attributes"]:
        if x["name"] == "selfReportedIncome":
            x["source"] = "measured"
    d = evaluate(a, _q(), proof=proof, now=NOW, verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.SIGNATURE_INVALID


def test_an_edited_commitment_refuses_on_the_signature():
    issued, other = _issue(), _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    a = json.loads(json.dumps(issued.attestation))
    for x in a["attributes"]:
        if x["name"] == "age":
            x["commitment"] = next(
                y["commitment"] for y in other.attestation["attributes"]
                if y["name"] == "age"
            )
    d = evaluate(a, _q(), proof=proof, now=NOW, verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.SIGNATURE_INVALID


def test_no_verifier_refuses_rather_than_accepting():
    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    d = _decide(issued, proof, keyed=False)
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.KEY_ABSENT


def test_malformed_attestation_refuses():
    issued = _issue()
    d = evaluate({"schema": "something.else/v0"}, _q(), proof=None, now=NOW,
                 verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.ATTESTATION_MALFORMED
    assert issued.attestation["schema"] == SCHEMA


def test_unknown_standing_is_malformed_not_silently_floored():
    issued = _issue()
    a = json.loads(json.dumps(issued.attestation))
    a["attributes"][0]["source"] = "certified_by_us"
    d = evaluate(a, _q(), proof=None, now=NOW, verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.ATTESTATION_MALFORMED


def test_a_commitment_that_is_not_a_curve_point_is_malformed():
    issued = _issue()
    a = json.loads(json.dumps(issued.attestation))
    a["attributes"][0]["commitment"] = "02" + "ff" * 32
    d = evaluate(a, _q(), proof=None, now=NOW, verifier=_verifier())
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.ATTESTATION_MALFORMED


# --- ordering ---------------------------------------------------------------


def test_a_broken_signature_outranks_an_expired_window():
    issued = _issue()
    a = json.loads(json.dumps(issued.attestation))
    a["attributes"][0]["sourceDetail"] = "edited"
    d = evaluate(a, _q(), proof=None, now=LATER, verifier=_verifier())
    assert d.reason is PredicateReason.SIGNATURE_INVALID


def test_an_expired_window_outranks_a_missing_proof():
    issued = _issue()
    d = _decide(issued, None, now=LATER)
    assert d.reason is PredicateReason.OUTSIDE_VALIDITY_WINDOW


def test_a_broken_proof_outranks_a_source_below_the_floor():
    # A forged proof is a worse fact than a weak source, and reporting the weak
    # source would let a forgery leave the room as merely insufficient.
    issued = _issue()
    opening = _opening(issued, "selfReportedIncome")
    predicate = Predicate(PredicateKind.AT_LEAST, lower=30000)
    proof = open_predicate(issued.attestation, opening, predicate)
    raw = bytearray(bytes.fromhex(proof["proof"]))
    raw[-1] ^= 0x01
    proof["proof"] = raw.hex()
    d = _decide(issued, proof, _q("selfReportedIncome", predicate,
                                  SourceStanding.MEASURED))
    assert d.state is AttributeState.REFUSED
    assert d.reason is PredicateReason.PROOF_INVALID


# --- the decision is portable ------------------------------------------------


def test_decision_serialises_to_the_wire_shape():
    from vaara.attestation._attest_canonical import canonical_json

    issued = _issue()
    proof = open_predicate(issued.attestation, _opening(issued), AT_LEAST_18)
    wire = _decide(issued, proof).to_dict()
    assert wire["state"] == "accepted"
    assert wire["reason"] == "predicate_proven"
    assert wire["source"] == "protocol_defined"
    assert wire["evaluatedAt"] == NOW
    assert json.loads(canonical_json(wire).decode()) == wire
