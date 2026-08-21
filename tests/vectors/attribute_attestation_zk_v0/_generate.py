"""Regenerate the attribute_attestation_zk_v0 conformance vectors.

Ten cases pin the property the format exists for: an issuer that commits to a
value and hands the opening to the holder has nothing left to sell, and a relying
party that learns only "the predicate holds, over a value sourced this strongly"
has learned everything it was entitled to.

pos_at_least_holds        age >= 18, proved over a hidden value  -> accepted
pos_in_range_holds        residency in [5, 20], two directions   -> accepted
neg_proof_absent          nothing was proved                     -> withheld
neg_attribute_absent      a predicate over an attribute not here -> withheld
neg_source_below_floor    sound proof, the subject typed it in   -> withheld
neg_expired_window        outside notBefore..notAfter            -> expired
neg_predicate_false       a proof of a statement that is untrue  -> refused
neg_proof_replayed        another document's proof presented here-> refused
neg_tampered_commitment   a commitment swapped after signing     -> refused
neg_tampered_standing     operator_declared edited to measured   -> refused

Determinism. Ed25519 signing is already deterministic, and the two other sources
of randomness are pinned here so the committed bytes reproduce exactly: the
commitment blinds are derived from a fixed seed and passed in, and the prover's
internal scalars are drawn from a seeded hash chain installed over
``random_scalar`` for the length of this script. Neither substitution exists in
the library, where a blind that is not fresh is a linkage bug.

Run: python3 tests/vectors/attribute_attestation_zk_v0/_generate.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from vaara.attestation.zk import _prove
from vaara.attestation.zk._group import N
from vaara.audit.signer import Ed25519Signer

HERE = Path(__file__).resolve().parent

ISSUER_SEED = bytes(range(32))
ISSUER = "henkilotodistus.example"
SUBJECT_KIND = "person-pseudonym"
SUBJECT_ID = "holder-7f3c"
NOT_BEFORE = "2026-08-21T04:13:00Z"
NOT_AFTER = "2026-09-21T04:13:00Z"
NOW_OPEN = "2026-08-21T06:00:00Z"
NOW_CLOSED = "2026-10-01T00:00:00Z"

DRBG_SEED = b"vaara/vectors/attribute-attestation-zk/v0"


class _SeededScalars:
    """A hash chain standing in for the CSPRNG, so a proof reproduces byte for
    byte. Installed over the prover's ``random_scalar`` and removed afterwards."""

    def __init__(self, label: bytes):
        self._label = label
        self._counter = 0

    def __call__(self) -> int:
        self._counter += 1
        digest = hashlib.sha256(
            DRBG_SEED + b"/" + self._label + b"/" + self._counter.to_bytes(8, "big")
        ).digest()
        return int.from_bytes(digest, "big") % N


def _blind(label: str) -> int:
    return int.from_bytes(
        hashlib.sha256(DRBG_SEED + b"/blind/" + label.encode()).digest(), "big"
    ) % N


def _key():
    return ed25519.Ed25519PrivateKey.from_private_bytes(ISSUER_SEED)


def _values():
    from vaara.attestation.attribute_zk import AttributeValue, SourceStanding

    return (
        AttributeValue("age", 37, SourceStanding.PROTOCOL_DEFINED, "passport MRZ"),
        AttributeValue("residencyYears", 12, SourceStanding.MEASURED,
                       "population register"),
        AttributeValue("selfReportedIncome", 48000,
                       SourceStanding.OPERATOR_DECLARED),
    )


def _issue(attestation_id: str, salt: str):
    from vaara.attestation.attribute_zk import Subject, issue

    values = _values()
    return issue(
        signer=Ed25519Signer(_key()),
        issuer=ISSUER,
        attestation_id=attestation_id,
        subject=Subject(kind=SUBJECT_KIND, id=SUBJECT_ID),
        values=values,
        not_before=NOT_BEFORE,
        not_after=NOT_AFTER,
        blinds={v.name: _blind(f"{salt}/{v.name}") for v in values},
    )


def _prove_with(label: str, fn):
    """Run one proving call under a seeded scalar source."""
    original = _prove.random_scalar
    _prove.random_scalar = _SeededScalars(label.encode())
    try:
        return fn()
    finally:
        _prove.random_scalar = original


def _case(attestation, *, name, predicate, minimum, state, reason,
          proof=None, now=NOW_OPEN, subject_id=None, accepted_issuers=None):
    return {
        "attestation": attestation,
        "expected_reason": reason,
        "expected_state": state,
        "issuer_key": "keys/ed25519_public.pem",
        "now": now,
        "proof": proof,
        "query": {
            "accepted_issuers": accepted_issuers,
            "minimum_source": minimum,
            "name": name,
            "predicate": predicate,
            "subject_id": subject_id,
        },
    }


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    from vaara.attestation._attribute_attestation_zk import (
        _range_prove,
        _transcript,
    )
    from vaara.attestation.attribute_zk import (
        PROOF_SCHEMA,
        PROOF_SYSTEM,
        Predicate,
        PredicateKind,
        attestation_digest,
        open_predicate,
    )
    from vaara.attestation.zk._params import params_digest

    keys = HERE / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    (keys / "ed25519_public.pem").write_bytes(
        _key().public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    cases = HERE / "cases"

    issued = _issue("att-holder-7f3c", "primary")
    att = issued.attestation
    openings = {o.name: o for o in issued.release_to_holder()}

    at_least_18 = Predicate(PredicateKind.AT_LEAST, lower=18)
    in_range_5_20 = Predicate(PredicateKind.IN_RANGE, lower=5, upper=20)
    at_least_30000 = Predicate(PredicateKind.AT_LEAST, lower=30000)

    age_proof = _prove_with(
        "age/at_least/18",
        lambda: open_predicate(att, openings["age"], at_least_18),
    )
    residency_proof = _prove_with(
        "residencyYears/in_range/5/20",
        lambda: open_predicate(att, openings["residencyYears"], in_range_5_20),
    )
    income_proof = _prove_with(
        "selfReportedIncome/at_least/30000",
        lambda: open_predicate(att, openings["selfReportedIncome"], at_least_30000),
    )

    _write(cases / "pos_at_least_holds.json",
           _case(att, name="age", predicate=at_least_18.to_dict(),
                 minimum="protocol_defined", proof=age_proof,
                 state="accepted", reason="predicate_proven"))

    _write(cases / "pos_in_range_holds.json",
           _case(att, name="residencyYears", predicate=in_range_5_20.to_dict(),
                 minimum="measured", proof=residency_proof,
                 state="accepted", reason="predicate_proven"))

    # Nothing was proved. Sound document, unanswered question, and the reason it
    # must not read as a forgery is that it is not one.
    _write(cases / "neg_proof_absent.json",
           _case(att, name="age", predicate=at_least_18.to_dict(),
                 minimum="protocol_defined", proof=None,
                 state="withheld", reason="proof_absent"))

    _write(cases / "neg_attribute_absent.json",
           _case(att, name="creditScore", predicate=at_least_18.to_dict(),
                 minimum="measured", proof=None,
                 state="withheld", reason="attribute_absent"))

    # The proof is sound and the value was typed in by the party being judged.
    _write(cases / "neg_source_below_floor.json",
           _case(att, name="selfReportedIncome", predicate=at_least_30000.to_dict(),
                 minimum="measured", proof=income_proof,
                 state="withheld", reason="source_below_floor"))

    _write(cases / "neg_expired_window.json",
           _case(att, name="age", predicate=at_least_18.to_dict(),
                 minimum="protocol_defined", proof=age_proof, now=NOW_CLOSED,
                 state="expired", reason="outside_validity_window"))

    # A proof of something untrue. The honest prover refuses to build this, so
    # the range argument is called directly on the false witness, which is what
    # a forger would have to do. The bits of a negative witness reconstruct to a
    # different point, so the weighted sum misses the target.
    false_predicate = Predicate(PredicateKind.AT_LEAST, lower=40)
    age_opening = openings["age"]
    forged_blob = _prove_with(
        "forgery/age/at_least/40",
        lambda: _range_prove(
            age_opening.value - 40,
            age_opening.blind,
            _transcript(attestation_digest(att), "age", false_predicate, "ge"),
        ),
    )
    _write(cases / "neg_predicate_false.json",
           _case(att, name="age", predicate=false_predicate.to_dict(),
                 minimum="protocol_defined",
                 proof={
                     "attestationDigest": attestation_digest(att),
                     "name": "age",
                     "predicate": false_predicate.to_dict(),
                     "proof": forged_blob.hex(),
                     "proofSystem": PROOF_SYSTEM,
                     "schema": PROOF_SCHEMA,
                     "verifierParamsDigest": params_digest(),
                 },
                 state="refused", reason="proof_invalid"))

    # A second issuance of the same values to the same subject, and its proof
    # presented here. The transcript names the document, so it does not travel.
    other = _issue("att-holder-7f3c-second", "secondary")
    other_openings = {o.name: o for o in other.release_to_holder()}
    replayed = _prove_with(
        "replay/age/at_least/18",
        lambda: open_predicate(
            other.attestation, other_openings["age"], at_least_18
        ),
    )
    _write(cases / "neg_proof_replayed.json",
           _case(att, name="age", predicate=at_least_18.to_dict(),
                 minimum="protocol_defined", proof=replayed,
                 state="refused", reason="proof_not_bound"))

    # Soundness before everything: the proof in both of these is valid, and the
    # document it was made over is not the document being presented.
    swapped = json.loads(json.dumps(att))
    for a in swapped["attributes"]:
        if a["name"] == "age":
            a["commitment"] = next(
                b["commitment"] for b in other.attestation["attributes"]
                if b["name"] == "age"
            )
    _write(cases / "neg_tampered_commitment.json",
           _case(swapped, name="age", predicate=at_least_18.to_dict(),
                 minimum="protocol_defined", proof=age_proof,
                 state="refused", reason="signature_invalid"))

    upgraded = json.loads(json.dumps(att))
    for a in upgraded["attributes"]:
        if a["name"] == "selfReportedIncome":
            a["source"] = "measured"
    _write(cases / "neg_tampered_standing.json",
           _case(upgraded, name="selfReportedIncome",
                 predicate=at_least_30000.to_dict(), minimum="measured",
                 proof=income_proof,
                 state="refused", reason="signature_invalid"))

    expected = {}
    for path in sorted(cases.glob("*.json")):
        c = json.loads(path.read_text(encoding="utf-8"))
        expected[path.stem] = {
            "expected_state": c["expected_state"],
            "expected_reason": c["expected_reason"],
        }
    _write(HERE / "expected.json", {"cases": expected})
    print(f"wrote {len(expected)} cases + expected.json")


if __name__ == "__main__":
    main()
