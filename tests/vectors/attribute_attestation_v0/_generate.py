"""Regenerate the attribute_attestation_v0 conformance vectors.

Eight cases pin the property the format exists for: a signed value is only worth
what its source is worth, and a value that falls short of what a relying party
asked for must not be reported the same way as a forged one.

pos_measured_clears_floor      a measured value against a measured floor -> accepted
neg_operator_declared_below    the supplier typed it in                  -> withheld
neg_attribute_absent           the attribute is not in the attestation   -> withheld
neg_subject_mismatch           right attribute, wrong subject            -> withheld
neg_issuer_not_accepted        right attribute, unaccepted issuer        -> withheld
neg_expired_window             outside notBefore..notAfter               -> expired
neg_tampered_value             a value edited after signing              -> refused
neg_unknown_standing           a standing the closed set does not carry  -> refused

The scenario is deliberately the Helsinki one: an alarm classified as an animal.
The same classification is evidence when a named model produced it and is not
evidence when the party being paid to reduce alarms typed it in. Both are signed,
both verify, and the corpus grades the difference.

Run: python3 tests/vectors/attribute_attestation_v0/_generate.py
"""
from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from vaara.attestation.attribute import (
    Attribute,
    SourceStanding,
    Subject,
    emit_attribute_attestation,
)
from vaara.audit.signer import Ed25519Signer

HERE = Path(__file__).resolve().parent

ISSUER_SEED = bytes(range(32))
ISSUER = "kiinteistoautomaatio.example"
SUBJECT = Subject(kind="detection-event", id="alarm-2026-08-21-0413-b17")
NOT_BEFORE = "2026-08-21T04:13:00Z"
NOT_AFTER = "2026-09-21T04:13:00Z"
NOW_OPEN = "2026-08-21T06:00:00Z"
NOW_CLOSED = "2026-10-01T00:00:00Z"

ATTRS = (
    Attribute("triggerClass", "animal", SourceStanding.MEASURED,
              "fauna-classifier v3.1.0"),
    Attribute("sensorId", "PIR-B17-04", SourceStanding.PROTOCOL_DEFINED),
    Attribute("confidence", "0.94", SourceStanding.MEASURED,
              "fauna-classifier v3.1.0"),
    Attribute("operatorNote", "toistuva", SourceStanding.OPERATOR_DECLARED),
)


def _key():
    return ed25519.Ed25519PrivateKey.from_private_bytes(ISSUER_SEED)


def _attestation():
    return emit_attribute_attestation(
        signer=Ed25519Signer(_key()),
        issuer=ISSUER,
        attestation_id="att-b17-0413",
        subject=SUBJECT,
        attributes=ATTRS,
        not_before=NOT_BEFORE,
        not_after=NOT_AFTER,
    )


def _case(att, *, name, minimum, state, reason,
          now=NOW_OPEN, subject_id=None, accepted_issuers=None):
    return {
        "attestation": att,
        "expected_reason": reason,
        "expected_state": state,
        "issuer_key": "keys/ed25519_public.pem",
        "now": now,
        "query": {
            "accepted_issuers": accepted_issuers,
            "minimum_source": minimum,
            "name": name,
            "subject_id": subject_id,
        },
    }


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    keys = HERE / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    (keys / "ed25519_public.pem").write_bytes(
        _key().public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    cases = HERE / "cases"
    att = _attestation()

    _write(cases / "pos_measured_clears_floor.json",
           _case(att, name="triggerClass", minimum="measured",
                 state="accepted", reason="attribute_attested"))

    _write(cases / "neg_operator_declared_below.json",
           _case(att, name="operatorNote", minimum="measured",
                 state="withheld", reason="source_below_floor"))

    _write(cases / "neg_attribute_absent.json",
           _case(att, name="modelLicence", minimum="measured",
                 state="withheld", reason="attribute_absent"))

    _write(cases / "neg_subject_mismatch.json",
           _case(att, name="triggerClass", minimum="measured",
                 subject_id="alarm-somewhere-else",
                 state="withheld", reason="subject_mismatch"))

    _write(cases / "neg_issuer_not_accepted.json",
           _case(att, name="triggerClass", minimum="measured",
                 accepted_issuers=["joku.muu.example"],
                 state="withheld", reason="issuer_not_accepted"))

    _write(cases / "neg_expired_window.json",
           _case(att, name="triggerClass", minimum="measured", now=NOW_CLOSED,
                 state="expired", reason="outside_validity_window"))

    tampered = json.loads(json.dumps(att))
    for a in tampered["attributes"]:
        if a["name"] == "triggerClass":
            a["value"] = "human"
    _write(cases / "neg_tampered_value.json",
           _case(tampered, name="triggerClass", minimum="measured",
                 state="refused", reason="signature_invalid"))

    unknown = json.loads(json.dumps(att))
    for a in unknown["attributes"]:
        if a["name"] == "operatorNote":
            a["source"] = "certified_by_us"
    _write(cases / "neg_unknown_standing.json",
           _case(unknown, name="triggerClass", minimum="measured",
                 state="refused", reason="attestation_malformed"))

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
