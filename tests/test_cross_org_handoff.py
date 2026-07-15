"""Cross-org evidence handoff: one org's record, another org's regulator.

Covers ``build_handoff`` / ``verify_handoff`` (content-addressed integrity, the
retained-record verdict routed through the C1 lens, the anchor-to-record
binding, the optional holder custody attestation, the producer-identity pin, and
strict mode), the ``cross_org_handoff_v0`` conformance vectors, and a standalone
Vaara-free checker that reproduces every verdict.

See ``docs/design/cross-org-handoff-spec.md``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation.receipt import (  # noqa: E402
    OutcomeDerived,
    build_handoff,
    emit_receipt,
    make_back_link,
    sign_manifest,
    verify_handoff,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)
from vaara.attestation._attest_canonical import canonical_json  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors" / "cross_org_handoff_v0"

DID = "did:web:vendor-a.example:billing"
KEYID = DID + "#key-2026"
HOLDER = "did:web:customer-b.example"
IAT = "2026-05-29T10:00:00Z"
ACTIVATED = "2026-01-01T00:00:00Z"
RETIRED = "2028-01-01T00:00:00Z"
ANCHOR_OK = "2026-05-29T10:05:00Z"


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _jwk(pk: ec.EllipticCurvePublicKey) -> dict:
    n = pk.public_numbers()
    return {"kty": "EC", "crv": "P-256",
            "x": _b64u(n.x.to_bytes(32, "big")), "y": _b64u(n.y.to_bytes(32, "big"))}


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="charge_card", server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"amount": 4200}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=payload, iss="issuer://test", sub="agent:billing",
        secret_version="v1", alg="HS256", signing_material=b"\x42" * 32,
        nonce="att-nonce-fixed-0001", iat="2026-05-29T09:59:59Z",
    )


def _receipt(key: ec.EllipticCurvePrivateKey) -> dict:
    return emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce="rcpt-nonce-fixed-0001", iat=IAT,
    ).to_dict()


def _method(keyid: str, jwk: dict, **markers: str) -> dict:
    m = {"id": keyid, "type": "JsonWebKey2020", "controller": DID, "publicKeyJwk": jwk}
    m.update(markers)
    return m


def _doc(*methods: dict) -> dict:
    return {"id": DID, "verificationMethod": list(methods)}


def _anchor(record: dict) -> dict:
    rec_hex = hashlib.sha256(canonical_json(record)).hexdigest()
    return {"chain_position": 0, "chain_head_hash": rec_hex, "backend": "eidas-tsa",
            "tsa_url": "https://tsa.example", "hash_algorithm": "sha256",
            "token_b64": "AA==", "anchored_time": ANCHOR_OK}


def _in_window_doc(key):
    return _doc(_method(KEYID, _jwk(key.public_key()),
                        validFrom=ACTIVATED, validUntil=RETIRED))


# ── build_handoff / verify_handoff ───────────────────────────────────────────


def test_clean_package_is_verifiable_not_corroborated():
    key = ec.generate_private_key(ec.SECP256R1())
    pkg = build_handoff(record=_receipt(key), did_document=_in_window_doc(key))
    v = verify_handoff(pkg)
    assert v.integrity_ok and v.verifiable and v.ok
    assert v.corroborated is False
    assert v.producer_identity_basis == "self_asserted_unpinned"
    assert v.custody == "unattested"


def test_bound_anchor_with_verified_time_corroborates():
    key = ec.generate_private_key(ec.SECP256R1())
    rec = _receipt(key)
    pkg = build_handoff(record=rec, did_document=_in_window_doc(key), anchor=_anchor(rec))
    v = verify_handoff(pkg, anchor_attested_time=ANCHOR_OK)
    assert v.anchor_present and v.anchor_binds and v.anchor_verified
    assert v.corroborated and v.ok


def test_anchor_present_but_unverified_stays_verifiable():
    key = ec.generate_private_key(ec.SECP256R1())
    rec = _receipt(key)
    pkg = build_handoff(record=rec, did_document=_in_window_doc(key), anchor=_anchor(rec))
    v = verify_handoff(pkg)  # no anchor_attested_time
    assert v.anchor_binds and v.anchor_verified is False
    assert v.verifiable and not v.corroborated


def test_anchor_over_a_different_record_does_not_bind():
    key = ec.generate_private_key(ec.SECP256R1())
    rec = _receipt(key)
    pkg = build_handoff(record=rec, did_document=_in_window_doc(key), anchor=_anchor(rec))
    pkg["evidence"]["anchor"]["chain_head_hash"] = "ab" * 32
    pkg["manifest"]["anchor_digest"] = (
        "sha256:" + hashlib.sha256(canonical_json(pkg["evidence"]["anchor"])).hexdigest())
    pkg["manifest_digest"] = (
        "sha256:" + hashlib.sha256(canonical_json(pkg["manifest"])).hexdigest())
    v = verify_handoff(pkg, anchor_attested_time=ANCHOR_OK)
    assert v.integrity_ok and v.anchor_binds is False
    assert v.verifiable and v.corroborated is False


def test_build_rejects_anchor_that_does_not_bind():
    key = ec.generate_private_key(ec.SECP256R1())
    bad = _anchor(_receipt(key))
    bad["chain_head_hash"] = "00" * 32
    with pytest.raises(ValueError, match="anchor does not bind"):
        build_handoff(record=_receipt(key), did_document=_in_window_doc(key), anchor=bad)


def test_tampered_component_fails_integrity():
    key = ec.generate_private_key(ec.SECP256R1())
    pkg = build_handoff(record=_receipt(key), did_document=_in_window_doc(key))
    pkg["evidence"]["did_document"]["verificationMethod"][0]["validUntil"] = \
        "2099-01-01T00:00:00Z"
    v = verify_handoff(pkg)
    assert v.integrity_ok is False and v.ok is False
    drift = [c.name for c in v.components if not c.ok]
    assert "did_document" in drift and "key_history" in drift


def test_swapped_producer_breaks_coherence():
    key = ec.generate_private_key(ec.SECP256R1())
    pkg = build_handoff(record=_receipt(key), did_document=_in_window_doc(key))
    pkg["manifest"]["producer"] = "did:web:attacker.example"
    v = verify_handoff(pkg)
    assert v.integrity_ok is False


def test_holder_attestation_verifies_but_does_not_gate():
    key = ec.generate_private_key(ec.SECP256R1())
    holder_key = ec.generate_private_key(ec.SECP256R1())
    rec = _receipt(key)
    seed = build_handoff(record=rec, did_document=_in_window_doc(key))
    att = sign_manifest(
        seed["manifest"], alg="ES256", keyid=HOLDER + "#k1",
        signing_material=holder_key, verifying_jwk=_jwk(holder_key.public_key()))
    pkg = build_handoff(record=rec, did_document=_in_window_doc(key),
                        holder_attestation=att)
    v = verify_handoff(pkg)
    assert v.custody == "holder_attested_selfsupplied"
    assert v.holder_keyid == HOLDER + "#k1"
    # Corrupting the custody signature never touches the record verdict.
    pkg["holder_attestation"]["signature"] = "00" * 64
    bad = verify_handoff(pkg)
    assert bad.custody == "holder_attestation_failed"
    assert bad.verifiable and bad.ok


def test_trusted_document_pins_producer_identity():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _in_window_doc(key)
    pkg = build_handoff(record=_receipt(key), did_document=doc)
    pinned = verify_handoff(pkg, trusted_did_document=doc)
    assert pinned.producer_identity_basis == "pinned"
    other = ec.generate_private_key(ec.SECP256R1())
    wrong = verify_handoff(pkg, trusted_did_document=_in_window_doc(other))
    assert wrong.producer_identity_basis == "pin_mismatch"


def test_strict_mode_requires_corroboration_and_pin():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _in_window_doc(key)
    rec = _receipt(key)
    pkg = build_handoff(record=rec, did_document=doc, anchor=_anchor(rec),
                        revocations={"version": 1, "entries": []})
    strict = verify_handoff(pkg, anchor_attested_time=ANCHOR_OK,
                            trusted_did_document=doc, strict=True)
    assert strict.ok and strict.strict
    # Same package without a verified anchor fails strict but stays verifiable.
    weak = verify_handoff(pkg, trusted_did_document=doc, strict=True)
    assert weak.ok is False and weak.verifiable


def test_signed_after_retirement_not_verifiable():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()),
                       validFrom=ACTIVATED, validUntil="2026-05-01T00:00:00Z"))
    pkg = build_handoff(record=_receipt(key), did_document=doc)
    v = verify_handoff(pkg)
    assert v.integrity_ok and v.verifiable is False and v.ok is False


def test_noncanonical_key_history_override_pins_model_digest():
    key = ec.generate_private_key(ec.SECP256R1())
    split = {"version": 1, "keys": [
        {"keyid": KEYID, "not_before": "2026-05-01T00:00:00Z",
         "not_after": "2026-07-01T00:00:00Z"},
        {"keyid": KEYID, "not_before": "2026-01-01T00:00:00Z",
         "not_after": "2026-03-01T00:00:00Z"},
    ]}
    pkg = build_handoff(record=_receipt(key), did_document=_in_window_doc(key),
                        key_history=split)
    v = verify_handoff(pkg)
    assert v.integrity_ok and v.verifiable  # iat is inside the second window


def test_build_rejects_producer_mismatch():
    key = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(ValueError, match="producer"):
        build_handoff(record=_receipt(key), did_document=_in_window_doc(key),
                      producer="did:web:someone-else.example")


def test_verdict_is_json_serializable():
    key = ec.generate_private_key(ec.SECP256R1())
    pkg = build_handoff(record=_receipt(key), did_document=_in_window_doc(key))
    json.dumps(verify_handoff(pkg).to_dict())


def test_malformed_package_raises_valueerror():
    with pytest.raises(ValueError):
        verify_handoff({"schema": "x", "manifest": {}})  # no evidence


def test_malformed_evidence_fails_closed_with_valueerror():
    # A hostile package must fail closed with a named ValueError, never a
    # traceback: a shape-valid but malformed receipt, and override blocks of the
    # wrong shape, are all normalised to ValueError (regression for the
    # AttestationError / AttributeError leak the adversarial review found).
    key = ec.generate_private_key(ec.SECP256R1())
    good = build_handoff(record=_receipt(key), did_document=_in_window_doc(key))

    broken_receipt = copy.deepcopy(good)
    broken_receipt["evidence"]["record"] = {"version": 1}  # missing required blocks
    with pytest.raises(ValueError):
        verify_handoff(broken_receipt)

    bad_revs = copy.deepcopy(good)
    bad_revs["evidence"]["revocations"] = {"entries": [123]}
    with pytest.raises(ValueError):
        verify_handoff(bad_revs)

    bad_kh = copy.deepcopy(good)
    bad_kh["evidence"]["key_history"] = ["not", "an", "object"]
    with pytest.raises(ValueError):
        verify_handoff(bad_kh)


# ── Conformance vectors ──────────────────────────────────────────────────────


def _cases():
    return json.loads((VECTORS / "cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_vaara_reproduces_vector_verdict(case):
    expected = json.loads((VECTORS / "expected.json").read_text())[case["name"]]
    verdict = verify_handoff(
        case["package"],
        anchor_attested_time=case.get("anchoredTime"),
        trusted_did_document=case.get("trustedDidDocument"),
        strict=case.get("strict", False),
    )
    got = verdict.to_dict()
    assert {k: got[k] for k in expected} == expected


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "coh_checker", VECTORS / "_check_independent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_matches_vaara_on_corroborated_package():
    chk = _load_checker()
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _in_window_doc(key)
    rec = _receipt(key)
    pkg = build_handoff(record=rec, did_document=doc, anchor=_anchor(rec),
                        revocations={"version": 1, "entries": []})
    case = {"package": pkg, "anchoredTime": ANCHOR_OK, "trustedDidDocument": doc}
    vaara = verify_handoff(pkg, anchor_attested_time=ANCHOR_OK,
                           trusted_did_document=doc).to_dict()
    got = chk._evaluate(case)
    assert got["corroborated"] is True and got["producer_identity_basis"] == "pinned"
    assert {k: vaara[k] for k in chk.COMPARE} == got
