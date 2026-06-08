"""Verify a retained record under a rotated or retired key (the 7-year problem).

Covers the `KeyHistory` validity-window model, the `verify_receipt_retained`
lens that composes binding, validity window, revocation, and time-anchor
corroboration, and the `key_rotation_v0` conformance vectors (Vaara and a
standalone Vaara-free checker both reproduce the verdicts).

See `docs/design/key-rotation-retention-spec.md`.
"""

from __future__ import annotations

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

import base64  # noqa: E402

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation.receipt import (  # noqa: E402
    KeyHistory,
    KeyValidity,
    OutcomeDerived,
    RevocationEntry,
    RevocationRegistry,
    emit_receipt,
    make_back_link,
    parse_receipt,
    verify_receipt_retained,
    within_validity,
)
from vaara.attestation.sep2787 import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

VECTORS = Path(__file__).resolve().parent / "vectors" / "key_rotation_v0"

DID = "did:web:agents.example.com:billing"
KEYID = DID + "#key-2026"
IAT = "2026-05-29T10:00:00Z"
ACTIVATED = "2026-01-01T00:00:00Z"
RETIRED = "2028-01-01T00:00:00Z"
ANCHOR_OK = "2026-05-29T10:05:00Z"
ANCHOR_LATE = "2028-06-01T00:00:00Z"
BEFORE_IAT = "2026-05-29T09:30:00Z"


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _jwk(public_key: ec.EllipticCurvePublicKey) -> dict:
    nums = public_key.public_numbers()
    return {"kty": "EC", "crv": "P-256",
            "x": _b64u(nums.x.to_bytes(32, "big")),
            "y": _b64u(nums.y.to_bytes(32, "big"))}


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


def _receipt(key: ec.EllipticCurvePrivateKey):
    return emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce="rcpt-nonce-fixed-0001", iat=IAT,
    )


def _method(keyid: str, jwk: dict, **markers: str) -> dict:
    m = {"id": keyid, "type": "JsonWebKey2020", "controller": DID,
         "publicKeyJwk": jwk}
    m.update(markers)
    return m


def _doc(*methods: dict) -> dict:
    return {"id": DID, "verificationMethod": list(methods)}


# ── KeyHistory: the validity-window model ────────────────────────────────────


def test_within_validity_half_open_window():
    assert within_validity(IAT, ACTIVATED, RETIRED) is True
    # not_before is inclusive, not_after is exclusive.
    assert within_validity(ACTIVATED, ACTIVATED, RETIRED) is True
    assert within_validity(RETIRED, ACTIVATED, RETIRED) is False
    assert within_validity("2025-01-01T00:00:00Z", ACTIVATED, RETIRED) is False


def test_within_validity_fails_closed_on_unparseable():
    assert within_validity("not-a-date", ACTIVATED, RETIRED) is False
    assert within_validity(IAT, "garbage", RETIRED) is False
    assert within_validity(IAT, ACTIVATED, "garbage") is False
    # An absent bound is unbounded on that side.
    assert within_validity(IAT, None, None) is True
    assert within_validity(IAT, ACTIVATED, None) is True


def test_key_history_unknown_key_is_unbounded():
    kh = KeyHistory([KeyValidity(KEYID, ACTIVATED, RETIRED)])
    status = kh.validity("did:web:other#k", IAT)
    assert status.within is True
    assert status.recorded is False


def test_key_history_no_keyid_not_checked():
    kh = KeyHistory([KeyValidity(KEYID, ACTIVATED, RETIRED)])
    status = kh.validity(None, IAT)
    assert status.within is True and status.recorded is False


def test_key_history_reports_governing_window():
    kh = KeyHistory([KeyValidity(KEYID, ACTIVATED, RETIRED)])
    status = kh.validity(KEYID, IAT)
    assert status.within is True and status.recorded is True
    assert status.not_before == ACTIVATED and status.not_after == RETIRED
    out = kh.validity(KEYID, "2030-01-01T00:00:00Z")
    assert out.within is False and out.recorded is True


def test_key_history_multiple_windows_any_admits():
    kh = KeyHistory([
        KeyValidity(KEYID, "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"),
        KeyValidity(KEYID, "2026-05-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    ])
    assert kh.validity(KEYID, IAT).within is True               # in the second
    assert kh.validity(KEYID, "2026-04-01T00:00:00Z").within is False  # the gap


def test_key_history_from_did_document_reads_markers_and_aliases():
    jwk = _jwk(ec.generate_private_key(ec.SECP256R1()).public_key())
    doc = _doc(
        _method(KEYID, jwk, validFrom=ACTIVATED, validUntil=RETIRED),
        _method(DID + "#alias", jwk, notBefore=ACTIVATED, notAfter=RETIRED),
        _method(DID + "#plain", jwk),  # no markers -> contributes no entry
    )
    kh = KeyHistory.from_did_document(doc)
    assert len(kh) == 2
    assert kh.validity(KEYID, IAT).not_after == RETIRED
    assert kh.validity(DID + "#alias", IAT).within is True
    assert kh.validity(DID + "#plain", IAT).recorded is False


def test_key_history_digest_stable_and_roundtrips():
    kh = KeyHistory([KeyValidity(KEYID, ACTIVATED, RETIRED)])
    assert kh.digest() == KeyHistory(list(kh.entries)).digest()
    assert KeyHistory.from_dict(kh.to_dict()).to_dict() == kh.to_dict()


def test_key_validity_from_dict_rejects_bad_shapes():
    with pytest.raises(ValueError):
        KeyValidity.from_dict({"keyid": ""})
    with pytest.raises(ValueError):
        KeyValidity.from_dict({"keyid": KEYID, "not_before": 123})


# ── verify_receipt_retained: the retained-record lens ────────────────────────


def test_retired_key_still_verifies_uncorroborated():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()),
                       validFrom=ACTIVATED, validUntil=RETIRED))
    r = verify_receipt_retained(_receipt(key), doc)
    assert r.bound and r.within_window and r.verifiable
    assert r.corroborated is False
    assert r.time_basis == "self_asserted"
    assert r.keyid == KEYID


def test_anchor_before_retirement_corroborates():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()),
                       validFrom=ACTIVATED, validUntil=RETIRED))
    r = verify_receipt_retained(_receipt(key), doc, anchored_time=ANCHOR_OK)
    assert r.verifiable and r.corroborated
    assert r.anchored_before_retirement and r.anchored_before_revocation
    assert r.time_basis == "anchored"


def test_anchor_after_retirement_does_not_corroborate():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()),
                       validFrom=ACTIVATED, validUntil=RETIRED))
    r = verify_receipt_retained(_receipt(key), doc, anchored_time=ANCHOR_LATE)
    assert r.verifiable is True
    assert r.corroborated is False
    assert r.anchored_before_retirement is False


def test_signed_after_retirement_is_out_of_window():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()),
                       validFrom=ACTIVATED, validUntil="2026-05-01T00:00:00Z"))
    r = verify_receipt_retained(_receipt(key), doc)
    assert r.bound is True
    assert r.within_window is False
    assert r.verifiable is False


def test_signed_before_activation_is_out_of_window():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()),
                       validFrom="2026-06-01T00:00:00Z"))
    r = verify_receipt_retained(_receipt(key), doc)
    assert r.within_window is False and r.verifiable is False


def test_revocation_overrides_a_valid_window():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()), validFrom=ACTIVATED,
                       validUntil=RETIRED, revoked=BEFORE_IAT))
    r = verify_receipt_retained(_receipt(key), doc, anchored_time=ANCHOR_OK)
    assert r.within_window is True
    assert r.revoked is True
    assert r.verifiable is False
    assert r.corroborated is False


def test_unbounded_key_verifies_unchanged():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key())))
    r = verify_receipt_retained(_receipt(key), doc)
    assert r.verifiable is True
    assert r.window_recorded is False
    assert r.not_before is None and r.not_after is None


def test_wrong_key_does_not_bind():
    signer = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(other.public_key()), validFrom=ACTIVATED))
    r = verify_receipt_retained(_receipt(signer), doc)
    assert r.bound is False and r.verifiable is False and r.keyid is None


def test_explicit_key_history_and_revocations_override_the_document():
    key = ec.generate_private_key(ec.SECP256R1())
    # The document records no window, but an out-of-band key history retires
    # the key before iat, and an out-of-band registry is empty.
    doc = _doc(_method(KEYID, _jwk(key.public_key())))
    kh = KeyHistory([KeyValidity(KEYID, ACTIVATED, "2026-05-01T00:00:00Z")])
    r = verify_receipt_retained(
        _receipt(key), doc, key_history=kh, revocations=RevocationRegistry(),
    )
    assert r.within_window is False and r.verifiable is False


def test_out_of_band_identity_revocation_applies():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()),
                       validFrom=ACTIVATED, validUntil=RETIRED))
    reg = RevocationRegistry([
        RevocationEntry(scope="identity", subject=DID, revoked_at=BEFORE_IAT),
    ])
    r = verify_receipt_retained(_receipt(key), doc, revocations=reg)
    assert r.revoked is True and r.verifiable is False


def test_to_dict_is_json_serializable():
    key = ec.generate_private_key(ec.SECP256R1())
    doc = _doc(_method(KEYID, _jwk(key.public_key()), validFrom=ACTIVATED))
    json.dumps(verify_receipt_retained(_receipt(key), doc).to_dict())


# ── Conformance vectors ──────────────────────────────────────────────────────


def _cases():
    return json.loads((VECTORS / "cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_vaara_reproduces_vector_verdict(case):
    expected = json.loads((VECTORS / "expected.json").read_text())[case["name"]]
    result = verify_receipt_retained(
        parse_receipt(case["receipt"]), case["didDocument"],
        anchored_time=case.get("anchoredTime"),
    )
    got = result.to_dict()
    assert {k: got[k] for k in expected} == expected


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ── Checker fidelity on documents the corpus does not carry ──────────────────
# The published vectors are well-formed (unique method ids). These exercise the
# malformed-but-defended case of two methods sharing one id, proving the
# Vaara-free checker mirrors KeyHistory / RevocationRegistry same-keyid
# aggregation (any window admits; earliest revocation wins) rather than reading
# only the first bound method.


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "kr_checker", VECTORS / "_check_independent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_mirrors_vaara_on_split_window_same_keyid():
    chk = _load_checker()
    key = ec.generate_private_key(ec.SECP256R1())
    jwk = _jwk(key.public_key())
    doc = _doc(
        _method(KEYID, jwk, validFrom="2026-01-01T00:00:00Z",
                validUntil="2026-03-01T00:00:00Z"),   # gap excludes iat
        _method(KEYID, jwk, validFrom="2026-05-01T00:00:00Z",
                validUntil="2026-07-01T00:00:00Z"),   # admits iat
    )
    case = {"receipt": _receipt(key).to_dict(), "didDocument": doc}
    vaara = verify_receipt_retained(parse_receipt(case["receipt"]), doc).to_dict()
    got = chk._evaluate(case)
    assert vaara["within_window"] is True and got["within_window"] is True
    assert {k: vaara[k] for k in chk.COMPARE} == got


def test_checker_mirrors_vaara_on_revocation_on_non_first_method():
    chk = _load_checker()
    key = ec.generate_private_key(ec.SECP256R1())
    jwk = _jwk(key.public_key())
    doc = _doc(
        _method(KEYID, jwk, validFrom=ACTIVATED, validUntil=RETIRED),  # in window
        _method(KEYID, jwk, revoked=BEFORE_IAT),  # revoked, on the non-first method
    )
    case = {"receipt": _receipt(key).to_dict(), "didDocument": doc}
    vaara = verify_receipt_retained(parse_receipt(case["receipt"]), doc).to_dict()
    got = chk._evaluate(case)
    assert vaara["revoked"] is True and got["revoked"] is True
    assert vaara["verifiable"] is False and got["verifiable"] is False
    assert {k: vaara[k] for k in chk.COMPARE} == got
