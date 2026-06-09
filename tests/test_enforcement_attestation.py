"""Attested enforcement: bind a SEV-SNP report to a SEP-2828 record.

Covers ``bind_record_to_report_data`` / ``verify_enforcement`` (the verdict
tiers, the measurement pin, the substitution and signature-malleability
defences, and the v0 honesty fields), the ``enforcement_attestation_v0``
conformance vectors, and a standalone Vaara-free checker that reproduces every
verdict.

See ``docs/design/enforcement-attestation-spec.md``.
"""

from __future__ import annotations

import base64
import copy
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

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation.receipt import (  # noqa: E402
    OutcomeDerived,
    bind_record_to_report_data,
    emit_receipt,
    make_back_link,
    verify_enforcement,
)
from vaara.attestation.sep2787 import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)
from vaara.attestation.tee import (  # noqa: E402
    SEV_SNP_BODY_SIZE,
    MockSEVSNPAttester,
)

VECTORS = Path(__file__).resolve().parent / "vectors" / "enforcement_attestation_v0"
DID = "did:web:vendor-a.example:billing"
IAT = "2026-05-29T10:00:00Z"
MEASUREMENT = bytes(range(1, 49))


def _record(nonce: str = "rcpt-nonce-fixed-0001") -> dict:
    att = emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=PayloadDerived(tool_calls=(ToolCallBinding(
            name="charge_card", server_fingerprint="sha256:" + "1" * 64,
            args=make_args_digest({"amount": 4200}),
        ),)),
        iss="issuer://test", sub="agent:billing", secret_version="v1",
        alg="HS256", signing_material=b"\x42" * 32,
        nonce="att-nonce-fixed-0001", iat="2026-05-29T09:59:59Z",
    )
    key = ec.generate_private_key(ec.SECP256R1())
    return emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce=nonce, iat=IAT,
    ).to_dict()


@pytest.fixture
def vcek_key():
    return ec.generate_private_key(ec.SECP384R1())


@pytest.fixture
def vcek_pem(vcek_key):
    return vcek_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _report(vcek_key, record, measurement=MEASUREMENT):
    attester = MockSEVSNPAttester(vcek_key, measurement=measurement)
    return attester.emit(bind_record_to_report_data(record))


# ── bind_record_to_report_data ───────────────────────────────────────────────


def test_binding_is_64_bytes_and_deterministic():
    rec = _record()
    a = bind_record_to_report_data(rec)
    assert len(a) == 64
    assert a == bind_record_to_report_data(rec)


def test_binding_differs_across_records():
    assert bind_record_to_report_data(_record("a")) != \
        bind_record_to_report_data(_record("b"))


def test_binding_covers_the_signature_field():
    # The full-record preimage: changing only the signature changes the binding,
    # so a report bound to a genuine record never binds a sig-stripped variant.
    rec = _record()
    variant = copy.deepcopy(rec)
    variant["signature"] = "00" * 64
    assert bind_record_to_report_data(rec) != bind_record_to_report_data(variant)


# ── verify_enforcement tiers ─────────────────────────────────────────────────


def test_clean_report_is_bound_not_attested(vcek_key, vcek_pem):
    rec = _record()
    v = verify_enforcement(rec, _report(vcek_key, rec), vcek_pem)
    assert v.tier == "bound" and v.ok
    assert v.signature_valid and v.bound
    assert v.measurement_basis == "unpinned"
    assert v.vcek_chain_basis == "caller_supplied_unverified"
    assert v.enforcement_logic_basis == "not_established"


def test_pinned_measurement_match(vcek_key, vcek_pem):
    rec = _record()
    v = verify_enforcement(rec, _report(vcek_key, rec), vcek_pem,
                           expected_measurement=MEASUREMENT.hex())
    assert v.tier == "measurement_pinned" and v.ok
    assert v.measurement_basis == "pinned"


def test_pin_mismatch_fails_even_in_default_mode(vcek_key, vcek_pem):
    rec = _record()
    v = verify_enforcement(rec, _report(vcek_key, rec), vcek_pem,
                           expected_measurement="ff" * 48)
    assert v.measurement_basis == "pin_mismatch"
    assert v.tier == "bound" and v.ok is False


def test_report_bound_to_a_different_record_does_not_verify(vcek_key, vcek_pem):
    rec, other = _record("a"), _record("b")
    report = _report(vcek_key, rec)  # bound to rec
    v = verify_enforcement(other, report, vcek_pem)
    assert v.signature_valid is True  # the report is genuinely signed
    assert v.bound is False and v.tier == "unverified" and v.ok is False


def test_signature_malleable_variant_does_not_verify(vcek_key, vcek_pem):
    rec = _record()
    report = _report(vcek_key, rec)
    variant = copy.deepcopy(rec)
    variant["signature"] = "00" * 64
    v = verify_enforcement(variant, report, vcek_pem)
    assert v.signature_valid is True and v.bound is False
    assert v.tier == "unverified"


def test_tampered_signature_fails(vcek_key, vcek_pem):
    rec = _record()
    report = bytearray(_report(vcek_key, rec))
    report[SEV_SNP_BODY_SIZE + 4] ^= 0x01
    v = verify_enforcement(rec, bytes(report), vcek_pem)
    assert v.signature_valid is False and v.tier == "unverified"


def test_wrong_vcek_fails(vcek_key):
    rec = _record()
    report = _report(vcek_key, rec)
    other_pem = ec.generate_private_key(ec.SECP384R1()).public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    v = verify_enforcement(rec, report, other_pem)
    assert v.signature_valid is False and v.tier == "unverified"


def test_unsupported_signature_algo_fails(vcek_key, vcek_pem):
    rec = _record()
    report = bytearray(_report(vcek_key, rec))
    report[0x034:0x038] = b"\x00\x00\x00\x00"
    v = verify_enforcement(rec, bytes(report), vcek_pem)
    assert v.signature_algo_ok is False and v.tier == "unverified"


def test_truncated_report_does_not_traceback(vcek_key, vcek_pem):
    rec = _record()
    v = verify_enforcement(rec, _report(vcek_key, rec)[:100], vcek_pem)
    assert v.parsed is False and v.tier == "unverified" and v.ok is False
    assert v.report_version is None and v.measurement is None


def test_strict_is_unreachable_in_v0(vcek_key, vcek_pem):
    rec = _record()
    v = verify_enforcement(rec, _report(vcek_key, rec), vcek_pem,
                           expected_measurement=MEASUREMENT.hex(), strict=True)
    # The cryptography all holds, but strict needs the chain-rooted tier.
    assert v.tier == "measurement_pinned" and v.strict and v.ok is False


def test_non_dict_record_raises_valueerror(vcek_key, vcek_pem):
    with pytest.raises(ValueError):
        verify_enforcement(["not", "a", "dict"], _report(vcek_key, _record()),
                           vcek_pem)


def test_verdict_is_json_serializable(vcek_key, vcek_pem):
    rec = _record()
    json.dumps(verify_enforcement(rec, _report(vcek_key, rec), vcek_pem).to_dict())


# ── Conformance vectors ──────────────────────────────────────────────────────


def _jwk_p384(pub) -> dict:
    n = pub.public_numbers()

    def _b(value: int) -> str:
        return base64.urlsafe_b64encode(value.to_bytes(48, "big")).rstrip(
            b"=").decode("ascii")

    return {"kty": "EC", "crv": "P-384", "x": _b(n.x), "y": _b(n.y)}


def _jwk_to_pem(jwk: dict) -> bytes:
    def _i(v: str) -> int:
        return int.from_bytes(
            base64.urlsafe_b64decode(v + "=" * (-len(v) % 4)), "big")

    pub = ec.EllipticCurvePublicNumbers(
        _i(jwk["x"]), _i(jwk["y"]), ec.SECP384R1()).public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _cases():
    return json.loads((VECTORS / "cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_vaara_reproduces_vector_verdict(case):
    expected = json.loads((VECTORS / "expected.json").read_text())[case["name"]]
    verdict = verify_enforcement(
        case["record"],
        base64.b64decode(case["report_b64"]),
        _jwk_to_pem(case["vcek_jwk"]),
        expected_measurement=case["expected_measurement"],
        strict=case["strict"],
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
        "enf_checker", VECTORS / "_check_independent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_matches_vaara_on_fresh_report(vcek_key, vcek_pem):
    chk = _load_checker()
    rec = _record()
    report = _report(vcek_key, rec)
    case = {
        "record": rec,
        "report_b64": base64.b64encode(report).decode("ascii"),
        "vcek_jwk": _jwk_p384(vcek_key.public_key()),
        "expected_measurement": MEASUREMENT.hex(),
        "strict": False,
    }
    vaara = verify_enforcement(
        rec, report, vcek_pem, expected_measurement=MEASUREMENT.hex(),
    ).to_dict()
    got = chk._evaluate(case)
    assert got["tier"] == "measurement_pinned" and got["ok"] is True
    assert {k: vaara[k] for k in chk.COMPARE} == got
