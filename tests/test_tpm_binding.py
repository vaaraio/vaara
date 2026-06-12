"""TPM binding: bind a TPM 2.0 quote + IMA log to a SEP-2828 record.

Covers ``bind_record_to_extra_data`` / ``verify_tpm_binding`` (the verdict tiers,
the IMA-PCR pin, the four binding links and the v0 honesty fields), the
``vaara.tpm-evidence-bundle/v0`` round-trip, and the structural rejection of a
malformed bundle. The wire path is exercised end to end through ``MockTPMQuoter``,
which marshals and ECDSA-signs a real ``TPMS_ATTEST`` so no TPM is needed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct

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
    bind_record_to_extra_data,
    build_tpm_bundle_document,
    emit_receipt,
    make_back_link,
    verify_tpm_binding,
    verify_tpm_bundle,
)
from vaara.attestation.sep2787 import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)
from vaara.attestation._tpm import IMA_PCR, MockTPMQuoter  # noqa: E402

DID = "did:web:vendor-a.example:billing"
IAT = "2026-06-12T10:00:00Z"


def _record(nonce: str = "rcpt-nonce-fixed-0001") -> dict:
    att = emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=PayloadDerived(tool_calls=(ToolCallBinding(
            name="charge_card", server_fingerprint="sha256:" + "1" * 64,
            args=make_args_digest({"amount": 4200}),
        ),)),
        iss="issuer://test", sub="agent:billing", secret_version="v1",
        alg="HS256", signing_material=b"\x42" * 32,
        nonce="att-nonce-fixed-0001", iat="2026-06-12T09:59:59Z",
    )
    key = ec.generate_private_key(ec.SECP256R1())
    return emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce=nonce, iat=IAT,
    ).to_dict()


def _ima(n: int = 4) -> tuple[str, bytes]:
    """A synthetic sha256 IMA ascii log and the PCR 10 it replays to."""
    acc = bytes(32)
    lines = []
    for i in range(n):
        th = hashlib.sha256(b"entry-%d" % i).digest()
        lines.append(f"10 {th.hex()} ima-ng sha256:{'00' * 32} /usr/bin/x{i}")
        acc = hashlib.sha256(acc + th).digest()
    return "\n".join(lines) + "\n", acc


@pytest.fixture
def ak_key():
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture
def ak_pem(ak_key):
    return ak_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _quote(ak_key, record, pcr10):
    q = MockTPMQuoter(ak_key)
    return q.quote(bind_record_to_extra_data(record), {IMA_PCR: pcr10})


def _verify(ak_key, ak_pem, record=None, ima=None, **kw):
    record = record if record is not None else _record()
    log, pcr10 = ima if ima is not None else _ima()
    attest, sig = _quote(ak_key, record, pcr10)
    return verify_tpm_binding(
        record, attest, sig, ak_pem,
        pcr_values={IMA_PCR: pcr10}, ima_log=log, **kw,
    )


# ── bind_record_to_extra_data ────────────────────────────────────────────────


def test_binding_is_64_bytes_and_deterministic():
    rec = _record()
    a = bind_record_to_extra_data(rec)
    assert len(a) == 64
    assert a == bind_record_to_extra_data(rec)


def test_binding_differs_across_records():
    assert bind_record_to_extra_data(_record("a")) != \
        bind_record_to_extra_data(_record("b"))


def test_binding_covers_the_signature_field():
    rec = _record()
    variant = copy.deepcopy(rec)
    variant["signature"] = "00" * 64
    assert bind_record_to_extra_data(rec) != bind_record_to_extra_data(variant)


# ── verify_tpm_binding tiers ─────────────────────────────────────────────────


def test_clean_quote_is_bound_not_attested(ak_key, ak_pem):
    v = _verify(ak_key, ak_pem)
    assert v.tier == "bound" and v.ok
    assert v.signature_valid and v.bound
    assert v.pcr_digest_recomputed and v.ima_replayed
    assert v.pcr_pin_basis == "unpinned"
    assert v.ak_chain_basis == "caller_supplied_unverified"
    assert v.decision_logic_basis == "not_established"
    assert v.freshness_basis == "not_established"


def test_pinned_ima_pcr_match(ak_key, ak_pem):
    log, pcr10 = _ima()
    v = _verify(ak_key, ak_pem, ima=(log, pcr10), expected_ima_pcr=pcr10.hex())
    assert v.tier == "pcr_pinned" and v.ok
    assert v.pcr_pin_basis == "pinned"


def test_pin_mismatch_fails_even_in_default_mode(ak_key, ak_pem):
    v = _verify(ak_key, ak_pem, expected_ima_pcr="ff" * 32)
    assert v.pcr_pin_basis == "pin_mismatch"
    assert v.tier == "bound" and v.ok is False


def test_quote_bound_to_a_different_record_does_not_verify(ak_key, ak_pem):
    rec, other = _record("a"), _record("b")
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    v = verify_tpm_binding(other, attest, sig, ak_pem,
                           pcr_values={IMA_PCR: pcr10}, ima_log=log)
    assert v.signature_valid is True  # genuinely signed
    assert v.bound is False and v.tier == "unverified" and v.ok is False


def test_signature_malleable_variant_does_not_verify(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    variant = copy.deepcopy(rec)
    variant["signature"] = "00" * 64
    v = verify_tpm_binding(variant, attest, sig, ak_pem,
                           pcr_values={IMA_PCR: pcr10}, ima_log=log)
    assert v.signature_valid is True and v.bound is False
    assert v.tier == "unverified"


def test_tampered_signature_fails(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    sig = bytearray(sig)
    sig[-1] ^= 0x01
    v = verify_tpm_binding(rec, attest, bytes(sig), ak_pem,
                           pcr_values={IMA_PCR: pcr10}, ima_log=log)
    assert v.signature_valid is False and v.tier == "unverified"


def test_wrong_ak_fails(ak_key):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    other = ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    v = verify_tpm_binding(rec, attest, sig, other,
                           pcr_values={IMA_PCR: pcr10}, ima_log=log)
    assert v.signature_valid is False and v.tier == "unverified"


def test_non_ecdsa_signature_algo_is_flagged(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, _ = _quote(ak_key, rec, pcr10)
    rsassa = struct.pack(">HH", 0x0014, 0x000B)  # TPM_ALG_RSASSA, sha256
    v = verify_tpm_binding(rec, attest, rsassa, ak_pem,
                           pcr_values={IMA_PCR: pcr10}, ima_log=log)
    assert v.signature_algo_ok is False and v.tier == "unverified"


def test_bad_magic_is_flagged(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    attest = bytearray(attest)
    attest[0] ^= 0x01  # corrupt TPM_GENERATED_VALUE
    v = verify_tpm_binding(rec, bytes(attest), sig, ak_pem,
                           pcr_values={IMA_PCR: pcr10}, ima_log=log)
    assert v.magic_ok is False and v.tier == "unverified"


def test_tampered_ima_log_breaks_replay(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    lines = log.splitlines()
    parts = lines[1].split()
    parts[1] = ("f" if parts[1][0] != "f" else "0") + parts[1][1:]
    lines[1] = " ".join(parts)
    bad = "\n".join(lines) + "\n"
    v = verify_tpm_binding(rec, attest, sig, ak_pem,
                           pcr_values={IMA_PCR: pcr10}, ima_log=bad)
    assert v.ima_replayed is False and v.tier == "unverified"


def test_wrong_pcr_value_breaks_digest(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    v = verify_tpm_binding(rec, attest, sig, ak_pem,
                           pcr_values={IMA_PCR: bytes(32)}, ima_log=log)
    assert v.pcr_digest_recomputed is False and v.tier == "unverified"


def test_truncated_quote_does_not_traceback(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    v = verify_tpm_binding(rec, attest[:12], sig, ak_pem,
                           pcr_values={IMA_PCR: pcr10}, ima_log=log)
    assert v.parsed is False and v.tier == "unverified" and v.ok is False


def test_strict_is_unreachable_in_v0(ak_key, ak_pem):
    log, pcr10 = _ima()
    v = _verify(ak_key, ak_pem, ima=(log, pcr10),
                expected_ima_pcr=pcr10.hex(), strict=True)
    assert v.tier == "pcr_pinned" and v.strict and v.ok is False


def test_non_dict_record_raises_valueerror(ak_key, ak_pem):
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, _record(), pcr10)
    with pytest.raises(ValueError):
        verify_tpm_binding(["not", "a", "dict"], attest, sig, ak_pem,
                           pcr_values={IMA_PCR: pcr10}, ima_log=log)


def test_verdict_is_json_serializable(ak_key, ak_pem):
    json.dumps(_verify(ak_key, ak_pem).to_dict())


# ── bundle round-trip ────────────────────────────────────────────────────────


def test_bundle_round_trip(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    doc = build_tpm_bundle_document(
        rec, attest, sig, ak_pem, {IMA_PCR: pcr10}, log,
        bank="sha256", expected_ima_pcr=pcr10.hex(),
    )
    v = verify_tpm_bundle(json.loads(json.dumps(doc)))
    assert v.tier == "pcr_pinned" and v.ok


def test_bundle_missing_field_raises(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    doc = build_tpm_bundle_document(rec, attest, sig, ak_pem, {IMA_PCR: pcr10}, log)
    del doc["quote"]["attest_b64"]
    with pytest.raises(ValueError):
        verify_tpm_bundle(doc)


def test_bundle_wrong_schema_raises(ak_key, ak_pem):
    rec = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(ak_key, rec, pcr10)
    doc = build_tpm_bundle_document(rec, attest, sig, ak_pem, {IMA_PCR: pcr10}, log)
    doc["schema"] = "something/else"
    with pytest.raises(ValueError):
        verify_tpm_bundle(doc)
