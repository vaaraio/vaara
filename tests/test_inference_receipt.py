# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Inference-receipt round-trip, tamper, float-normalization, and back-link tests.

Tier A (integrity) coverage: a signed model+input+output binding that any
Vaara verifier checks with no new crypto. See
``research/inference_receipts_design_20260614.md``.
"""

from __future__ import annotations

import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation.inference import (  # noqa: E402
    InferenceOutcome,
    ModelDerived,
    RequestDeclared,
    emit_inference_attestation,
    emit_inference_receipt,
    make_inference_back_link,
    make_output_commitment,
    make_request_commitment,
    normalize_inference_request,
    parse_inference_attestation,
    parse_inference_receipt,
    verify_inference_attestation,
    verify_inference_attestation_detail,
    verify_inference_back_link,
    verify_inference_receipt_signature,
)
from vaara.attestation.sep2787 import verify_args_commitment  # noqa: E402

HS_SECRET = b"\x42" * 32
MESSAGES = [{"role": "user", "content": "Summarize the Q3 report."}]
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 42}
OUTPUT = {"content": "The Q3 report covers three regions.", "toolCalls": []}
MODEL = ModelDerived(
    model_ref="qwen3:30b-a3b",
    manifest_digest="sha256:" + "a" * 64,
    gguf_metadata_hash="sha256:" + "b" * 64,
    quantization="Q4_K_M",
    param_count="30B",
)


def _attestation(alg="HS256", signing_material=HS_SECRET, **over):
    rd = RequestDeclared(
        intent="inference/chat/qwen3:30b-a3b",
        request_commitment=make_request_commitment(messages=MESSAGES, sampling=SAMPLING),
    )
    kwargs = dict(
        request_declared=rd,
        model_derived=MODEL,
        iss="vaara-infer-proxy",
        sub="vaara/homeserver",
        secret_version="v1",
        alg=alg,
        signing_material=signing_material,
    )
    kwargs.update(over)
    return emit_inference_attestation(**kwargs)


def _outcome(commit_output=True):
    return InferenceOutcome(
        status="completed",
        completed_at="2026-06-14T22:00:00Z",
        tier="integrity",
        output_commitment=make_output_commitment(OUTPUT) if commit_output else None,
        eval_stats={"promptEvalCount": 11, "evalCount": 256, "evalDurationNs": 9123456},
    )


def _receipt(att, alg="HS256", signing_material=HS_SECRET, outcome=None):
    return emit_inference_receipt(
        back_link=make_inference_back_link(att),
        outcome_derived=outcome or _outcome(),
        iss="vaara-infer-proxy",
        sub="vaara/homeserver",
        secret_version="v1",
        alg=alg,
        signing_material=signing_material,
    )


# --- round trip -------------------------------------------------------------


def test_attestation_round_trip_hs256():
    att = _attestation()
    assert verify_inference_attestation(att, verifying_material=HS_SECRET)
    assert parse_inference_attestation(att.to_dict()) == att


def test_receipt_round_trip_hs256():
    att = _attestation()
    receipt = _receipt(att)
    assert verify_inference_receipt_signature(receipt, verifying_material=HS_SECRET)
    assert parse_inference_receipt(receipt.to_dict()) == receipt


def test_round_trip_es256():
    key = ec.generate_private_key(ec.SECP256R1())
    att = _attestation(alg="ES256", signing_material=key)
    receipt = _receipt(att, alg="ES256", signing_material=key)
    pub = key.public_key()
    assert verify_inference_attestation(att, verifying_material=pub)
    assert verify_inference_receipt_signature(receipt, verifying_material=pub)
    assert verify_inference_back_link(receipt, attestation=att)


# --- tamper detection -------------------------------------------------------


def test_tampered_model_fails_attestation():
    att = _attestation()
    forged = att.to_dict()
    forged["modelDerived"]["manifestDigest"] = "sha256:" + "c" * 64
    rebuilt = parse_inference_attestation(forged)
    assert not verify_inference_attestation(rebuilt, verifying_material=HS_SECRET)


def test_tampered_status_fails_receipt():
    att = _attestation()
    receipt = _receipt(att)
    forged = receipt.to_dict()
    forged["outcomeDerived"]["status"] = "refused"
    rebuilt = parse_inference_receipt(forged)
    assert not verify_inference_receipt_signature(rebuilt, verifying_material=HS_SECRET)


def test_output_commitment_binds_response():
    att = _attestation()
    receipt = _receipt(att)
    good = verify_args_commitment(
        receipt.outcome_derived.output_commitment, runtime_arguments=OUTPUT
    )
    assert good.ok and good.projection_match
    tampered = verify_args_commitment(
        receipt.outcome_derived.output_commitment,
        runtime_arguments={"content": "different", "toolCalls": []},
    )
    assert not tampered.ok


# --- float normalization ----------------------------------------------------


def test_float_sampling_does_not_raise_and_binds():
    # canonical_json rejects raw floats; the normalizer must rewrite them.
    commitment = make_request_commitment(messages=MESSAGES, sampling=SAMPLING)
    runtime = normalize_inference_request(messages=MESSAGES, sampling=SAMPLING)
    result = verify_args_commitment(commitment, runtime_arguments=runtime)
    assert result.ok and result.projection_match


def test_different_sampling_breaks_request_commitment():
    att = _attestation()
    other = normalize_inference_request(
        messages=MESSAGES, sampling={**SAMPLING, "temperature": 0.8}
    )
    result = verify_args_commitment(
        att.request_declared.request_commitment, runtime_arguments=other
    )
    assert not result.ok


# --- back-link pin ----------------------------------------------------------


def test_back_link_pins_correct_attestation():
    att = _attestation()
    receipt = _receipt(att)
    assert verify_inference_back_link(receipt, attestation=att)
    other = _attestation()  # fresh nonce -> different digest
    assert not verify_inference_back_link(receipt, attestation=other)


# --- TTL --------------------------------------------------------------------


def test_expired_attestation_rejected():
    att = _attestation(iat="2020-01-01T00:00:00Z", exp_seconds=300)
    assert not verify_inference_attestation(att, verifying_material=HS_SECRET)


def test_future_dated_attestation_rejected():
    att = _attestation(iat="2999-01-01T00:00:00Z", exp_seconds=300)
    assert not verify_inference_attestation(att, verifying_material=HS_SECRET)


# --- signature vs freshness split (the verifier must not cry wolf) ----------


def test_detail_live_is_signature_and_fresh():
    att = _attestation()
    d = verify_inference_attestation_detail(att, verifying_material=HS_SECRET)
    assert d["signatureValid"] is True
    assert d["fresh"] is True
    assert d["iatValid"] is True


def test_detail_expired_keeps_valid_signature():
    # An authentic but stale credential: the signature stays valid, only the
    # live TTL window has lapsed. An archived receipt must read this way, not
    # as a signature failure.
    att = _attestation(iat="2020-01-01T00:00:00Z", exp_seconds=300)
    d = verify_inference_attestation_detail(att, verifying_material=HS_SECRET)
    assert d["signatureValid"] is True
    assert d["fresh"] is False
    assert d["ageSeconds"] is not None and d["ageSeconds"] > 300
    # strict bool wrapper still rejects an expired attestation (back-compat).
    assert not verify_inference_attestation(att, verifying_material=HS_SECRET)


def test_detail_tampered_fails_signature():
    att = _attestation()
    forged = att.to_dict()
    forged["modelDerived"]["manifestDigest"] = "sha256:" + "c" * 64
    rebuilt = parse_inference_attestation(forged)
    d = verify_inference_attestation_detail(rebuilt, verifying_material=HS_SECRET)
    assert d["signatureValid"] is False


# --- closed schema ----------------------------------------------------------


def test_unknown_key_rejected():
    from vaara.attestation._sep2787_types import AttestationError

    att = _attestation()
    forged = att.to_dict()
    forged["modelDerived"]["rogue"] = "x"
    with pytest.raises(AttestationError):
        parse_inference_attestation(forged)


def test_eval_stats_float_rejected():
    from vaara.attestation._sep2787_types import AttestationError

    att = _attestation()
    receipt = _receipt(att)
    forged = receipt.to_dict()
    forged["outcomeDerived"]["evalStats"]["evalCount"] = 1.5
    with pytest.raises(AttestationError):
        parse_inference_receipt(forged)
