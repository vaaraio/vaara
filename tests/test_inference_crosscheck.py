# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Tests for the model-diversity cross-check (vaara.inference-crosscheck/v0).

The whole wire path runs through a ``StubJudge`` (no inference
server), exactly as the TPM weld tests run on ``MockTPMQuoter`` with no TPM.
Subject attestation+receipt pairs are HS256-signed with a fixed test secret.
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

from vaara.attestation._inference_crosscheck import (  # noqa: E402
    CROSSCHECK_METHOD,
    CROSSCHECK_SCHEMA,
    JudgeOutcome,
    build_crosscheck,
    parse_crosscheck,
    response_matches_receipt,
    verify_crosscheck,
    verify_crosscheck_signature,
)
from vaara.attestation._inference_types import (  # noqa: E402
    InferenceOutcome,
    ModelDerived,
    RequestDeclared,
)
from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.attestation.inference import (  # noqa: E402
    emit_inference_attestation,
    emit_inference_receipt,
    make_inference_back_link,
    make_output_commitment,
    make_request_commitment,
)

SECRET = b"crosscheck-test-secret-0123456789"
ALG = "HS256"
MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]
RESPONSE = {"content": "The capital of France is Paris.", "thinking": ""}

SUBJECT_MODEL = ModelDerived("qwen3:1.7b", "sha256:" + "1" * 64, "sha256:" + "a" * 64)
DIVERSE_MODEL = ModelDerived("llama3.2:3b", "sha256:" + "2" * 64, "sha256:" + "b" * 64)


def _subject_pair(response=RESPONSE, *, status="completed"):
    rd = RequestDeclared(
        intent="inference/chat/qwen3:1.7b",
        request_commitment=make_request_commitment(messages=MESSAGES),
    )
    att = emit_inference_attestation(
        request_declared=rd, model_derived=SUBJECT_MODEL, iss="vaara-infer-proxy",
        sub="qwen3:1.7b", secret_version="v1", alg=ALG, signing_material=SECRET,
    )
    oc = make_output_commitment(response) if status == "completed" else None
    outcome = InferenceOutcome(
        status=status, completed_at="2026-06-16T05:00:00Z", tier="integrity",
        output_commitment=oc, eval_stats={"totalTokens": 10},
    )
    receipt = emit_inference_receipt(
        back_link=make_inference_back_link(att), outcome_derived=outcome,
        iss="vaara-infer-proxy", sub="qwen3:1.7b", secret_version="v1",
        alg=ALG, signing_material=SECRET,
    )
    return att, receipt


class StubJudge:
    def __init__(self, agreement, model=DIVERSE_MODEL, raw=None):
        self.agreement = agreement
        self.model = model
        self.raw = raw if raw is not None else f"VERDICT: {agreement}"

    def __call__(self, *, messages, candidate_response):
        return JudgeOutcome(self.agreement, self.raw, self.model)


def _build(judge, response=RESPONSE):
    att, receipt = _subject_pair(response)
    return att, receipt, build_crosscheck(
        attestation=att, receipt=receipt, messages=MESSAGES, response=RESPONSE,
        judge=judge, secret_version="v1", alg=ALG, signing_material=SECRET,
    )


def test_response_matches_receipt():
    _, receipt = _subject_pair()
    assert response_matches_receipt(receipt, RESPONSE) is True
    assert response_matches_receipt(receipt, {"content": "Lyon."}) is False
    _, refusal = _subject_pair(status="refused")
    assert response_matches_receipt(refusal, RESPONSE) is False


def test_build_and_verify_corroborated():
    _, receipt, record = _build(StubJudge("equivalent"))
    assert record.diverse is True
    v = verify_crosscheck(record, subject_receipt=receipt, verifying_material=SECRET)
    assert v["corroborated"] is True
    assert v["signatureValid"] is True
    assert v["receiptBinds"] is True
    assert v["method"] == CROSSCHECK_METHOD


def test_roundtrip_parse_and_signature():
    _, _, record = _build(StubJudge("equivalent"))
    doc = record.to_dict()
    assert doc["schema"] == CROSSCHECK_SCHEMA
    reparsed = parse_crosscheck(doc)
    assert reparsed.agreement == "equivalent"
    assert reparsed.subject_receipt_digest == record.subject_receipt_digest
    assert verify_crosscheck_signature(reparsed, verifying_material=SECRET) is True


def test_build_rejects_substituted_response():
    att, receipt = _subject_pair(RESPONSE)
    with pytest.raises(AttestationError, match="does not match the receipt"):
        build_crosscheck(
            attestation=att, receipt=receipt, messages=MESSAGES,
            response={"content": "A substituted answer."}, judge=StubJudge("equivalent"),
            secret_version="v1", alg=ALG, signing_material=SECRET,
        )


def test_build_rejects_mismatched_backlink():
    att_a, _ = _subject_pair()
    _, receipt_b = _subject_pair({"content": "different run", "thinking": ""})
    with pytest.raises(AttestationError, match="back-link"):
        build_crosscheck(
            attestation=att_a, receipt=receipt_b, messages=MESSAGES, response=RESPONSE,
            judge=StubJudge("equivalent"), secret_version="v1", alg=ALG,
            signing_material=SECRET,
        )


def test_build_rejects_unknown_agreement():
    att, receipt = _subject_pair()
    with pytest.raises(AttestationError, match="unknown agreement"):
        build_crosscheck(
            attestation=att, receipt=receipt, messages=MESSAGES, response=RESPONSE,
            judge=StubJudge("definitely-right"), secret_version="v1", alg=ALG,
            signing_material=SECRET,
        )


def test_non_diverse_not_corroborated():
    _, receipt, record = _build(StubJudge("equivalent", model=SUBJECT_MODEL))
    assert record.diverse is False
    v = verify_crosscheck(record, subject_receipt=receipt, verifying_material=SECRET)
    assert v["corroborated"] is False
    assert "same identity" in v["reason"]


def test_divergent_not_corroborated():
    _, receipt, record = _build(StubJudge("divergent"))
    v = verify_crosscheck(record, subject_receipt=receipt, verifying_material=SECRET)
    assert v["corroborated"] is False
    assert "divergent" in v["reason"]


def test_verify_detects_wrong_receipt():
    _, _, record = _build(StubJudge("equivalent"))
    _, other_receipt = _subject_pair({"content": "another inference", "thinking": ""})
    v = verify_crosscheck(record, subject_receipt=other_receipt, verifying_material=SECRET)
    assert v["receiptBinds"] is False
    assert v["corroborated"] is False


def test_parse_rejects_unknown_field():
    _, _, record = _build(StubJudge("equivalent"))
    doc = record.to_dict()
    doc["rogue"] = 1
    with pytest.raises(AttestationError, match="closed"):
        parse_crosscheck(doc)


def test_parse_rejects_tampered_diverse_flag():
    _, _, record = _build(StubJudge("equivalent"))
    doc = record.to_dict()
    assert doc["crosscheck"]["diverse"] is True
    doc["crosscheck"]["diverse"] = False  # models still differ -> inconsistent
    with pytest.raises(AttestationError, match="diverse"):
        parse_crosscheck(doc)


def test_tampered_payload_fails_signature():
    _, receipt, record = _build(StubJudge("equivalent"))
    doc = record.to_dict()
    doc["crosscheck"]["checkedAt"] = "2099-01-01T00:00:00Z"  # outside the signed bytes
    tampered = parse_crosscheck(doc)  # still self-consistent, parses fine
    v = verify_crosscheck(tampered, subject_receipt=receipt, verifying_material=SECRET)
    assert v["signatureValid"] is False
    assert v["corroborated"] is False


def test_keyless_structural_verify():
    _, receipt, record = _build(StubJudge("equivalent"))
    v = verify_crosscheck(record, subject_receipt=receipt)
    assert "signatureValid" not in v
    assert v["corroborated"] is True  # diverse + equivalent + receipt binds


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("VERDICT: equivalent — looks right", "equivalent"),
        ("I think VERDICT: Divergent, wrong sum", "divergent"),
        ("VERDICT: uncertain, cannot tell", "uncertain"),
        ("VERDICT: not equivalent, it is wrong", "uncertain"),  # hedge != equivalent
        ("the answer is equivalent to mine", "uncertain"),  # no marker -> fail safe
        ("hmm, hard to say", "uncertain"),
    ],
)
def test_parse_agreement_fail_safe(reply, expected):
    from vaara.attestation._inference_crosscheck import parse_agreement

    assert parse_agreement(reply) == expected
