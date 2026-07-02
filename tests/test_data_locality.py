"""The reference emitter produces records the independent corpus checker grades.

This is the tie between the open ``data_locality_v0`` vector corpus and the
in-tree emitter: rather than trusting the emitter's own verifier, every record
it produces here is handed to the corpus's dependency-free
``_check_independent.py`` and must earn the expected verdict. The emitter signs
with the same seeds the checker derives, so a mismatch means the emitter drifted
from the published format.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.attestation.data_locality import (
    TransferFacts,
    emit_data_locality_record,
    emit_from_interception,
    payload_digest,
    region_attestation,
    verify_record_signature,
)
from vaara.audit.signer import Ed25519Signer, Ed25519Verifier

REPO = Path(__file__).resolve().parent.parent
CHECKER_PATH = REPO / "tests" / "vectors" / "data_locality_v0" / "_check_independent.py"

# Same seed labels the corpus checker derives its trust anchors from.
ISSUER_SEED = b"vaara-data-locality-issuer/v0"
ATTESTER_SEED = b"vaara-region-attester/v0"

PII = {"subject": "user-42", "text": "personal data payload"}
POLICY = "eu-inference-only@v1"


def _signer(seed_label: bytes) -> Ed25519Signer:
    key = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed_label).digest())
    return Ed25519Signer(key)


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("dl_checker", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def issuer():
    return _signer(ISSUER_SEED)


@pytest.fixture(scope="module")
def attester():
    return _signer(ATTESTER_SEED)


def _transfer(region: str, payload=PII, data_class: str = "personal_data") -> TransferFacts:
    return TransferFacts(
        action_id="act-test",
        data_class=data_class,
        endpoint=f"https://api.{region}.model.example/v1/infer",
        endpoint_region=region,
        payload_digest=payload_digest(payload),
        tls_cert_sha256="sha256:" + "0" * 64,
    )


def _case(record, payload=PII):
    return {"record": record, "payload": payload, "expected_verdict": None}


def test_attested_eu_grades_ok_attested(checker, issuer, attester):
    att = region_attestation(attester, attester="provider-tee-01",
                             attested_region="eu-central-1", nonce="n1")
    record = emit_data_locality_record(
        signer=issuer, issuer="vaara-locality-emitter",
        transfer=_transfer("eu-central-1"), decision="allow",
        policy_id=POLICY, region_attestation=att,
    )
    assert checker._verify(_case(record)) == "ok_attested"


def test_no_attestation_grades_ok_asserted(checker, issuer):
    record = emit_data_locality_record(
        signer=issuer, issuer="vaara-locality-emitter",
        transfer=_transfer("eu-west-1"), decision="allow", policy_id=POLICY,
    )
    assert checker._verify(_case(record)) == "ok_asserted"


def test_wrong_decision_caught_by_independent_policy_recompute(checker, issuer):
    # Emitter records allow for PII to a US region; the checker recomputes block.
    record = emit_data_locality_record(
        signer=issuer, issuer="vaara-locality-emitter",
        transfer=_transfer("us-east-1"), decision="allow", policy_id=POLICY,
    )
    assert checker._verify(_case(record)) == "policy_mismatch"


def test_tampered_signature_caught(checker, issuer):
    record = emit_data_locality_record(
        signer=issuer, issuer="vaara-locality-emitter",
        transfer=_transfer("eu-central-1"), decision="allow", policy_id=POLICY,
    )
    record["transfer"]["endpointRegion"] = "us-east-1"  # mutate a signed field
    assert checker._verify(_case(record)) == "bad_signature"


def test_runtime_payload_mismatch_caught(checker, issuer):
    record = emit_data_locality_record(
        signer=issuer, issuer="vaara-locality-emitter",
        transfer=_transfer("eu-central-1"), decision="allow", policy_id=POLICY,
    )
    # A different payload than the one whose digest was committed.
    assert checker._verify(_case(record, payload={"other": 1})) == "payload_mismatch"


def test_attester_region_contradiction_caught(checker, issuer, attester):
    att = region_attestation(attester, attester="provider-tee-01",
                             attested_region="us-east-1", nonce="n8")
    record = emit_data_locality_record(
        signer=issuer, issuer="vaara-locality-emitter",
        transfer=_transfer("eu-central-1"), decision="allow",
        policy_id=POLICY, region_attestation=att,
    )
    assert checker._verify(_case(record)) == "attestation_region_mismatch"


def test_verify_record_signature_roundtrip(issuer):
    record = emit_data_locality_record(
        signer=issuer, issuer="vaara-locality-emitter",
        transfer=_transfer("eu-central-1"), decision="allow", policy_id=POLICY,
    )
    verifier = Ed25519Verifier(issuer.public_key_bytes())
    assert verify_record_signature(record, verifier=verifier) is True
    record["signature"] = record["signature"][:-2] + ("00" if record["signature"][-2:] != "00" else "11")
    assert verify_record_signature(record, verifier=verifier) is False


def test_emit_from_interception_maps_block(issuer):
    class _Result:
        allowed = False
        action_id = "act-99"

    record = emit_from_interception(
        _Result(), signer=issuer, issuer="vaara-locality-emitter",
        data_class="personal_data",
        endpoint="https://api.us-east-1.model.example/v1/infer",
        endpoint_region="us-east-1", payload=PII,
        tls_cert_sha256="sha256:" + "0" * 64, policy_id=POLICY,
    )
    assert record["decision"]["decision"] == "block"
    assert record["transfer"]["actionId"] == "act-99"
