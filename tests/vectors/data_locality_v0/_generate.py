"""Regenerate the data_locality_v0 conformance vectors.

Eight cases pin the data-locality evidence contract: a signed record binds an
agent action's cross-border transfer facts (data class, endpoint, endpoint
region, TLS cert fingerprint, payload digest) to the policy decision that ran,
and optionally carries a region attestation signed by a party distinct from the
issuer (a TEE or the receiving provider).

The suite is two-tier and the verdicts name the tier a record reaches:

  Tier A (proof, no trusted party) — record signature verifies, payload digest
    recomputes from the payload bytes, and the recorded allow/block decision
    recomputes from the transfer facts under the named policy.
  Tier B (carried claim) — a region attestation signed by an attester key
    verifies against that key and agrees with the claimed region. Present and
    valid -> ok_attested; absent -> ok_asserted (location asserted, not
    attested, stated plainly).

Both keys are Ed25519, derived deterministically from published seed labels so
the corpus carries no randomness; the seeds are corpus-only. The sibling
_check_independent.py reproduces every verdict with no Vaara import.

Run: .venv/bin/python tests/vectors/data_locality_v0/_generate.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
SCHEMA = "vaara.data-locality/v0"

# Corpus-only seed labels. sha256 -> 32-byte Ed25519 seed. Not deployed keys.
ISSUER_SEED = b"vaara-data-locality-issuer/v0"
ATTESTER_SEED = b"vaara-region-attester/v0"

# Policy eu-inference-only@v1: personal data may only leave to an EU region;
# non-personal data is unconstrained.
POLICY_ID = "eu-inference-only@v1"
EU_REGIONS = ("eu-central-1", "eu-north-1", "eu-west-1")

PII = {"subject": "user-42", "text": "personal data payload"}
NONPII = {"metric": "latency_ms", "value": 12}


def _key(seed_label: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed_label).digest())


ISSUER = _key(ISSUER_SEED)
ATTESTER = _key(ATTESTER_SEED)


def _pub_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_jcs(obj) -> str:
    return "sha256:" + hashlib.sha256(_jcs(obj)).hexdigest()


def _attestation(region: str, nonce: str, *, corrupt: bool = False) -> dict:
    body = {"attestedRegion": region, "attester": "provider-tee-01", "nonce": nonce}
    sig = ATTESTER.sign(_jcs(body)).hex()
    if corrupt:
        sig = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    return {**body, "sig": sig}


def _record(spec: dict) -> dict:
    region = spec["region"]
    payload = spec["payload"]
    transfer = {
        "actionId": spec["action_id"],
        "dataClass": spec["data_class"],
        "endpoint": f"https://api.{region}.model.example/v1/infer",
        "endpointRegion": region,
        "payloadDigest": _sha256_jcs(payload),
        "tlsCertSha256": _sha256_jcs({"cert": region}),  # stand-in fingerprint
    }
    record = {
        "alg": "Ed25519",
        "decision": {"decision": spec["decision"], "policyId": POLICY_ID},
        "issuer": "vaara-locality-emitter",
        "schema": SCHEMA,
        "transfer": transfer,
        "version": 1,
    }
    att = spec.get("att")
    if att is not None:
        record["regionAttestation"] = _attestation(*att[:2], corrupt=att[2])
    signed = {k: v for k, v in record.items() if k != "signature"}
    record["signature"] = ISSUER.sign(_jcs(signed)).hex()
    if spec.get("tamper"):
        record["transfer"]["endpointRegion"] = "us-east-1"  # break the signature
    return record


# (name, data_class, region, payload, decision, att|None, flags, runtime_payload,
#  expected_verdict, attested). att = (region, nonce, corrupt_bool).
SPECS = [
    dict(name="pos_attested_eu", action_id="act-001", data_class="personal_data",
         region="eu-central-1", payload=PII, decision="allow",
         att=("eu-central-1", "nonce-eu-1", False),
         expected="ok_attested", attested=True),
    dict(name="pos_asserted_eu_no_attestation", action_id="act-002",
         data_class="personal_data", region="eu-west-1", payload=PII,
         decision="allow", att=None, expected="ok_asserted", attested=False),
    dict(name="pos_nonpersonal_us", action_id="act-003", data_class="non_personal",
         region="us-east-1", payload=NONPII, decision="allow", att=None,
         expected="ok_asserted", attested=False),
    dict(name="neg_policy_mismatch_pii_us", action_id="act-004",
         data_class="personal_data", region="us-east-1", payload=PII,
         decision="allow", att=None, expected="policy_mismatch", attested=False),
    dict(name="neg_bad_signature", action_id="act-005", data_class="personal_data",
         region="eu-central-1", payload=PII, decision="allow", att=None,
         tamper=True, expected="bad_signature", attested=False),
    dict(name="neg_payload_tampered", action_id="act-006", data_class="personal_data",
         region="eu-central-1", payload=PII, decision="allow", att=None,
         runtime_payload=NONPII, expected="payload_mismatch", attested=False),
    dict(name="neg_attestation_bad_sig", action_id="act-007",
         data_class="personal_data", region="eu-central-1", payload=PII,
         decision="allow", att=("eu-central-1", "nonce-eu-7", True),
         expected="attestation_bad_sig", attested=False),
    dict(name="neg_attestation_region_mismatch", action_id="act-008",
         data_class="personal_data", region="eu-central-1", payload=PII,
         decision="allow", att=("us-east-1", "nonce-us-8", False),
         expected="attestation_region_mismatch", attested=True),
]


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build() -> None:
    expected_cases = {}
    for spec in SPECS:
        record = _record(spec)
        case = {
            "record": record,
            "payload": spec.get("runtime_payload") or spec["payload"],
            "expected_verdict": spec["expected"],
            "attested": spec["attested"],
        }
        _write(CASES / f"{spec['name']}.json", case)
        expected_cases[spec["name"]] = {
            "expected_verdict": spec["expected"], "attested": spec["attested"]
        }

    _write(HERE / "expected.json", {
        "keys": {
            "issuerPublicKey": _pub_hex(ISSUER),
            "attesterPublicKey": _pub_hex(ATTESTER),
        },
        "policy": {"id": POLICY_ID, "euRegions": list(EU_REGIONS)},
        "cases": expected_cases,
    })
    print(f"wrote {len(SPECS)} cases + expected.json to {HERE}")


if __name__ == "__main__":
    build()
