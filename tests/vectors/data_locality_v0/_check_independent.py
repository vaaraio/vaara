"""Independent conformance checker for data_locality_v0 vectors.

No Vaara import. Uses only hashlib, json, rfc8785, and cryptography (Ed25519).
Every verdict is a property of the bytes in each case file, not of the Vaara
codebase. The checker derives the public keys from the same published seed
labels the generator used, then verifies against the PUBLIC key only: a real
relying party ships trust anchors, not seeds.

Two tiers, in evaluation order:
  1. record signature (Tier A integrity) ....... bad_signature
  2. payload digest recompute (Tier A) ......... payload_mismatch
  3. policy decision recompute (Tier A) ........ policy_mismatch
  4. carried region attestation (Tier B) ....... attestation_bad_sig,
                                                 attestation_region_mismatch,
                                                 ok_attested
     no attestation ............................ ok_asserted

Run: .venv/bin/python tests/vectors/data_locality_v0/_check_independent.py
Exit 0 = all cases match expected. Non-zero = mismatch printed and raised.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

HERE = Path(__file__).resolve().parent

# Published corpus seeds (see README). Derive keys, keep only the public half.
ISSUER_SEED = b"vaara-data-locality-issuer/v0"
ATTESTER_SEED = b"vaara-region-attester/v0"

POLICY_ID = "eu-inference-only@v1"
EU_REGIONS = frozenset({"eu-central-1", "eu-north-1", "eu-west-1"})


def _public_key(seed_label: bytes) -> Ed25519PublicKey:
    priv = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed_label).digest())
    return priv.public_key()


ISSUER_PUB = _public_key(ISSUER_SEED)
ATTESTER_PUB = _public_key(ATTESTER_SEED)


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_jcs(obj) -> str:
    return "sha256:" + hashlib.sha256(_jcs(obj)).hexdigest()


def _verify_ed25519(pub: Ed25519PublicKey, payload: bytes, sig_hex: str) -> bool:
    try:
        pub.verify(bytes.fromhex(sig_hex), payload)
        return True
    except (InvalidSignature, ValueError):
        return False


def _policy_decision(data_class: str, endpoint_region: str) -> str:
    """Recompute eu-inference-only@v1 from transfer facts alone."""
    if data_class != "personal_data":
        return "allow"
    return "allow" if endpoint_region in EU_REGIONS else "block"


def _verify(case: dict) -> str:
    record = case.get("record")
    if not isinstance(record, dict) or "signature" not in record:
        return "missing_record"

    # 1. record signature over JCS of everything but the signature itself.
    signed = {k: v for k, v in record.items() if k != "signature"}
    if not _verify_ed25519(ISSUER_PUB, _jcs(signed), record["signature"]):
        return "bad_signature"

    transfer = record["transfer"]

    # 2. payload digest binds the record to the payload presented at runtime.
    if transfer["payloadDigest"] != _sha256_jcs(case["payload"]):
        return "payload_mismatch"

    # 3. recorded decision must equal the independently recomputed verdict.
    recomputed = _policy_decision(transfer["dataClass"], transfer["endpointRegion"])
    if record["decision"]["decision"] != recomputed:
        return "policy_mismatch"

    # 4. Tier B: carried region attestation, verified against the attester key.
    att = record.get("regionAttestation")
    if att is None:
        return "ok_asserted"
    body = {"attestedRegion": att["attestedRegion"], "attester": att["attester"],
            "nonce": att["nonce"]}
    if not _verify_ed25519(ATTESTER_PUB, _jcs(body), att["sig"]):
        return "attestation_bad_sig"
    if att["attestedRegion"] != transfer["endpointRegion"]:
        return "attestation_region_mismatch"
    return "ok_attested"


def main() -> int:
    expected_path = HERE / "expected.json"
    if not expected_path.exists():
        print("expected.json not found — run _generate.py first", file=sys.stderr)
        return 1

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    cases_dir = HERE / "cases"
    failures = []

    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        computed = _verify(case)
        want = case["expected_verdict"]
        status = "PASS" if computed == want else "FAIL"
        print(f"  {status}  {path.stem:<34}  computed={computed!r}  expected={want!r}")
        if computed != want:
            failures.append(path.stem)

    # Cross-check every declared case exists and matches.
    for name, meta in expected["cases"].items():
        path = cases_dir / f"{name}.json"
        if not path.exists():
            print(f"  MISSING  {name}", file=sys.stderr)
            failures.append(name)
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        if _verify(case) != meta["expected_verdict"]:
            failures.append(f"{name}(expected.json cross-check)")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1

    print(f"\nall {len(list(cases_dir.glob('*.json')))} cases pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
