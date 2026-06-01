#!/usr/bin/env python3
"""Independent conformance checker for the SEP-2828 decision/outcome vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``.
It does not import Vaara. It reads the committed fixtures from disk and
reproduces signature verification, attestation back-link verification,
and decision/outcome pairing for each case, then compares against
``expected.json``.

A second implementation that can run this file (or reproduce its logic)
demonstrates the decision/outcome record format is consumable without
depending on Vaara. Run:
``python tests/vectors/decision_pairing_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent
KEYS = HERE / "keys"

_DECISION_BLOCKS = ("version", "alg", "backLink", "decisionDerived",
                    "issuerAsserted")
_RECEIPT_BLOCKS = ("version", "alg", "backLink", "outcomeDerived",
                   "receiptAsserted")


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _payload(record: dict, blocks) -> bytes:
    return _jcs({k: record[k] for k in blocks})


def verify_signature(record: dict, blocks) -> bool:
    payload = _payload(record, blocks)
    alg, sig = record["alg"], record["signature"]
    if alg == "HS256":
        secret = (KEYS / "hs256_secret.bin").read_bytes()
        expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    if alg == "ES256":
        pub = serialization.load_pem_public_key(
            (KEYS / "es256_public.pem").read_bytes())
        if len(sig) != 128:
            return False
        raw = bytes.fromhex(sig)
        der = encode_dss_signature(
            int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
        try:
            pub.verify(der, payload, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False
    return False


def verify_back_link(record: dict, attestation: dict) -> bool:
    expected = _sha256_hex(_jcs(attestation))
    bl = record["backLink"]
    if not hmac.compare_digest(bl["attestationDigest"], expected):
        return False
    return bl["attestationNonce"] == attestation["issuerAsserted"]["nonce"]


def records_paired(decision: dict, receipt: dict) -> bool:
    db, rb = decision["backLink"], receipt["backLink"]
    if not hmac.compare_digest(db["attestationDigest"], rb["attestationDigest"]):
        return False
    return db["attestationNonce"] == rb["attestationNonce"]


def _load(case: Path, name: str):
    p = case / name
    return json.loads(p.read_text()) if p.exists() else None


# Declarative keys carried in expected.json that are documentation, not
# crypto verdicts; the checker passes them through verbatim.
_DOC_KEYS = {"outcome_required", "winner", "open_contract", "note"}


def _verdicts(case: Path, expected: dict) -> dict:
    att = _load(case, "attestation.json")
    dec = _load(case, "decision.json")
    rec = _load(case, "receipt.json")
    got: dict = {}
    for key in expected:
        if key in _DOC_KEYS:
            got[key] = expected[key]
        elif key == "decision_signature_ok":
            got[key] = verify_signature(dec, _DECISION_BLOCKS)
        elif key == "decision_back_link_ok":
            got[key] = verify_back_link(dec, att)
        elif key == "receipt_signature_ok":
            got[key] = verify_signature(rec, _RECEIPT_BLOCKS)
        elif key in ("receipt_back_link_ok",
                     "receipt_back_link_ok_against_stored_attestation"):
            got[key] = verify_back_link(rec, att)
        elif key == "records_paired":
            got[key] = records_paired(dec, rec)
        elif key == "receipt_present":
            got[key] = rec is not None
        elif key == "both_signatures_ok":
            got[key] = (verify_signature(_load(case, "decision_a.json"),
                                         _DECISION_BLOCKS)
                        and verify_signature(_load(case, "decision_b.json"),
                                             _DECISION_BLOCKS))
        elif key == "both_back_links_ok":
            got[key] = (verify_back_link(_load(case, "decision_a.json"), att)
                        and verify_back_link(_load(case, "decision_b.json"), att))
        elif key == "replayed_receipt_signature_ok":
            got[key] = verify_signature(_load(case, "receipt_replayed.json"),
                                        _RECEIPT_BLOCKS)
        else:
            raise ValueError(f"unknown expected key: {key!r}")
    return got


def main() -> int:
    failures = 0
    cases = sorted((HERE / "normative").iterdir())
    for case in cases:
        if not case.is_dir():
            continue
        expected = json.loads((case / "expected.json").read_text())
        got = _verdicts(case, expected)
        ok = got == expected
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {case.name}: {got}")
    print(f"\n{len(cases) - failures}/{len(cases)} cases matched expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
