#!/usr/bin/env python3
"""Regenerate the pq_hybrid_v0 vectors.

Builds execution receipts with a classical signature (ES256 / RS256) and a
parallel ML-DSA-65 signature, plus the DID document carrying both the classical
and the AKP (ML-DSA) verification methods, then records the quantum-resistance
verdict Vaara assigns each. The independent checker
(``_check_independent.py``) must reproduce every verdict without importing
Vaara. See ``docs/design/pq-hybrid-signing-spec.md``.

Signatures are randomized (ECDSA, and ML-DSA's hedged variant), so re-running
this writes fresh bytes. The committed ``cases.json`` / ``expected.json`` are
the frozen fixtures; regenerate only when the envelope changes.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec, rsa
from dilithium_py.ml_dsa import ML_DSA_65

from vaara.attestation._receipt_pq import _b64u_encode
from vaara.attestation.receipt import (
    BackLink,
    OutcomeDerived,
    attach_pq_signature,
    emit_receipt,
    parse_receipt,
    pq_verdict,
)

HERE = Path(__file__).resolve().parent
COMPARE = (
    "tier", "classical_bound", "pq_bound", "suite", "pq_keyid",
    "quantum_resistant", "downgrade_resistant",
)
ISS = "did:web:issuer.example"
ES_KEYID = ISS + "#es256-2026"
RS_KEYID = ISS + "#rs256-2026"
PQ_KEYID = ISS + "#mldsa-2026"
IAT = "2026-06-11T12:00:00Z"
NONCE = "nonce-pq-0"


def _b64u_int(n: int, size: int) -> str:
    return base64.urlsafe_b64encode(n.to_bytes(size, "big")).rstrip(b"=").decode()


def _ec_jwk(key: ec.EllipticCurvePrivateKey) -> dict:
    nums = key.public_key().public_numbers()
    return {"kty": "EC", "crv": "P-256", "x": _b64u_int(nums.x, 32),
            "y": _b64u_int(nums.y, 32)}


def _rsa_jwk(key: rsa.RSAPrivateKey) -> dict:
    nums = key.public_key().public_numbers()
    size = (nums.n.bit_length() + 7) // 8
    return {"kty": "RSA", "n": _b64u_int(nums.n, size),
            "e": _b64u_int(nums.e, (nums.e.bit_length() + 7) // 8)}


def main() -> int:
    es_key = ec.generate_private_key(ec.SECP256R1())
    rs_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pq_pub, pq_sec = ML_DSA_65.keygen()
    wrong_pub, _wrong_sec = ML_DSA_65.keygen()  # for the wrong-key case

    def method(keyid: str, jwk: dict) -> dict:
        return {"id": keyid, "type": "JsonWebKey2020", "publicKeyJwk": jwk}

    def akp(pub: bytes) -> dict:
        return {"kty": "AKP", "alg": "ML-DSA-65", "pub": _b64u_encode(pub)}

    doc = {"id": ISS, "verificationMethod": [
        method(ES_KEYID, _ec_jwk(es_key)),
        method(RS_KEYID, _rsa_jwk(rs_key)),
        method(PQ_KEYID, akp(pq_pub)),
    ]}
    # A document whose ML-DSA method holds the wrong public key.
    doc_wrong_pq = copy.deepcopy(doc)
    doc_wrong_pq["verificationMethod"][2] = method(PQ_KEYID, akp(wrong_pub))

    bl = BackLink(attestation_digest="sha256:" + "a" * 64, attestation_nonce="att-1")
    od = OutcomeDerived(status="executed", completed_at=IAT)

    def emit(alg, material, suite=None):
        return emit_receipt(
            back_link=bl, outcome_derived=od, iss=ISS, sub="tool:transfer",
            secret_version="v1", alg=alg, signing_material=material,
            iat=IAT, nonce=NONCE, sig_suite=suite,
        )

    es_hybrid = attach_pq_signature(
        emit("ES256", es_key, "ES256+ML-DSA-65"),
        pq_secret_key=pq_sec, pq_keyid=PQ_KEYID,
    )
    rs_hybrid = attach_pq_signature(
        emit("RS256", rs_key, "RS256+ML-DSA-65"),
        pq_secret_key=pq_sec, pq_keyid=PQ_KEYID,
    )
    classical = emit("ES256", es_key)
    pqc_present = attach_pq_signature(
        classical, pq_secret_key=pq_sec, pq_keyid=PQ_KEYID
    )
    unknown = emit("ES256", es_key, "ES256+BOGUS")
    mismatch = attach_pq_signature(
        emit("ES256", es_key, "RS256+ML-DSA-65"),
        pq_secret_key=pq_sec, pq_keyid=PQ_KEYID,
    )

    cases: list[dict] = []

    def add(name: str, receipt_dict: dict, document: dict) -> None:
        verdict = pq_verdict(parse_receipt(receipt_dict), document)
        cases.append({"name": name, "receipt": receipt_dict, "didDocument": document})
        expected[name] = {k: verdict.to_dict()[k] for k in COMPARE}

    expected: dict[str, dict] = {}

    add("es256_hybrid_clean", es_hybrid.to_dict(), doc)
    add("rs256_hybrid_clean", rs_hybrid.to_dict(), doc)
    add("classical_only", classical.to_dict(), doc)
    add("pqc_present", pqc_present.to_dict(), doc)

    stripped = es_hybrid.to_dict()
    stripped.pop("pqSignature")
    add("downgrade_stripped", stripped, doc)

    tampered_pq = es_hybrid.to_dict()
    s = bytearray.fromhex(tampered_pq["pqSignature"]["sig"])
    s[0] ^= 0x01
    tampered_pq["pqSignature"]["sig"] = bytes(s).hex()
    add("tampered_pq_sig", tampered_pq, doc)

    tampered_body = es_hybrid.to_dict()
    tampered_body["receiptAsserted"]["sub"] = "tool:exfiltrate"
    add("tampered_body", tampered_body, doc)

    add("unknown_suite", unknown.to_dict(), doc)
    add("suite_alg_mismatch", mismatch.to_dict(), doc)
    add("pq_wrong_key", es_hybrid.to_dict(), doc_wrong_pq)

    (HERE / "cases.json").write_text(
        json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n"
    )
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
