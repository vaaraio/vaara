#!/usr/bin/env python3
"""Independent checker for the pq_hybrid_v0 vectors.

Imports only the standard library plus ``cryptography``, ``rfc8785``, and the
one named post-quantum dependency ``dilithium-py``. It does not import Vaara.
For each committed case it reproduces the quantum-resistance verdict:

1. **Preimage.** RFC 8785 JCS over ``{version, alg, backLink, outcomeDerived,
   receiptAsserted}``. ``receiptAsserted.sigSuite`` rides inside it, which is
   what makes a committed hybrid suite tamper-evident.
2. **Classical bind.** Verify the ES256 / RS256 signature against a key the DID
   document lists whose type matches the receipt ``alg``.
3. **PQC bind.** When a ``pqSignature`` is present, verify the ML-DSA-65
   signature over the same preimage against the AKP method named by its
   ``keyid``.
4. **Tier.** Apply the allowlisted-suite logic, failing closed on a stripped,
   tampered, inconsistent, or unknown-suite record.

Verdicts are compared against ``expected.json`` (the non-normative ``reason``
is not compared). Exit code 0 means every case matched. Run:
``python tests/vectors/pq_hybrid_v0/_check_independent.py``.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

# A conformant environment for this suite carries the post-quantum extra
# (``vaara[pq]``). When an optional dependency is absent, exit with the standard
# skip code (77) so the aggregate runner reports SKIP with a reason rather than a
# false failure; the suite still runs and passes where the extra is installed.
_SKIP_EXIT_CODE = 77
try:
    import rfc8785
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.hazmat.primitives.asymmetric.utils import (
        encode_dss_signature,
    )
    from dilithium_py.ml_dsa import ML_DSA_65
except ModuleNotFoundError as exc:
    print(
        "SKIP: pq_hybrid_v0 needs an optional dependency not installed: "
        f"{exc.name} (pip install 'vaara[pq]')",
        file=sys.stderr,
    )
    raise SystemExit(_SKIP_EXIT_CODE) from None

HERE = Path(__file__).resolve().parent
_BLOCKS = ("version", "alg", "backLink", "outcomeDerived", "receiptAsserted")
_MLDSA65_PUBKEY_BYTES = 1952
_HYBRID_SUITES = {
    "ES256+ML-DSA-65": ("ES256", "ML-DSA-65"),
    "RS256+ML-DSA-65": ("RS256", "ML-DSA-65"),
}
_ALG_KTY = {"ES256": "EC", "RS256": "RSA"}
COMPARE = (
    "tier", "classical_bound", "pq_bound", "suite", "pq_keyid",
    "quantum_resistant", "downgrade_resistant",
)


def _b64u_to_int(value: str) -> int:
    pad = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + pad), "big")


def _b64u_to_bytes(value: str) -> bytes:
    pad = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + pad)


def _classical_key(jwk: dict, want_kty: str):
    # A malformed or partial method (missing x/y/e/n, non-string coord) is
    # treated as unusable and skipped, the way verify_receipt_identity does,
    # rather than crashing the whole offline check on one bad document key.
    if jwk.get("kty") != want_kty:
        return None
    try:
        if want_kty == "EC":
            if jwk.get("crv") != "P-256":
                return None
            return ec.EllipticCurvePublicNumbers(
                _b64u_to_int(jwk["x"]), _b64u_to_int(jwk["y"]), ec.SECP256R1()
            ).public_key()
        if want_kty == "RSA":
            return rsa.RSAPublicNumbers(
                _b64u_to_int(jwk["e"]), _b64u_to_int(jwk["n"])
            ).public_key()
    except (KeyError, ValueError, TypeError):
        return None
    return None


def _classical_verifies(alg: str, payload: bytes, sig_hex: str, key) -> bool:
    try:
        if alg == "ES256":
            if len(sig_hex) != 128:
                return False
            raw = bytes.fromhex(sig_hex)
            der = encode_dss_signature(
                int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
            )
            key.verify(der, payload, ec.ECDSA(hashes.SHA256()))
            return True
        if alg == "RS256":
            key.verify(bytes.fromhex(sig_hex), payload,
                       padding.PKCS1v15(), hashes.SHA256())
            return True
    except (InvalidSignature, ValueError):
        return False
    return False


def _classical_bound(receipt: dict, doc: dict, payload: bytes) -> bool:
    iss = receipt["receiptAsserted"]["iss"]
    if not iss.startswith("did:web:") or doc.get("id") != iss:
        return False
    want_kty = _ALG_KTY.get(receipt["alg"])
    if want_kty is None:
        return False
    for method in doc.get("verificationMethod", []):
        if not isinstance(method, dict):
            continue
        key = _classical_key(method.get("publicKeyJwk", {}), want_kty)
        if key is not None and _classical_verifies(
            receipt["alg"], payload, receipt["signature"], key
        ):
            return True
    return False


def _mldsa_pub(method: dict):
    jwk = method.get("publicKeyJwk")
    if not isinstance(jwk, dict) or jwk.get("kty") != "AKP" \
            or jwk.get("alg") != "ML-DSA-65":
        return None
    pub = jwk.get("pub")
    if not isinstance(pub, str) or not pub:
        return None
    try:
        raw = _b64u_to_bytes(pub)
    except (ValueError, TypeError):
        return None
    return raw if len(raw) == _MLDSA65_PUBKEY_BYTES else None


def _pq_bound(receipt: dict, doc: dict, payload: bytes):
    pq = receipt.get("pqSignature")
    if not isinstance(pq, dict):
        return False, None
    keyid = pq.get("keyid")
    if pq.get("alg") != "ML-DSA-65":
        return False, keyid
    try:
        sig = bytes.fromhex(pq.get("sig", ""))
    except ValueError:
        return False, keyid
    for method in doc.get("verificationMethod", []):
        if not isinstance(method, dict) or method.get("id") != keyid:
            continue
        raw = _mldsa_pub(method)
        if raw is None:
            continue
        try:
            if ML_DSA_65.verify(raw, payload, sig):
                return True, keyid
        except (ValueError, TypeError):
            return False, keyid
    return False, keyid


def _evaluate(case: dict) -> dict:
    receipt, doc = case["receipt"], case["didDocument"]
    payload = rfc8785.dumps({k: receipt[k] for k in _BLOCKS})
    classical_bound = _classical_bound(receipt, doc, payload)
    suite = receipt["receiptAsserted"].get("sigSuite")
    pq_bound, pq_keyid = _pq_bound(receipt, doc, payload)

    def verdict(tier, cb, pb, kid, qr, dr):
        return {"tier": tier, "classical_bound": cb, "pq_bound": pb,
                "suite": suite, "pq_keyid": kid,
                "quantum_resistant": qr, "downgrade_resistant": dr}

    if suite is not None:
        members = _HYBRID_SUITES.get(suite)
        if members is None:
            return verdict("hybrid-downgraded", classical_bound, pq_bound,
                           pq_keyid, False, False)
        if receipt["alg"] != members[0]:
            return verdict("hybrid-downgraded", classical_bound, pq_bound,
                           pq_keyid, False, False)
        if receipt.get("pqSignature") is None:
            return verdict("hybrid-downgraded", classical_bound, False,
                           None, False, False)
        if not pq_bound:
            return verdict("hybrid-downgraded", classical_bound, False,
                           pq_keyid, False, False)
        if not classical_bound:
            return verdict("hybrid-downgraded", False, pq_bound,
                           pq_keyid, False, False)
        return verdict("hybrid-verified", True, True, pq_keyid, True, True)

    if pq_bound:
        return verdict("pqc-present", classical_bound, True, pq_keyid, True, False)
    return verdict("classical-only", classical_bound, False, pq_keyid, False, False)


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for case in cases:
        name = case["name"]
        got = {k: _evaluate(case)[k] for k in COMPARE}
        want = {k: expected[name][k] for k in COMPARE}
        if got != want:
            failures.append(f"{name}: expected {want}, got {got}")
        else:
            print(f"{name}: OK {got['tier']}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall pq_hybrid_v0 cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
