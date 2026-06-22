#!/usr/bin/env python3
"""Independent checker for the Visa TAP <-> Vaara binding profile (tap_v0).

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. It reproduces, from the committed Trusted Agent Protocol (TAP)
request and the held vaara.receipt/v1 decision receipts alone, every verdict a
third party reaches with nothing but the issuer's public key. No live endpoint,
no service to trust: the verdict is recomputed offline from the bytes in hand.

Per lifecycle step (step0 in-progress, step1 terminal):

  action_ref_recomputes      sha256 over the JCS-canonical action tuple
                             (agentId, actionType, scope, timestampMs, seq,
                             terminal) equals request.actionRef, the join key.
  request_binding_resolves   sha256 over the JCS-canonical TAP request equals the
                             receipt's decisionDerived.evidenceRef.digest: the
                             receipt names that exact request by content address.
  receipt_signature_ok       the ES256 signature verifies over the canonical
                             (version, alg, backLink, decisionDerived,
                             issuerAsserted) blocks against the public key.

And one cross-step verdict:

  lifecycle_distinguishes_terminal
                             the in-progress and terminal steps have distinct
                             action_refs, carry terminal=false / terminal=true,
                             and the in-progress receipt does not resolve against
                             the terminal request. A mid-action receipt cannot be
                             passed off as the final one.

Run: tests/vectors/tap_v0/_check_independent.py
Exit 0 means every verdict matched expected.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.serialization import load_pem_public_key

HERE = Path(__file__).resolve().parent
KEYS = HERE / "keys"
_DECISION_BLOCKS = ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")
_ACTION_KEYS = ("agentId", "actionType", "scope", "timestampMs", "seq", "terminal")
_STEPS = ("step0", "step1")


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def action_ref_recomputes(request: dict) -> bool:
    action = {k: request[k] for k in _ACTION_KEYS}
    return _sha256_hex(_jcs(action)) == request["actionRef"]


def request_binding_resolves(receipt: dict, request: dict) -> bool:
    ref = receipt.get("decisionDerived", {}).get("evidenceRef")
    if not isinstance(ref, dict) or ref.get("canonicalization") != "JCS":
        return False
    return _sha256_hex(_jcs(request)) == ref.get("digest")


def receipt_signature_ok(receipt: dict, pub) -> bool:
    if receipt.get("alg") != "ES256":
        return False
    sig = receipt.get("signature", "")
    if len(sig) != 128:
        return False
    payload = _jcs({k: receipt[k] for k in _DECISION_BLOCKS})
    raw = bytes.fromhex(sig)
    der = encode_dss_signature(
        int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
    )
    try:
        pub.verify(der, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def _step_verdicts(request: dict, receipt: dict, pub) -> dict:
    return {
        "action_ref_recomputes": action_ref_recomputes(request),
        "request_binding_resolves": request_binding_resolves(receipt, request),
        "receipt_signature_ok": receipt_signature_ok(receipt, pub),
    }


def lifecycle_distinguishes_terminal(q0: dict, r0: dict, q1: dict) -> bool:
    # distinct join keys for the two lifecycle points
    if q0["actionRef"] == q1["actionRef"]:
        return False
    # the tuple carries an explicit, opposite terminal flag
    if q0.get("terminal") is not False or q1.get("terminal") is not True:
        return False
    # the in-progress receipt does not resolve against the terminal request:
    # a mid-action receipt cannot be presented where the final one is required
    ref0 = r0.get("decisionDerived", {}).get("evidenceRef", {})
    return _sha256_hex(_jcs(q1)) != ref0.get("digest")


def _load(step: str, name: str) -> dict:
    return json.loads((HERE / step / name).read_text(encoding="utf-8"))


def main() -> int:
    pub = load_pem_public_key((KEYS / "es256_public.pem").read_bytes())
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    q0, r0 = _load("step0", "request.json"), _load("step0", "receipt.json")
    q1, r1 = _load("step1", "request.json"), _load("step1", "receipt.json")
    got = {
        "step0": _step_verdicts(q0, r0, pub),
        "step1": _step_verdicts(q1, r1, pub),
        "lifecycle_distinguishes_terminal": lifecycle_distinguishes_terminal(q0, r0, q1),
    }

    ok = got == expected
    for step in _STEPS:
        for k, v in got[step].items():
            mark = "OK" if v == expected.get(step, {}).get(k) else "FAIL"
            print(f"[{mark}] {step}.{k}: {v}")
    lv = got["lifecycle_distinguishes_terminal"]
    mark = "OK" if lv == expected.get("lifecycle_distinguishes_terminal") else "FAIL"
    print(f"[{mark}] lifecycle_distinguishes_terminal: {lv}")
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
