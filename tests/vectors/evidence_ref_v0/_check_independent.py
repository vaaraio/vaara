#!/usr/bin/env python3
"""Independent conformance checker for the evidenceRef vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``.
It does not import Vaara. It reads the committed fixtures from disk and
reproduces, for each case, the decision signature verification and the
content-address resolution of the cited drift record, then compares
against ``expected.json``.

A second implementation that can run this file (or reproduce its logic)
demonstrates the evidenceRef binding is verifiable without depending on
Vaara, and that a detector and a decision issuer agree on the content
address without trusting each other. Run:
``python tests/vectors/evidence_ref_v0/_check_independent.py``.
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


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def verify_signature(record: dict, blocks) -> bool:
    payload = _jcs({k: record[k] for k in blocks})
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


# Canonicalization rules this checker can apply to recompute a content
# address. Named in evidenceRef.canonicalization so a later rule is an
# explicit value rather than a silent reinterpretation.
_CANONICALIZERS = {"JCS": _jcs}


def evidence_ref_resolves(decision: dict, drift_record: dict) -> bool:
    """The cited content address equals the address recomputed from the
    committed drift record bytes. Fails closed when the decision carries no
    evidenceRef, names a canonicalization this checker does not implement,
    or omits the digest: a verifier asked whether the decision's evidence
    reference resolves answers no when there is no resolvable reference."""
    ref = decision.get("decisionDerived", {}).get("evidenceRef")
    if not isinstance(ref, dict):
        return False
    canon = _CANONICALIZERS.get(ref.get("canonicalization"))
    digest = ref.get("digest")
    if canon is None or not isinstance(digest, str):
        return False
    return hmac.compare_digest(_sha256_hex(canon(drift_record)), digest)


def _load(case: Path, name: str):
    p = case / name
    return json.loads(p.read_text()) if p.exists() else None


# Declarative keys carried in expected.json that are documentation, not
# crypto verdicts; the checker passes them through verbatim.
_DOC_KEYS = {"note"}


def _verdicts(case: Path, expected: dict) -> dict:
    dec = _load(case, "decision.json")
    drift = _load(case, "drift_record.json")
    got: dict = {}
    for key in expected:
        if key in _DOC_KEYS:
            got[key] = expected[key]
        elif key == "decision_signature_ok":
            got[key] = verify_signature(dec, _DECISION_BLOCKS)
        elif key == "evidence_ref_resolves":
            got[key] = evidence_ref_resolves(dec, drift)
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
