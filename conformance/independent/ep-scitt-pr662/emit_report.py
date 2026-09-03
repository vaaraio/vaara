#!/usr/bin/env python3
"""Deterministic machine-readable report for the second-runner result on
EP-SCITT-STATEMENT-IDENTITY-v0.1.

Emits report.independent.json. Contains no timestamps and no random values, so
re-running on the same inputs produces byte-identical output and the same
SHA-256. Import surface is stdlib plus `cryptography`. Nothing from EMILIA,
nothing from Vaara.

Usage:  python emit_report.py > report.independent.json
"""

import base64
import hashlib
import json
import os
import platform
import sys

import cryptography
from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

HERE = os.path.dirname(os.path.abspath(__file__))
VECTORS = os.path.join(HERE, "vectors.reference.json")
VERIFIER = os.path.join(HERE, "independent_verify.py")


def b64u(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def tag(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def digest_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify(key, sig, msg):
    if len(sig) != 64:
        return False
    r = int.from_bytes(sig[:32], "big")
    s = int.from_bytes(sig[32:], "big")
    try:
        key.verify(utils.encode_dss_signature(r, s), msg, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


def main():
    raw = open(VECTORS, "rb").read()
    v = json.loads(raw)
    fx, exp = v["fixture"], v["expected"]

    jwk = fx["public_jwk"]
    key = ec.EllipticCurvePublicNumbers(
        int.from_bytes(b64u(jwk["x"]), "big"),
        int.from_bytes(b64u(jwk["y"]), "big"),
        ec.SECP256R1(),
    ).public_key()

    si = b64u(fx["sig_structure_base64url"])
    sa = b64u(fx["signature_a_base64url"])
    sb = b64u(fx["signature_b_base64url"])
    ea = b64u(fx["cose_sign1_a_base64url"])
    eb = b64u(fx["cose_sign1_b_base64url"])

    ra, va = int.from_bytes(sa[:32], "big"), int.from_bytes(sa[32:], "big")
    rb, vb = int.from_bytes(sb[:32], "big"), int.from_bytes(sb[32:], "big")

    positive = {
        "signature_a_valid": verify(key, sa, si),
        "signature_b_valid": verify(key, sb, si),
        "statement_entry_digest_a": tag(ea),
        "statement_entry_digest_b": tag(eb),
        "entry_digests_differ": tag(ea) != tag(eb),
        "signing_input_digest": tag(si),
        "same_r": ra == rb,
        "s_b_equals_n_minus_s_a": vb == N - va,
        "a_is_high_s": va > N // 2,
        "b_is_low_s": vb <= N // 2,
    }
    positive["classification"] = (
        "same_signing_input_different_envelope"
        if positive["signature_a_valid"]
        and positive["signature_b_valid"]
        and positive["entry_digests_differ"]
        and positive["same_r"]
        else "other"
    )

    expected = {
        "signature_a_valid": exp["signature_a_valid"],
        "signature_b_valid": exp["signature_b_valid"],
        "statement_entry_digest_a": exp["statement_entry_digest_a"],
        "statement_entry_digest_b": exp["statement_entry_digest_b"],
        "entry_digests_differ": True,
        "signing_input_digest": exp["signing_input_digest"],
        "same_r": True,
        "s_b_equals_n_minus_s_a": True,
        "a_is_high_s": True,
        "b_is_low_s": True,
        "classification": exp["classification"],
    }

    tampered = bytearray(si)
    tampered[-1] ^= 0x01
    other_key = ec.EllipticCurvePublicNumbers(
        int.from_bytes(b64u(jwk["x"]), "big") ^ 1,
        int.from_bytes(b64u(jwk["y"]), "big"),
        ec.SECP256R1(),
    )
    hostile = {
        "payload_bit_flip_vs_sig_a": verify(key, sa, bytes(tampered)),
        "payload_bit_flip_vs_sig_b": verify(key, sb, bytes(tampered)),
        "reversed_s_bytes": verify(key, sa[:32] + sb[32:64][::-1], si),
        "all_zero_signature": verify(key, b"\x00" * 64, si),
        "s_plus_group_order": verify(
            key, sa[:32] + ((va + N) % (1 << 256)).to_bytes(32, "big"), si
        ),
        "envelope_a_bytes_as_signing_input": verify(key, sa, ea),
    }
    try:
        hostile["valid_sig_wrong_key"] = verify(other_key.public_key(), sa, si)
    except ValueError:
        # x^1 is not on the curve; use a deterministic on-curve alternative.
        alt = ec.derive_private_key(2, ec.SECP256R1()).public_key()
        hostile["valid_sig_wrong_key"] = verify(alt, sa, si)

    report = {
        "report_version": "independent-second-runner-v1",
        "profile": v["profile"],
        "vectors_version": v["@version"],
        "consumed_input": {
            "file": "vectors.reference.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source": (
                "https://github.com/emiliaprotocol/emilia-protocol/tree/"
                "e507acdf8efbe8951cb4294801d4c440f0b86a5a/conformance/"
                "composition/scitt-statement-identity-v0.1"
            ),
            "branch": "feat/aic-ccs-partner-artifacts",
        },
        "verifier": {
            "file": "independent_verify.py",
            "sha256": digest_file(VERIFIER),
            "language": "Python",
            "imports_emilia_code": False,
            "imports_vaara_code": False,
            "third_party_imports": ["cryptography"],
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "cryptography": cryptography.__version__,
            "openssl": backend.openssl_version_text(),
            "os": "Linux",
            "kernel": platform.release(),
            "machine": platform.machine(),
            "libc": platform.libc_ver()[0] + " " + platform.libc_ver()[1],
        },
        "positive": positive,
        "positive_expected": expected,
        "positive_all_match": all(positive[k] == expected[k] for k in expected),
        "hostile_accepted": hostile,
        "hostile_all_refused": not any(hostile.values()),
        "establishes": (
            "Independent implementation over pinned, author-supplied vectors. "
            "The published vectors reproduce in a different language on a "
            "different platform, and the three identities remain separate when "
            "computed by a second party."
        ),
        "does_not_establish": [
            "independently derived vectors",
            "EP profile verification",
            "Transparency Service registration",
            "validation of the specification text",
        ],
        "run_kind": "independent-implementation",
        "runner": "Henri Sirkkavaara, Vaara",
    }

    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
