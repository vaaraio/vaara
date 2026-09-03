#!/usr/bin/env python3
"""Independent second runner for EP-SCITT-STATEMENT-IDENTITY-v0.1.

Consumes vectors.reference.json only. Imports nothing from EMILIA and nothing
from Vaara. Python stdlib plus `cryptography` for raw P-256 verification.

Reports what Iman asked a second runner to report:
  1. both positive signatures
  2. the distinct exact-entry digests
  3. the common signing-input digest
  4. the hostile substitutions
"""

import base64
import hashlib
import json
import os
import platform
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

# SEC p256r1 group order.
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

# The vectors sit next to this file. Pass a path to point it somewhere else.
HERE = os.path.dirname(os.path.abspath(__file__))
VECTORS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "vectors.reference.json")


def b64u(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sha256_tag(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def load_key(jwk):
    return ec.EllipticCurvePublicNumbers(
        int.from_bytes(b64u(jwk["x"]), "big"),
        int.from_bytes(b64u(jwk["y"]), "big"),
        ec.SECP256R1(),
    ).public_key()


def verify_p1363(key, sig_raw, msg):
    """ES256: IEEE P1363 r||s, converted to DER for the library."""
    if len(sig_raw) != 64:
        return False
    r = int.from_bytes(sig_raw[:32], "big")
    s = int.from_bytes(sig_raw[32:], "big")
    try:
        key.verify(
            utils.encode_dss_signature(r, s),
            msg,
            ec.ECDSA(hashes.SHA256()),
        )
        return True
    except Exception:
        return False


def rs(sig_raw):
    return (
        int.from_bytes(sig_raw[:32], "big"),
        int.from_bytes(sig_raw[32:], "big"),
    )


def main():
    raw = open(VECTORS, "rb").read()
    v = json.loads(raw)
    fx, exp = v["fixture"], v["expected"]

    key = load_key(fx["public_jwk"])
    sig_struct = b64u(fx["sig_structure_base64url"])
    sig_a = b64u(fx["signature_a_base64url"])
    sig_b = b64u(fx["signature_b_base64url"])
    env_a = b64u(fx["cose_sign1_a_base64url"])
    env_b = b64u(fx["cose_sign1_b_base64url"])

    print(f"profile          {v['profile']}")
    print("runner           independent Python, cryptography only")
    print(f"platform         {platform.platform()}")
    print(f"python           {sys.version.split()[0]}")
    print(f"vectors sha-256  {hashlib.sha256(raw).hexdigest()}")
    print()

    results = []

    # 1. both positive signatures
    ok_a = verify_p1363(key, sig_a, sig_struct)
    ok_b = verify_p1363(key, sig_b, sig_struct)
    results.append(("signature_a_valid", ok_a, exp["signature_a_valid"]))
    results.append(("signature_b_valid", ok_b, exp["signature_b_valid"]))

    # 2. distinct exact-entry digests
    da, db = sha256_tag(env_a), sha256_tag(env_b)
    results.append(("statement_entry_digest_a", da, exp["statement_entry_digest_a"]))
    results.append(("statement_entry_digest_b", db, exp["statement_entry_digest_b"]))
    results.append(("entry_digests_differ", da != db, True))

    # 3. common signing-input digest
    si = sha256_tag(sig_struct)
    results.append(("signing_input_digest", si, exp["signing_input_digest"]))

    # the malleability relation itself
    ra, sa = rs(sig_a)
    rb, sb = rs(sig_b)
    results.append(("same_r", ra == rb, True))
    results.append(("s_b_equals_n_minus_s_a", sb == N - sa, True))
    results.append(("a_is_high_s", sa > N // 2, True))
    results.append(("b_is_low_s", sb <= N // 2, True))

    cls = (
        "same_signing_input_different_envelope"
        if (ok_a and ok_b and da != db and ra == rb)
        else "other"
    )
    results.append(("classification", cls, exp["classification"]))

    print("=== POSITIVE CASES ===")
    for name, got, want in results:
        mark = "PASS" if got == want else "FAIL"
        print(f"  [{mark}] {name:34s} got={got}")

    # 4. hostile substitutions
    print("\n=== HOSTILE SUBSTITUTIONS (all must be refused) ===")
    hostile = []

    tampered = bytearray(sig_struct)
    tampered[-1] ^= 0x01
    hostile.append(("payload_bit_flip_vs_sig_a", verify_p1363(key, sig_a, bytes(tampered))))
    hostile.append(("payload_bit_flip_vs_sig_b", verify_p1363(key, sig_b, bytes(tampered))))

    swapped = sig_a[:32] + sig_b[32:64][::-1]
    hostile.append(("reversed_s_bytes", verify_p1363(key, swapped, sig_struct)))

    zero = b"\x00" * 64
    hostile.append(("all_zero_signature", verify_p1363(key, zero, sig_struct)))

    s_plus_n = sig_a[:32] + ((sa + N) % (1 << 256)).to_bytes(32, "big")
    hostile.append(("s_plus_group_order", verify_p1363(key, s_plus_n, sig_struct)))

    other = ec.generate_private_key(ec.SECP256R1()).public_key()
    hostile.append(("valid_sig_wrong_key", verify_p1363(other, sig_a, sig_struct)))

    hostile.append(("envelope_a_bytes_as_signing_input", verify_p1363(key, sig_a, env_a)))

    for name, accepted in hostile:
        mark = "PASS" if not accepted else "FAIL"
        print(f"  [{mark}] {name:34s} accepted={accepted}")

    pos_fail = [r[0] for r in results if r[1] != r[2]]
    hos_fail = [h[0] for h in hostile if h[1]]
    print()
    print(f"positive: {len(results) - len(pos_fail)}/{len(results)} matched")
    print(f"hostile:  {len(hostile) - len(hos_fail)}/{len(hostile)} refused")
    if pos_fail or hos_fail:
        print(f"FAILURES: {pos_fail + hos_fail}")
        return 1
    print("RESULT: independent verifier agrees with vectors.reference.json in full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
