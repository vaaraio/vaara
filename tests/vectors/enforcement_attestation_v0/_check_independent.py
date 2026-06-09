#!/usr/bin/env python3
"""Independent checker for the enforcement_attestation_v0 vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
not import Vaara. For each committed case it reproduces the verdict
``verify_enforcement`` returns:

1. Parse the SEV-SNP report at the AMD ABI rev 1.55 (Table 22) offsets: VERSION
   at 0x000, SIGNATURE_ALGO at 0x034, REPORT_DATA at 0x050 (64 bytes),
   MEASUREMENT at 0x090 (48 bytes), the signed body is 0x000:0x2A0, and the
   ECDSA signature (r || s, each 72-byte little-endian, 48 significant) is
   0x2A0:0x2A0+512. A length other than 1184 is unparseable.
2. Compute the expected REPORT_DATA as SHA-512 of the RFC 8785 (JCS) bytes of
   the FULL record (including its ``signature`` field) and byte-compare all 64
   bytes. The full-record preimage is what defeats both substitution and the
   signature-malleable variant.
3. Verify the ECDSA-P384-SHA384 signature over the body against the VCEK PEM,
   only when the version is supported and the algorithm is ECDSA-P384-SHA384.
4. Pin the measurement against ``expected_measurement`` when supplied.

The VCEK is trusted as supplied; its chain to AMD's ARK is not validated (the
same v0 deferral the verdict discloses), so ``vcek_chain_basis`` is always
``caller_supplied_unverified`` and the chain-rooted ``attested`` tier (and a
strict pass) is unreachable. Verdicts are compared against ``expected.json``
(the non-normative ``reason`` is not compared). Run:
``python tests/vectors/enforcement_attestation_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent

REPORT_SIZE = 1184
BODY_SIZE = 0x2A0
SIG_SIZE = 512
SUPPORTED_VERSIONS = frozenset({2})
ALGO_ECDSA_P384_SHA384 = 1
COMPARE = (
    "tier", "parsed", "report_version", "signature_algo_ok", "signature_valid",
    "bound", "report_data_expected", "report_data_actual", "measurement",
    "measurement_basis", "vcek_chain_basis", "enforcement_logic_basis",
    "strict", "ok",
)


def _normalize_hex(value):
    if not isinstance(value, str):
        return None
    try:
        return bytes.fromhex(value.strip())
    except ValueError:
        return None


def _reject_floats(value) -> None:
    """Raise ValueError on any IEEE-754 float, mirroring canonical_json.

    Vaara's canonicaliser rejects floats at the boundary, so ``verify_enforcement``
    refuses a float-bearing record rather than emitting a verdict. The checker
    mirrors that, so it never silently produces a verdict Vaara would not.
    """
    if isinstance(value, float):
        raise ValueError("IEEE-754 float in record is prohibited")
    if isinstance(value, dict):
        for v in value.values():
            _reject_floats(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _reject_floats(v)


def _verify_p384(body: bytes, sig_field: bytes, vcek_pem: bytes) -> bool:
    """ECDSA-P384-SHA384 over the body against the VCEK.

    Mirrors ``verify_sev_snp_report_signature``: a verifier-side input error (an
    unloadable PEM, a non-EC key, or the wrong curve) raises ``ValueError``;
    only a genuine signature mismatch returns False. Vaara raises
    ``TEEAttestationError`` for the same inputs, so neither side silently turns a
    bad VCEK into a verdict.
    """
    try:
        pub = serialization.load_pem_public_key(vcek_pem)
    except ValueError as exc:
        raise ValueError(f"failed to load VCEK PEM: {exc}") from exc
    if not isinstance(pub, ec.EllipticCurvePublicKey):
        raise ValueError("VCEK is not an EC public key")
    if not isinstance(pub.curve, ec.SECP384R1):
        raise ValueError(f"VCEK curve is {pub.curve.name}; expected secp384r1")
    r = int.from_bytes(sig_field[:48], "little")
    s = int.from_bytes(sig_field[72:72 + 48], "little")
    der = encode_dss_signature(r, s)
    try:
        pub.verify(der, body, ec.ECDSA(hashes.SHA384()))
        return True
    except InvalidSignature:
        return False


def _evaluate(case: dict) -> dict:
    record = case["record"]
    report = base64.b64decode(case["report_b64"])
    vcek_pem = case["vcek_pem"].encode("ascii")
    expected_measurement = case["expected_measurement"]
    strict = bool(case["strict"])

    _reject_floats(record)
    try:
        canonical = rfc8785.dumps(record)
    except Exception as exc:  # noqa: BLE001 - bignum / non-string key, like canonical_json
        raise ValueError(f"cannot canonicalise record: {exc}") from exc
    expected_report_data = hashlib.sha512(canonical).digest()
    report_data_expected = expected_report_data.hex()

    parsed = len(report) == REPORT_SIZE
    report_version = None
    version_ok = False
    signature_algo_ok = False
    signature_valid = False
    bound = False
    report_data_actual = None
    measurement = None

    if parsed:
        report_version = int.from_bytes(report[0x000:0x004], "little")
        algo = int.from_bytes(report[0x034:0x038], "little")
        report_data = report[0x050:0x090]
        measurement_bytes = report[0x090:0x0C0]
        body = report[:BODY_SIZE]
        sig_field = report[BODY_SIZE:BODY_SIZE + SIG_SIZE]
        report_data_actual = report_data.hex()
        measurement = measurement_bytes.hex()
        version_ok = report_version in SUPPORTED_VERSIONS
        signature_algo_ok = algo == ALGO_ECDSA_P384_SHA384
        bound = report_data == expected_report_data
        if version_ok and signature_algo_ok:
            signature_valid = _verify_p384(body, sig_field, vcek_pem)

    if expected_measurement is None:
        measurement_basis = "unpinned"
    else:
        exp = _normalize_hex(expected_measurement)
        if parsed and exp is not None and report[0x090:0x0C0] == exp:
            measurement_basis = "pinned"
        else:
            measurement_basis = "pin_mismatch"

    vcek_chain_basis = "caller_supplied_unverified"
    enforcement_logic_basis = "not_established"

    crypto_ok = (
        parsed and version_ok and signature_algo_ok and signature_valid and bound
    )
    if not crypto_ok:
        tier = "unverified"
    elif measurement_basis == "pinned":
        tier = "measurement_pinned"
    else:
        tier = "bound"

    if strict:
        ok = bool(
            crypto_ok
            and measurement_basis == "pinned"
            and vcek_chain_basis == "kds_verified"
        )
    else:
        ok = bool(crypto_ok and measurement_basis != "pin_mismatch")

    return {
        "tier": tier,
        "parsed": parsed,
        "report_version": report_version,
        "signature_algo_ok": signature_algo_ok,
        "signature_valid": signature_valid,
        "bound": bound,
        "report_data_expected": report_data_expected,
        "report_data_actual": report_data_actual,
        "measurement": measurement,
        "measurement_basis": measurement_basis,
        "vcek_chain_basis": vcek_chain_basis,
        "enforcement_logic_basis": enforcement_logic_basis,
        "strict": strict,
        "ok": ok,
    }


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for case in cases:
        name = case["name"]
        got = _evaluate(case)
        want = expected[name]
        if {k: got[k] for k in COMPARE} != want:
            failures.append(f"{name}:\n    expected {want}\n    got      {got}")
        else:
            print(f"{name}: OK tier={got['tier']} ok={got['ok']} "
                  f"bound={got['bound']} sig={got['signature_valid']}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall enforcement_attestation_v0 cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
