#!/usr/bin/env python3
"""Independent checker for the key_rotation_v0 vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It
does not import Vaara. For each committed case it reproduces verification of a
retained record under a key that was later rotated out:

1. **Bind.** Verify the ES256 receipt signature against a verification key the
   archived DID document lists (level-2, offline). The bound method's id is
   the keyid.
2. **Validity window.** Read the bound method's ``validFrom`` / ``validUntil``
   (or ``notBefore`` / ``notAfter``) markers and test the receipt ``iat``
   against the half-open window ``[not_before, not_after)``.
3. **Revocation.** Read the bound method's ``revoked`` instant and apply the
   revocation-in-time rule at ``iat``; revocation overrides graceful retirement.
4. **Anchor corroboration.** When the case supplies an attested
   ``anchoredTime``, test whether it predates the key's retirement and any
   revocation: that is what makes the in-window claim impossible to forge later
   with a stolen retired key.

Verdicts are compared against ``expected.json`` (the non-normative ``reason``
string is not compared). Run:
``python tests/vectors/key_rotation_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent
_RECEIPT_BLOCKS = ("version", "alg", "backLink", "outcomeDerived", "receiptAsserted")
_NOT_BEFORE_KEYS = ("validFrom", "notBefore")
_NOT_AFTER_KEYS = ("validUntil", "notAfter")
COMPARE = (
    "bound", "keyid", "within_window", "window_recorded", "not_before",
    "not_after", "revoked", "revoked_at", "time_basis",
    "anchored_before_retirement", "anchored_before_revocation",
    "verifiable", "corroborated",
)


def _b64u_to_int(value: str) -> int:
    pad = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + pad), "big")


def _jwk_to_ec_key(jwk: dict):
    if jwk.get("kty") != "EC" or jwk.get("crv") != "P-256":
        return None
    numbers = ec.EllipticCurvePublicNumbers(
        _b64u_to_int(jwk["x"]), _b64u_to_int(jwk["y"]), ec.SECP256R1()
    )
    return numbers.public_key()


def _es256_verifies(payload: bytes, signature_hex: str, public_key) -> bool:
    if len(signature_hex) != 128:
        return False
    try:
        raw = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    der = encode_dss_signature(
        int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
    )
    try:
        public_key.verify(der, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def _parse_iso(value):
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _first(method: dict, keys):
    for key in keys:
        value = method.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _within(at_time, not_before, not_after) -> bool:
    at = _parse_iso(at_time)
    if at is None:
        return False
    if not_before is not None:
        nb = _parse_iso(not_before)
        if nb is None or at < nb:
            return False
    if not_after is not None:
        na = _parse_iso(not_after)
        if na is None or at >= na:
            return False
    return True


def _anchor_before(anchored_time, boundary) -> bool:
    if boundary is None:
        return True
    if anchored_time is None:
        return False
    a = _parse_iso(anchored_time)
    b = _parse_iso(boundary)
    if a is None or b is None:
        return False
    return a < b


def _miss() -> dict:
    return dict(zip(COMPARE, (
        False, None, False, False, None, None, False, None,
        "self_asserted", False, False, False, False,
    )))


def _evaluate(case: dict) -> dict:
    receipt = case["receipt"]
    doc = case["didDocument"]
    iss = receipt["receiptAsserted"]["iss"]
    iat = receipt["receiptAsserted"]["iat"]
    anchored = case.get("anchoredTime")

    out = _miss()
    out["time_basis"] = "anchored" if anchored is not None else "self_asserted"
    if not iss.startswith("did:web:") or doc.get("id") != iss \
            or receipt.get("alg") != "ES256":
        return out

    payload = rfc8785.dumps({k: receipt[k] for k in _RECEIPT_BLOCKS})
    bound_method = None
    for method in doc.get("verificationMethod", []):
        key = _jwk_to_ec_key(method.get("publicKeyJwk", {}))
        if key is not None and _es256_verifies(payload, receipt["signature"], key):
            bound_method = method
            break
    if bound_method is None:
        return out

    # Mirror KeyHistory / RevocationRegistry, which key by method id and
    # aggregate across every verification method carrying that id (a key may
    # be split across more than one method, e.g. a reactivation after a gap).
    keyid = bound_method.get("id")
    same = [m for m in doc.get("verificationMethod", [])
            if isinstance(m, dict) and m.get("id") == keyid]

    # Validity window: the key is in window if iat falls inside ANY recorded
    # window for it. Report the admitting window, else the first recorded.
    windows = [(_first(m, _NOT_BEFORE_KEYS), _first(m, _NOT_AFTER_KEYS))
               for m in same]
    windows = [(nb, na) for (nb, na) in windows if nb is not None or na is not None]
    if not windows:
        not_before = not_after = None
        window_recorded = False
        within_window = True
    else:
        window_recorded = True
        admitting = next(((nb, na) for (nb, na) in windows if _within(iat, nb, na)),
                         None)
        within_window = admitting is not None
        not_before, not_after = admitting if admitting is not None else windows[0]

    # Revocation: earliest revocation at or before iat wins; an unparseable
    # instant fails closed (counts as revoked).
    issued = _parse_iso(iat)
    matched = []
    for m in same:
        marker = m.get("revoked")
        if not isinstance(marker, str) or not marker:
            continue
        parsed = _parse_iso(marker)
        if parsed is None or issued is None or parsed <= issued:
            matched.append((parsed, marker))
    revoked, revoked_at = False, None
    if matched:
        parseable = [(p, m) for (p, m) in matched if p is not None]
        revoked = True
        revoked_at = (min(parseable, key=lambda t: t[0])[1] if parseable
                      else matched[0][1])

    verifiable = within_window and not revoked
    abr = _anchor_before(anchored, not_after) if anchored is not None else False
    abrev = _anchor_before(anchored, revoked_at) if anchored is not None else False
    corroborated = verifiable and anchored is not None and abr and abrev

    out.update({
        "bound": True, "keyid": bound_method.get("id"),
        "within_window": within_window, "window_recorded": window_recorded,
        "not_before": not_before, "not_after": not_after,
        "revoked": revoked, "revoked_at": revoked_at,
        "anchored_before_retirement": abr, "anchored_before_revocation": abrev,
        "verifiable": verifiable, "corroborated": corroborated,
    })
    return out


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for case in cases:
        name = case["name"]
        got = _evaluate(case)
        want = expected[name]
        if got != want:
            failures.append(f"{name}: expected {want}, got {got}")
        else:
            print(f"{name}: OK verifiable={got['verifiable']} "
                  f"corroborated={got['corroborated']}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall key_rotation_v0 cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
