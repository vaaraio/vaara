#!/usr/bin/env python3
"""Independent verifier for the v0 credential-grant vectors.

A second implementation of the brokered-credential rules, written from the
wire spec alone with no ``import vaara``. For each committed set it
re-derives, from the bytes on disk:

* the HS256 signature over the JCS-canonical grant preimage
  ``{version, alg, scope, binding, asserted}``;
* the bound attestation digest (sha256 over the JCS of ``attestation.json``);
* the args commitment (the two-step hash-only projection Vaara uses);

then reproduces the gateway verdict in the same precedence
(bad_signature -> expired -> scope_mismatch -> binding_unknown) and compares
it to ``expected.json``. ``rfc8785`` provides the one shared primitive (JCS);
everything else is the standard library. All vectors are HS256; ES256 / RS256
verify the identical preimage with a public key, so the format claim carries.

Run: ``python conformance/sep2828/credential_grant_v0/_check_independent.py``.
Exit 0 means every set reproduced its expected verdict.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve().parent
CLOCK_SKEW_SECONDS = 30


def _jcs(value) -> bytes:
    return rfc8785.dumps(value)


def _sha256_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _args_commitment(args) -> str:
    inner = _sha256_digest(_jcs(args))
    return _sha256_digest(_jcs({"digest": inner}))


def _preimage(grant) -> bytes:
    return _jcs(
        {
            "version": grant["version"],
            "alg": grant["alg"],
            "scope": grant["scope"],
            "binding": grant["binding"],
            "asserted": grant["asserted"],
        }
    )


def _iso_to_epoch(iso: str):
    from datetime import datetime

    try:
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return None


def verdict(grant, attestation, inputs) -> dict:
    """Reproduce the gateway verdict for one set (HS256 only)."""
    key = bytes.fromhex(inputs["keyHex"])
    expected_sig = hmac.new(key, _preimage(grant), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, grant.get("signature", "")):
        return {"ok": False, "reason": "bad_signature"}

    asserted = grant["asserted"]
    iat_epoch = _iso_to_epoch(asserted["iat"])
    now = inputs["now"]
    if iat_epoch is None:
        return {"ok": False, "reason": "expired"}
    deadline = iat_epoch + asserted["expSeconds"] + CLOCK_SKEW_SECONDS
    if iat_epoch > now + CLOCK_SKEW_SECONDS or now > deadline:
        return {"ok": False, "reason": "expired"}

    scope = grant["scope"]
    if scope["toolName"] != inputs["toolName"]:
        return {"ok": False, "reason": "scope_mismatch"}
    if scope["tenantId"] != inputs["tenantId"]:
        return {"ok": False, "reason": "scope_mismatch"}
    if scope["argsCommitment"] != _args_commitment(inputs["args"]):
        return {"ok": False, "reason": "scope_mismatch"}

    known = {_sha256_digest(_jcs(attestation))}
    if grant["binding"]["attestationDigest"] not in known:
        return {"ok": False, "reason": "binding_unknown"}

    return {"ok": True, "reason": "ok"}


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text())
    failures = 0
    for name in sorted(expected):
        d = HERE / "sets" / name
        grant = json.loads((d / "grant.json").read_text())
        attestation = json.loads((d / "attestation.json").read_text())
        inputs = json.loads((d / "inputs.json").read_text())
        got = verdict(grant, attestation, inputs)
        ok = got == expected[name]
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {got['reason']}")
        if not ok:
            print("  want:", expected[name])
            print("  got :", got)
    print(f"\n{len(expected) - failures}/{len(expected)} sets matched expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
