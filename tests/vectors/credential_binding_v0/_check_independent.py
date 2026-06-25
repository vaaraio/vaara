"""Independent conformance checker for credential_binding_v0 vectors.

No Vaara import.  Uses only hmac, hashlib, json, rfc8785, and stdlib datetime.
Every verification step replicates the gateway logic from first principles so
the passing verdict is a property of the bytes in each case file, not of the
Vaara codebase.

Run: python3 tests/vectors/credential_binding_v0/_check_independent.py
Exit 0 = all cases match expected. Non-zero = mismatch printed and raised.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
from datetime import datetime
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve().parent

# Must match the key used in _generate.py.
KEY = b"x" * 32

CLOCK_SKEW = 30  # seconds; matches gateway default


# ── helpers ───────────────────────────────────────────────────────────────────

def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _iso_to_epoch(iso: str) -> float:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso).timestamp()


def _args_commitment(args: dict) -> str:
    """Replicate make_args_digest from _sep2787_canonical.py.

    Step 1: sha256 over JCS of args.
    Step 2: sha256 over JCS of {"digest": "<step1>"}.
    """
    args_digest_hex = "sha256:" + hashlib.sha256(_jcs(args)).hexdigest()
    projection_bytes = _jcs({"digest": args_digest_hex})
    return "sha256:" + hashlib.sha256(projection_bytes).hexdigest()


def _signing_payload(cred: dict) -> bytes:
    """JCS of {version, alg, scope, binding, asserted} — capabilities included when present."""
    body = {
        "alg": cred["alg"],
        "asserted": cred["asserted"],
        "binding": cred["binding"],
        "scope": cred["scope"],
        "version": cred["version"],
    }
    if cred.get("capabilities"):
        body["capabilities"] = cred["capabilities"]
    return _jcs(body)


def _verify_hs256(payload: bytes, signature_hex: str) -> bool:
    expected = hmac.new(KEY, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)


def _verify(case: dict) -> str:
    """Return the computed verdict string."""
    cred = case.get("credential")

    if cred is None:
        return "missing_credential"

    # 1. signature
    payload = _signing_payload(cred)
    if not _verify_hs256(payload, cred["signature"]):
        return "bad_signature"

    # 2. expiry
    asserted = cred["asserted"]
    iat_epoch = _iso_to_epoch(asserted["iat"])
    now = float(case["now"])
    deadline = iat_epoch + asserted["expSeconds"] + CLOCK_SKEW
    if iat_epoch > now + CLOCK_SKEW or now > deadline:
        return "expired"

    # 3. scope — tool name
    scope = cred["scope"]
    if scope["toolName"] != case["runtime_tool_name"]:
        return "scope_mismatch"

    # 4. scope — tenant
    if scope["tenantId"] != case["runtime_tenant_id"]:
        return "scope_mismatch"

    # 5. scope — args commitment (capabilities mode skipped: corpus uses exact-args grants)
    runtime_commitment = _args_commitment(case["runtime_args"])
    if scope["argsCommitment"] != runtime_commitment:
        return "scope_mismatch"

    # 6. binding
    known = set(case.get("known_attestation_digests", []))
    if cred["binding"]["attestationDigest"] not in known:
        return "binding_unknown"

    return "ok"


def main() -> int:
    expected_path = HERE / "expected.json"
    if not expected_path.exists():
        print("expected.json not found — run _generate.py first", file=sys.stderr)
        return 1

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    cases_dir = HERE / "cases"
    failures = []

    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        computed = _verify(case)
        want = case["expected_verdict"]
        status = "PASS" if computed == want else "FAIL"
        print(f"  {status}  {path.stem:<30}  computed={computed!r}  expected={want!r}")
        if computed != want:
            failures.append(path.stem)

    # Cross-check against expected.json
    for name, meta in expected["cases"].items():
        want = meta["expected_verdict"]
        path = cases_dir / f"{name}.json"
        if not path.exists():
            print(f"  MISSING  {name}", file=sys.stderr)
            failures.append(name)
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        computed = _verify(case)
        if computed != want:
            failures.append(f"{name}(expected.json cross-check)")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1

    print(f"\nall {len(list(cases_dir.glob('*.json')))} cases pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
