"""Independent conformance checker for capability_scope_v0 vectors.

No Vaara import. Uses only hmac, hashlib, decimal, json, rfc8785, and stdlib
datetime. Every verification step replicates the gateway logic from principles
so the passing verdict is a property of the bytes in each case file, not of
the Vaara codebase.

Run: python3 tests/vectors/capability_scope_v0/_check_independent.py
Exit 0 = all cases match expected. Non-zero = mismatch printed and raised.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve().parent

KEY = b"x" * 32
CLOCK_SKEW = 30


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _iso_to_epoch(iso: str) -> float:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso).timestamp()


def _signing_payload(cred: dict) -> bytes:
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


def _evaluate_capabilities(capabilities: list, runtime_args: object) -> str:
    """Replicate evaluate() from _grant_capability.py without importing it.

    Returns "ok", "capability_exceeded", or "capability_uncovered".
    Coverage is CLOSED: every runtime arg must be named by a capability.
    """
    if not isinstance(runtime_args, dict):
        return "capability_exceeded"
    named = {c["arg"] for c in capabilities}
    for key in runtime_args:
        if key not in named:
            return "capability_uncovered"
    for cap in capabilities:
        arg = cap["arg"]
        op = cap["op"]
        value = cap["value"]
        if arg not in runtime_args:
            return "capability_exceeded"
        actual = runtime_args[arg]
        if op == "eq":
            if isinstance(actual, bool) or str(actual) != value:
                return "capability_exceeded"
        elif op == "in":
            if isinstance(actual, bool) or str(actual) not in value:
                return "capability_exceeded"
        elif op in ("le", "ge"):
            if isinstance(actual, bool) or not isinstance(actual, (int, float, str)):
                return "capability_exceeded"
            try:
                a = Decimal(str(actual))
                bound = Decimal(value)
            except InvalidOperation:
                return "capability_exceeded"
            if op == "le" and not (a <= bound):
                return "capability_exceeded"
            if op == "ge" and not (a >= bound):
                return "capability_exceeded"
        else:
            return "capability_exceeded"
    return "ok"


def _verify(case: dict) -> str:
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

    # 5. capability enforcement (closed coverage)
    caps = cred.get("capabilities")
    if not caps:
        return "scope_mismatch"
    reason = _evaluate_capabilities(caps, case["runtime_args"])
    if reason != "ok":
        return reason

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
