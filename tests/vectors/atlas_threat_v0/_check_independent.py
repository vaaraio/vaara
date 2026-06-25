"""Independent conformance checker for atlas_threat_v0 vectors.

No Vaara import. Uses only hmac, hashlib, json, rfc8785, and stdlib.
Every verification step replicates the threat-detection logic from first
principles so a passing verdict is a property of the bytes in each case
file, not of the Vaara codebase.

Run: python3 tests/vectors/atlas_threat_v0/_check_independent.py
Exit 0 = all cases match expected. Non-zero = mismatch printed and raised.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve().parent

# Must match the key used in _generate.py.
KEY = b"y" * 32


def _jcs(obj: object) -> bytes:
    return rfc8785.dumps(obj)


def _args_commitment(args: dict) -> str:
    """Replicate make_args_digest from _sep2787_canonical.py.

    Step 1: sha256 over JCS of args.
    Step 2: sha256 over JCS of {"digest": "<step1>"}.
    """
    step1 = "sha256:" + hashlib.sha256(_jcs(args)).hexdigest()
    step2 = _jcs({"digest": step1})
    return "sha256:" + hashlib.sha256(step2).hexdigest()


def _signing_payload(receipt: dict) -> bytes:
    """JCS of all receipt fields excluding 'signature'."""
    return _jcs({k: v for k, v in receipt.items() if k != "signature"})


def _verify_hmac(receipt: dict) -> bool:
    expected = hmac.new(KEY, _signing_payload(receipt), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, receipt.get("signature", ""))


def _check(case: dict) -> str:
    receipt = case["receipt"]
    auth = case["authorization"]
    now_ms = int(case["now_ms"])
    window_ms = int(case["freshness_window_ms"])

    # 1. signature
    if not _verify_hmac(receipt):
        return "bad_signature"

    # 2. actionType (tool substitution detection)
    if receipt["actionType"] != auth["actionType"]:
        return "tool_mismatch"

    # 3. agentId
    if receipt["agentId"] != auth["agentId"]:
        return "agent_mismatch"

    # 4. argsCommitment (prompt injection / args mutation detection)
    if receipt["argsCommitment"] != auth["argsCommitment"]:
        return "args_tampered"

    # 5. scope (scope escalation detection)
    if receipt["scope"] != auth["scope"]:
        return "scope_exceeded"

    # 6. freshness (replay detection)
    age_ms = abs(now_ms - int(receipt["timestampMs"]))
    if age_ms > window_ms:
        return "stale"

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
        computed = _check(case)
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
        computed = _check(case)
        if computed != want:
            failures.append(f"{name}(expected.json cross-check)")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1

    print(f"\nall {len(list(cases_dir.glob('*.json')))} cases pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
