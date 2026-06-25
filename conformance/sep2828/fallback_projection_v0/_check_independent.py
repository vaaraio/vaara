#!/usr/bin/env python3
"""Independent checker for fallback_projection_v0 vectors.

Verifies, from the standard library alone (no Vaara import), that:

1. For each projection vector, JCS-encoding the projection object
   (RFC 8785: sorted keys, no whitespace, UTF-8) and hashing it with
   SHA-256 reproduces the committed attestationDigest.

2. observer_stable_a and observer_stable_b yield identical
   attestationDigests — the key portability property: two honest
   observers of the same call carrying different _meta sidecars
   (progress tokens, trace IDs) produce the same digest because
   the fallback projection excludes those transport fields.

3. neg_different_tool yields a distinct attestationDigest from the
   observer_stable pair — different toolName changes the digest.

Run: python3 conformance/sep2828/fallback_projection_v0/_check_independent.py
Exit 0 = all vectors match expected.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def jcs(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text())
    failures = 0

    for name, want in sorted(expected.items()):
        proj = json.loads((HERE / "projections" / f"{name}.json").read_text())
        canonical = jcs(proj)
        got_bytes = canonical.decode("utf-8")
        got_digest = sha256(canonical)

        bytes_ok = got_bytes == want["projectionBytes"]
        digest_ok = got_digest == want["attestationDigest"]
        ok = bytes_ok and digest_ok

        if not ok:
            failures += 1
            if not bytes_ok:
                print(f"[FAIL] {name}: projectionBytes mismatch")
                print(f"  want: {want['projectionBytes']}")
                print(f"  got:  {got_bytes}")
            if not digest_ok:
                print(f"[FAIL] {name}: attestationDigest mismatch")
                print(f"  want: {want['attestationDigest']}")
                print(f"  got:  {got_digest}")
        else:
            print(f"[OK]   {name}: {got_digest}")

    # Observer stability: same projection regardless of excluded _meta sidecar
    a = expected["observer_stable_a"]["attestationDigest"]
    b = expected["observer_stable_b"]["attestationDigest"]
    if a != b:
        print(f"[FAIL] observer stability: a={a} b={b}")
        failures += 1
    else:
        print(f"[OK]   observer stability: both observers → {a}")

    # Negative: different toolName must produce different digest
    neg = expected["neg_different_tool"]["attestationDigest"]
    if neg == a:
        print("[FAIL] neg_different_tool: digest should differ from observer_stable")
        failures += 1
    else:
        print(f"[OK]   neg_different_tool diverges from observer_stable")

    total = len(expected) + 2  # +2 for the cross-vector assertions
    passed = total - failures
    print(f"\n{passed}/{total} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
