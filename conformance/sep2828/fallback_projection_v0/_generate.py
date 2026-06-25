#!/usr/bin/env python3
"""Generate fallback_projection_v0 conformance vectors.

Writes projections/<name>.json and expected.json.
Run once to regenerate; check output into source control.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTIONS_DIR = HERE / "projections"
PROJECTIONS_DIR.mkdir(exist_ok=True)

VECTORS: dict[str, dict] = {
    # Positive: no authBinding — the explicit unauthenticated fallback profile.
    # Projection = {arguments, toolName} with authBinding absent.
    # Raw _meta example: {"progressToken": "pt-1", "traceId": "tr-001"}
    # (no "authorization_binding" key → authBinding absent from projection).
    "basic_no_auth_binding": {
        "arguments": {"action": "read", "path": "/docs/report.pdf"},
        "toolName": "filesystem_read",
    },
    # Positive: authBinding present (scope + policy carried through).
    "with_auth_binding": {
        "arguments": {"bucket": "prod-data", "object": "reports/q1.csv"},
        "authBinding": {
            "capabilityGrant": "read-only",
            "policyId": "pol-2026-001",
            "scope": "storage",
        },
        "toolName": "gcs_read",
    },
    # Observer-stability pair (item 1).
    # Raw calls have DIFFERENT _meta sidecars:
    #   observer_stable_a raw _meta: {"progressToken": "pt-aaa", "traceId": "tr-001",
    #                                  "authorization_binding": {"policyId": "pol-abc",
    #                                                            "scope": "read"}}
    #   observer_stable_b raw _meta: {"progressToken": "pt-bbb", "x-injected-id": "inj-999",
    #                                  "authorization_binding": {"policyId": "pol-abc",
    #                                                            "scope": "read"}}
    # progressToken and x-injected-id are stripped; authorization_binding becomes authBinding.
    # Checker asserts: observer_stable_a.digest == observer_stable_b.digest.
    "observer_stable_a": {
        "arguments": {"file": "invoice.pdf"},
        "authBinding": {"policyId": "pol-abc", "scope": "read"},
        "toolName": "document_fetch",
    },
    "observer_stable_b": {
        "arguments": {"file": "invoice.pdf"},
        "authBinding": {"policyId": "pol-abc", "scope": "read"},
        "toolName": "document_fetch",
    },
    # Negative: different toolName → different digest (same other fields).
    "neg_different_tool": {
        "arguments": {"file": "invoice.pdf"},
        "authBinding": {"policyId": "pol-abc", "scope": "read"},
        "toolName": "document_fetch_v2",
    },
    # Negative (item 2): same toolName and authBinding, different arguments → different digest.
    # Isolates the arguments contribution to the projection preimage.
    "neg_different_args": {
        "arguments": {"file": "statement.pdf"},
        "authBinding": {"policyId": "pol-abc", "scope": "read"},
        "toolName": "document_fetch",
    },
    # Negative (item 3): same toolName and arguments, different authBinding → different digest.
    # Isolates the authBinding contribution to the projection preimage.
    "neg_different_auth_binding": {
        "arguments": {"file": "invoice.pdf"},
        "authBinding": {"policyId": "pol-xyz", "scope": "write"},
        "toolName": "document_fetch",
    },
}


def jcs(obj: object) -> bytes:
    """RFC 8785 JCS: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def main() -> None:
    expected: dict[str, dict] = {}
    for name, proj in VECTORS.items():
        canonical = jcs(proj)
        digest = sha256(canonical)
        (PROJECTIONS_DIR / f"{name}.json").write_text(
            json.dumps(proj, indent=2, sort_keys=True) + "\n"
        )
        expected[name] = {
            "projectionBytes": canonical.decode("utf-8"),
            "attestationDigest": digest,
        }

    (HERE / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(expected)} vectors to {HERE}")
    a = expected["observer_stable_a"]["attestationDigest"]
    b = expected["observer_stable_b"]["attestationDigest"]
    assert a == b, f"observer stability broken: {a} != {b}"
    n = expected["neg_different_tool"]["attestationDigest"]
    assert a != n, "neg_different_tool should differ from observer_stable"
    d_args = expected["neg_different_args"]["attestationDigest"]
    assert a != d_args, "neg_different_args should differ from observer_stable (args change moves digest)"
    d_auth = expected["neg_different_auth_binding"]["attestationDigest"]
    assert a != d_auth, "neg_different_auth_binding should differ from observer_stable (authBinding change moves digest)"
    assert d_args != d_auth, "neg_different_args and neg_different_auth_binding must not collide"
    print("observer stability: OK")
    print("neg_different_tool divergence: OK")
    print("neg_different_args divergence: OK")
    print("neg_different_auth_binding divergence: OK")


if __name__ == "__main__":
    main()
