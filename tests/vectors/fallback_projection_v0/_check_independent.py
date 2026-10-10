#!/usr/bin/env python3
"""Independent checker for fallback_projection_v0.

The SEP-2828 fallback projection, version
tools_call_params_plus_meta_authorization_binding_v1: when no SEP-2787
attestation exists for a call, backLink.attestationDigest is the digest of

    {"projection": <version>, "name": <params.name>,
     "arguments": <params.arguments>,
     "authorizationBinding": <params._meta.authorization_binding>}

canonicalized with RFC 8785 (JCS). The binding block is required and must be
an object with a non-empty string nonce; name and arguments are required.
Every other _meta member is left out, so two observers of one call agree.

Imports the standard library and rfc8785 only, never Vaara. Checks per case:
the projection bytes and digest match expected.json, or the case is refused
where expected.json says malformed. Then: provider_view and gateway_view give
one digest; different_tool, different_arguments and replayed_binding each give
another.

Run: tests/vectors/fallback_projection_v0/_check_independent.py
Exit 0 means every verdict matched.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve().parent
V1 = "tools_call_params_plus_meta_authorization_binding_v1"


class Malformed(ValueError):
    pass


def projection(env: dict, version: str) -> dict:
    if version != V1:
        raise Malformed(f"unsupported projection {version!r}")
    meta = env.get("_meta")
    binding = meta.get("authorization_binding") if isinstance(meta, dict) else None
    if not isinstance(binding, dict):
        raise Malformed("authorization_binding absent or not an object")
    if not isinstance(binding.get("nonce"), str) or not binding["nonce"]:
        raise Malformed("authorization_binding.nonce absent")
    if "name" not in env or "arguments" not in env:
        raise Malformed("name or arguments missing")
    return {"projection": version, "name": env["name"], "arguments": env["arguments"],
            "authorizationBinding": binding}


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    digests: dict[str, str] = {}
    failures = 0
    for name, want in sorted(expected.items()):
        env = json.loads((HERE / "envelopes" / f"{name}.json").read_text(encoding="utf-8"))
        try:
            canon = rfc8785.dumps(projection(env, want["version"]))
        except Malformed as exc:
            ok = want.get("malformed") is True
            print(f"[{'OK' if ok else 'FAIL'}] {name}: refused ({exc})")
            failures += not ok
            continue
        digest = "sha256:" + hashlib.sha256(canon).hexdigest()
        digests[name] = digest
        ok = (not want.get("malformed") and canon.decode("utf-8") == want["projectionBytes"]
              and digest == want["attestationDigest"])
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {digest}")
        failures += not ok

    base = digests.get("provider_view")
    ok = base is not None and digests.get("gateway_view") == base
    print(f"[{'OK' if ok else 'FAIL'}] provider and gateway views agree")
    failures += not ok
    for name in ("different_tool", "different_arguments", "replayed_binding"):
        ok = name in digests and digests[name] != base
        print(f"[{'OK' if ok else 'FAIL'}] {name} diverges from provider_view")
        failures += not ok

    print(f"\n{'all verdicts matched expected' if not failures else f'{failures} mismatch(es)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
