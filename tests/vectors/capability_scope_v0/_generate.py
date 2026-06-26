"""Regenerate the capability_scope_v0 conformance vectors.

Five cases pin the capability-mode enforcement contract (Track 1, Phase C):

pos_valid_grant       — all caps satisfied (amount le 500, vendor in {acme,globex}) → ok
neg_amount_exceeded   — amount 600 > bound 500 (le violated) → capability_exceeded
neg_vendor_not_in_set — vendor not in allowed set → capability_exceeded
neg_uncovered_arg     — runtime arg with no matching capability → capability_uncovered
neg_missing_credential — no credential in _meta → missing_credential

Each fixture is self-contained JSON. _check_independent.py reproduces every
verdict with no Vaara import — only hmac, hashlib, decimal, json, rfc8785
— so a passing check is a property of the bytes, not of this script.

Run: python3 tests/vectors/capability_scope_v0/_generate.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from vaara.credential import (
    Capability,
    GrantBinding,
    GrantScope,
    emit_grant,
)

HERE = Path(__file__).resolve().parent

KEY = b"x" * 32
SECRET_VERSION = "corpus-key-v0"

TOOL = "transfer_funds"
TENANT = "mcp-tenant-01"
ATT_DIGEST = "sha256:" + "a" * 64
ATT_NONCE = "nonce-cap-v0-01"

CAPS = (
    Capability("amount", "le", "500"),
    Capability("vendor", "in", ("acme", "globex")),
)

NOW = time.time()


def _grant_dict(**kwargs) -> dict:
    scope = GrantScope(
        tool_name=TOOL,
        args_commitment="",
        tenant_id=TENANT,
    )
    binding = GrantBinding(
        attestation_digest=ATT_DIGEST,
        attestation_nonce=ATT_NONCE,
    )
    cred = emit_grant(
        scope=scope,
        binding=binding,
        iss="vaara.io",
        sub=TENANT,
        secret_version=SECRET_VERSION,
        alg="HS256",
        signing_material=KEY,
        exp_seconds=300,
        capabilities=CAPS,
        **kwargs,
    )
    return cred.to_dict()


def _case(
    credential: dict | None,
    runtime_args: dict,
    runtime_tenant: str = TENANT,
    expected_verdict: str = "ok",
) -> dict:
    return {
        "credential": credential,
        "expected_verdict": expected_verdict,
        "known_attestation_digests": [ATT_DIGEST],
        "now": NOW,
        "runtime_args": runtime_args,
        "runtime_tenant_id": runtime_tenant,
        "runtime_tool_name": TOOL,
    }


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    cases_dir = HERE / "cases"
    valid_args = {"amount": 400, "vendor": "acme"}

    _write(
        cases_dir / "pos_valid_grant.json",
        _case(_grant_dict(), valid_args, expected_verdict="ok"),
    )

    _write(
        cases_dir / "neg_amount_exceeded.json",
        _case(
            _grant_dict(),
            {"amount": 600, "vendor": "acme"},
            expected_verdict="capability_exceeded",
        ),
    )

    _write(
        cases_dir / "neg_vendor_not_in_set.json",
        _case(
            _grant_dict(),
            {"amount": 400, "vendor": "evilcorp"},
            expected_verdict="capability_exceeded",
        ),
    )

    _write(
        cases_dir / "neg_uncovered_arg.json",
        _case(
            _grant_dict(),
            {"amount": 400, "vendor": "acme", "memo": "hidden"},
            expected_verdict="capability_uncovered",
        ),
    )

    _write(
        cases_dir / "neg_missing_credential.json",
        _case(None, valid_args, expected_verdict="missing_credential"),
    )

    expected_cases = {}
    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        expected_cases[path.stem] = {"expected_verdict": case["expected_verdict"]}
    _write(HERE / "expected.json", {"cases": expected_cases})

    print(f"wrote {len(expected_cases)} cases + expected.json")


if __name__ == "__main__":
    main()
