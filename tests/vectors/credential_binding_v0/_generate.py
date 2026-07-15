"""Regenerate the credential_binding_v0 conformance vectors.

Five cases pin the gateway enforcement contract (Track 1, MCP proxy):

  pos_valid_grant      — valid credential + matching arg commitment → ok
  neg_args_changed     — arg commitment mismatch at runtime → scope_mismatch
  neg_expired          — credential past its expiry window → expired
  neg_wrong_tenant     — tenantId in scope does not match runtime call → scope_mismatch
  neg_no_credential    — vaara/credential absent from _meta → missing_credential

Each fixture is self-contained JSON.  The sibling _check_independent.py
reproduces every verdict with no Vaara import — only hmac, hashlib, json, rfc8785
— so a passing check is a property of the bytes, not of this script.

Run from repo root: python3 tests/vectors/credential_binding_v0/_generate.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vaara.attestation._attest_canonical import make_args_digest
from vaara.credential._grant_emit import emit_grant
from vaara.credential._grant_types import GrantBinding, GrantScope

HERE = Path(__file__).resolve().parent

# ── signing key (HS256, raw bytes) ────────────────────────────────────────────
KEY = b"x" * 32
SECRET_VERSION = "corpus-key-v0"

# ── tool / tenant / args ──────────────────────────────────────────────────────
TOOL = "read_file"
TENANT = "mcp-tenant-01"
ARGS = {"path": "/tmp/vaara/read_file_test"}
ARGS_ALT = {"path": "/etc/passwd"}

# ── attestation identity (stable fake for corpus fixtures) ────────────────────
ATTEST_NONCE = "dGVzdG5vbmNlMTIz"
ATTEST_DIGEST = "sha256:" + hashlib.sha256(b"vaara-corpus-attest-v0").hexdigest()

# ── pinned timestamps ─────────────────────────────────────────────────────────
# IAT_EPOCH = 1779200000  →  2026-05-19T14:13:20Z
# NOW = IAT_EPOCH + 30  (within the 60 s expiry window of pos_valid_grant)
IAT = "2026-05-19T14:13:20Z"
IAT_OLD = "2025-05-19T14:13:20Z"   # one year earlier; expired at NOW
NOW = 1779200030


def _grant_dict(*, iat: str = IAT, tenant: str = TENANT, args: dict = ARGS,
                exp_seconds: int = 60) -> dict:
    args_commitment = make_args_digest(args).projection_digest
    cred = emit_grant(
        scope=GrantScope(tool_name=TOOL, args_commitment=args_commitment, tenant_id=tenant),
        binding=GrantBinding(attestation_digest=ATTEST_DIGEST, attestation_nonce=ATTEST_NONCE),
        iss="vaara-mcp-proxy",
        sub="corpus/test",
        secret_version=SECRET_VERSION,
        alg="HS256",
        signing_material=KEY,
        exp_seconds=exp_seconds,
        iat=iat,
    )
    return cred.to_dict()


def _case(credential, *, runtime_args: dict = ARGS, runtime_tenant: str = TENANT,
          expected_verdict: str) -> dict:
    return {
        "credential": credential,
        "expected_verdict": expected_verdict,
        "known_attestation_digests": [ATTEST_DIGEST],
        "now": NOW,
        "runtime_args": runtime_args,
        "runtime_tenant_id": runtime_tenant,
        "runtime_tool_name": TOOL,
    }


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cases_dir = HERE / "cases"

    _write(cases_dir / "pos_valid_grant.json",
           _case(_grant_dict(), expected_verdict="ok"))

    _write(cases_dir / "neg_args_changed.json",
           _case(_grant_dict(), runtime_args=ARGS_ALT, expected_verdict="scope_mismatch"))

    _write(cases_dir / "neg_expired.json",
           _case(_grant_dict(iat=IAT_OLD), expected_verdict="expired"))

    _write(cases_dir / "neg_wrong_tenant.json",
           _case(_grant_dict(), runtime_tenant="other-tenant", expected_verdict="scope_mismatch"))

    _write(cases_dir / "neg_no_credential.json",
           _case(None, expected_verdict="missing_credential"))

    expected_cases = {}
    for path in sorted(cases_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        expected_cases[path.stem] = {
            "expected_verdict": obj["expected_verdict"],
            "signature_ok": obj["credential"] is not None,
        }
    _write(HERE / "expected.json", {"cases": expected_cases})
    print(f"wrote {len(expected_cases)} credential_binding_v0 vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
