#!/usr/bin/env python3
"""Regenerate the authorization_v0 conformance vectors.

This script imports Vaara to MINT the vectors; the sibling
``_check_independent.py`` imports no Vaara and only RECOMPUTES them. Keeping the
two apart is the point: the producer and the auditor share no code, so a passing
independent check means the receipt stands on its own.

It mints two cases against one capability grant (amount <= 500, vendor in
{acme, globex}, destination == 0xABC):

* ``allow`` - runtime args within bounds; the gateway permits, decision ``allow``.
* ``deny``  - amount over the cap; the gateway refuses, decision ``block`` with
  reason ``capability_exceeded``. This is the headline artifact: a signed,
  portable proof of a refused action.

The signing key is derived from a fixed scalar so the public key is stable
across runs. ECDSA signatures still vary per run (random k) but always verify,
so a regenerate may show a diff only in the signature hex.

Run: tests/vectors/authorization_v0/_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation._attest_canonical import iso8601_to_epoch, make_args_digest
from vaara.credential import (
    Capability,
    GrantBinding,
    GrantScope,
    emit_grant,
    verify_grant,
)
from vaara.credential._authorization_receipt import mint_authorization_receipt

HERE = Path(__file__).resolve().parent

# Fixed scalar -> stable public key across regenerations. Test-only key.
_SCALAR = 0x42C0FFEE_1337_0BADBEEF_CAFEBABE_0DDF00D_1234567890ABCDEF_42424242
DIGEST = "sha256:" + "ab" * 32
NONCE = "att-nonce-xyz"
IAT = "2026-06-18T12:00:00Z"
DECIDED_AT = "2026-06-18T12:00:05Z"

CAPS = (
    Capability("amount", "le", "500"),
    Capability("vendor", "in", ("acme", "globex")),
    Capability("destination", "eq", "0xABC"),
)
MINT_ARGS = {"amount": 100, "vendor": "acme", "destination": "0xABC"}
COMMIT = make_args_digest(MINT_ARGS).projection_digest

CASES = {
    "allow": {"amount": 400, "vendor": "globex", "destination": "0xABC"},
    "deny": {"amount": 600, "vendor": "acme", "destination": "0xABC"},
}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    priv = ec.derive_private_key(_SCALAR, ec.SECP256R1())
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _write_bytes = (HERE / "keys" / "es256_public.pem")
    _write_bytes.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes.write_bytes(pub_pem)

    grant = emit_grant(
        scope=GrantScope(tool_name="pay.send", args_commitment=COMMIT, tenant_id="tenant-a"),
        binding=GrantBinding(attestation_digest=DIGEST, attestation_nonce=NONCE),
        iss="vaara-mcp-proxy",
        sub="tenant-a/upstream",
        secret_version="key-v1",
        alg="ES256",
        signing_material=priv,
        exp_seconds=60,
        iat=IAT,
        nonce="grant-nonce-1",
        capabilities=CAPS,
    )

    for case, runtime_args in CASES.items():
        verdict = verify_grant(
            grant,
            verifying_material=priv.public_key(),
            runtime_tool_name="pay.send",
            runtime_args=runtime_args,
            runtime_tenant_id="tenant-a",
            known_attestation_digests=frozenset({DIGEST}),
            now=iso8601_to_epoch(IAT) + 5,
        )
        auth = mint_authorization_receipt(
            credential=grant,
            runtime_args=runtime_args,
            verdict=verdict,
            iss="vaara-mcp-proxy",
            sub="tenant-a/upstream",
            secret_version="key-v1",
            alg="ES256",
            signing_material=priv,
            decided_at=DECIDED_AT,
            nonce=f"decision-nonce-{case}",
        )
        _write(HERE / case / "grant.json", grant.to_dict())
        _write(HERE / case / "args.json", runtime_args)
        _write(HERE / case / "evidence.json", auth.evidence)
        _write(HERE / case / "receipt.json", auth.record.to_dict())

    expected = {
        case: {
            "grant_fingerprint_recomputes": True,
            "args_commitment_recomputes": True,
            "verdict_recomputes": True,
            "evidence_binding_resolves": True,
            "receipt_signature_ok": True,
        }
        for case in CASES
    }
    _write(HERE / "expected.json", expected)
    print("wrote authorization_v0 vectors:", ", ".join(sorted(CASES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
