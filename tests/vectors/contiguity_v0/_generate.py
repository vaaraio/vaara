#!/usr/bin/env python3
"""Regenerate the contiguity_v0 conformance vectors.

This script imports Vaara to MINT the vectors; the sibling
``_check_independent.py`` imports no Vaara and only RECOMPUTES them. Keeping the
two apart is the point: the producer and the auditor share no code, so a passing
independent check means the gap is provable from the receipts alone.

It mints one capability grant and a stream of five authorization receipts under
a single coverage boundary, each carrying a signed ``completeness`` block
(``boundaryId`` / ``seq`` / ``runningCount``). The same five receipts back two
cases:

* ``complete`` - all five held; seq 0..4 is contiguous and the running count
  matches, so a third party confirms the stream is whole.
* ``dropped``  - the seq-2 receipt is withheld. The four held receipts still
  carry the signed running count that says five exist, so seq 2 is a provable
  gap with no issuer access and no external witness.

The signing key is derived from a fixed scalar so the public key is stable
across runs. ECDSA signatures still vary per run (random k) but always verify,
so a regenerate may show a diff only in the signature hex. The ``dropped`` case
reuses the byte-identical receipts from ``complete`` minus one file, so the gap
is a genuine omission, not a re-mint.

Run: tests/vectors/contiguity_v0/_generate.py
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
BOUNDARY = "vaara-mcp-proxy"
STREAM_LEN = 5
DROPPED_SEQ = 2

CAPS = (
    Capability("amount", "le", "500"),
    Capability("vendor", "in", ("acme", "globex")),
    Capability("destination", "eq", "0xABC"),
)
MINT_ARGS = {"amount": 100, "vendor": "acme", "destination": "0xABC"}
COMMIT = make_args_digest(MINT_ARGS).projection_digest
RUNTIME_ARGS = {"amount": 400, "vendor": "globex", "destination": "0xABC"}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    priv = ec.derive_private_key(_SCALAR, ec.SECP256R1())
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _key = HERE / "keys" / "es256_public.pem"
    _key.parent.mkdir(parents=True, exist_ok=True)
    _key.write_bytes(pub_pem)

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
    verdict = verify_grant(
        grant,
        verifying_material=priv.public_key(),
        runtime_tool_name="pay.send",
        runtime_args=RUNTIME_ARGS,
        runtime_tenant_id="tenant-a",
        known_attestation_digests=frozenset({DIGEST}),
        now=iso8601_to_epoch(IAT) + 5,
    )

    _write(HERE / "grant.json", grant.to_dict())

    # Mint the stream once; both cases share the byte-identical receipts.
    receipts: list[dict] = []
    for seq in range(STREAM_LEN):
        auth = mint_authorization_receipt(
            credential=grant,
            runtime_args=RUNTIME_ARGS,
            verdict=verdict,
            iss="vaara-mcp-proxy",
            sub="tenant-a/upstream",
            secret_version="key-v1",
            alg="ES256",
            signing_material=priv,
            decided_at=DECIDED_AT,
            nonce=f"decision-nonce-{seq}",
            completeness={
                "boundaryId": BOUNDARY,
                "seq": seq,
                "runningCount": seq + 1,
            },
        )
        receipts.append({"record": auth.record.to_dict(), "evidence": auth.evidence})

    for seq, authz in enumerate(receipts):
        name = f"{BOUNDARY}-{seq:04d}-authz.json"
        _write(HERE / "complete" / name, authz)
        if seq != DROPPED_SEQ:
            _write(HERE / "dropped" / name, authz)

    expected = {
        "complete": {
            "all_signatures_ok": True,
            "all_evidence_bindings_resolve": True,
            "contiguity": {
                "ok": True,
                "present": STREAM_LEN,
                "expected": STREAM_LEN,
                "missingSeqs": [],
            },
        },
        "dropped": {
            "all_signatures_ok": True,
            "all_evidence_bindings_resolve": True,
            "contiguity": {
                "ok": False,
                "present": STREAM_LEN - 1,
                "expected": STREAM_LEN,
                "missingSeqs": [DROPPED_SEQ],
            },
        },
    }
    _write(HERE / "expected.json", expected)
    print("wrote contiguity_v0 vectors: complete, dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
