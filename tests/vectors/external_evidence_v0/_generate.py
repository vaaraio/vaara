#!/usr/bin/env python3
"""Regenerate the external_evidence_v0 conformance vectors.

Generic "external execution evidence" binding profile: a verifier that carries an
``external_execution_evidence`` slot (linked_call_id / evidence_hash /
evidence_type, the shape used by agentrust trace-spec #34 and cMCP #301) resolves
that slot against a ``vaara.receipt/v1`` authorization receipt as the recomputable
producer. The slot's ``evidenceHash`` content-addresses the receipt's evidence
record; the slot's ``linkedCallId`` matches the call the receipt names; the slot's
``evidenceType`` is the receipt's evidence schema.

Imports Vaara to MINT the vaara.receipt/v1 side; the sibling
``_check_independent.py`` imports no Vaara and only RECOMPUTES. See README.md for
the full mapping.

The differentiator is the half the slot alone cannot give. A per-call
``evidence_hash`` proves a given call's evidence exists and recomputes; it cannot
prove no call's evidence was dropped from the trace. The held receipts carry a
signed per-boundary ``completeness`` block, so the ``dropped`` case (seq 1
withheld, slot and receipt both) is still a provable gap from the running count,
while every held slot resolves cleanly.

Run: tests/vectors/external_evidence_v0/_generate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation._sep2787_canonical import iso8601_to_epoch, make_args_digest
from vaara.credential import Capability, GrantBinding, GrantScope, emit_grant, verify_grant
from vaara.credential._authorization_receipt import AUTHORIZATION_SCHEMA, mint_authorization_receipt

HERE = Path(__file__).resolve().parent

_SCALAR = 0x42C0FFEE_1337_0BADBEEF_CAFEBABE_0DDF00D_1234567890ABCDEF_42424242  # test key
DIGEST = "sha256:" + "ab" * 32
NONCE = "att-nonce-extev"
IAT = "2026-06-22T09:00:00Z"
DECIDED_AT = "2026-06-22T09:00:05Z"
BOUNDARY = "trace:agent-run-9f2c"  # the external trace = coverage boundary
SERVER_FINGERPRINT = "manifest:sha256:" + "cd" * 32
STREAM_LEN = 3
DROPPED_SEQ = 1

CAPS = (
    Capability("region", "in", ("EU", "US")),
    Capability("items", "le", "10"),
    Capability("orderRef", "eq", "cart-9f2c"),
)
MINT_ARGS = {"region": "EU", "items": 3, "orderRef": "cart-9f2c"}
COMMIT = make_args_digest(MINT_ARGS).projection_digest
RUNTIME_ARGS = {"region": "EU", "items": 3, "orderRef": "cart-9f2c"}


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _call_id(seq: int) -> str:
    return f"call-9f2c-{seq:04d}"


def _slot(call_id: str, evidence_hash: str) -> dict:
    """The external_execution_evidence slot a trace-spec #34 / cMCP #301 verifier
    holds. Representative shape; the binding rests on the content-addressing
    discipline (evidenceHash over the named artifact), not these field names."""
    return {
        "schema": "exec.evidence/v0",
        "linkedCallId": call_id,
        "evidenceHash": evidence_hash,
        "evidenceType": AUTHORIZATION_SCHEMA,
    }


def main() -> int:
    priv = ec.derive_private_key(_SCALAR, ec.SECP256R1())
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    (HERE / "keys").mkdir(parents=True, exist_ok=True)
    (HERE / "keys" / "es256_public.pem").write_bytes(pub_pem)

    grant = emit_grant(
        scope=GrantScope(
            tool_name="fulfillment.dispatch", args_commitment=COMMIT, tenant_id="tenant-a"
        ),
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
        runtime_tool_name="fulfillment.dispatch",
        runtime_args=RUNTIME_ARGS,
        runtime_tenant_id="tenant-a",
        known_attestation_digests=frozenset({DIGEST}),
        now=iso8601_to_epoch(IAT) + 5,
    )
    _write(HERE / "grant.json", grant.to_dict())

    coverage = {
        "boundary": BOUNDARY,
        "serverFingerprint": SERVER_FINGERPRINT,
        "scope": "only calls routed through this agent run are observed",
    }
    held = []  # mint once; both cases share the byte-identical held files
    for seq in range(STREAM_LEN):
        call_id = _call_id(seq)
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
            ref=f"mcp:call/{call_id}",
            coverage=coverage,
            completeness={"boundaryId": BOUNDARY, "seq": seq, "runningCount": seq + 1},
        )
        evidence_hash = _sha256_hex(_jcs(auth.evidence))
        held.append(
            {
                "record": auth.record.to_dict(),
                "evidence": auth.evidence,
                "slot": _slot(call_id, evidence_hash),
            }
        )

    slug = BOUNDARY.replace(":", "-").replace("/", "-")
    for seq, item in enumerate(held):
        name = f"{slug}-{seq:04d}-authz.json"
        _write(HERE / "complete" / name, item)
        if seq != DROPPED_SEQ:
            _write(HERE / "dropped" / name, item)

    base = {
        "all_signatures_ok": True,
        "all_evidence_bindings_resolve": True,
        "all_slots_resolve": True,
    }
    expected = {
        "complete": {**base, "contiguity": {
            "ok": True, "present": STREAM_LEN, "expected": STREAM_LEN, "missingSeqs": []}},
        "dropped": {**base, "contiguity": {
            "ok": False, "present": STREAM_LEN - 1, "expected": STREAM_LEN,
            "missingSeqs": [DROPPED_SEQ]}},
    }
    _write(HERE / "expected.json", expected)
    print("wrote external_evidence_v0 vectors: complete, dropped, grant.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
