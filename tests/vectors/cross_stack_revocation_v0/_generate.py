#!/usr/bin/env python3
"""Generate the cross_stack_revocation_v0 conformance vectors.

One receipt, several revocation registries, three lenses. Each case asserts
that the receipt-verifier lens, the transparency-log lens, and the
export-digest lens reach the same revoked verdict for the same receipt and
registry, the cross-stack guarantee of ``docs/design/cross-stack-revocation-spec.md``.

ECDSA signatures are randomized; re-running overwrites the cases with fresh
but equivalent vectors. The committed JSON is the vector; the independent
checker verifies whatever is committed. Run from the repo root:
``python tests/vectors/cross_stack_revocation_v0/_generate.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation import InProcessTransparencyLog
from vaara.attestation.receipt import (
    OutcomeDerived,
    RevocationEntry,
    RevocationRegistry,
    emit_receipt,
    make_back_link,
    receipt_leaf_bytes,
)
from vaara.attestation.sep2787 import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HERE = Path(__file__).resolve().parent
DID = "did:web:agents.example.com:billing"
KEYID = DID + "#key-2026"
HS_SECRET = b"\x42" * 32
SCALAR_A = 0x1F2E3D4C5B6A79887766554433221100FFEEDDCCBBAA99887766554433221101

IAT = "2026-05-29T10:00:00Z"
REVOKED_BEFORE = "2026-05-29T09:30:00Z"
REVOKED_AFTER = "2026-05-29T11:00:00Z"


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="charge_card",
        server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"amount": 4200}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:billing",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
        nonce="att-nonce-fixed-0001",
        iat="2026-05-29T09:59:59Z",
    )


def main() -> None:
    key_a = ec.derive_private_key(SCALAR_A, ec.SECP256R1())
    receipt = emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID,
        sub=DID,
        secret_version="v1",
        alg="ES256",
        signing_material=key_a,
        nonce="rcpt-nonce-fixed-0001",
        iat=IAT,
    )
    receipt_dict = receipt.to_dict()

    # Registries from different sources, all consulted by the same predicate.
    reg_key_before = RevocationRegistry(
        [RevocationEntry("key", KEYID, REVOKED_BEFORE)]
    )
    reg_key_after = RevocationRegistry(
        [RevocationEntry("key", KEYID, REVOKED_AFTER)]
    )
    reg_identity_before = RevocationRegistry(
        [RevocationEntry("identity", DID, REVOKED_BEFORE)]
    )
    reg_clean = RevocationRegistry([])

    # One transparency log holding the receipt at a non-trivial index, so the
    # inclusion proof carries real siblings.
    log = InProcessTransparencyLog()
    leaf = receipt_leaf_bytes(receipt)
    log.append(b"filler-0")
    log.append(b"filler-1")
    entry = log.append(leaf)
    log.append(b"filler-3")
    log.append(b"filler-4")
    proof = log.inclusion_proof(entry.log_index)
    root = log.root_hash

    inclusion = {
        "log_index": proof.log_index,
        "tree_size": proof.tree_size,
        "siblings_hex": [s.hex() for s in proof.siblings],
        "root_hex": root.hex(),
    }

    def case(name, registry, keyid, revoked):
        return {
            "name": name,
            "receipt": receipt_dict,
            "keyid": keyid,
            "registry": registry.to_dict(),
            "registry_digest": registry.digest(),
            "inclusion": inclusion,
            "revoked": revoked,
        }

    cases = [
        case("key_revoked_in_time", reg_key_before, KEYID, True),
        case("key_revoked_after", reg_key_after, KEYID, False),
        case("identity_revoked_in_time", reg_identity_before, None, True),
        case("clean", reg_clean, KEYID, False),
    ]

    expected = {}
    for c in cases:
        revoked = c["revoked"]
        expected[c["name"]] = {
            # Receipt-verifier lens.
            "receipt_lens_revoked": revoked,
            # Transparency-log lens: always included; ok only when not revoked.
            "log_lens_included": True,
            "log_lens_ok": not revoked,
            # Export lens: the registry digest the bundle would pin.
            "registry_digest": c["registry_digest"],
        }

    (HERE / "cases.json").write_text(
        json.dumps({"did": DID, "keyid": KEYID, "cases": cases}, indent=2, sort_keys=True)
        + "\n"
    )
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )
    print("wrote cases.json, expected.json")


if __name__ == "__main__":
    main()
