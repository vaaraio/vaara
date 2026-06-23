#!/usr/bin/env python3
"""Regenerate the class_gate_v0 conformance vectors.

Enforcement-time consumption of a sealed worst-case class. A chain recipient gates
its own next unattended action on the
boundary's sealed ``maxClass`` (the v1.7.0 seal): it holds a policy set of action
classes it will proceed under (``permitted_classes``) and permits iff the sealed
class is a member of that set, failing closed when no class is sealed. The gate is
**membership**, not an ordering over class labels (SPEC 5.3 computes no ordering).

The load-bearing case is ``permit_gap_bounded``: an interior receipt is withheld,
so the boundary has a provable gap, yet the gate still permits because the seal
bounds the missing record's worst case at the permitted class. The gate consumes
the committed bound; it does not re-derive the chain or query a log.

Imports Vaara to MINT the receipts and the signed seal; the sibling
``_check_independent.py`` imports no Vaara and only RECOMPUTES (signatures,
contiguity, the sealed class, and the gate decision). See README.md.

Run: tests/vectors/class_gate_v0/_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation._sep2787_canonical import iso8601_to_epoch, make_args_digest
from vaara.credential import Capability, GrantBinding, GrantScope, emit_grant, verify_grant
from vaara.credential._authorization_receipt import mint_authorization_receipt

HERE = Path(__file__).resolve().parent

_SCALAR = 0x6A11C1A55_C0FFEE_F00D_BADC0DE_1234_5678_9ABC_DEF0_2468ACE0_13579BDF  # test key
DIGEST = "sha256:" + "ef" * 32
IAT = "2026-06-22T15:00:00Z"
DECIDED_AT = "2026-06-22T15:00:05Z"
BOUNDARY = "chain:agent-handoff-7a1d"  # the chain a recipient gates its next step under
STREAM_LEN = 3
DROPPED_SEQ = 1
# The consumer's policy set, baked into the vector and re-applied by the checker.
PERMITTED_CLASSES = ["data.read", "data.write"]

CAPS = (
    Capability("region", "in", ("EU", "US")),
    Capability("items", "le", "10"),
    Capability("orderRef", "eq", "cart-7a1d"),
)
RUNTIME_ARGS = {"region": "EU", "items": 2, "orderRef": "cart-7a1d"}
COMMIT = make_args_digest(RUNTIME_ARGS).projection_digest

# case -> (held seqs, sealed maxClass | None)
_CASES = {
    "permit": ([0, 1, 2], "data.read"),
    "permit_gap_bounded": ([0, 2], "data.read"),
    "deny_class": ([0, 1, 2], "tx.transfer"),
    "deny_unbounded": ([0, 2], None),
}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate(max_class):
    """The decision the checker re-derives; spelled here only to build expected."""
    if max_class is None:
        return False, "unbounded_no_sealed_class"
    permit = max_class in PERMITTED_CLASSES
    return permit, "permitted" if permit else "class_not_permitted"


def main() -> int:
    priv = ec.derive_private_key(_SCALAR, ec.SECP256R1())
    (HERE / "keys").mkdir(parents=True, exist_ok=True)
    (HERE / "keys" / "es256_public.pem").write_bytes(
        priv.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    grant = emit_grant(
        scope=GrantScope(
            tool_name="handoff.dispatch", args_commitment=COMMIT, tenant_id="tenant-a"
        ),
        binding=GrantBinding(attestation_digest=DIGEST, attestation_nonce="att-nonce-classgate"),
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
        runtime_tool_name="handoff.dispatch",
        runtime_args=RUNTIME_ARGS,
        runtime_tenant_id="tenant-a",
        known_attestation_digests=frozenset({DIGEST}),
        now=iso8601_to_epoch(IAT) + 5,
    )
    _write(HERE / "grant.json", grant.to_dict())

    coverage = {"boundary": BOUNDARY, "scope": "only steps in this handoff chain are observed"}

    def _mint(completeness: dict, ref: str, nonce: str) -> dict:
        auth = mint_authorization_receipt(
            credential=grant, runtime_args=RUNTIME_ARGS, verdict=verdict,
            iss="vaara-mcp-proxy", sub="tenant-a/upstream", secret_version="key-v1",
            alg="ES256", signing_material=priv, decided_at=DECIDED_AT, nonce=nonce,
            ref=ref, coverage=coverage, completeness=completeness,
        )
        return {"record": auth.record.to_dict(), "evidence": auth.evidence}

    # Per-call receipts, minted once and shared byte-identically across cases.
    seq_items = {
        seq: _mint(
            {"boundaryId": BOUNDARY, "seq": seq, "runningCount": seq + 1},
            ref=f"mcp:call/call-7a1d-{seq:04d}", nonce=f"decision-nonce-{seq}",
        )
        for seq in range(STREAM_LEN)
    }

    # One signed terminal seal per distinct maxClass: the bound rides under signature.
    def _seal(max_class) -> dict:
        completeness = {"boundaryId": BOUNDARY, "sealed": True, "total": STREAM_LEN}
        if max_class is not None:
            completeness["maxClass"] = max_class
        suffix = max_class.replace(".", "-") if max_class else "none"
        return _mint(completeness, ref=f"mcp:seal/{BOUNDARY}", nonce=f"seal-nonce-{suffix}")

    slug = BOUNDARY.replace(":", "-").replace("/", "-")
    seal_name = f"{slug}-9999-seal-authz.json"
    expected: dict = {}
    for case, (held_seqs, max_class) in _CASES.items():
        for seq in held_seqs:
            _write(HERE / case / f"{slug}-{seq:04d}-authz.json", seq_items[seq])
        _write(HERE / case / seal_name, _seal(max_class))
        present = len(held_seqs)
        missing = sorted(set(range(STREAM_LEN)) - set(held_seqs))
        permit, reason = _gate(max_class)
        expected[case] = {
            "all_signatures_ok": True,
            "all_evidence_bound": True,
            "permit": permit,
            "reason": reason,
            "worstCaseClass": max_class,
            "contiguity": {
                "ok": not missing, "present": present,
                "expected": STREAM_LEN, "missingSeqs": missing,
            },
        }

    # deny_relabeled: the adversary Mayur021 named. Mint an honest tx.transfer
    # seal (correctly denied), then relabel maxClass DOWN to a permitted class
    # WITHOUT re-signing. The record signature still verifies (it never covered
    # the evidence block); the evidence no longer recomputes to the signed
    # evidenceRef.digest. A binding-checking gate drops the unbound seal and fails
    # closed; a gate that reads maxClass raw is tricked into permit.
    relabeled = "deny_relabeled"
    for seq in range(STREAM_LEN):
        _write(HERE / relabeled / f"{slug}-{seq:04d}-authz.json", seq_items[seq])
    tampered = _seal("tx.transfer")
    tampered["evidence"]["completeness"]["maxClass"] = "data.read"
    _write(HERE / relabeled / seal_name, tampered)
    expected[relabeled] = {
        "all_signatures_ok": True,
        "all_evidence_bound": False,
        "permit": False,
        "reason": "unbounded_no_sealed_class",
        "worstCaseClass": None,
        "contiguity": {
            "ok": True, "present": STREAM_LEN,
            "expected": STREAM_LEN, "missingSeqs": [],
        },
    }

    _write(HERE / "expected.json", {"permittedClasses": PERMITTED_CLASSES, "cases": expected})
    print("wrote class_gate_v0 vectors: " + ", ".join([*_CASES, relabeled]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
