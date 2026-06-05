"""Cross-stack revocation: one rule, consulted by every verification lens.

Covers the `RevocationRegistry` predicate, the receipt-verifier and
transparency-log lenses, the Article-12 export pin, and the
`cross_stack_revocation_v0` conformance vectors (Vaara and a standalone
Vaara-free checker both reproduce the verdicts).

See `docs/design/cross-stack-revocation-spec.md`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)

from vaara.attestation import InProcessTransparencyLog  # noqa: E402
from vaara.attestation.receipt import (  # noqa: E402
    OutcomeDerived,
    RevocationEntry,
    RevocationRegistry,
    check_receipt_revocation,
    emit_receipt,
    make_back_link,
    parse_receipt,
    receipt_leaf_bytes,
    revoked_in_time,
    verify_logged_receipt,
    verify_receipt_identity_live,
)
from vaara.attestation.sep2787 import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

VECTORS = Path(__file__).resolve().parent / "vectors" / "cross_stack_revocation_v0"

DID = "did:web:agents.example.com:billing"
KEYID = DID + "#key-2026"
IAT = "2026-05-29T10:00:00Z"
BEFORE = "2026-05-29T09:30:00Z"
AFTER = "2026-05-29T11:00:00Z"


# ── The shared rule ──────────────────────────────────────────────────────────


def test_revoked_in_time_rule():
    assert revoked_in_time(BEFORE, IAT) is True
    assert revoked_in_time(IAT, IAT) is True  # at issuance binds
    assert revoked_in_time(AFTER, IAT) is False
    # Unparseable instants fail closed.
    assert revoked_in_time("not-a-date", IAT) is True
    assert revoked_in_time(BEFORE, "not-a-date") is True


def test_registry_key_scope_needs_keyid():
    reg = RevocationRegistry([RevocationEntry("key", KEYID, BEFORE)])
    # Without a keyid a key-scope entry cannot match.
    assert reg.status(DID, IAT).revoked is False
    s = reg.status(DID, IAT, keyid=KEYID)
    assert s.revoked is True
    assert s.matched_by == "key"
    assert s.revoked_at == BEFORE


def test_registry_identity_scope_matches_iss():
    reg = RevocationRegistry([RevocationEntry("identity", DID, BEFORE)])
    s = reg.status(DID, IAT)
    assert s.revoked is True and s.matched_by == "identity"
    assert reg.status("did:web:other", IAT).revoked is False


def test_registry_earliest_revocation_and_key_specificity_win():
    reg = RevocationRegistry([
        RevocationEntry("identity", DID, AFTER),
        RevocationEntry("identity", DID, BEFORE),
        RevocationEntry("key", KEYID, BEFORE),
    ])
    s = reg.status(DID, IAT, keyid=KEYID)
    # Earliest revoked-in-time wins; on the BEFORE tie the key-scope entry is
    # reported because it is the more specific statement.
    assert s.revoked is True
    assert s.revoked_at == BEFORE
    assert s.matched_by == "key"


def test_registry_roundtrip_and_digest_stable():
    reg = RevocationRegistry([
        RevocationEntry("identity", DID, BEFORE),
        RevocationEntry("key", KEYID, AFTER),
    ])
    # Construction order does not affect the canonical digest.
    other = RevocationRegistry([
        RevocationEntry("key", KEYID, AFTER),
        RevocationEntry("identity", DID, BEFORE),
    ])
    assert reg.digest() == other.digest()
    assert RevocationRegistry.from_dict(reg.to_dict()).digest() == reg.digest()


def test_from_did_document_ignores_deactivation():
    doc = {
        "id": DID,
        "deactivated": True,
        "verificationMethod": [
            {"id": KEYID, "revoked": BEFORE},
            {"id": DID + "#live"},  # no revoked instant -> no entry
        ],
    }
    reg = RevocationRegistry.from_did_document(doc, DID)
    assert len(reg) == 1
    assert reg.entries[0].scope == "key"
    assert reg.entries[0].subject == KEYID


# ── A real receipt across the lenses ─────────────────────────────────────────


def _receipt(key, *, iat=IAT):
    att = emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=PayloadDerived(tool_calls=(ToolCallBinding(
            name="charge_card",
            server_fingerprint="sha256:" + "1" * 64,
            args=make_args_digest({"amount": 4200}),
        ),)),
        iss="issuer://test", sub="agent:billing", secret_version="v1",
        alg="HS256", signing_material=b"\x42" * 32,
        nonce="att-nonce-fixed-0001", iat="2026-05-29T09:59:59Z",
    )
    return emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(status="executed", completed_at=iat),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce="rcpt-nonce-fixed-0001", iat=iat,
    )


def _ec_doc(key, *, revoked):
    import base64
    nums = key.public_key().public_numbers()

    def _b64u(n):
        return base64.urlsafe_b64encode(n.to_bytes(32, "big")).rstrip(b"=").decode()

    method = {
        "id": KEYID, "type": "JsonWebKey2020", "controller": DID,
        "publicKeyJwk": {"kty": "EC", "crv": "P-256",
                         "x": _b64u(nums.x), "y": _b64u(nums.y)},
    }
    if revoked is not None:
        method["revoked"] = revoked
    return {"id": DID, "verificationMethod": [method]}


def test_receipt_and_log_lenses_agree_with_identity_lens():
    key = ec.generate_private_key(ec.SECP256R1())
    receipt = _receipt(key)
    doc = _ec_doc(key, revoked=BEFORE)
    registry = RevocationRegistry.from_did_document(doc, DID)

    # Identity lens (level 3) derives revocation from the document.
    raw = json.dumps(doc).encode()
    live = verify_receipt_identity_live(receipt, fetcher=lambda url: raw)
    assert live.revoked is True and live.trusted is False

    # Receipt-verifier lens, registry built from the same document, agrees.
    assert check_receipt_revocation(receipt, registry, keyid=KEYID).revoked is True

    # Transparency-log lens: included but not ok, because it is revoked.
    log = InProcessTransparencyLog()
    log.append(b"a")
    entry = log.append(receipt_leaf_bytes(receipt))
    verdict = verify_logged_receipt(
        receipt=receipt, proof=log.inclusion_proof(entry.log_index),
        expected_root=log.root_hash, registry=registry, keyid=KEYID,
    )
    assert verdict.included is True
    assert verdict.revocation.revoked is True
    assert verdict.ok is False


def test_log_lens_ok_when_not_revoked():
    key = ec.generate_private_key(ec.SECP256R1())
    receipt = _receipt(key)
    registry = RevocationRegistry([RevocationEntry("key", KEYID, AFTER)])
    log = InProcessTransparencyLog()
    entry = log.append(receipt_leaf_bytes(receipt))
    verdict = verify_logged_receipt(
        receipt=receipt, proof=log.inclusion_proof(entry.log_index),
        expected_root=log.root_hash, registry=registry, keyid=KEYID,
    )
    assert verdict.included is True and verdict.ok is True


def test_log_lens_not_included_for_wrong_root():
    key = ec.generate_private_key(ec.SECP256R1())
    receipt = _receipt(key)
    log = InProcessTransparencyLog()
    entry = log.append(receipt_leaf_bytes(receipt))
    verdict = verify_logged_receipt(
        receipt=receipt, proof=log.inclusion_proof(entry.log_index),
        expected_root=b"\x00" * 32,
        registry=RevocationRegistry([]), keyid=KEYID,
    )
    assert verdict.included is False and verdict.ok is False


# ── Article-12 export lens ───────────────────────────────────────────────────


def _trail(n=3):
    from vaara.audit.trail import AuditTrail
    from vaara.taxonomy.actions import ActionRequest, create_default_registry

    reg = create_default_registry()
    tx = reg.get("tx.transfer")
    trail = AuditTrail()
    for i in range(n):
        trail.record_action_requested(ActionRequest(
            agent_id=f"agent-{i}", tool_name="send_funds", action_type=tx,
            parameters={"to": f"0xabc{i}", "amount": 10 * i},
        ))
    return trail


def test_export_without_revocation_is_unchanged(tmp_path):
    from vaara.audit.export import export_signed
    from vaara.audit.signer import Ed25519Signer

    signer = Ed25519Signer(Ed25519PrivateKey.generate())
    out = tmp_path / "t.zip"
    res = export_signed(_trail(), out, signer=signer)
    assert "revocation" not in res.manifest
    with zipfile.ZipFile(out) as zf:
        assert "revocation.json" not in zf.namelist()


def test_export_pins_revocation_and_standalone_verifier_accepts(tmp_path):
    from vaara.audit.export import export_signed
    from vaara.audit.signer import Ed25519Signer

    signer = Ed25519Signer(Ed25519PrivateKey.generate())
    registry = RevocationRegistry([RevocationEntry("key", KEYID, BEFORE)])
    out = tmp_path / "t.zip"
    res = export_signed(_trail(), out, signer=signer, revocation=registry)

    assert res.manifest["revocation"]["entry_count"] == 1
    assert res.manifest["revocation"]["registry_sha256"] == registry.digest()
    with zipfile.ZipFile(out) as zf:
        assert "revocation.json" in zf.namelist()
        rebuilt = RevocationRegistry.from_dict(json.loads(zf.read("revocation.json")))
        assert rebuilt.digest() == registry.digest()

    proc = subprocess.run(
        [sys.executable, "scripts/verify_vaara_trail.py", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_standalone_verifier_rejects_tampered_revocation(tmp_path):
    from vaara.audit.export import export_signed
    from vaara.audit.signer import Ed25519Signer

    signer = Ed25519Signer(Ed25519PrivateKey.generate())
    registry = RevocationRegistry([RevocationEntry("key", KEYID, BEFORE)])
    out = tmp_path / "t.zip"
    export_signed(_trail(), out, signer=signer, revocation=registry)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(out) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "revocation.json":
                data = json.dumps(
                    RevocationRegistry([RevocationEntry("key", KEYID, AFTER)]).to_dict()
                ).encode()
            zout.writestr(name, data)

    proc = subprocess.run(
        [sys.executable, "scripts/verify_vaara_trail.py", str(tampered)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "revocation.json digest does not match" in (proc.stdout + proc.stderr)


# ── Conformance vectors ──────────────────────────────────────────────────────


def _cases():
    return json.loads((VECTORS / "cases.json").read_text())["cases"]


def _expected():
    return json.loads((VECTORS / "expected.json").read_text())


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_vaara_reproduces_vector_verdict(case):
    from vaara.attestation.transparency_log import InclusionProof

    want = _expected()[case["name"]]
    receipt = parse_receipt(case["receipt"])
    registry = RevocationRegistry.from_dict(case["registry"])
    keyid = case.get("keyid")

    rstat = check_receipt_revocation(receipt, registry, keyid=keyid)
    assert rstat.revoked is want["receipt_lens_revoked"]
    assert registry.digest() == want["registry_digest"]

    inc = case["inclusion"]
    proof = InclusionProof(
        log_index=inc["log_index"], tree_size=inc["tree_size"],
        siblings=tuple(bytes.fromhex(h) for h in inc["siblings_hex"]),
    )
    verdict = verify_logged_receipt(
        receipt=receipt, proof=proof,
        expected_root=bytes.fromhex(inc["root_hex"]),
        registry=registry, keyid=keyid,
    )
    assert verdict.included is want["log_lens_included"]
    assert verdict.ok is want["log_lens_ok"]


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
