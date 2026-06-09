#!/usr/bin/env python3
"""Generate the cross_org_handoff_v0 conformance vectors.

Emits cross-org handoff packages that exercise ``verify_handoff``: one vendor's
signed record, the archived DID document listing its now-retired key, key
history, revocations, and an optional eIDAS RFC 3161 anchor, all pinned by
content digest, checked by another organisation's regulator.

The cases cover the clean verifiable and corroborated tiers, an anchor bound to
a *different* record (rejected, never a silent corroboration), tampered DID
document and manifest (integrity drift), a self-supplied holder custody
attestation (good and corrupted), a non-canonical key-history override (the
model-digest trap), a record signed after key retirement, the producer-identity
pin, and strict mode.

ECDSA signatures are randomized, so re-running overwrites the cases with fresh
but equivalent vectors. ``expected.json`` is produced by Vaara; the committed
``_check_independent.py`` reproduces every verdict without importing Vaara. Run
from the repo root:
``python tests/vectors/cross_org_handoff_v0/_generate.py``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation.receipt import (
    OutcomeDerived,
    build_handoff,
    emit_receipt,
    make_back_link,
    sign_manifest,
    verify_handoff,
)
from vaara.attestation.sep2787 import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)
from vaara.attestation._sep2787_canonical import canonical_json

HERE = Path(__file__).resolve().parent
DID = "did:web:vendor-a.example:billing"
KEYID = DID + "#key-2026"
HOLDER = "did:web:customer-b.example"
HOLDER_KEYID = HOLDER + "#k1"
IAT = "2026-05-29T10:00:00Z"
ACTIVATED = "2026-01-01T00:00:00Z"
RETIRED = "2028-01-01T00:00:00Z"
ANCHOR_OK = "2026-05-29T10:05:00Z"
SCALAR = 0x1F2E3D4C5B6A79887766554433221100FFEEDDCCBBAA99887766554433221101
HOLDER_SCALAR = 0x0A1B2C3D4E5F60718293A4B5C6D7E8F9001122334455667788990AABBCCDDEEFF

COMPARE = (
    "integrity_ok", "producer_identity_basis", "bound", "keyid",
    "window_recorded", "revocation_source_present", "revoked", "anchor_present",
    "anchor_verified", "anchor_binds", "verifiable", "corroborated", "custody",
    "holder_keyid", "strict", "ok",
)
EMPTY_REVS = {"version": 1, "entries": []}


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _jwk(pk: ec.EllipticCurvePublicKey) -> dict:
    n = pk.public_numbers()
    return {"kty": "EC", "crv": "P-256",
            "x": _b64u(n.x.to_bytes(32, "big")), "y": _b64u(n.y.to_bytes(32, "big"))}


def _method(keyid: str, jwk: dict, **markers: str) -> dict:
    m = {"id": keyid, "type": "JsonWebKey2020", "controller": DID, "publicKeyJwk": jwk}
    m.update(markers)
    return m


def _doc(*methods: dict) -> dict:
    return {"id": DID, "verificationMethod": list(methods)}


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="charge_card", server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"amount": 4200}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=payload, iss="issuer://test", sub="agent:billing",
        secret_version="v1", alg="HS256", signing_material=b"\x42" * 32,
        nonce="att-nonce-fixed-0001", iat="2026-05-29T09:59:59Z",
    )


def _receipt(key: ec.EllipticCurvePrivateKey) -> dict:
    return emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce="rcpt-nonce-fixed-0001", iat=IAT,
    ).to_dict()


def _anchor(record: dict) -> dict:
    rec_hex = hashlib.sha256(canonical_json(record)).hexdigest()
    return {"chain_position": 0, "chain_head_hash": rec_hex, "backend": "eidas-tsa",
            "tsa_url": "https://tsa.example", "hash_algorithm": "sha256",
            "token_b64": "AA==", "anchored_time": ANCHOR_OK}


def _cover() -> dict:
    return {"system": "BillingBot", "action": "charge_card", "provider": "Vendor A",
            "deployer": "Customer B",
            "obligation": "deployer log retention, AI Act Art 26(6)"}


def _refresh_manifest_digest(pkg: dict) -> None:
    pkg["manifest_digest"] = (
        "sha256:" + hashlib.sha256(canonical_json(pkg["manifest"])).hexdigest()
    )


def main() -> None:
    key = ec.derive_private_key(SCALAR, ec.SECP256R1())
    holder_key = ec.derive_private_key(HOLDER_SCALAR, ec.SECP256R1())
    jwk = _jwk(key.public_key())
    holder_jwk = _jwk(holder_key.public_key())
    rec = _receipt(key)
    in_window = _doc(_method(KEYID, jwk, validFrom=ACTIVATED, validUntil=RETIRED))

    def base(**kw) -> dict:
        # Deep-copy every input so no two packages alias a mutable dict: a later
        # in-place tamper on one case must not corrupt another (build_handoff
        # stores the objects it is given by reference).
        kw.setdefault("record", rec)
        kw.setdefault("did_document", in_window)
        kw.setdefault("holder", HOLDER)
        kw.setdefault("cover", _cover())
        kw = {k: (copy.deepcopy(v) if isinstance(v, (dict, list)) else v)
              for k, v in kw.items()}
        return build_handoff(**kw)

    # Holder-attested: build once for the manifest, sign it, rebuild with it.
    seed = base(anchor=_anchor(rec), revocations=EMPTY_REVS)
    att_block = sign_manifest(seed["manifest"], alg="ES256", keyid=HOLDER_KEYID,
                              signing_material=holder_key, verifying_jwk=holder_jwk)
    holder_attested = base(anchor=_anchor(rec), revocations=EMPTY_REVS,
                           holder_attestation=att_block)
    holder_failed = copy.deepcopy(holder_attested)
    sig = holder_failed["holder_attestation"]["signature"]
    holder_failed["holder_attestation"]["signature"] = ("00" * 8) + sig[16:]

    # Anchor over a DIFFERENT record: keep integrity green (refresh the pinned
    # digests) so only the record binding fails.
    diff_anchor = base(anchor=_anchor(rec), revocations=EMPTY_REVS)
    diff_anchor["evidence"]["anchor"]["chain_head_hash"] = "ab" * 32
    diff_anchor["manifest"]["anchor_digest"] = (
        "sha256:" + hashlib.sha256(
            canonical_json(diff_anchor["evidence"]["anchor"])).hexdigest()
    )
    _refresh_manifest_digest(diff_anchor)

    # Tampered DID document: edit a window after sealing, leave digests stale.
    tampered_did = base(anchor=_anchor(rec), revocations=EMPTY_REVS)
    tampered_did["evidence"]["did_document"]["verificationMethod"][0]["validUntil"] = \
        "2099-01-01T00:00:00Z"

    # Tampered manifest: swap the producer, leave manifest_digest stale.
    tampered_manifest = base(anchor=_anchor(rec), revocations=EMPTY_REVS)
    tampered_manifest["manifest"]["producer"] = "did:web:attacker.example"

    # Non-canonical key-history override: entries reversed from sorted order; the
    # pinned digest is the MODEL digest, so a raw-bytes hash would mismatch.
    split = {"version": 1, "keys": [
        {"keyid": KEYID, "not_before": "2026-05-01T00:00:00Z",
         "not_after": "2026-07-01T00:00:00Z"},
        {"keyid": KEYID, "not_before": "2026-01-01T00:00:00Z",
         "not_after": "2026-03-01T00:00:00Z"},
    ]}
    noncanonical_kh = base(key_history=split, revocations=EMPTY_REVS)

    # Signed after retirement: the window closes before iat.
    retired_doc = _doc(_method(KEYID, jwk, validFrom=ACTIVATED,
                               validUntil="2026-05-01T00:00:00Z"))
    after_retirement = base(did_document=retired_doc, anchor=_anchor(rec),
                            revocations=EMPTY_REVS)

    cases = [
        {"name": "clean_no_anchor", "package": base(revocations=EMPTY_REVS)},
        {"name": "anchored_not_verified",
         "package": base(anchor=_anchor(rec), revocations=EMPTY_REVS)},
        {"name": "corroborated",
         "package": base(anchor=_anchor(rec), revocations=EMPTY_REVS),
         "anchoredTime": ANCHOR_OK},
        {"name": "pinned_corroborated",
         "package": base(anchor=_anchor(rec), revocations=EMPTY_REVS),
         "anchoredTime": ANCHOR_OK, "trustedDidDocument": in_window},
        {"name": "strict_pass",
         "package": base(anchor=_anchor(rec), revocations=EMPTY_REVS),
         "anchoredTime": ANCHOR_OK, "trustedDidDocument": in_window, "strict": True},
        {"name": "strict_unmet_no_anchor",
         "package": base(revocations=EMPTY_REVS),
         "trustedDidDocument": in_window, "strict": True},
        {"name": "anchor_over_different_record", "package": diff_anchor,
         "anchoredTime": ANCHOR_OK},
        {"name": "tampered_did_document", "package": tampered_did},
        {"name": "tampered_manifest_producer", "package": tampered_manifest},
        {"name": "holder_attested", "package": holder_attested},
        {"name": "holder_attestation_failed", "package": holder_failed},
        {"name": "noncanonical_key_history", "package": noncanonical_kh},
        {"name": "signed_after_retirement", "package": after_retirement},
    ]

    expected = {}
    for case in cases:
        verdict = verify_handoff(
            case["package"],
            anchor_attested_time=case.get("anchoredTime"),
            trusted_did_document=case.get("trustedDidDocument"),
            strict=case.get("strict", False),
        )
        d = verdict.to_dict()
        expected[case["name"]] = {k: d[k] for k in COMPARE}

    (HERE / "cases.json").write_text(
        json.dumps({"cases": cases, "did": DID, "keyid": KEYID},
                   indent=2, sort_keys=True) + "\n")
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases and expected.json")


if __name__ == "__main__":
    main()
