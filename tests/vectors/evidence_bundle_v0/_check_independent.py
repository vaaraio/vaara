#!/usr/bin/env python3
"""Independent conformance checker for the evidence_bundle_v0 vectors.

Imports only the standard library plus ``rfc8785`` and ``cryptography``. It
does not import Vaara. For each committed bundle it reproduces the single
``verify_evidence_bundle`` entrypoint: it runs each of the six lenses whose
evidence is present (identity, signature, back-link, inclusion, consistency,
revocation), threads the identity-resolved keyid into the revocation lens,
and aggregates one verdict the same way the reference does.

The aggregation rule under test: a lens with no evidence is *not applicable*
and does not count; ``ok`` is True only when the receipt signature was
established (by the identity lens binding it to a document key, or the
signature lens verifying it under supplied key material) AND every
applicable lens passed. The committed ``expected.json`` carries the
reference verdict; exit code 0 means every bundle's recomputed verdict
matched it.

Run: ``python tests/vectors/evidence_bundle_v0/_check_independent.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent
_RECEIPT_BLOCKS = ("version", "alg", "backLink", "outcomeDerived", "receiptAsserted")
_RECEIPT_KEYS = (*_RECEIPT_BLOCKS, "signature")


def _b64u_to_int(value: str) -> int:
    pad = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + pad), "big")


def _jwk_to_ec_key(jwk: dict):
    if not isinstance(jwk, dict) or jwk.get("kty") != "EC" or jwk.get("crv") != "P-256":
        return None
    try:
        numbers = ec.EllipticCurvePublicNumbers(
            _b64u_to_int(jwk["x"]), _b64u_to_int(jwk["y"]), ec.SECP256R1()
        )
        return numbers.public_key()
    except (KeyError, ValueError):
        return None


def _es256_verifies(payload: bytes, signature_hex: str, public_key) -> bool:
    if len(signature_hex) != 128:
        return False
    try:
        raw = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    der = encode_dss_signature(
        int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
    )
    try:
        public_key.verify(der, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def _parse_iso(value):
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _revoked_in_time(revoked_at, issued_at) -> bool:
    r = _parse_iso(revoked_at)
    i = _parse_iso(issued_at)
    if r is None or i is None:
        return True
    return r <= i


def _registry_revoked(registry: dict, iss: str, issued_at: str, keyid) -> bool:
    for entry in registry.get("entries", []):
        scope = entry.get("scope")
        subject = entry.get("subject")
        if scope == "key":
            if keyid is None or subject != keyid:
                continue
        elif scope == "identity":
            if subject != iss:
                continue
        else:
            continue
        if _revoked_in_time(entry.get("revoked_at"), issued_at):
            return True
    return False


def _hash_leaf(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _verify_inclusion(leaf: bytes, log_index, tree_size, siblings, root: bytes) -> bool:
    if not (0 <= log_index < tree_size):
        return False
    node = _hash_leaf(leaf)
    idx, size = log_index, tree_size
    it = iter(siblings)
    while size > 1:
        last = size - 1
        if idx == last and idx % 2 == 0:
            pass
        else:
            sib = next(it, None)
            if sib is None:
                return False
            node = _hash_node(node, sib) if idx % 2 == 0 else _hash_node(sib, node)
        idx //= 2
        size = (size + 1) // 2
    if next(it, None) is not None:
        return False
    return node == root


def _verify_consistency(first_size, first_root, second_size, second_root, proof) -> bool:
    if first_size > second_size:
        return False
    if first_size == second_size:
        return not proof and first_root == second_root
    if first_size == 0:
        return not proof
    path = list(proof)
    if first_size & (first_size - 1) == 0:
        path = [first_root, *path]
    if not path:
        return False
    fn, sn = first_size - 1, second_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    nodes = iter(path)
    fr = sr = next(nodes)
    for sibling in nodes:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = _hash_node(sibling, fr)
            sr = _hash_node(sibling, sr)
            while fn != 0 and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            sr = _hash_node(sr, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and fr == first_root and sr == second_root


def _lens(applicable: bool, ok: bool) -> dict:
    return {"applicable": applicable, "ok": ok}


def _identity_lens(bundle: dict):
    """Returns (lens_dict, resolved_keyid)."""
    doc = bundle.get("did_document")
    if doc is None:
        return _lens(False, False), None
    receipt = bundle["receipt"]
    iss = receipt["receiptAsserted"]["iss"]
    if not iss.startswith("did:web:") or doc.get("id") != iss or receipt["alg"] != "ES256":
        return _lens(True, False), None
    payload = rfc8785.dumps({k: receipt[k] for k in _RECEIPT_BLOCKS})
    expected_keyid = bundle.get("expected_keyid")
    for method in doc.get("verificationMethod", []):
        if expected_keyid is not None and method.get("id") != expected_keyid:
            continue
        key = _jwk_to_ec_key(method.get("publicKeyJwk", {}))
        if key is None:
            continue
        if _es256_verifies(payload, receipt["signature"], key):
            return _lens(True, True), method.get("id")
    return _lens(True, False), None


def _signature_lens(bundle: dict) -> dict:
    jwk = bundle.get("verifying_jwk")
    if jwk is None:
        return _lens(False, False)
    receipt = bundle["receipt"]
    key = _jwk_to_ec_key(jwk)
    if key is None or receipt["alg"] != "ES256":
        return _lens(True, False)
    payload = rfc8785.dumps({k: receipt[k] for k in _RECEIPT_BLOCKS})
    return _lens(True, _es256_verifies(payload, receipt["signature"], key))


def _back_link_lens(bundle: dict) -> dict:
    att = bundle.get("attestation")
    if att is None:
        return _lens(False, False)
    receipt = bundle["receipt"]
    digest = "sha256:" + hashlib.sha256(rfc8785.dumps(att)).hexdigest()
    back = receipt["backLink"]
    ok = (
        back["attestationDigest"] == digest
        and back["attestationNonce"] == att["issuerAsserted"]["nonce"]
    )
    return _lens(True, ok)


def _inclusion_lens(bundle: dict) -> dict:
    inc = bundle.get("inclusion")
    if inc is None:
        return _lens(False, False)
    receipt = bundle["receipt"]
    leaf = rfc8785.dumps({k: receipt[k] for k in _RECEIPT_KEYS})
    siblings = [bytes.fromhex(h) for h in inc["siblings_hex"]]
    ok = _verify_inclusion(
        leaf, inc["log_index"], inc["tree_size"], siblings, bytes.fromhex(inc["root_hex"])
    )
    return _lens(True, ok)


def _consistency_lens(bundle: dict) -> dict:
    con = bundle.get("consistency")
    if con is None:
        return _lens(False, False)
    proof = [bytes.fromhex(h) for h in con["hashes_hex"]]
    ok = _verify_consistency(
        con["first_size"], bytes.fromhex(con["first_root_hex"]),
        con["second_size"], bytes.fromhex(con["second_root_hex"]), proof,
    )
    return _lens(True, ok)


def _revocation_lens(bundle: dict, keyid) -> dict:
    registry = bundle.get("registry")
    if registry is None:
        return _lens(False, False)
    receipt = bundle["receipt"]
    asserted = receipt["receiptAsserted"]
    revoked = _registry_revoked(registry, asserted["iss"], asserted["iat"], keyid)
    return _lens(True, not revoked)


def _evaluate(bundle: dict) -> dict:
    identity, resolved_keyid = _identity_lens(bundle)
    keyid = resolved_keyid if resolved_keyid is not None else bundle.get("expected_keyid")
    lenses = {
        "identity": identity,
        "signature": _signature_lens(bundle),
        "back_link": _back_link_lens(bundle),
        "inclusion": _inclusion_lens(bundle),
        "consistency": _consistency_lens(bundle),
        "revocation": _revocation_lens(bundle, keyid),
    }
    authenticity = lenses["identity"]["ok"] or lenses["signature"]["ok"]
    failures = [
        name for name, res in lenses.items() if res["applicable"] and not res["ok"]
    ]
    return {
        "ok": authenticity and not failures,
        "authenticity_established": authenticity,
        "lenses": lenses,
    }


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for case in cases:
        name = case["name"]
        got = _evaluate(case["bundle"])
        want = expected[name]
        if got != want:
            failures.append(f"{name}: expected {want}, got {got}")
        else:
            print(f"{name}: OK ok={got['ok']}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall evidence_bundle_v0 cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
