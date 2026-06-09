#!/usr/bin/env python3
"""Independent checker for the cross_org_handoff_v0 vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
not import Vaara. For each committed handoff package it reproduces the verdict a
regulator's ``verify_handoff`` returns:

1. **Integrity.** Recompute each pinned component digest from the enclosed
   bytes. ``record`` / ``did_document`` / ``anchor`` are plain
   ``sha256(jcs(component))``; ``key_history`` and ``revocations`` are MODEL
   digests over the canonicalised, re-sorted model (an override else the
   document), so a non-canonical override still matches. Recompute the manifest
   fingerprint and confirm the producer is coherent.
2. **Record.** Bind the ES256 signature to a key the archived document lists,
   then judge the bound key's validity window and revocation at the claimed
   ``iat`` (the retained-record arithmetic).
3. **Anchor.** Confirm an enclosed anchor's imprint equals
   ``sha256(jcs(record))`` over sha256 (a byte compare). When the case supplies
   a pre-verified ``anchoredTime`` and the anchor binds, judge whether it
   predates retirement and revocation (corroboration).
4. **Custody.** Verify a holder ES256/RS256 attestation over ``jcs(manifest)``.
5. **Producer pin.** With a ``trustedDidDocument`` the bound key must match it.

The one step this checker does NOT perform is verifying the RFC 3161 CMS token
signature itself, which needs the timeanchor extra and a trusted TSA chain;
exactly as the key_rotation_v0 checker, the attested time is taken pre-verified
from the case. The anchor-to-record binding IS reproduced here. Verdicts are
compared against ``expected.json`` (the non-normative ``reason`` is not
compared). Run:
``python tests/vectors/cross_org_handoff_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
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
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent
_RECEIPT_BLOCKS = ("version", "alg", "backLink", "outcomeDerived", "receiptAsserted")
_NOT_BEFORE_KEYS = ("validFrom", "notBefore")
_NOT_AFTER_KEYS = ("validUntil", "notAfter")
COMPARE = (
    "integrity_ok", "producer_identity_basis", "bound", "keyid",
    "window_recorded", "revocation_source_present", "revoked", "anchor_present",
    "anchor_verified", "anchor_binds", "verifiable", "corroborated", "custody",
    "holder_keyid", "strict", "ok",
)


def _jcs_digest(value) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _b64u_to_int(value: str) -> int:
    pad = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + pad), "big")


def _ec_key(jwk: dict):
    if not isinstance(jwk, dict) or jwk.get("kty") != "EC" or jwk.get("crv") != "P-256":
        return None
    nums = ec.EllipticCurvePublicNumbers(
        _b64u_to_int(jwk["x"]), _b64u_to_int(jwk["y"]), ec.SECP256R1())
    return nums.public_key()


def _es256_verifies(payload: bytes, sig_hex, public_key) -> bool:
    if not isinstance(sig_hex, str) or len(sig_hex) != 128:
        return False
    try:
        raw = bytes.fromhex(sig_hex)
    except ValueError:
        return False
    der = encode_dss_signature(int.from_bytes(raw[:32], "big"),
                               int.from_bytes(raw[32:], "big"))
    try:
        public_key.verify(der, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def _rsa_key(jwk: dict):
    from cryptography.hazmat.primitives.asymmetric import rsa
    if not isinstance(jwk, dict) or jwk.get("kty") != "RSA":
        return None
    return rsa.RSAPublicNumbers(_b64u_to_int(jwk["e"]),
                                _b64u_to_int(jwk["n"])).public_key()


def _rs256_verifies(payload: bytes, sig_hex, public_key) -> bool:
    try:
        sig = bytes.fromhex(sig_hex)
    except (ValueError, TypeError):
        return False
    try:
        public_key.verify(sig, payload, padding.PKCS1v15(), hashes.SHA256())
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


def _first(method: dict, keys):
    for key in keys:
        value = method.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _within(at_time, not_before, not_after) -> bool:
    at = _parse_iso(at_time)
    if at is None:
        return False
    if not_before is not None:
        nb = _parse_iso(not_before)
        if nb is None or at < nb:
            return False
    if not_after is not None:
        na = _parse_iso(not_after)
        if na is None or at >= na:
            return False
    return True


def _anchor_before(anchored_time, boundary) -> bool:
    if boundary is None:
        return True
    if anchored_time is None:
        return False
    a, b = _parse_iso(anchored_time), _parse_iso(boundary)
    if a is None or b is None:
        return False
    return a < b


def _hex_eq(a, b) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    try:
        return bytes.fromhex(a) == bytes.fromhex(b)
    except ValueError:
        return False


def _kh_entries(evidence: dict, doc: dict):
    """Effective key-history entries: an override, else the document's markers."""
    override = evidence.get("key_history")
    if override is not None:
        return [{"keyid": e.get("keyid"), "not_before": e.get("not_before"),
                 "not_after": e.get("not_after")} for e in override.get("keys", [])]
    out = []
    for m in doc.get("verificationMethod", []):
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        nb, na = _first(m, _NOT_BEFORE_KEYS), _first(m, _NOT_AFTER_KEYS)
        if isinstance(mid, str) and mid and (nb is not None or na is not None):
            out.append({"keyid": mid, "not_before": nb, "not_after": na})
    return out


def _kh_digest(entries) -> str:
    keys = []
    for e in entries:
        d = {"keyid": e["keyid"]}
        if e.get("not_before") is not None:
            d["not_before"] = e["not_before"]
        if e.get("not_after") is not None:
            d["not_after"] = e["not_after"]
        keys.append(d)
    keys.sort(key=lambda d: (d["keyid"], d.get("not_before", ""), d.get("not_after", "")))
    return _jcs_digest({"version": 1, "keys": keys})


def _rev_entries(evidence: dict, doc: dict):
    """Effective revocation entries: an override, else the document's markers."""
    override = evidence.get("revocations")
    if override is not None:
        return [{"scope": e.get("scope"), "subject": e.get("subject"),
                 "revoked_at": e.get("revoked_at")} for e in override.get("entries", [])]
    out = []
    for m in doc.get("verificationMethod", []):
        if not isinstance(m, dict):
            continue
        revoked, mid = m.get("revoked"), m.get("id")
        if isinstance(revoked, str) and revoked and isinstance(mid, str) and mid:
            out.append({"scope": "key", "subject": mid, "revoked_at": revoked})
    return out


def _rev_digest(entries) -> str:
    rows = [{"scope": e["scope"], "subject": e["subject"], "revoked_at": e["revoked_at"]}
            for e in entries]
    rows.sort(key=lambda d: (d["scope"], d["subject"], d["revoked_at"]))
    return _jcs_digest({"version": 1, "entries": rows})


def _bind(record: dict, doc: dict):
    """The id of the method whose EC key verifies the ES256 receipt signature."""
    iss = record.get("receiptAsserted", {}).get("iss")
    if not isinstance(iss, str) or not iss.startswith("did:web:") \
            or doc.get("id") != iss or record.get("alg") != "ES256":
        return None
    payload = rfc8785.dumps({k: record[k] for k in _RECEIPT_BLOCKS})
    for m in doc.get("verificationMethod", []):
        if not isinstance(m, dict):
            continue
        key = _ec_key(m.get("publicKeyJwk", {}))
        if key is not None and _es256_verifies(payload, record.get("signature"), key):
            return m.get("id")
    return None


def _window(entries, keyid, iat):
    """(within, recorded, not_after) for ``keyid`` over the effective entries."""
    matching = [e for e in entries if e.get("keyid") == keyid]
    windows = [(e.get("not_before"), e.get("not_after")) for e in matching
               if e.get("not_before") is not None or e.get("not_after") is not None]
    if not windows:
        return True, False, None
    admit = next(((nb, na) for (nb, na) in windows if _within(iat, nb, na)), None)
    if admit is not None:
        return True, True, admit[1]
    return False, True, windows[0][1]


def _revocation(entries, iss, keyid, iat):
    """(revoked, revoked_at) earliest binding revocation at or before ``iat``."""
    issued = _parse_iso(iat)
    matched = []
    for e in entries:
        scope, subject, mark = e.get("scope"), e.get("subject"), e.get("revoked_at")
        if scope == "key":
            if subject != keyid:
                continue
        elif subject != iss:
            continue
        parsed = _parse_iso(mark)
        if parsed is None or issued is None or parsed <= issued:
            matched.append((parsed, mark))
    if not matched:
        return False, None
    rankable = [(p, m) for (p, m) in matched if p is not None]
    return True, (min(rankable, key=lambda t: t[0])[1] if rankable else matched[0][1])


def _bound_jwk(doc: dict, keyid):
    for m in doc.get("verificationMethod", []):
        if isinstance(m, dict) and m.get("id") == keyid and isinstance(
                m.get("publicKeyJwk"), dict):
            return m["publicKeyJwk"]
    return None


def _custody(package: dict, manifest: dict):
    att = package.get("holder_attestation")
    if att is None:
        return "unattested", None
    keyid = att.get("keyid")
    keyid = keyid if isinstance(keyid, str) else None
    alg, sig, jwk = att.get("alg"), att.get("signature"), att.get("verifying_jwk")
    payload = rfc8785.dumps(manifest)
    ok = False
    if alg == "ES256" and isinstance(jwk, dict):
        key = _ec_key(jwk)
        ok = key is not None and _es256_verifies(payload, sig, key)
    elif alg == "RS256" and isinstance(jwk, dict):
        key = _rsa_key(jwk)
        ok = key is not None and _rs256_verifies(payload, sig, key)
    return ("holder_attested_selfsupplied" if ok else "holder_attestation_failed"), keyid


def _evaluate(case: dict) -> dict:
    package = case["package"]
    evidence, manifest = package["evidence"], package["manifest"]
    record, doc = evidence["record"], evidence["did_document"]
    iss = record["receiptAsserted"]["iss"]
    iat = record["receiptAsserted"]["iat"]
    anchor = evidence.get("anchor")
    anchor_present = anchor is not None

    kh_entries = _kh_entries(evidence, doc)
    rev_entries = _rev_entries(evidence, doc)

    # integrity: each pinned digest recomputed, plus the manifest fingerprint
    # and producer coherence.
    comp = {
        "record": _jcs_digest(record) == manifest.get("record_digest"),
        "did_document": _jcs_digest(doc) == manifest.get("did_document_digest"),
        "key_history": _kh_digest(kh_entries) == manifest.get("key_history_digest"),
        "revocations": _rev_digest(rev_entries) == manifest.get("revocations_digest"),
    }
    if anchor_present:
        comp["anchor"] = _jcs_digest(anchor) == manifest.get("anchor_digest")
    else:
        comp["anchor"] = manifest.get("anchor_digest") is None
    manifest_ok = _jcs_digest(manifest) == package.get("manifest_digest")
    producer_coherent = (manifest.get("producer") == iss and doc.get("id") == iss)
    integrity_ok = all(comp.values()) and manifest_ok and producer_coherent

    # record verdict
    keyid = _bind(record, doc)
    bound = keyid is not None
    if bound:
        within, window_recorded, not_after = _window(kh_entries, keyid, iat)
        revoked, revoked_at = _revocation(rev_entries, iss, keyid, iat)
    else:
        within, window_recorded, not_after, revoked, revoked_at = \
            False, False, None, False, None
    verifiable = bound and within and not revoked

    # anchor binding + corroboration
    anchor_binds = False
    if anchor_present:
        anchor_binds = anchor.get("hash_algorithm") == "sha256" and _hex_eq(
            anchor.get("chain_head_hash"),
            hashlib.sha256(rfc8785.dumps(record)).hexdigest())
    anchored_time = case.get("anchoredTime")
    anchor_verified = anchor_present and anchored_time is not None
    eff_time = anchored_time if (anchor_binds and anchor_verified) else None
    corroborated = bool(
        verifiable and eff_time is not None
        and _anchor_before(eff_time, not_after)
        and _anchor_before(eff_time, revoked_at))

    revocation_source_present = (evidence.get("revocations") is not None
                                 or len(rev_entries) > 0)

    custody, holder_keyid = _custody(package, manifest)

    basis = "self_asserted_unpinned"
    trusted = case.get("trustedDidDocument")
    if trusted is not None:
        enclosed = _bound_jwk(doc, keyid) if bound else None
        ref = _bound_jwk(trusted, keyid) if bound else None
        basis = "pinned" if (enclosed is not None and enclosed == ref) else "pin_mismatch"

    strict = bool(case.get("strict", False))
    if strict:
        ok = bool(integrity_ok and corroborated and window_recorded
                  and revocation_source_present and basis == "pinned")
    else:
        ok = bool(integrity_ok and verifiable)

    return {
        "integrity_ok": integrity_ok, "producer_identity_basis": basis,
        "bound": bound, "keyid": keyid, "window_recorded": window_recorded,
        "revocation_source_present": revocation_source_present, "revoked": revoked,
        "anchor_present": anchor_present, "anchor_verified": anchor_verified,
        "anchor_binds": anchor_binds, "verifiable": verifiable,
        "corroborated": corroborated, "custody": custody,
        "holder_keyid": holder_keyid, "strict": strict, "ok": ok,
    }


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for case in cases:
        name = case["name"]
        got = _evaluate(case)
        want = expected[name]
        if got != want:
            failures.append(f"{name}:\n    expected {want}\n    got      {got}")
        else:
            print(f"{name}: OK ok={got['ok']} integrity={got['integrity_ok']} "
                  f"verifiable={got['verifiable']} corroborated={got['corroborated']}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall cross_org_handoff_v0 cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
