#!/usr/bin/env python3
"""Zero-install checker for the governance_decision_v0 conformance vectors.

Same verdicts as ``_check_independent.py``, with **nothing installed**: this file
imports only the Python standard library. No ``cryptography``, no ``rfc8785``, no
Vaara. Copy this one file next to the vectors and run it:

    python3 _check_zerodep.py

Why it exists. ``_check_independent.py`` already imports no Vaara code, so it
proves the verdicts are properties of the bytes rather than of our runtime. It
still asks a reviewer to install two packages first, and in a procurement or
audit setting that install is the step that does not happen. This file removes it:
RFC 8785 canonicalization, ES256 verification and SPKI key parsing are all
implemented here against the specifications, so a reviewer with a bare Python can
reproduce every verdict offline.

``_check_independent.py`` remains the reference checker and its bytes are pinned by
downstream specifications. This is a sibling, never a replacement: when the two
disagree, that disagreement is the finding, and ``tests/test_vectors_zerodep_checker.py``
fails on it in CI.

Verification only. Nothing here can sign, and nothing here reads a private key.

Specifications implemented:
  RFC 8785  JSON Canonicalization Scheme (key ordering, string escaping, UTF-8 output)
  RFC 6090 / FIPS 186-4  ECDSA over NIST P-256, verification only
  RFC 5280  SubjectPublicKeyInfo, enough of the DER to reach the public point
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_STREAM_CASES = ("complete", "dropped", "tail_sealed", "tail_unsealed")
_SEALED = {"complete", "dropped", "tail_sealed"}  # tail_unsealed deliberately has no seal

_SAFE_INT = 2 ** 53 - 1  # beyond this, JSON numbers stop being exactly representable


# --- RFC 8785 canonicalization, standard library only -------------------------

_ESCAPES = {0x08: "\\b", 0x09: "\\t", 0x0A: "\\n", 0x0C: "\\f", 0x0D: "\\r",
            0x22: '\\"', 0x5C: "\\\\"}


def _jcs_string(s: str) -> str:
    """RFC 8785 section 3.2.2.2. Escape only what must be escaped; emit the rest literally.

    Everything above U+001F other than quote and backslash stays as its own character, so the
    output carries real UTF-8 rather than the ``\\uXXXX`` sequences ``json.dumps`` emits by
    default. That difference is exactly what the unicode_scope vector exists to catch.
    """
    out = ['"']
    for ch in s:
        cp = ord(ch)
        if cp in _ESCAPES:
            out.append(_ESCAPES[cp])
        elif cp < 0x20:
            out.append(f"\\u{cp:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _jcs_number(n) -> str:
    """Integers only, which is all this corpus contains.

    RFC 8785 defers to ECMAScript ``Number::toString`` for non-integers. Rather than ship a
    partial float serializer that could diverge from the reference implementation on some
    value and be discovered later, this raises: an unsupported input is a loud failure here,
    never a silently different digest.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"this checker canonicalizes integers only, got {type(n).__name__}: {n!r}")
    if abs(n) > _SAFE_INT:
        raise ValueError(f"integer outside the exactly-representable JSON range: {n}")
    return str(n)


def jcs(obj) -> bytes:
    """Canonicalize to RFC 8785 bytes.

    Object keys sort by their UTF-16 code unit sequence, which ``k.encode("utf-16-be")``
    reproduces exactly: comparing those big-endian bytes is comparing code units in order,
    including the surrogate pairs that make astral keys sort below U+E000.
    """
    def enc(o) -> str:
        if o is None:
            return "null"
        if o is True:
            return "true"
        if o is False:
            return "false"
        if isinstance(o, str):
            return _jcs_string(o)
        if isinstance(o, int):
            return _jcs_number(o)
        if isinstance(o, float):
            return _jcs_number(o)  # raises, with the reason
        if isinstance(o, list):
            return "[" + ",".join(enc(v) for v in o) + "]"
        if isinstance(o, dict):
            items = sorted(o.items(), key=lambda kv: kv[0].encode("utf-16-be"))
            return "{" + ",".join(_jcs_string(k) + ":" + enc(v) for k, v in items) + "}"
        raise ValueError(f"not JSON data: {type(o).__name__}")

    return enc(obj).encode("utf-8")


# --- ECDSA over P-256, verification only --------------------------------------

_P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
_A = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC
_B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
_G = (0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
      0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5)


def _on_curve(pt) -> bool:
    if pt is None:
        return True
    x, y = pt
    return 0 <= x < _P and 0 <= y < _P and (y * y - (x * x * x + _A * x + _B)) % _P == 0


def _add(p, q):
    if p is None:
        return q
    if q is None:
        return p
    x1, y1 = p
    x2, y2 = q
    if x1 == x2 and (y1 + y2) % _P == 0:
        return None
    if p == q:
        lam = (3 * x1 * x1 + _A) * pow(2 * y1, -1, _P) % _P
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, _P) % _P
    x3 = (lam * lam - x1 - x2) % _P
    return (x3, (lam * (x1 - x3) - y1) % _P)


def _mul(k: int, pt):
    """Double-and-add. Not constant time, and it does not need to be: every input is public."""
    if k % _N == 0 or pt is None:
        return None
    result, addend = None, pt
    while k:
        if k & 1:
            result = _add(result, addend)
        addend = _add(addend, addend)
        k >>= 1
    return result


def es256_verify(message: bytes, sig_hex: str, pubkey: tuple[int, int]) -> bool:
    """Verify a raw ``r||s`` ES256 signature (128 hex characters) over SHA-256 of ``message``."""
    try:
        if len(sig_hex) != 128:
            return False
        raw = bytes.fromhex(sig_hex)
        r = int.from_bytes(raw[:32], "big")
        s = int.from_bytes(raw[32:], "big")
        if not (1 <= r < _N and 1 <= s < _N):
            return False
        e = int.from_bytes(hashlib.sha256(message).digest(), "big")
        w = pow(s, -1, _N)
        pt = _add(_mul(e * w % _N, _G), _mul(r * w % _N, pubkey))
        return pt is not None and pt[0] % _N == r
    except ValueError:
        return False  # non-hex signature is a failed signature, not a crash


# --- RFC 5280 SubjectPublicKeyInfo, only as far as the point ------------------


def _der_read(buf: memoryview, pos: int) -> tuple[int, memoryview, int]:
    """Read one TLV. Returns (tag, contents, next position). Definite lengths only."""
    tag = buf[pos]
    pos += 1
    n = buf[pos]
    pos += 1
    if n & 0x80:
        count = n & 0x7F
        if count == 0 or count > 4:
            raise ValueError("unsupported DER length encoding")
        n = int.from_bytes(bytes(buf[pos:pos + count]), "big")
        pos += count
    return tag, buf[pos:pos + n], pos + n


def load_p256_pem(pem: str) -> tuple[int, int]:
    """Parse a PEM SPKI public key and return the P-256 point, checking it is on the curve."""
    body = "".join(re.findall(r"-----BEGIN PUBLIC KEY-----(.*?)-----END PUBLIC KEY-----", pem, re.S))
    if not body:
        raise ValueError("no PEM public key block found")
    der = memoryview(base64.b64decode("".join(body.split())))
    tag, spki, _ = _der_read(der, 0)
    if tag != 0x30:
        raise ValueError("SubjectPublicKeyInfo is not a SEQUENCE")
    _, _alg, pos = _der_read(spki, 0)          # AlgorithmIdentifier, not inspected further
    tag, bitstring, _ = _der_read(spki, pos)   # subjectPublicKey BIT STRING
    if tag != 0x03 or bitstring[0] != 0:
        raise ValueError("subjectPublicKey is not a whole-octet BIT STRING")
    point = bytes(bitstring[1:])
    if len(point) != 65 or point[0] != 0x04:
        raise ValueError("expected a 65-byte uncompressed EC point")
    pt = (int.from_bytes(point[1:33], "big"), int.from_bytes(point[33:], "big"))
    if not _on_curve(pt):
        raise ValueError("public key is not a point on P-256")
    return pt


# --- the checks themselves, mirroring _check_independent.py --------------------


def _sha(obj) -> str:
    return "sha256:" + hashlib.sha256(jcs(obj)).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sig(pub, wrapped: dict) -> bool:
    record = wrapped.get("record")
    sig = wrapped.get("signature", "")
    if not isinstance(record, dict) or len(sig) != 128:
        return False
    return es256_verify(jcs(record), sig, pub)


def _recompute_refs(record: dict) -> dict:
    intent_digest = _sha({
        "action_type": record["action_type"],
        "normalized_scope": record["normalized_scope"],
        "params_hash": record["params_hash"],
    })
    intent_ref = _sha({
        "schema": record["schema"],
        "agent_id": record["agent_id"],
        "action_type": record["action_type"],
        "normalized_scope": record["normalized_scope"],
        "intent_digest": intent_digest,
    })
    decision_context_hash = _sha({
        "policy_refs": record["policy_refs"],
        "target_state_digest": record["target_state_digest"],
        "continuation_id": record["continuation_id"],
        "normalization_id": record["normalization_id"],
    })
    receipt_ref = _sha({
        "intent_ref": intent_ref,
        "target_state_digest": record["target_state_digest"],
        "continuation_id": record["continuation_id"],
        "seq": record["completeness"]["seq"],
        "timestamp_ms": record["timestamp_ms"],
        "idempotency_key": record["idempotency_key"],
    })
    return {"intent_digest": intent_digest, "intent_ref": intent_ref,
            "decision_context_hash": decision_context_hash, "receipt_ref": receipt_ref}


def _derivations_consistent(record: dict) -> bool:
    return all(record[k] == v for k, v in _recompute_refs(record).items())


def _contiguity(records: list[dict]) -> dict:
    blocks = [r["record"]["completeness"] for r in records]
    seqs = [int(b["seq"]) for b in blocks]
    counts = [int(b["runningCount"]) for b in blocks]
    expected = max((max(seqs) + 1 if seqs else 0), (max(counts) if counts else 0))
    missing = sorted(set(range(expected)) - set(seqs))
    mismatch = any(int(b["runningCount"]) != int(b["seq"]) + 1 for b in blocks)
    present = len(blocks)
    ok = not missing and not mismatch and len(seqs) == len(set(seqs)) and present == expected
    return {"ok": ok, "present": present, "expected": expected, "missingSeqs": missing}


def _sealed_contiguity(records: list[dict], seal: dict | None) -> dict:
    base = _contiguity(records)
    if seal is None:
        return base
    total = int(seal["record"]["total"])
    seqs = {int(r["record"]["completeness"]["seq"]) for r in records}
    missing = sorted(set(range(total)) - seqs)
    ok = not missing and base["ok"] and base["expected"] == total
    return {"ok": ok, "present": len(records), "expected": total, "missingSeqs": missing}


def _load_stream(case: str) -> tuple[list[dict], dict | None]:
    case_dir = HERE / "stream" / case
    records = [_load(p) for p in sorted(case_dir.glob("*-decision.json"))]
    seal_path = case_dir / "seal.json"
    return records, (_load(seal_path) if seal_path.exists() else None)


def _verdict_exact_intent_mismatch(case: dict) -> str:
    a = _recompute_refs(case["approved"]["record"])["intent_ref"]
    c = _recompute_refs(case["candidate"]["record"])["intent_ref"]
    return "deny" if a != c else "allow"


def _verdict_target_state_drift(case: dict) -> str:
    ar, cr = case["approved"]["record"], case["candidate"]["record"]
    same_intent = _recompute_refs(ar)["intent_ref"] == _recompute_refs(cr)["intent_ref"]
    drift = ar["target_state_digest"] != cr["target_state_digest"]
    return "revise" if (same_intent and drift) else "deny"


def _verdict_continuation_mismatch(case: dict) -> str:
    ar, cr = case["approved"]["record"], case["candidate"]["record"]
    same_intent = _recompute_refs(ar)["intent_ref"] == _recompute_refs(cr)["intent_ref"]
    moved = (ar["continuation_id"] != cr["continuation_id"]
             and _recompute_refs(ar)["decision_context_hash"]
             != _recompute_refs(cr)["decision_context_hash"])
    return "deny" if (same_intent and moved) else "allow"


def _verdict_duplicate_outcome(case: dict) -> str:
    f, s = case["first"]["record"], case["second"]["record"]
    replay = f["receipt_ref"] == s["receipt_ref"] and f["idempotency_key"] == s["idempotency_key"]
    return "deny" if replay else "allow"


_VERDICTS = {
    "exact_intent_mismatch": _verdict_exact_intent_mismatch,
    "target_state_drift": _verdict_target_state_drift,
    "continuation_mismatch": _verdict_continuation_mismatch,
    "duplicate_outcome": _verdict_duplicate_outcome,
}
_CASE_WRAPPED = {
    "exact_intent_mismatch": ("approved", "candidate"),
    "target_state_drift": ("approved", "candidate"),
    "continuation_mismatch": ("approved", "candidate"),
    "duplicate_outcome": ("first", "second"),
}


def main() -> int:
    pub = load_p256_pem((HERE / "keys" / "es256_public.pem").read_text(encoding="utf-8"))
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    results: list[tuple[str, bool]] = []

    def check(label: str, ok: bool) -> None:
        results.append((label, ok))
        print(f"[{'OK' if ok else 'FAIL'}] {label}")

    for case in _STREAM_CASES:
        records, seal = _load_stream(case)
        sigs_ok = all(_verify_sig(pub, r) for r in records)
        check(f"stream.{case}.all_signatures_ok",
              sigs_ok == expected["stream"][case]["all_signatures_ok"])
        check(f"stream.{case}.derivations_consistent",
              all(_derivations_consistent(r["record"]) for r in records))
        got = _sealed_contiguity(records, seal if case in _SEALED else None)
        check(f"stream.{case}.sealed_contiguity",
              got == expected["stream"][case]["sealed_contiguity"])

    for name, fn in _VERDICTS.items():
        case = _load(HERE / "cases" / f"{name}.json")
        sigs_ok = all(_verify_sig(pub, case[k]) for k in _CASE_WRAPPED[name])
        check(f"cases.{name}.signatures_ok",
              sigs_ok == expected["cases"][name]["signatures_ok"])
        check(f"cases.{name}.verdict", fn(case) == expected["cases"][name]["expected_verdict"])

    uni = _load(HERE / "cases" / "unicode_scope.json")
    urec = uni["record"]
    check("unicode_scope.signature_ok",
          _verify_sig(pub, uni) == expected["unicode_scope"]["signature_ok"])
    check("unicode_scope.intent_ref",
          _recompute_refs(urec)["intent_ref"] == expected["unicode_scope"]["intent_ref"]
          == urec["intent_ref"])
    check("unicode_scope.params_hash",
          urec["params_hash"] == expected["unicode_scope"]["params_hash"])

    ok = all(v for _, v in results)
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'} "
          f"({len(results)} checks, standard library only)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
