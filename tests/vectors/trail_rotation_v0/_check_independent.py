#!/usr/bin/env python3
"""Independent checker for the vaara.trail-rotation/v0 vectors.

Imports the standard library, ``rfc8785`` and ``cryptography``. It does NOT
import Vaara. From the records, the public keys and the archive bytes alone it
reproduces, for every case:

  chain        every record's hash recomputes from its content and links to the
               one before it. The first record may link to a hash outside the
               trail only when a rotation record in the trail names that hash
               (its ``last_hash``) and carries a valid signature; otherwise
               the chain is broken at the first record.
  seam state   each ``trail_rotated`` record is ``signed`` (an Ed25519
               signature over the RFC 8785 payload of its fields verifies under
               the export's key, or the named rotation key), ``unsigned``,
               ``bad_signature`` or ``malformed``. Only a signed one accounts
               for the first record's link.
  archive      the sha256 of the archive file against the digest the record
               states.
  verdict      pass when the chain is whole, no rotation record is bad or
               malformed, and an archive that was given matches its digest.

Run: tests/vectors/trail_rotation_v0/_check_independent.py
Exit 0 means every verdict matched the committed cases; 77 means a dependency
is not installed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

try:
    import rfc8785
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
except ImportError as exc:
    print(f"SKIP: {exc}")
    sys.exit(77)

HERE = Path(__file__).resolve().parent
SIGNATURE_TYPE = "vaara.trail-rotation/v1"
TEXT_FIELDS = ("last_hash", "archive_sha256", "boundary_utc", "rotated_at", "signer_fingerprint")
ARCHIVE_KINDS = {"zip", "sealed"}
SIGNATURE_KEYS = {"type", "alg", "signature"}


def record_hash(rec: dict) -> str:
    content = {k: rec.get(k) for k in (
        "record_id", "action_id", "event_type", "timestamp", "agent_id", "tool_name")}
    content["data"] = rec.get("data", {})
    content["regulatory_articles"] = rec.get("regulatory_articles", [])
    content["previous_hash"] = rec.get("previous_hash", "")
    version = rec.get("chain_version", 1)
    if isinstance(version, int) and version >= 2:
        content["tenant_id"] = rec.get("tenant_id", "")
        content["chain_version"] = version
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def payload(data: dict) -> bytes:
    return rfc8785.dumps({
        "type": SIGNATURE_TYPE,
        "lastHash": data["last_hash"],
        "purgedRecords": data["purged_records"],
        "archiveSha256": data["archive_sha256"],
        "archiveRecords": data["archive_records"],
        "archiveKind": data["archive_kind"],
        "boundaryUtc": data["boundary_utc"],
        "rotatedAt": data["rotated_at"],
        "signerFingerprint": data["signer_fingerprint"],
    })


def seam_state(data: dict, key: Ed25519PublicKey) -> str:
    purged, archived = data.get("purged_records"), data.get("archive_records")
    well_formed = (
        all(isinstance(data.get(k), str) and data[k] for k in TEXT_FIELDS)
        and type(purged) is int and purged >= 1
        and type(archived) is int and archived >= 0
        and data.get("archive_kind") in ARCHIVE_KINDS
    )
    if not well_formed:
        return "malformed"
    block = data.get("rotation_signature")
    if block is None:
        return "unsigned"
    if (not isinstance(block, dict) or set(block) != SIGNATURE_KEYS
            or block["type"] != SIGNATURE_TYPE or block["alg"] != "Ed25519"
            or not isinstance(block["signature"], str)):
        return "malformed"
    try:
        key.verify(bytes.fromhex(block["signature"]), payload(data))
    except (InvalidSignature, ValueError):
        return "bad_signature"
    return "signed"


def judge(doc: dict, case: dict) -> dict:
    def public(name: str) -> Ed25519PublicKey:
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(doc["signers"][name]["public_hex"]))

    key = public(case["rotation_key"] or doc["export_signer"])
    records = case["records"]
    seams = [{"index": i, "state": seam_state(r["data"], key), "last_hash": r["data"].get("last_hash"),
              "digest": r["data"].get("archive_sha256")}
             for i, r in enumerate(records) if r["event_type"] == "trail_rotated"]
    link = records[0]["previous_hash"] if records else ""
    accounted = next((s for s in seams
                      if link and s["last_hash"] == link and s["state"] == "signed"), None)

    chain, prev = "ok", link if accounted else ""
    for rec in records:
        if rec["previous_hash"] != prev or record_hash(rec) != rec["record_hash"]:
            chain = "break"
            break
        prev = rec["record_hash"]

    archive_check = None
    if case["archive"] is not None and accounted is not None:
        data = bytes.fromhex(doc["archive_hex" if case["archive"] == "match" else "altered_archive_hex"])
        archive_check = ("digest_only" if hashlib.sha256(data).hexdigest() == accounted["digest"]
                         else "digest_mismatch")

    bad = any(s["state"] in ("bad_signature", "malformed") for s in seams)
    return {
        "verdict": "pass" if chain == "ok" and not bad and archive_check != "digest_mismatch" else "fail",
        "chain": chain,
        "begins_at_seam": accounted is not None,
        "seams": [{"index": s["index"], "state": s["state"], "begins_here": s is accounted}
                  for s in seams],
        "archive_check": archive_check,
    }


def main() -> int:
    doc = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    failed = 0
    for case in doc["cases"]:
        got = judge(doc, case)
        if got == case["expected"]:
            print(f"ok    {case['name']}")
        else:
            failed += 1
            print(f"FAIL  {case['name']}\n  expected {case['expected']}\n  got      {got}")
    total = len(doc["cases"])
    print(f"{total - failed}/{total} cases reproduced")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
