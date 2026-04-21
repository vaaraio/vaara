#!/usr/bin/env python3
"""Standalone verifier for a Vaara signed audit-trail export.

No Vaara install required. Only dependency:

    pip install cryptography

Usage:

    ./verify_vaara_trail.py path/to/trail.zip
    ./verify_vaara_trail.py path/to/trail.zip --pubkey signer.pem

A regulator, auditor, or internal conformity reviewer can copy this one
file to any machine and run it — the trail zip is entirely self-describing
(it ships its own public key so you can verify integrity offline), but you
SHOULD pass --pubkey with a key you received out-of-band to check that the
signer is who they claim to be, not just that the trail is internally
consistent.

The verifier exits 0 if every check passes, 1 if any check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
except ImportError:
    print(
        "This verifier requires the 'cryptography' package.\n"
        "Install with: pip install cryptography",
        file=sys.stderr,
    )
    sys.exit(2)


REQUIRED_FILES = {"trail.jsonl", "manifest.json", "trail.sig"}


def _load_pubkey(data: bytes) -> Ed25519PublicKey:
    loaded = serialization.load_pem_public_key(data)
    if not isinstance(loaded, Ed25519PublicKey):
        raise ValueError("public key is not Ed25519")
    return loaded


def _verify_chain(trail_bytes: bytes) -> str | None:
    prev_hash = ""
    for idx, raw in enumerate(ln for ln in trail_bytes.splitlines() if ln.strip()):
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as e:
            return f"record {idx}: malformed JSON ({e})"
        if rec.get("previous_hash", "") != prev_hash:
            return f"record {idx}: previous_hash mismatch"
        content = {
            "record_id": rec.get("record_id"),
            "action_id": rec.get("action_id"),
            "event_type": rec.get("event_type"),
            "timestamp": rec.get("timestamp"),
            "agent_id": rec.get("agent_id"),
            "tool_name": rec.get("tool_name"),
            "data": rec.get("data", {}),
            "regulatory_articles": rec.get("regulatory_articles", []),
            "previous_hash": prev_hash,
        }
        canonical = json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        if rec.get("record_hash") != expected:
            return f"record {idx}: hash mismatch (record tampered)"
        prev_hash = rec["record_hash"]
    return None


def verify(zip_path: Path, pubkey_path: Path | None) -> tuple[bool, list[str], dict | None]:
    errors: list[str] = []
    if not zip_path.exists():
        return False, [f"file not found: {zip_path}"], None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            missing = REQUIRED_FILES - names
            if missing:
                return False, [f"missing files in zip: {sorted(missing)}"], None
            trail_bytes = zf.read("trail.jsonl")
            manifest_bytes = zf.read("manifest.json")
            signature = zf.read("trail.sig")
            embedded_pubkey = (
                zf.read("signer_pubkey.pem") if "signer_pubkey.pem" in names else None
            )
    except zipfile.BadZipFile as e:
        return False, [f"corrupt zip: {e}"], None

    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as e:
        return False, [f"manifest.json invalid JSON: {e}"], None

    if manifest.get("schema_version") != "1":
        errors.append(f"unsupported schema_version: {manifest.get('schema_version')!r}")

    actual_sha = hashlib.sha256(trail_bytes).hexdigest()
    if manifest.get("trail_sha256") != actual_sha:
        errors.append("trail.jsonl SHA-256 does not match manifest.trail_sha256 (tampered)")

    if pubkey_path is not None:
        pk_data = pubkey_path.read_bytes()
    elif embedded_pubkey is not None:
        pk_data = embedded_pubkey
    else:
        return False, errors + ["no --pubkey given and no signer_pubkey.pem in zip"], manifest

    pk = _load_pubkey(pk_data)
    to_verify = hashlib.sha256(trail_bytes + manifest_bytes).digest()
    try:
        pk.verify(signature, to_verify)
    except InvalidSignature:
        errors.append("Ed25519 signature did not verify")

    chain_error = _verify_chain(trail_bytes)
    if chain_error is not None:
        errors.append(f"hash chain check failed: {chain_error}")

    lines = [ln for ln in trail_bytes.splitlines() if ln.strip()]
    if manifest.get("record_count") != len(lines):
        errors.append(
            f"record_count mismatch: manifest={manifest.get('record_count')}, actual={len(lines)}"
        )
    if lines:
        try:
            first = json.loads(lines[0]).get("record_hash", "")
            last = json.loads(lines[-1]).get("record_hash", "")
            if manifest.get("first_hash") != first:
                errors.append("first_hash mismatch")
            if manifest.get("last_hash") != last:
                errors.append("last_hash mismatch")
        except json.JSONDecodeError:
            errors.append("could not parse first/last record")

    return (not errors), errors, manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("zip_path", help="Path to the signed trail zip")
    p.add_argument(
        "--pubkey",
        default=None,
        help="Optional PEM public key received out-of-band. If omitted, uses the "
        "public key embedded in the zip (good for integrity check, not identity).",
    )
    args = p.parse_args(argv)

    ok, errors, manifest = verify(
        Path(args.zip_path),
        Path(args.pubkey) if args.pubkey else None,
    )

    if manifest:
        print("Manifest:")
        for k in (
            "schema_version",
            "vaara_version",
            "created_utc",
            "record_count",
            "signer_pubkey_fingerprint",
            "agent_id",
        ):
            print(f"  {k:<30} {manifest.get(k)}")

    if ok:
        print("\nVerification: OK")
        return 0
    print("\nVerification: FAILED", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
