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
    from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
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
CORE_FILES = {"trail.jsonl", "manifest.json"}
THRESHOLD_PREFIX = "threshold-"


def _load_pubkey(data: bytes) -> Ed25519PublicKey:
    try:
        loaded = serialization.load_pem_public_key(data)
    except (ValueError, UnsupportedAlgorithm) as e:
        # cryptography 47.0+ may raise UnsupportedAlgorithm where 46.x raised
        # ValueError; the embedded signer_pubkey.pem inside a zip is
        # untrusted input, so we collapse to ValueError with a clear message.
        raise ValueError(f"public key could not be parsed: {e}") from e
    if not isinstance(loaded, Ed25519PublicKey):
        raise ValueError("public key is not Ed25519")
    return loaded


def _raw_ed25519(loaded: Ed25519PublicKey) -> bytes:
    if hasattr(loaded, "public_bytes_raw"):
        return loaded.public_bytes_raw()
    return loaded.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _verify_threshold(
    manifest: dict, to_verify: bytes, sigs: dict, pubkeys: dict,
    pinned_fp: str | None = None,
) -> list[str]:
    """k-of-n threshold verification (Ed25519 members). Mirrors
    vaara.audit.verify._verify_threshold so this standalone file gives the
    same verdict with only the 'cryptography' package installed.

    ``pinned_fp`` is the member fingerprint of an out-of-band --pubkey. When
    set, that key must itself be one of the members that signed, else the
    signer set is attacker-chosen and the export is not authenticated."""
    errors: list[str] = []
    member_algorithm = manifest.get("member_algorithm")
    k = manifest.get("threshold_k")
    fps = manifest.get("signer_fingerprints")
    if not isinstance(k, int) or k < 1:
        return [f"invalid threshold_k: {k!r}"]
    if not isinstance(fps, list) or not fps or not all(isinstance(f, str) for f in fps):
        return ["manifest signer_fingerprints missing or malformed"]
    if manifest.get("signers_n") != len(fps):
        errors.append("signers_n does not match signer_fingerprints length")
    if k > len(fps):
        return errors + [f"threshold_k ({k}) exceeds n ({len(fps)})"]
    if member_algorithm != "Ed25519":
        return errors + [
            f"this standalone verifier supports Ed25519 members only, "
            f"got {member_algorithm!r}"
        ]

    authorized: dict = {}
    for fp in fps:
        raw = pubkeys.get(fp)
        if raw is None:
            errors.append(f"missing pubkeys/ entry for authorized signer {fp}")
            continue
        try:
            loaded = _load_pubkey(raw)
        except ValueError as e:
            errors.append(f"pubkey for {fp}: {e}")
            continue
        raw_norm = _raw_ed25519(loaded)
        if hashlib.sha256(raw_norm).hexdigest()[:32] != fp:
            errors.append(f"pubkey for {fp} does not match its fingerprint (substituted key)")
            continue
        authorized[fp] = raw_norm

    valid = set()
    for fp, sig in sigs.items():
        raw_norm = authorized.get(fp)
        if raw_norm is None:
            continue
        try:
            Ed25519PublicKey.from_public_bytes(raw_norm).verify(sig, to_verify)
            valid.add(fp)
        except (InvalidSignature, ValueError):
            pass
    if len(valid) < k:
        errors.append(
            f"threshold not met: {len(valid)} valid signature(s) from the "
            f"authorized set, need at least {k} of {len(fps)}"
        )
    if pinned_fp is not None and pinned_fp not in valid:
        errors.append(
            "pinned --pubkey is not among the valid threshold signers: the "
            "signer set is attacker-suppliable unless the supplied key is one "
            "of the signers, so this export is not authenticated (integrity only)"
        )
    return errors


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
        # Chain v2 (v0.47+) binds tenant_id and chain_version into the hash.
        # Legacy v1 records omit both, so pre-v0.47 trails re-verify
        # unchanged. Mirrors vaara.audit.verify._verify_chain_bytes.
        chain_version = rec.get("chain_version", 1)
        if isinstance(chain_version, int) and chain_version >= 2:
            content["tenant_id"] = rec.get("tenant_id", "")
            content["chain_version"] = chain_version
        canonical = json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        if rec.get("record_hash") != expected:
            return f"record {idx}: hash mismatch (record tampered)"
        prev_hash = rec["record_hash"]
    return None


def _collect_key_lifecycle(zip_path: Path) -> list[dict]:
    """Custodian key-lifecycle records in the trail, in chain order.

    Informational only; the records are covered by the hash-chain check, so
    a tampered lifecycle record fails on the chain, not here.
    """
    out: list[dict] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trail_bytes = zf.read("trail.jsonl")
    except (zipfile.BadZipFile, KeyError, FileNotFoundError):
        return out
    for raw in (ln for ln in trail_bytes.splitlines() if ln.strip()):
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if rec.get("event_type") != "key_lifecycle":
            continue
        data = rec.get("data", {}) or {}
        out.append({
            "action": data.get("action"),
            "fingerprint": data.get("fingerprint"),
            "threshold_k": data.get("threshold_k"),
            "signers_n": data.get("signers_n"),
            "reason": data.get("reason", ""),
        })
    return out


def verify(zip_path: Path, pubkey_path: Path | None) -> tuple[bool, list[str], dict | None]:
    errors: list[str] = []
    if not zip_path.exists():
        return False, [f"file not found: {zip_path}"], None

    threshold_sigs: dict = {}
    threshold_pubkeys: dict = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            missing = CORE_FILES - names
            if missing:
                return False, [f"missing files in zip: {sorted(missing)}"], None
            trail_bytes = zf.read("trail.jsonl")
            manifest_bytes = zf.read("manifest.json")
            signature = zf.read("trail.sig") if "trail.sig" in names else None
            embedded_pubkey = (
                zf.read("signer_pubkey.pem") if "signer_pubkey.pem" in names else None
            )
            revocation_bytes = (
                zf.read("revocation.json") if "revocation.json" in names else None
            )
            for name in names:
                if name.startswith("sigs/") and name.endswith(".sig"):
                    threshold_sigs[name[len("sigs/"):-len(".sig")]] = zf.read(name)
                elif name.startswith("pubkeys/") and name != "pubkeys/":
                    fp = name[len("pubkeys/"):].rsplit(".", 1)[0]
                    threshold_pubkeys[fp] = zf.read(name)
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

    # Optional revocation registry (v0.55). The manifest pins the registry
    # digest, and the manifest itself is signed, so checking revocation.json
    # against manifest.revocation.registry_sha256 confirms the revocation
    # state in the bundle is the one the signer attested.
    rev_manifest = manifest.get("revocation")
    if rev_manifest is not None or revocation_bytes is not None:
        if rev_manifest is None:
            errors.append("revocation.json present but manifest has no revocation block")
        elif revocation_bytes is None:
            errors.append("manifest declares revocation but revocation.json is missing")
        else:
            expected = rev_manifest.get("registry_sha256")
            actual_rev = "sha256:" + hashlib.sha256(revocation_bytes).hexdigest()
            if expected != actual_rev:
                errors.append(
                    "revocation.json digest does not match "
                    "manifest.revocation.registry_sha256 (tampered)"
                )

    algorithm = manifest.get("signature_algorithm", "Ed25519")
    to_verify = hashlib.sha256(trail_bytes + manifest_bytes).digest()

    if algorithm.startswith(THRESHOLD_PREFIX):
        # Only an out-of-band --pubkey pins authenticity; the embedded key is
        # in-zip (attacker-controllable) and must not serve as a pin.
        pinned_fp = None
        if pubkey_path is not None:
            try:
                loaded_pin = _load_pubkey(pubkey_path.read_bytes())
                pinned_fp = hashlib.sha256(_raw_ed25519(loaded_pin)).hexdigest()[:32]
            except (OSError, ValueError) as e:
                errors.append(f"--pubkey could not be loaded: {e}")
        errors.extend(
            _verify_threshold(
                manifest, to_verify, threshold_sigs, threshold_pubkeys, pinned_fp
            )
        )
    else:
        if pubkey_path is not None:
            pk_data = pubkey_path.read_bytes()
        elif embedded_pubkey is not None:
            pk_data = embedded_pubkey
        else:
            return False, errors + ["no --pubkey given and no signer_pubkey.pem in zip"], manifest
        pk = _load_pubkey(pk_data)
        if signature is None:
            errors.append("missing trail.sig in zip")
        else:
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

        events = _collect_key_lifecycle(Path(args.zip_path))
        if events:
            print(f"  {'key_lifecycle_events':<30} {len(events)}")
            for ev in events:
                line = f"    - {ev['action']} {ev['fingerprint']}"
                if ev.get("threshold_k") is not None:
                    line += f" (quorum now {ev['threshold_k']}-of-{ev['signers_n']})"
                if ev.get("reason"):
                    line += f": {ev['reason']}"
                print(line)

    if ok:
        print("\nVerification: OK")
        return 0
    print("\nVerification: FAILED", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
