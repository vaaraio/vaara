"""Verifier for signed audit-trail exports produced by :mod:`vaara.audit.export`.

This module is intentionally self-contained — it imports only the standard
library plus ``cryptography``. No other Vaara modules are imported. A
regulator can copy ``scripts/verify_vaara_trail.py`` (which wraps this
logic) to any machine and run it without installing Vaara.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

try:
    from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey as _PK

logger = logging.getLogger(__name__)

_REQUIRED_FILES = {"trail.jsonl", "manifest.json", "trail.sig"}
# Single-signer exports carry trail.sig; threshold exports carry sigs/ +
# pubkeys/ instead. Both always carry these two.
_CORE_FILES = {"trail.jsonl", "manifest.json"}
_THRESHOLD_PREFIX = "threshold-"
_INSTALL_HINT = (
    "Signed-trail verification requires the 'cryptography' package. "
    "Install with: pip install 'vaara[export]' (or 'pip install cryptography' "
    "if using the standalone verifier script)."
)


@dataclass
class VerifyResult:
    """Return value of :func:`verify_signed`.

    ``ok``: whether every check passed.
    ``errors``: list of human-readable error strings (empty if ``ok``).
    ``manifest``: the manifest dict from the zip (``None`` if unreadable).
    """
    ok: bool
    errors: list[str] = field(default_factory=list)
    manifest: Optional[dict] = None


def _require_crypto() -> None:
    if not _HAS_CRYPTO:
        raise ImportError(_INSTALL_HINT)


def _load_public_key(
    key: Union[str, Path, bytes, "_PK"],
) -> "Ed25519PublicKey":
    _require_crypto()
    if isinstance(key, Ed25519PublicKey):
        return key
    if isinstance(key, (str, Path)):
        key = Path(key).read_bytes()
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(
            f"public_key must be a Path, str, bytes, or Ed25519PublicKey — got {type(key).__name__}"
        )
    try:
        loaded = serialization.load_pem_public_key(bytes(key))
    except (ValueError, UnsupportedAlgorithm) as e:
        # cryptography 47.0+ may raise UnsupportedAlgorithm where 46.x raised
        # ValueError; collapse both to ValueError so callers see one shape.
        # This matters when the embedded signer_pubkey.pem in a zip is
        # malformed or uses an unsupported algorithm (untrusted input).
        raise ValueError(f"public_key could not be parsed as a PEM public key: {e}") from e
    if not isinstance(loaded, Ed25519PublicKey):
        raise ValueError("public_key must be an Ed25519 public key")
    return loaded


def _verify_chain_bytes(trail_bytes: bytes) -> Optional[str]:
    """Re-verify the hash chain from the exported JSONL bytes.

    Returns None if intact, else a human-readable error string. Uses only
    stdlib — mirrors :meth:`AuditTrail.verify_chain` without requiring
    vaara imports, so the standalone verifier can run it.
    """
    prev_hash = ""
    line_iter = (ln for ln in trail_bytes.splitlines() if ln.strip())
    for idx, raw in enumerate(line_iter):
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as e:
            return f"record {idx}: malformed JSON ({e})"
        if rec.get("previous_hash", "") != prev_hash:
            return (
                f"record {idx}: previous_hash mismatch "
                f"(expected {prev_hash!r}, got {rec.get('previous_hash')!r})"
            )
        # Recompute hash over the canonical content. Mirrors
        # AuditRecord.compute_hash — keep the two in lockstep.
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
        # Records with no chain_version key are legacy v1 and omit both, so
        # pre-v0.47 trails re-verify unchanged.
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


def _raw_ed25519_bytes(loaded: "Ed25519PublicKey") -> bytes:
    """Raw 32-byte encoding of an Ed25519 public key, version-tolerant."""
    if hasattr(loaded, "public_bytes_raw"):
        return loaded.public_bytes_raw()
    return loaded.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _member_fingerprint(raw_pubkey: bytes) -> str:
    """32-hex-char fingerprint of a raw public key (mirrors the signer)."""
    return hashlib.sha256(raw_pubkey).hexdigest()[:32]


def _verify_threshold(
    *,
    manifest: dict,
    to_verify: bytes,
    sigs: dict[str, bytes],
    pubkeys: dict[str, bytes],
) -> list[str]:
    """Verify a k-of-n threshold export. Returns a list of error strings.

    Requires at least ``threshold_k`` distinct authorized signers to carry
    a valid signature over ``to_verify``. The authorized fingerprint list
    lives in the signed manifest, and each stored public key is bound to
    its manifest fingerprint, so a substituted pubkey is rejected and an
    unauthorized extra signature is ignored rather than counted.
    """
    errors: list[str] = []
    member_algorithm = manifest.get("member_algorithm")
    k = manifest.get("threshold_k")
    n = manifest.get("signers_n")
    fps = manifest.get("signer_fingerprints")

    if not isinstance(k, int) or k < 1:
        return [f"invalid threshold_k: {k!r}"]
    if not isinstance(fps, list) or not fps or not all(isinstance(f, str) for f in fps):
        return ["manifest signer_fingerprints missing or malformed"]
    if len(set(fps)) != len(fps):
        errors.append("signer_fingerprints contains duplicates")
    if n != len(fps):
        errors.append(
            f"signers_n ({n}) does not match signer_fingerprints length ({len(fps)})"
        )
    if k > len(fps):
        return errors + [f"threshold_k ({k}) exceeds n ({len(fps)})"]
    if member_algorithm not in ("Ed25519", "ML-DSA-65"):
        return errors + [f"unsupported member_algorithm: {member_algorithm!r}"]

    # Bind each authorized pubkey to its manifest fingerprint.
    authorized: dict[str, bytes] = {}
    for fp in fps:
        raw = pubkeys.get(fp)
        if raw is None:
            errors.append(f"missing pubkeys/ entry for authorized signer {fp}")
            continue
        if member_algorithm == "Ed25519":
            try:
                loaded = serialization.load_pem_public_key(raw)
            except (ValueError, UnsupportedAlgorithm) as e:
                errors.append(f"pubkey for {fp} could not be parsed: {e}")
                continue
            if not isinstance(loaded, Ed25519PublicKey):
                errors.append(f"pubkey for {fp} is not Ed25519")
                continue
            raw_norm = _raw_ed25519_bytes(loaded)
        else:
            raw_norm = raw
        if _member_fingerprint(raw_norm) != fp:
            errors.append(
                f"pubkey for {fp} does not match its fingerprint (substituted key)"
            )
            continue
        authorized[fp] = raw_norm

    mldsa_cls = None
    if member_algorithm == "ML-DSA-65":
        try:
            from vaara.audit.signer import MLDSA65Verifier as mldsa_cls  # type: ignore
        except ImportError as exc:
            return errors + [f"ML-DSA-65 verify requires the pq extra: {exc}"]

    # Count distinct authorized signers with a valid signature.
    valid: set[str] = set()
    for fp, sig in sigs.items():
        raw_norm = authorized.get(fp)
        if raw_norm is None:
            continue  # unknown or unauthorized signer; not counted
        if member_algorithm == "Ed25519":
            try:
                Ed25519PublicKey.from_public_bytes(raw_norm).verify(sig, to_verify)
                valid.add(fp)
            except (InvalidSignature, ValueError):
                pass
        elif mldsa_cls(raw_norm).verify(to_verify, sig):
            valid.add(fp)

    if len(valid) < k:
        errors.append(
            f"threshold not met: {len(valid)} valid signature(s) from the "
            f"authorized set, need at least {k} of {len(fps)}"
        )
    return errors


def verify_signed(
    zip_path: Union[str, Path],
    public_key: Optional[Union[str, Path, bytes, "_PK"]] = None,
) -> VerifyResult:
    """Verify a signed audit-trail export.

    Checks, in order:

    1. Zip contains the required files (``trail.jsonl``, ``manifest.json``,
       ``trail.sig``, and optionally ``signer_pubkey.pem``).
    2. Manifest parses as JSON and has the expected schema version.
    3. ``manifest.trail_sha256`` matches the actual SHA-256 of
       ``trail.jsonl``.
    4. Ed25519 signature over ``sha256(trail.jsonl || manifest.json)``
       verifies under ``public_key`` (if given) or the embedded
       ``signer_pubkey.pem``.
    5. The hash chain re-verifies from the exported bytes.
    6. Manifest endpoints (``first_hash``, ``last_hash``, ``record_count``)
       match the exported JSONL.

    Args:
        zip_path: Path to the zip produced by :func:`export_signed`.
        public_key: Ed25519 public key — Path, PEM bytes, or loaded key
            object. If omitted, uses ``signer_pubkey.pem`` from inside
            the zip (convenient for local verification; for production,
            always pass a trusted public key out-of-band).

    Returns:
        VerifyResult. On success, ``ok=True`` and ``errors`` is empty.
    """
    _require_crypto()

    zip_path = Path(zip_path)
    errors: list[str] = []
    manifest: Optional[dict] = None

    if not zip_path.exists():
        return VerifyResult(ok=False, errors=[f"file not found: {zip_path}"])

    threshold_sigs: dict[str, bytes] = {}
    threshold_pubkeys: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            # trail.jsonl + manifest.json are required for every export;
            # the signature surface differs (single trail.sig vs threshold
            # sigs/ + pubkeys/) and is resolved after the manifest is read.
            missing = _CORE_FILES - names
            if missing:
                return VerifyResult(
                    ok=False,
                    errors=[f"missing files in zip: {sorted(missing)}"],
                )
            trail_bytes = zf.read("trail.jsonl")
            manifest_bytes = zf.read("manifest.json")
            signature = zf.read("trail.sig") if "trail.sig" in names else None
            embedded_pubkey_pem = (
                zf.read("signer_pubkey.pem")
                if "signer_pubkey.pem" in names
                else None
            )
            embedded_pubkey_bin = (
                zf.read("signer_pubkey.bin")
                if "signer_pubkey.bin" in names
                else None
            )
            for name in names:
                if name.startswith("sigs/") and name.endswith(".sig"):
                    threshold_sigs[name[len("sigs/"):-len(".sig")]] = zf.read(name)
                elif name.startswith("pubkeys/") and name != "pubkeys/":
                    stem = name[len("pubkeys/"):]
                    fp = stem.rsplit(".", 1)[0]
                    threshold_pubkeys[fp] = zf.read(name)
    except zipfile.BadZipFile as e:
        return VerifyResult(ok=False, errors=[f"corrupt zip: {e}"])

    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as e:
        return VerifyResult(ok=False, errors=[f"manifest.json is not valid JSON: {e}"])

    if manifest.get("schema_version") != "1":
        errors.append(
            f"unsupported schema_version: {manifest.get('schema_version')!r}"
        )

    actual_sha = hashlib.sha256(trail_bytes).hexdigest()
    if manifest.get("trail_sha256") != actual_sha:
        errors.append(
            "trail.jsonl SHA-256 does not match manifest.trail_sha256 "
            "(trail was tampered after export)"
        )

    algorithm = manifest.get("signature_algorithm", "Ed25519")
    to_verify = hashlib.sha256(trail_bytes + manifest_bytes).digest()

    if algorithm.startswith(_THRESHOLD_PREFIX):
        errors.extend(
            _verify_threshold(
                manifest=manifest,
                to_verify=to_verify,
                sigs=threshold_sigs,
                pubkeys=threshold_pubkeys,
            )
        )
    elif algorithm == "Ed25519":
        if public_key is None:
            if embedded_pubkey_pem is None:
                return VerifyResult(
                    ok=False,
                    errors=errors + [
                        "no public_key argument and no signer_pubkey.pem in zip",
                    ],
                    manifest=manifest,
                )
            pk = _load_public_key(embedded_pubkey_pem)
        else:
            pk = _load_public_key(public_key)
        if signature is None:
            errors.append("missing trail.sig in zip")
        else:
            try:
                pk.verify(signature, to_verify)
            except InvalidSignature:
                errors.append(
                    "Ed25519 signature did not verify "
                    "(wrong public key, tampered manifest, or tampered trail)"
                )
    elif algorithm == "ML-DSA-65":
        try:
            from vaara.audit.signer import MLDSA65Verifier
        except ImportError as exc:
            return VerifyResult(
                ok=False,
                errors=errors + [
                    f"ML-DSA-65 verify requires the pq extra: {exc}",
                ],
                manifest=manifest,
            )
        if public_key is None:
            if embedded_pubkey_bin is None:
                return VerifyResult(
                    ok=False,
                    errors=errors + [
                        "no public_key argument and no signer_pubkey.bin in zip",
                    ],
                    manifest=manifest,
                )
            pk_bytes = embedded_pubkey_bin
        elif isinstance(public_key, (bytes, bytearray)):
            pk_bytes = bytes(public_key)
        else:
            return VerifyResult(
                ok=False,
                errors=errors + [
                    "ML-DSA-65 verify requires public_key as raw bytes",
                ],
                manifest=manifest,
            )
        if signature is None:
            errors.append("missing trail.sig in zip")
        else:
            verifier = MLDSA65Verifier(pk_bytes)
            if not verifier.verify(to_verify, signature):
                errors.append(
                    "ML-DSA-65 signature did not verify "
                    "(wrong public key, tampered manifest, or tampered trail)"
                )
    else:
        errors.append(f"unknown signature_algorithm: {algorithm!r}")

    chain_error = _verify_chain_bytes(trail_bytes)
    if chain_error is not None:
        errors.append(f"hash chain check failed: {chain_error}")

    lines = [ln for ln in trail_bytes.splitlines() if ln.strip()]
    if manifest.get("record_count") != len(lines):
        errors.append(
            f"record_count mismatch: manifest={manifest.get('record_count')}, "
            f"actual={len(lines)}"
        )
    if lines:
        try:
            first = json.loads(lines[0]).get("record_hash", "")
            last = json.loads(lines[-1]).get("record_hash", "")
            if manifest.get("first_hash") != first:
                errors.append("first_hash in manifest does not match first record")
            if manifest.get("last_hash") != last:
                errors.append("last_hash in manifest does not match last record")
        except json.JSONDecodeError:
            errors.append("could not parse first/last trail record")

    return VerifyResult(ok=not errors, errors=errors, manifest=manifest)
