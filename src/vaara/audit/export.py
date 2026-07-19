# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Signed, regulator-handoff export of the audit trail.

Produces a single zip containing:

- ``trail.jsonl``   — canonical event log (one JSON record per line).
- ``manifest.json`` — schema version, record count, first/last hashes,
  signer public-key fingerprint, vaara version, UTC timestamp, and
  whether the hash chain verified intact at export time.
- ``trail.sig``     — Ed25519 detached signature over
  ``sha256(trail.jsonl_bytes || manifest.json_bytes)``.

A regulator verifies the package with ``vaara.audit.verify.verify_signed``
or the standalone ``scripts/verify_vaara_trail.py`` — no Vaara install
required by the verifier.

Requires ``pip install vaara[export]`` (pulls ``cryptography``). The
core library stays zero-dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

try:
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

if TYPE_CHECKING:
    from vaara.attestation import RevocationRegistry
    from vaara.audit.signer import Signer
    from vaara.audit.trail import AuditTrail

logger = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = "1"
_INSTALL_HINT = (
    "Signed export requires the 'cryptography' package. "
    "Install with: pip install 'vaara[export]'"
)


@dataclass(frozen=True)
class ExportResult:
    """Return value of :func:`export_signed`.

    ``path``: path to the written zip.
    ``manifest``: the manifest dict embedded in the zip.
    ``chain_intact``: whether the hash chain verified at export time.
    """
    path: Path
    manifest: dict
    chain_intact: bool


def _require_crypto() -> None:
    if not _HAS_CRYPTO:
        raise ImportError(_INSTALL_HINT)


def _load_private_key(
    key: Union[str, Path, bytes, "Ed25519PrivateKey"],
) -> "Ed25519PrivateKey":
    """Accept a PEM path, PEM bytes, or an already-loaded key object."""
    _require_crypto()
    if isinstance(key, Ed25519PrivateKey):
        return key
    if isinstance(key, (str, Path)):
        key = Path(key).read_bytes()
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(
            f"signer_key must be a Path, str, bytes, or Ed25519PrivateKey — got {type(key).__name__}"
        )
    try:
        loaded = serialization.load_pem_private_key(bytes(key), password=None)
    except (ValueError, UnsupportedAlgorithm) as e:
        # cryptography 47.0+ may raise UnsupportedAlgorithm where 46.x raised
        # ValueError; collapse both to ValueError so callers see one shape.
        raise ValueError(f"signer_key could not be parsed as a PEM private key: {e}") from e
    if not isinstance(loaded, Ed25519PrivateKey):
        raise ValueError("signer_key must be an Ed25519 private key")
    return loaded


def _pubkey_fingerprint(public_key: "Ed25519PublicKey") -> str:
    """SHA-256 fingerprint (hex, first 16 bytes) of the raw public key.

    Matches what ``vaara keygen`` prints so operators can cross-check which
    key signed a given export.
    """
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()[:32]


def _build_manifest(
    trail_jsonl_bytes: bytes,
    record_count: int,
    first_hash: str,
    last_hash: str,
    chain_intact: bool,
    *,
    signer_fingerprint: str,
    signature_algorithm: str,
    vaara_version: str,
    agent_id: str = "",
) -> dict:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "vaara_version": vaara_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record_count": record_count,
        "first_hash": first_hash,
        "last_hash": last_hash,
        "chain_intact_at_export": chain_intact,
        "trail_sha256": hashlib.sha256(trail_jsonl_bytes).hexdigest(),
        "signer_pubkey_fingerprint": signer_fingerprint,
        "signature_algorithm": signature_algorithm,
        "signature_over": "sha256(trail.jsonl_bytes || manifest.json_bytes_without_signature)",
        "agent_id": agent_id or "",
    }


def _snapshot_trail(
    trail: "AuditTrail", out_path: Path
) -> tuple[bytes, int, str, str, bool]:
    """Snapshot a trail to bytes plus its endpoints and chain state.

    Shared by the single-signer and threshold export paths. Returns
    ``(trail_bytes, record_count, first_hash, last_hash, chain_intact)``.
    """
    from vaara.audit.trail import AuditTrail

    if not isinstance(trail, AuditTrail):
        raise TypeError(
            f"trail must be an AuditTrail instance, got {type(trail).__name__}"
        )
    if not out_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {out_path.parent}"
        )

    # Snapshot under the trail's own lock via its existing exporter. Use a
    # temp path next to the final zip so large trails stream to disk.
    tmp_jsonl = out_path.with_suffix(out_path.suffix + ".tmp.jsonl")
    try:
        record_count = trail.export_jsonl(tmp_jsonl)
        trail_bytes = tmp_jsonl.read_bytes()
    finally:
        tmp_jsonl.unlink(missing_ok=True)

    chain_error = trail.verify_chain()
    chain_intact = chain_error is None
    first_hash = ""
    last_hash = ""
    if record_count > 0:
        lines = trail_bytes.splitlines()
        if lines:
            first_hash = json.loads(lines[0]).get("record_hash", "")
            last_hash = json.loads(lines[-1]).get("record_hash", "")

    return trail_bytes, record_count, first_hash, last_hash, chain_intact


def export_signed(
    trail: "AuditTrail",
    out_path: Union[str, Path],
    signer_key: Union[str, Path, bytes, "Ed25519PrivateKey", None] = None,
    agent_id: str = "",
    *,
    signer: Optional["Signer"] = None,
    revocation: Optional["RevocationRegistry"] = None,
) -> ExportResult:
    """Produce a signed, regulator-handoff zip of the audit trail.

    Args:
        trail: The :class:`AuditTrail` to export.
        out_path: Where to write the zip file. Parent directory must exist.
        signer_key: Ed25519 private key — accepts a PEM path, PEM bytes, or
            a loaded ``Ed25519PrivateKey`` object. Backward-compatible
            shortcut for the default Ed25519 path. Mutually exclusive
            with ``signer``.
        agent_id: Optional scope hint written to the manifest. Does not
            filter the exported records (export is whole-trail).
        signer: A ``vaara.audit.signer.Signer`` instance — currently
            ``Ed25519Signer`` or ``MLDSA65Signer`` (v0.14.0). Set this
            when promoting an operator's retention horizon past the
            credible quantum threshold. Manifest carries the algorithm
            identifier so verifiers dispatch automatically.
        revocation: An optional
            ``vaara.attestation.RevocationRegistry``. When supplied, the
            registry's canonical bytes are written to a ``revocation.json``
            member and its digest plus entry count are pinned into the
            signed manifest (``revocation.registry_sha256``). This makes the
            revocation state at export time part of the tamper-evident
            bundle, so a regulator recomputes every receipt's
            revocation-in-time verdict against the exact registry the
            exporter used. Omitting it produces a byte-identical manifest to
            earlier versions.

    Returns:
        ExportResult with the output path, manifest dict, and chain status.

    Raises:
        ImportError: If the ``cryptography`` package is not installed
            for the Ed25519 path, or ``dilithium-py`` for the ML-DSA-65
            path (install with ``pip install 'vaara[pq]'``).
        FileNotFoundError: If ``out_path`` parent directory does not exist.
        ValueError: If both ``signer_key`` and ``signer`` are supplied,
            or neither is.
    """
    from vaara.audit.signer import Ed25519Signer

    if signer is not None and signer_key is not None:
        raise ValueError(
            "export_signed: pass `signer` or `signer_key`, not both"
        )
    if signer is None and signer_key is None:
        raise ValueError(
            "export_signed: one of `signer` or `signer_key` is required"
        )

    if signer is None:
        _require_crypto()
        signer = Ed25519Signer(_load_private_key(signer_key))

    # Local import avoids a package-level dep on the trail module for tooling
    # that only imports the export helpers.
    from vaara import __version__ as vaara_version

    out_path = Path(out_path)
    trail_bytes, record_count, first_hash, last_hash, chain_intact = (
        _snapshot_trail(trail, out_path)
    )

    manifest = _build_manifest(
        trail_jsonl_bytes=trail_bytes,
        record_count=record_count,
        first_hash=first_hash,
        last_hash=last_hash,
        chain_intact=chain_intact,
        signer_fingerprint=signer.public_key_fingerprint()[:32],
        signature_algorithm=signer.algorithm,
        vaara_version=vaara_version,
        agent_id=agent_id,
    )

    revocation_bytes: Optional[bytes] = None
    if revocation is not None:
        revocation_bytes = revocation.canonical_bytes()
        manifest["revocation"] = {
            "entry_count": len(revocation),
            "registry_sha256": revocation.digest(),
        }

    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")

    to_sign = hashlib.sha256(trail_bytes + manifest_bytes).digest()
    signature = signer.sign(to_sign)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trail.jsonl", trail_bytes)
        zf.writestr("manifest.json", manifest_bytes)
        zf.writestr("trail.sig", signature)
        if revocation_bytes is not None:
            # Bound to the signed manifest through registry_sha256, so the
            # file's integrity is covered transitively by trail.sig.
            zf.writestr("revocation.json", revocation_bytes)
        if signer.algorithm == "Ed25519":
            # Preserve the existing PEM-encoded public-key file for back-
            # compatible verification by older Vaara clients.
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
            pub = Ed25519PublicKey.from_public_bytes(signer.public_key_bytes())
            zf.writestr("signer_pubkey.pem", pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ))
        else:
            zf.writestr("signer_pubkey.bin", signer.public_key_bytes())

    logger.info(
        "Exported signed trail: %s (%d records, chain_intact=%s, fingerprint=%s)",
        out_path,
        record_count,
        chain_intact,
        manifest["signer_pubkey_fingerprint"],
    )

    return ExportResult(path=out_path, manifest=manifest, chain_intact=chain_intact)


_THRESHOLD_PREFIX = "threshold-"


def _short_fp(raw_pubkey: bytes) -> str:
    """32-hex-char (16-byte) fingerprint of a raw public key.

    Matches the ``signer_pubkey_fingerprint`` convention used elsewhere in
    the manifest (``public_key_fingerprint()[:32]``).
    """
    return hashlib.sha256(raw_pubkey).hexdigest()[:32]


def _pubkey_member_bytes(raw_pubkey: bytes, member_algorithm: str) -> tuple[str, bytes]:
    """Serialize an authorized member public key for the zip.

    Returns ``(filename_ext, file_bytes)``: ``("pem", <PEM>)`` for Ed25519
    so any cryptography client can load it, ``("bin", <raw>)`` otherwise.
    """
    if member_algorithm == "Ed25519":
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        pub = Ed25519PublicKey.from_public_bytes(raw_pubkey)
        return "pem", pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    return "bin", raw_pubkey


def export_signed_threshold(
    trail: "AuditTrail",
    out_path: Union[str, Path],
    *,
    signers: list["Signer"],
    threshold_k: int,
    authorized_pubkeys: Optional[list[bytes]] = None,
    member_algorithm: str = "Ed25519",
    agent_id: str = "",
) -> ExportResult:
    """Produce a k-of-n threshold-signed audit-trail export.

    Unlike :func:`export_signed`, which carries one signature, this writes
    one detached signature per available custodian. Verification requires
    at least ``threshold_k`` valid signatures from the authorized set, so
    no single key-holder can issue or forge a trail on their own.

    See ``docs/design/threshold-signing-spec.md``.

    Args:
        trail: The :class:`AuditTrail` to export.
        out_path: Where to write the zip. Parent directory must exist.
        signers: The custodian ``Signer`` instances available to sign now.
            Must number at least ``threshold_k`` and each must be a member
            of the authorized set. All share ``member_algorithm``.
        threshold_k: The quorum. ``1 <= threshold_k <= n``.
        authorized_pubkeys: Raw public-key bytes of all ``n`` authorized
            custodians. If omitted, the authorized set is exactly the
            ``signers`` provided (the common "everyone signs" case) and
            ``n == len(signers)``.
        member_algorithm: Signature algorithm shared by every member
            (``"Ed25519"`` by default; ``"ML-DSA-65"`` reserved for the PQ
            extra). Mixed-algorithm sets are not supported.
        agent_id: Optional scope hint written to the manifest. Does not
            filter the exported records (export is whole-trail).

    Returns:
        ExportResult with the output path, manifest dict, and chain status.

    Raises:
        ValueError: If ``threshold_k`` is out of range, fewer than
            ``threshold_k`` signers are supplied, a signer is not in the
            authorized set, or any signer's algorithm differs from
            ``member_algorithm``.
        FileNotFoundError: If ``out_path`` parent directory does not exist.
    """
    if not signers:
        raise ValueError("export_signed_threshold: `signers` must be non-empty")
    for s in signers:
        if s.algorithm != member_algorithm:
            raise ValueError(
                "export_signed_threshold: signer algorithm "
                f"{s.algorithm!r} != member_algorithm {member_algorithm!r}"
            )

    # Resolve the authorized set (the n custodians). Default: everyone
    # supplied is a member. De-duplicate by fingerprint, preserving order.
    authorized_raw = (
        [s.public_key_bytes() for s in signers]
        if authorized_pubkeys is None
        else list(authorized_pubkeys)
    )
    authorized_by_fp: dict[str, bytes] = {}
    for raw in authorized_raw:
        authorized_by_fp.setdefault(_short_fp(raw), raw)
    n = len(authorized_by_fp)

    if not isinstance(threshold_k, int) or threshold_k < 1:
        raise ValueError("export_signed_threshold: threshold_k must be >= 1")
    if threshold_k > n:
        raise ValueError(
            f"export_signed_threshold: threshold_k ({threshold_k}) exceeds n ({n})"
        )

    # Collect signing custodians, de-duplicated by fingerprint, each a
    # member of the authorized set. A custodian counts once.
    signing_by_fp: dict[str, "Signer"] = {}
    for s in signers:
        fp = s.public_key_fingerprint()[:32]
        if fp not in authorized_by_fp:
            raise ValueError(
                f"export_signed_threshold: signer {fp} is not in the authorized set"
            )
        signing_by_fp.setdefault(fp, s)

    if len(signing_by_fp) < threshold_k:
        raise ValueError(
            f"export_signed_threshold: have {len(signing_by_fp)} distinct "
            f"signer(s), need at least threshold_k ({threshold_k})"
        )

    return _write_threshold_zip(
        trail=trail,
        out_path=Path(out_path),
        signing_by_fp=signing_by_fp,
        authorized_by_fp=authorized_by_fp,
        threshold_k=threshold_k,
        n=n,
        member_algorithm=member_algorithm,
        agent_id=agent_id,
    )


def _write_threshold_zip(
    *,
    trail: "AuditTrail",
    out_path: Path,
    signing_by_fp: dict,
    authorized_by_fp: dict,
    threshold_k: int,
    n: int,
    member_algorithm: str,
    agent_id: str,
) -> ExportResult:
    """Build the manifest, collect member signatures, write the zip."""
    from vaara import __version__ as vaara_version

    trail_bytes, record_count, first_hash, last_hash, chain_intact = (
        _snapshot_trail(trail, out_path)
    )

    # The authorized fingerprint list is part of the signed manifest, so k,
    # n, and the membership set cannot be altered without invalidating every
    # member signature. Sort for a stable, canonical set identity.
    signer_fingerprints = sorted(authorized_by_fp)
    set_fingerprint = hashlib.sha256(
        "".join(signer_fingerprints).encode("ascii")
    ).hexdigest()[:32]

    manifest = _build_manifest(
        trail_jsonl_bytes=trail_bytes,
        record_count=record_count,
        first_hash=first_hash,
        last_hash=last_hash,
        chain_intact=chain_intact,
        signer_fingerprint=set_fingerprint,
        signature_algorithm=f"{_THRESHOLD_PREFIX}{member_algorithm}",
        vaara_version=vaara_version,
        agent_id=agent_id,
    )
    manifest["threshold_k"] = threshold_k
    manifest["signers_n"] = n
    manifest["member_algorithm"] = member_algorithm
    manifest["signer_fingerprints"] = signer_fingerprints
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")

    # Every custodian signs the identical digest over the trail and the
    # manifest (which now pins k/n and the authorized set).
    to_sign = hashlib.sha256(trail_bytes + manifest_bytes).digest()

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trail.jsonl", trail_bytes)
        zf.writestr("manifest.json", manifest_bytes)
        for fp, s in signing_by_fp.items():
            zf.writestr(f"sigs/{fp}.sig", s.sign(to_sign))
        for fp, raw in authorized_by_fp.items():
            ext, body = _pubkey_member_bytes(raw, member_algorithm)
            zf.writestr(f"pubkeys/{fp}.{ext}", body)

    logger.info(
        "Exported threshold-signed trail: %s (%d records, chain_intact=%s, "
        "%d-of-%d, set=%s)",
        out_path,
        record_count,
        chain_intact,
        threshold_k,
        n,
        set_fingerprint,
    )

    return ExportResult(path=out_path, manifest=manifest, chain_intact=chain_intact)
