# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
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
import math
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


class _Refusal(ValueError):
    """JSON that two readers could read differently, or that has no canonical form."""


def _refuse_repeats(pairs: list[tuple[str, object]]) -> dict:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise _Refusal(f'duplicate member "{key}"')
        seen.add(key)
    return dict(pairs)


def _refuse_constant(name: str) -> None:
    raise _Refusal(f"non-finite number {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _Refusal(f"non-finite number {text}")
    return value


def _loads(raw: Union[bytes, str]) -> dict:
    """``json.loads`` for a trail line or the manifest: a JSON object, or a refusal.

    A member repeated in one object reads as its last value to Python and its
    first to a first-wins parser, and the hash is over only one of them. NaN,
    Infinity and a literal that overflows to one cannot be re-encoded for the
    hash at all. A line that parses to an array, number, string, boolean or
    null is no record: it has no members to read. All are refused as
    ``_Refusal`` wherever the export is read. Mirrored in
    ``scripts/verify_vaara_trail.py``.
    """
    value = json.loads(raw, object_pairs_hook=_refuse_repeats,
                       parse_constant=_refuse_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise _Refusal("not a JSON object")
    return value


_UNREADABLE = (json.JSONDecodeError, _Refusal)


def _data_of(rec: dict) -> dict:
    """A record's ``data`` when it is an object, else empty. The hash covers
    whatever is there, so a reader that wants fields cannot trust its shape."""
    data = rec.get("data")
    return data if isinstance(data, dict) else {}


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
    key_lifecycle: list[dict] = field(default_factory=list)
    """Custodian key-lifecycle records found in the trail (rotations,
    revocations, additions), in chain order. Informational: surfaced so a
    reviewer sees the custodian set's history inline. Does not affect ``ok``.
    """
    erased: list[dict] = field(default_factory=list)
    """Records whose data was erased (``vaara trail erase``): the slot keeps
    its stored hash, so the link verifies, and a later ``record_erased``
    record names it by id and that hash. What the record held cannot be
    checked. Each entry: ``index``, ``record_id``, ``record_hash``,
    ``erased_by_index``. The chain is not intact, but not broken either:
    does not affect ``ok``.
    """
    erasures: list[dict] = field(default_factory=list)
    """One entry per ``record_erased`` record, from :func:`check_erasures`:
    the legal ground, who erased, and the state of the eraser's signature.
    ``bad_signature`` and ``malformed`` fail ``ok``; so does ``unsigned``
    when an ``erasure_key`` was given.
    """
    cage_lifecycle: list[dict] = field(default_factory=list)
    """Cage launch starts and stops (``vaara cage run`` / ``stop``, ``vaara
    run``), in chain order. A stop that names a record which is not the start
    of the same driver and launch, or a start that comes after it, is an
    error: the trail never writes one, so it was written past the trail.
    """
    cage_open: list[dict] = field(default_factory=list)
    """Starts in this export that no stop closes: still running when the
    export was taken, or ended without a record. Does not affect ``ok``."""
    cage_unpaired_stops: list[dict] = field(default_factory=list)
    """Stops whose start is not in this export (an export of a later range,
    or a launch started outside Vaara). Does not affect ``ok``."""
    unrecorded_windows: list[dict] = field(default_factory=list)
    """Stretches where the audit store refused writes (``trail_not_recording``
    records), in chain order. Each entry: ``index``, ``unrecorded_writes`` (a
    lower bound), ``first_failure_utc``, ``last_failure_utc``, ``stage``,
    ``error``, ``resumed_record_id`` and, when that record is in the export,
    ``resumed_index`` and ``after_index`` (the record before it: the window lies
    between the two, which sit side by side on the chain). Calls in the window
    ran with no record. The chain is whole around it, so this does not affect
    ``ok``; a mark whose fields are nonsense does."""
    rotation_seams: list[dict] = field(default_factory=list)
    """Retention rotations (``trail_rotated`` records, written by ``vaara
    trail rotate`` before it purges), in chain order, from
    :func:`check_rotations`. ``begins_here`` is True for the one whose
    ``last_hash`` the first record of the export links to and whose signature
    checks out: the export then begins at a rotation seam, and the records
    before it are in the archive named by ``archive_sha256``. A first record
    that links to no such record is a chain break. ``archive_check`` says what
    was done when an archive was given: ``verified``, ``digest_only``
    (sealed), or why it failed. ``bad_signature`` and ``malformed`` fail
    ``ok`` wherever the record sits."""


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


#: Mirrors vaara.audit.trail.ERASED_DATA_KEY; kept here so this module stays
#: free of vaara imports.
_ERASED_DATA_KEY = "_vaara_erased"


def _erasure_pointer(rec: dict) -> Optional[str]:
    """The erasure record id a tombstoned record points at, else None."""
    data = rec.get("data")
    if not isinstance(data, dict) or set(data) != {_ERASED_DATA_KEY}:
        return None
    inner = data[_ERASED_DATA_KEY]
    if not isinstance(inner, dict):
        return None
    pointer = inner.get("erasure_record_id")
    return pointer if isinstance(pointer, str) and pointer else None


ERASURE_RECORD_TYPE = "vaara.record-erasure/v1"
_ERASURE_SIGNATURE_KEYS = frozenset({"type", "alg", "signature"})
#: Signature states that fail verification.
_ERASURE_FINDINGS = frozenset({"bad_signature", "malformed"})


def _jcs_string(value: str) -> str:
    # RFC 8785 escapes only the quote, the backslash and C0 controls, with
    # the same short forms json uses; everything else is raw UTF-8.
    return json.dumps(value, ensure_ascii=False)


def _jcs_fields(fields: dict) -> bytes:
    """RFC 8785 (JCS) over an object of strings, integers and null, written out
    by hand so this module stays on the standard library. Field names are
    ASCII, so the JCS key order is the plain sort order."""
    parts = []
    for name in sorted(fields):
        value = fields[name]
        if value is None:
            encoded = "null"
        elif isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"{name} is not a string or integer")
        elif isinstance(value, int):
            encoded = str(value)
        else:
            encoded = _jcs_string(value)
        parts.append(f"{_jcs_string(name)}:{encoded}")
    return ("{" + ",".join(parts) + "}").encode("utf-8")


def erasure_payload(data: dict) -> bytes:
    """The exact bytes an eraser's signature covers, from a ``record_erased``
    record's ``data``. JCS, see :func:`_jcs_fields`.
    """
    return _jcs_fields({
        "type": ERASURE_RECORD_TYPE,
        "erasedRecordId": data.get("erased_record_id"),
        "erasedRecordHash": data.get("erased_record_hash"),
        "erasedSeq": data.get("erased_seq"),
        "erasedEventType": data.get("erased_event_type"),
        "erasedActionId": data.get("erased_action_id"),
        "reason": data.get("reason"),
        "legalGround": data.get("legal_ground"),
        "erasedBy": data.get("erased_by"),
        "erasedAt": data.get("erased_at"),
    })


ROTATION_RECORD_TYPE = "vaara.trail-rotation/v1"
_ROTATION_SIGNATURE_KEYS = frozenset({"type", "alg", "signature"})
_ROTATION_ARCHIVE_KINDS = frozenset({"zip", "sealed"})
_ROTATION_TEXT_FIELDS = ("last_hash", "archive_sha256", "boundary_utc", "rotated_at",
                         "signer_fingerprint")
#: Signature states that fail verification wherever the record sits.
_ROTATION_FINDINGS = frozenset({"bad_signature", "malformed"})
#: States in which a rotation record accounts for the start of the trail.
_ROTATION_ACCEPTED = frozenset({"signed", "unpinned"})


def rotation_payload(data: dict) -> bytes:
    """The exact bytes a rotation signature covers, from a ``trail_rotated``
    record's ``data``: the hash the first survivor links to, how many records
    went, the archive's digest, count and kind, the cut-off and signing times,
    and the fingerprint of the signing key. JCS, see :func:`_jcs_fields`.
    """
    return _jcs_fields({
        "type": ROTATION_RECORD_TYPE,
        "lastHash": data.get("last_hash"),
        "purgedRecords": data.get("purged_records"),
        "archiveSha256": data.get("archive_sha256"),
        "archiveRecords": data.get("archive_records"),
        "archiveKind": data.get("archive_kind"),
        "boundaryUtc": data.get("boundary_utc"),
        "rotatedAt": data.get("rotated_at"),
        "signerFingerprint": data.get("signer_fingerprint"),
    })


def _load_erasure_key(key: Union[str, Path, bytes, object]) -> object:
    """An EC P-256 or RSA public key from PEM bytes, a path, or a key object."""
    _require_crypto()
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    if isinstance(key, (str, Path)):
        key = Path(key).read_bytes()
    if isinstance(key, (bytes, bytearray)):
        try:
            key = serialization.load_pem_public_key(bytes(key))
        except (ValueError, UnsupportedAlgorithm) as e:
            raise ValueError(f"erasure key could not be parsed as a PEM public key: {e}") from e
    if isinstance(key, ec.EllipticCurvePublicKey) and isinstance(key.curve, ec.SECP256R1):
        return key
    if isinstance(key, rsa.RSAPublicKey):
        return key
    raise ValueError("erasure key must be an EC P-256 or RSA public key")


def _verify_erasure_signature(payload: bytes, alg: str, signature_hex: str,
                              key: object) -> bool:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    try:
        raw = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    try:
        if alg == "ES256" and isinstance(key, ec.EllipticCurvePublicKey):
            if len(raw) != 64:
                return False
            der = encode_dss_signature(int.from_bytes(raw[:32], "big"),
                                       int.from_bytes(raw[32:], "big"))
            key.verify(der, payload, ec.ECDSA(hashes.SHA256()))
            return True
        if alg == "RS256" and isinstance(key, rsa.RSAPublicKey):
            key.verify(raw, payload, padding.PKCS1v15(), hashes.SHA256())
            return True
    except InvalidSignature:
        return False
    return False


def _erasure_state(data: dict, key: Optional[object]) -> str:
    block = data.get("eraser_signature")
    if block is None:
        return "unsigned"
    if (not isinstance(block, dict) or set(block) != _ERASURE_SIGNATURE_KEYS
            or block.get("type") != ERASURE_RECORD_TYPE
            or not all(isinstance(block[k], str) for k in _ERASURE_SIGNATURE_KEYS)
            or not isinstance(data.get("erased_by"), str) or not data["erased_by"]):
        return "malformed"
    if key is None:
        return "unpinned"
    try:
        payload = erasure_payload(data)
    except ValueError:
        return "malformed"
    if not _verify_erasure_signature(payload, block["alg"], block["signature"], key):
        return "bad_signature"
    return "signed"


def check_erasures(records: list[dict], public_key: Optional[object] = None) -> list[dict]:
    """Judge the eraser's signature on every ``record_erased`` record.

    ``records`` are trail records as dicts (``record_id``, ``event_type``,
    ``data``). ``public_key`` is the eraser's EC P-256 or RSA public key, held
    out of band; the record never carries one. Each erasure lands in one state:

    * ``signed``: verifies under ``public_key`` over every signed field.
    * ``unsigned``: no signature (every erasure written before signing existed).
    * ``unpinned``: signed, but no key was given, so nobody can say whose.
    * ``bad_signature``: does not verify under ``public_key``.
    * ``malformed``: the block is not the closed ``type``/``alg``/``signature``
      shape, or the record names no eraser.

    This judges who signed. Whether the record itself was altered is the
    chain walk's job.
    """
    key = _load_erasure_key(public_key) if public_key is not None else None
    out: list[dict] = []
    for index, rec in enumerate(records):
        if rec.get("event_type") != "record_erased":
            continue
        data = _data_of(rec)
        out.append({
            "index": index,
            "erasure_record_id": rec.get("record_id"),
            "erased_record_id": data.get("erased_record_id"),
            "legal_ground": data.get("legal_ground") or "",
            "erased_by": data.get("erased_by") or "",
            "erased_at": data.get("erased_at") or "",
            "state": _erasure_state(data, key),
        })
    return out


def _rotation_fields_ok(data: dict) -> bool:
    purged, archived = data.get("purged_records"), data.get("archive_records")
    return (
        all(isinstance(data.get(k), str) and data[k] for k in _ROTATION_TEXT_FIELDS)
        and isinstance(purged, int) and not isinstance(purged, bool) and purged >= 1
        and isinstance(archived, int) and not isinstance(archived, bool) and archived >= 0
        and data.get("archive_kind") in _ROTATION_ARCHIVE_KINDS
    )


def _rotation_state(data: dict, key: Optional["Ed25519PublicKey"]) -> str:
    if not _rotation_fields_ok(data):
        return "malformed"
    block = data.get("rotation_signature")
    if block is None:
        return "unsigned"
    if (not isinstance(block, dict) or set(block) != _ROTATION_SIGNATURE_KEYS
            or block.get("type") != ROTATION_RECORD_TYPE or block.get("alg") != "Ed25519"
            or not isinstance(block.get("signature"), str)):
        return "malformed"
    if key is None:
        return "unpinned"
    try:
        signature = bytes.fromhex(block["signature"])
        key.verify(signature, rotation_payload(data))
    except (ValueError, InvalidSignature):
        return "bad_signature"
    return "signed"


def check_rotations(records: list[dict],
                    public_key: Optional["Ed25519PublicKey"] = None) -> list[dict]:
    """Judge every ``trail_rotated`` record: the claim and the signature on it.

    ``records`` are trail records as dicts (``record_id``, ``event_type``,
    ``data``); anything else in the list is skipped, so a caller can pass the
    whole trail or only the lines that name the type. ``public_key`` is the
    Ed25519 key that signed the rotation, the one that signs the exports. Each
    record lands in one state:

    * ``signed``: the signature verifies under ``public_key`` over every field.
    * ``unsigned``: well formed, no signature. It accounts for nothing.
    * ``unpinned``: signed, but there is no key to check it with (a threshold
      export carries no single signer), so nobody can say whose.
    * ``bad_signature``: does not verify under ``public_key``.
    * ``malformed``: a field is missing or of the wrong type, or the signature
      block is not the closed ``type``/``alg``/``signature`` shape.

    Each entry also carries ``begins_here`` and ``archive_check``, both unset
    here: they depend on where the trail begins, which :func:`verify_signed`
    reads. This judges the claim. Whether the record itself was altered is the
    chain walk's job.
    """
    out: list[dict] = []
    for index, rec in enumerate(records):
        if rec.get("event_type") != "trail_rotated":
            continue
        data = _data_of(rec)
        out.append({
            "index": index,
            "record_id": rec.get("record_id"),
            "last_hash": data.get("last_hash"),
            "purged_records": data.get("purged_records"),
            "archive_sha256": data.get("archive_sha256"),
            "archive_records": data.get("archive_records"),
            "archive_kind": data.get("archive_kind"),
            "boundary_utc": data.get("boundary_utc"),
            "rotated_at": data.get("rotated_at"),
            "signer_fingerprint": data.get("signer_fingerprint"),
            "state": _rotation_state(data, public_key),
            "begins_here": False,
            "archive_check": None,
        })
    return out


def _verify_chain_bytes(trail_bytes: bytes, *,
                        start_hash: str = "") -> tuple[Optional[str], list[dict]]:
    """Re-verify the hash chain from the exported JSONL bytes.

    Returns ``(error, erased)``: error is None unless the chain is broken.
    A tombstoned record whose hash no longer recomputes is held until the
    walk has seen every ``record_erased`` record after it; one naming its id
    and stored hash makes it an erased slot, otherwise it is a hash mismatch.
    ``start_hash`` is the hash the first record must link to: empty for a trail
    that begins at its first record, and the ``last_hash`` of a signed rotation
    record for one that begins at a rotation seam (the caller decides that).
    Uses only stdlib — mirrors :meth:`AuditTrail.verify_chain` without
    requiring vaara imports, so the standalone verifier can run it.
    """
    held: list[tuple[int, dict]] = []
    erasures: dict[tuple[str, str], int] = {}
    prev_hash = start_hash
    line_iter = (ln for ln in trail_bytes.splitlines() if ln.strip())
    for idx, raw in enumerate(line_iter):
        try:
            rec = _loads(raw)
        except json.JSONDecodeError as e:
            return f"record {idx}: malformed JSON ({e})", []
        except _Refusal as e:
            return f"record {idx}: {e}", []
        if rec.get("previous_hash", "") != prev_hash:
            return (
                f"record {idx}: previous_hash mismatch "
                f"(expected {prev_hash!r}, got {rec.get('previous_hash')!r})"
            ), []
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
            if _erasure_pointer(rec) is None or not isinstance(rec.get("record_hash"), str):
                return f"record {idx}: hash mismatch (record tampered)", []
            held.append((idx, rec))
        if rec.get("event_type") == "record_erased":
            data = _data_of(rec)
            rid, rhash = data.get("erased_record_id"), data.get("erased_record_hash")
            if isinstance(rid, str) and isinstance(rhash, str):
                erasures.setdefault((rid, rhash), idx)
        prev_hash = rec["record_hash"]
    erased = []
    for idx, rec in held:
        rid = rec.get("record_id")
        by = erasures.get((rid, rec["record_hash"])) if isinstance(rid, str) else None
        if by is None or by <= idx:
            return (f"record {idx}: hash mismatch (erasure tombstone that no later "
                    f"record_erased record accounts for)"), []
        erased.append({"index": idx, "record_id": rec.get("record_id"),
                       "record_hash": rec["record_hash"], "erased_by_index": by})
    return None, erased


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
    pinned_fp: Optional[str] = None,
) -> list[str]:
    """Verify a k-of-n threshold export. Returns a list of error strings.

    Requires at least ``threshold_k`` distinct authorized signers to carry
    a valid signature over ``to_verify``. The authorized fingerprint list
    lives in the signed manifest, and each stored public key is bound to
    its manifest fingerprint, so a substituted pubkey is rejected and an
    unauthorized extra signature is ignored rather than counted.

    ``pinned_fp`` is the member fingerprint of a public key the caller
    supplied out-of-band. When set, it establishes *authenticity* rather
    than mere internal consistency: the pinned key must itself be one of
    the members that carried a valid signature, else the export is a
    self-consistent forgery under an attacker-chosen signer set and is
    rejected. When ``None`` the result is integrity/consistency only — the
    signer set is whatever the (untrusted) manifest declares.
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
        member_key = authorized.get(fp)
        if member_key is None:
            continue  # unknown or unauthorized signer; not counted
        if member_algorithm == "Ed25519":
            try:
                Ed25519PublicKey.from_public_bytes(member_key).verify(sig, to_verify)
                valid.add(fp)
            except (InvalidSignature, ValueError):
                pass
        elif mldsa_cls is not None and mldsa_cls(member_key).verify(to_verify, sig):
            valid.add(fp)

    if len(valid) < k:
        errors.append(
            f"threshold not met: {len(valid)} valid signature(s) from the "
            f"authorized set, need at least {k} of {len(fps)}"
        )
    if pinned_fp is not None and pinned_fp not in valid:
        errors.append(
            "pinned public_key is not among the valid threshold signers: "
            "the signer set is attacker-suppliable unless a caller-supplied "
            "key is one of the signers, so this export is not authenticated "
            "against the pinned key (integrity only)"
        )
    return errors


def _check_revocation_registry(manifest: dict,
                               revocation_bytes: Optional[bytes]) -> list[str]:
    """The optional revocation registry against the digest the manifest pins.

    The manifest is signed, so ``revocation.json`` matching
    ``manifest.revocation.registry_sha256`` confirms the revocation state in
    the bundle is the one the signer attested. Same three findings, same
    wording, as ``scripts/verify_vaara_trail.py``.
    """
    block = manifest.get("revocation")
    if block is None and revocation_bytes is None:
        return []
    if block is not None and not isinstance(block, dict):
        return ["manifest revocation block is not an object"]
    if block is None:
        return ["revocation.json present but manifest has no revocation block"]
    if revocation_bytes is None:
        return ["manifest declares revocation but revocation.json is missing"]
    actual = "sha256:" + hashlib.sha256(revocation_bytes).hexdigest()
    if block.get("registry_sha256") != actual:
        return [
            "revocation.json digest does not match "
            "manifest.revocation.registry_sha256 (tampered)"
        ]
    return []


def verify_signed(
    zip_path: Union[str, Path],
    public_key: Optional[Union[str, Path, bytes, "_PK"]] = None,
    erasure_key: Optional[Union[str, Path, bytes, object]] = None,
    *,
    rotation_key: Optional[Union[str, Path, bytes, "_PK"]] = None,
    archive: Optional[Union[str, Path]] = None,
) -> VerifyResult:
    """Verify a signed audit-trail export.

    Checks, in order:

    1. Zip contains the required files (``trail.jsonl``, ``manifest.json``,
       ``trail.sig``, and optionally ``signer_pubkey.pem``).
    2. Manifest parses as JSON and has the expected schema version.
    3. ``manifest.trail_sha256`` matches the actual SHA-256 of
       ``trail.jsonl``.
    4. ``revocation.json`` and the manifest's ``revocation`` block go
       together, and the file's SHA-256 matches
       ``manifest.revocation.registry_sha256`` (an export that pins no
       registry has neither).
    5. Ed25519 signature over ``sha256(trail.jsonl || manifest.json)``
       verifies under ``public_key`` (if given) or the embedded
       ``signer_pubkey.pem``.
    6. The hash chain re-verifies from the exported bytes.
    7. Manifest endpoints (``first_hash``, ``last_hash``, ``record_count``)
       match the exported JSONL.
    8. A first record that links to a hash outside the export is accepted only
       when the export holds a ``trail_rotated`` record naming that hash whose
       signature verifies (see :func:`check_rotations`); then the export
       begins at a rotation seam. With ``archive``, that archive is checked
       against the digest the record states.

    Args:
        zip_path: Path to the zip produced by :func:`export_signed`.
        public_key: Ed25519 public key — Path, PEM bytes, or loaded key
            object. If omitted, uses ``signer_pubkey.pem`` from inside
            the zip (convenient for local verification; for production,
            always pass a trusted public key out-of-band).
        rotation_key: Ed25519 public key that signed the rotation records,
            when it is not the key that signed this export (a rotation
            signed under an earlier key). Defaults to the export's own key;
            a threshold export has none, so its rotation records are
            ``unpinned`` unless this is given.
        archive: The retention archive the export begins after: the zip, or
            the sealed file of an encrypted rotation. Its SHA-256 must be the
            one the rotation record states; a zip must also verify, hold the
            stated number of records and hold the record the export links to.

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
            revocation_bytes = (
                zf.read("revocation.json") if "revocation.json" in names else None
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
        manifest = _loads(manifest_bytes)
    except json.JSONDecodeError as e:
        return VerifyResult(ok=False, errors=[f"manifest.json is not valid JSON: {e}"])
    except _Refusal as e:
        return VerifyResult(ok=False, errors=[f"manifest.json: {e}"])

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

    errors.extend(_check_revocation_registry(manifest, revocation_bytes))

    algorithm = manifest.get("signature_algorithm", "Ed25519")
    to_verify = hashlib.sha256(trail_bytes + manifest_bytes).digest()
    export_key = None

    if isinstance(algorithm, str) and algorithm.startswith(_THRESHOLD_PREFIX):
        # A caller-supplied key pins authenticity: it must be one of the
        # members that signed, else the threshold set is attacker-chosen.
        pinned_fp: Optional[str] = None
        if public_key is not None:
            try:
                loaded_pin = _load_public_key(public_key)
            except (ValueError, TypeError) as exc:
                errors.append(f"pinned public_key could not be loaded: {exc}")
                loaded_pin = None
            if loaded_pin is not None:
                if isinstance(loaded_pin, Ed25519PublicKey):
                    pinned_fp = _member_fingerprint(_raw_ed25519_bytes(loaded_pin))
                else:
                    errors.append(
                        "pinned public_key is not Ed25519; cannot match it to a "
                        "threshold member"
                    )
        errors.extend(
            _verify_threshold(
                manifest=manifest,
                to_verify=to_verify,
                sigs=threshold_sigs,
                pubkeys=threshold_pubkeys,
                pinned_fp=pinned_fp,
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
        export_key = pk
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

    lines = [ln for ln in trail_bytes.splitlines() if ln.strip()]
    seam_key = _load_public_key(rotation_key) if rotation_key is not None else export_key
    rotations = check_rotations(_rotation_candidates(lines), seam_key)
    start = _first_link(lines)
    begun = [r for r in rotations if start and r["last_hash"] == start]
    accounted = next((r for r in begun if r["state"] in _ROTATION_ACCEPTED), None)
    if accounted is not None:
        accounted["begins_here"] = True
    chain_error, erased = _verify_chain_bytes(
        trail_bytes, start_hash=start if accounted is not None else "")
    if chain_error is not None:
        errors.append(f"hash chain check failed: {chain_error}"
                      + _unaccounted_start(chain_error, start, begun))
    for r in rotations:
        if r["state"] in _ROTATION_FINDINGS:
            errors.append(f"record {r['index']}: rotation record is {r['state']}")
    if archive is not None:
        errors.extend(_check_archive(
            Path(archive), accounted, public_key=public_key, rotation_key=rotation_key))
    if manifest.get("record_count") != len(lines):
        errors.append(
            f"record_count mismatch: manifest={manifest.get('record_count')}, "
            f"actual={len(lines)}"
        )
    if lines:
        try:
            first = _loads(lines[0]).get("record_hash", "")
            last = _loads(lines[-1]).get("record_hash", "")
            if manifest.get("first_hash") != first:
                errors.append("first_hash in manifest does not match first record")
            if manifest.get("last_hash") != last:
                errors.append("last_hash in manifest does not match last record")
        except _UNREADABLE:
            errors.append("could not parse first/last trail record")

    lifecycle = _collect_key_lifecycle(lines)
    erasures = _check_export_erasures(lines, erasure_key, errors)
    cages, cage_open, unpaired, cage_errors = _check_cage_lifecycle(lines)
    errors.extend(cage_errors)
    windows, window_errors = _check_unrecorded_windows(lines)
    errors.extend(window_errors)

    return VerifyResult(
        ok=not errors,
        errors=errors,
        manifest=manifest,
        key_lifecycle=lifecycle,
        erased=erased,
        erasures=erasures,
        cage_lifecycle=cages,
        cage_open=cage_open,
        cage_unpaired_stops=unpaired,
        unrecorded_windows=windows,
        rotation_seams=rotations,
    )


def _rotation_candidates(lines: list[bytes]) -> list[dict]:
    """The trail as dicts for :func:`check_rotations`, parsing only the lines that
    name the type; the rest stay empty so indexes still match the chain."""
    records: list[dict] = []
    for ln in lines:
        if b"trail_rotated" not in ln:
            records.append({})
            continue
        try:
            records.append(_loads(ln))
        except _UNREADABLE:
            records.append({})
    return records


def _first_link(lines: list[bytes]) -> str:
    """The hash the first record links to, when it links to anything."""
    if not lines:
        return ""
    try:
        link = _loads(lines[0]).get("previous_hash", "")
    except _UNREADABLE:
        return ""
    return link if isinstance(link, str) else ""


def _unaccounted_start(chain_error: str, start: str, begun: list[dict]) -> str:
    """Why a first record that links outside the export was not accepted."""
    if not start or not chain_error.startswith("record 0: previous_hash mismatch"):
        return ""
    if begun:
        return (f" (rotation record {begun[0]['index']} names that hash, "
                f"but it is {begun[0]['state']})")
    return " (no rotation record in the trail names that hash)"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_archive(path: Path, seam: Optional[dict], *, public_key: object,
                   rotation_key: object) -> list[str]:
    """Check a retention archive against the rotation record the export begins at.

    Sets ``seam["archive_check"]`` and returns the errors. A zip must hash to
    the recorded digest, verify as an export of its own, hold the recorded
    number of records, and hold the record the export links to. A sealed file
    can only be checked by its digest: the records are encrypted.
    """
    if seam is None:
        return ["an archive was given, but this export does not begin at a "
                "signed rotation record, so there is nothing to check it against"]
    name, at = path.name, seam["index"]
    try:
        actual = _file_sha256(path)
    except OSError as exc:
        seam["archive_check"] = "unreadable"
        return [f"archive {name}: {exc.strerror or exc}"]
    if actual != seam["archive_sha256"]:
        seam["archive_check"] = "digest_mismatch"
        return [f"archive {name} does not match rotation record {at}: "
                f"SHA-256 {actual} != recorded {seam['archive_sha256']}"]
    if seam["archive_kind"] == "sealed":
        seam["archive_check"] = "digest_only"
        return []
    inner = verify_signed(path, public_key=public_key, rotation_key=rotation_key)
    if not inner.ok:
        seam["archive_check"] = "archive_invalid"
        return [f"archive {name} does not verify: {inner.errors[0]}"]
    held = (inner.manifest or {}).get("record_count")
    if held != seam["archive_records"]:
        seam["archive_check"] = "count_mismatch"
        return [f"archive {name} does not match rotation record {at}: "
                f"{held} records, the record says {seam['archive_records']}"]
    with zipfile.ZipFile(path) as zf:
        hashes = set()
        for ln in zf.read("trail.jsonl").splitlines():
            if ln.strip():
                try:
                    hashes.add(_loads(ln).get("record_hash"))
                except _UNREADABLE:
                    continue
    if seam["last_hash"] not in hashes:
        seam["archive_check"] = "link_missing"
        return [f"archive {name} does not hold the record rotation record {at} "
                f"names (last_hash {seam['last_hash']})"]
    seam["archive_check"] = "verified"
    return []


def describe_rotation_seams(result: VerifyResult) -> list[str]:
    """One line per rotation record: where the export begins, what was archived."""
    lines = []
    for s in result.rotation_seams:
        if s["state"] == "malformed":
            lines.append(f"rotation record {s['index']}: malformed (a field or the "
                         f"signature block does not have the expected shape)")
            continue
        n = s["purged_records"]
        what = f"{n} record{'s' if n != 1 else ''} archived as {s['archive_sha256']}"
        if s["begins_here"]:
            line = f"begins at a rotation seam: {what}"
            if s["state"] == "unpinned":
                line += " (signature not checked: pass --rotation-key)"
        else:
            line = f"rotation record {s['index']}: {what} (this trail does not begin there)"
        if s["archive_check"] == "verified":
            line += (f" (archive checked: digest matches, {s['archive_records']} records "
                     f"verify, holds the record this trail links to)")
        elif s["archive_check"] == "digest_only":
            line += " (archive checked: digest matches, contents not checked: sealed)"
        lines.append(line)
    return lines


def _check_export_erasures(lines: list[bytes], erasure_key: Optional[object],
                           errors: list[str]) -> list[dict]:
    records = []
    for ln in lines:
        try:
            records.append(_loads(ln))
        except _UNREADABLE:
            records.append({})
    erasures = check_erasures(records, public_key=erasure_key)
    for e in erasures:
        if e["state"] in _ERASURE_FINDINGS or (
                erasure_key is not None and e["state"] == "unsigned"):
            errors.append(f"record {e['index']}: erasure of "
                          f"{e['erased_record_id']} is {e['state']}")
    return erasures


def _collect_key_lifecycle(lines: list[bytes]) -> list[dict]:
    """Pull custodian key-lifecycle records out of the trail, in chain order.

    Informational surfacing only (see :class:`VerifyResult.key_lifecycle`);
    the records are already covered by the hash-chain check, so a tampered
    lifecycle record fails verification on the chain, not here.
    """
    out: list[dict] = []
    for ln in lines:
        try:
            rec = _loads(ln)
        except _UNREADABLE:
            continue
        if rec.get("event_type") != "key_lifecycle":
            continue
        data = _data_of(rec)
        out.append(
            {
                "timestamp": rec.get("timestamp"),
                "action": data.get("action"),
                "fingerprint": data.get("fingerprint"),
                "threshold_k": data.get("threshold_k"),
                "signers_n": data.get("signers_n"),
                "reason": data.get("reason", ""),
                "actor": data.get("actor", ""),
            }
        )
    return out


def _check_cage_lifecycle(
    lines: list[bytes],
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """Pair each cage stop with the start it names.

    Returns ``(records, open_starts, unpaired_stops, errors)``. Ids are read
    across every record in the export, so a stop naming some other record
    is told apart from a stop whose start lies outside the export.
    """
    index_of: dict[str, int] = {}
    parsed: list[Optional[dict]] = []
    for idx, ln in enumerate(lines):
        try:
            rec = _loads(ln)
        except _UNREADABLE:
            rec = None
        parsed.append(rec)
        if rec is not None and isinstance(rec.get("record_id"), str):
            index_of.setdefault(rec["record_id"], idx)

    records: list[dict] = []
    starts: dict[str, dict] = {}
    closed: set[str] = set()
    unpaired: list[dict] = []
    errors: list[str] = []
    for idx, rec in enumerate(parsed):
        if rec is None or rec.get("event_type") != "cage_lifecycle":
            continue
        data = _data_of(rec)
        entry = {
            "index": idx,
            "record_id": rec.get("record_id"),
            "timestamp": rec.get("timestamp"),
            "action": data.get("action"),
            "driver": data.get("driver"),
            "name": data.get("name"),
            "policy_digest": data.get("policy_digest", ""),
            "actor": data.get("actor", ""),
        }
        for key in ("started_record_id", "processes_ended", "exit_code", "reason"):
            if key in data:
                entry[key] = data[key]
        records.append(entry)
        if entry["action"] == "started":
            starts[str(entry["record_id"])] = entry
            continue
        if entry["action"] != "stopped":
            errors.append(f"cage_lifecycle record {idx}: unknown action {entry['action']!r}")
            continue
        sid = data.get("started_record_id")
        if not sid:
            unpaired.append(entry)
            continue
        at = index_of.get(sid) if isinstance(sid, str) else None
        if at is None:
            unpaired.append(entry)
            continue
        target = parsed[at] or {}
        tdata = _data_of(target)
        if (target.get("event_type") != "cage_lifecycle" or tdata.get("action") != "started"
                or tdata.get("driver") != entry["driver"] or tdata.get("name") != entry["name"]):
            errors.append(f"cage_lifecycle record {idx}: names record {at}, which is not "
                          f"the start of {entry['driver']}/{entry['name']}")
        elif at > idx:
            errors.append(f"cage_lifecycle record {idx}: a stop written before its start "
                          f"(record {at})")
        else:
            closed.add(sid)
    cage_open = [e for rid, e in starts.items() if rid not in closed]
    return records, cage_open, unpaired, errors


def describe_cage_lifecycle(result: VerifyResult) -> list[str]:
    """One plain line per cage start or stop, for the verify commands."""
    open_ids = {e["record_id"] for e in result.cage_open}
    unpaired_ids = {e["record_id"] for e in result.cage_unpaired_stops}
    lines = []
    for e in result.cage_lifecycle:
        head = f"record {e['index']} cage {e['driver']}/{e['name']}: {e['action']}"
        if e["action"] == "started":
            extra = [f"policy {e['policy_digest']}"] if e.get("policy_digest") else []
            if e.get("actor"):
                extra.append(f"by {e['actor']}")
            line = head + (f" ({', '.join(extra)})" if extra else "")
            if e["record_id"] in open_ids:
                line = head + ", no stop in this export" + line[len(head):]
        else:
            extra = [str(e.get("reason") or "no reason given")]
            if e.get("processes_ended") is not None:
                extra.append(f"{e['processes_ended']} processes ended")
            if e.get("exit_code") is not None:
                extra.append(f"exit {e['exit_code']}")
            line = head + f" ({', '.join(extra)})"
            if e["record_id"] in unpaired_ids:
                line += ", start not in this export"
        lines.append(line)
    return lines


_WINDOW_TEXT_FIELDS = ("first_failure_utc", "last_failure_utc", "resumed_record_id")
_WINDOW_CAUSE_LIMIT = 140


def _check_unrecorded_windows(lines: list[bytes]) -> tuple[list[dict], list[str]]:
    """Read each ``trail_not_recording`` record and place it on the chain.

    Returns ``(windows, errors)``. A record the trail did not write (a count
    that is not a whole number of at least 1, a missing time or record id, a
    record id that is not an earlier record) is an error and is not listed.
    A ``resumed_record_id`` outside the export is not an error: an export of a
    later range does not hold it.
    """
    index_of: dict[str, int] = {}
    parsed: list[Optional[dict]] = []
    for idx, ln in enumerate(lines):
        try:
            rec = _loads(ln)
        except _UNREADABLE:
            rec = None
        parsed.append(rec)
        if rec is not None and isinstance(rec.get("record_id"), str):
            index_of.setdefault(rec["record_id"], idx)

    windows: list[dict] = []
    errors: list[str] = []
    for idx, rec in enumerate(parsed):
        if rec is None or rec.get("event_type") != "trail_not_recording":
            continue
        data = _data_of(rec)
        count = data.get("unrecorded_writes")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            errors.append(f"trail_not_recording record {idx}: unrecorded_writes {count!r} "
                          "is not a whole number of at least 1")
            continue
        missing = [f for f in _WINDOW_TEXT_FIELDS
                   if not isinstance(data.get(f), str) or not data[f]]
        if missing:
            errors.append(f"trail_not_recording record {idx}: {missing[0]} is missing or "
                          "not text")
            continue
        resumed = index_of.get(data["resumed_record_id"])
        if resumed is not None and resumed >= idx:
            errors.append(f"trail_not_recording record {idx}: names record {resumed}, "
                          "which is not an earlier record")
            continue
        windows.append({
            "index": idx,
            "record_id": rec.get("record_id"),
            "timestamp": rec.get("timestamp"),
            "unrecorded_writes": count,
            "first_failure_utc": data["first_failure_utc"],
            "last_failure_utc": data["last_failure_utc"],
            "stage": str(data.get("stage") or ""),
            "error": str(data.get("error") or ""),
            "resumed_record_id": data["resumed_record_id"],
            "resumed_index": resumed,
            "after_index": resumed - 1 if resumed else None,
        })
    return windows, errors


def describe_unrecorded_windows(result: VerifyResult) -> list[str]:
    """One plain line per unrecorded window, for the verify commands."""
    lines = []
    for w in result.unrecorded_windows:
        if w["resumed_index"] is None:
            where = "before this export"
        elif w["after_index"] is None:
            where = f"before record {w['resumed_index']}"
        else:
            where = f"between record {w['after_index']} and record {w['resumed_index']}"
        cause = ": ".join(x for x in (w["stage"], w["error"]) if x)
        if len(cause) > _WINDOW_CAUSE_LIMIT:
            cause = cause[:_WINDOW_CAUSE_LIMIT].rstrip() + "..."
        lines.append(
            f"record {w['index']} trail not recording: the store refused at least "
            f"{w['unrecorded_writes']} write(s) {where}, {w['first_failure_utc']} to "
            f"{w['last_failure_utc']}{f' ({cause})' if cause else ''}; calls in that "
            "window ran with no record here"
        )
    return lines
