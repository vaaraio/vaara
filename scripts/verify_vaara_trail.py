#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone verifier for a Vaara signed audit-trail export.

No Vaara install required. Only dependency:

    pip install cryptography

Usage:

    ./verify_vaara_trail.py path/to/trail.zip
    ./verify_vaara_trail.py path/to/trail.zip --pubkey signer.pem
    ./verify_vaara_trail.py path/to/trail.zip --erasure-key eraser.pub.pem

A trail that honoured a GDPR Art. 17 erasure (`vaara trail erase`) verifies:
the erased slot is reported as erased (link verified, content not checkable),
and --erasure-key checks the eraser's signature on each erasure record.

A trail with cage launches (`vaara cage run`, `vaara cage stop`, `vaara run`)
is paired: each stop must name the start of the same driver and launch, written
before it, or the run fails. Starts with no stop and stops whose start is not
in the export are listed, not failed.

A trail whose store refused writes and then took them again carries a
`trail_not_recording` record. It is listed with the number of writes refused
(at least) and the two adjacent records the window lies between; calls in that
window ran with no record. A record whose fields are nonsense fails the run.

A trail line or manifest that repeats a member of one object, or holds NaN,
Infinity or a number that overflows to one, fails the run: two readers could
read it differently, and the hash covers only one reading. A line or manifest
that is valid JSON but not an object (`[1]`, `7`, `"x"`, `null`) fails it too.

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
import math
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


def _loads(raw: bytes | str) -> dict:
    """``json.loads`` for a trail line or the manifest: a JSON object, or a refusal.

    A member repeated in one object reads as its last value to Python and its
    first to a first-wins parser, and the hash is over only one of them. NaN,
    Infinity and a literal that overflows to one cannot be re-encoded for the
    hash at all. A line that parses to an array, number, string, boolean or
    null is no record: it has no members to read. All are refused as
    ``_Refusal`` wherever the file is read, mirroring ``vaara.audit.verify._loads``.
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


# Erasure (GDPR Art. 17, `vaara trail erase`). The erased slot keeps its stored
# hash so the next record still chains to it, a later record_erased record
# names it, and the eraser may sign that record. Mirrors vaara.audit.verify
# (_erasure_pointer, erasure_payload, check_erasures) so this file gives the
# same verdict with only the 'cryptography' package installed.
_ERASED_DATA_KEY = "_vaara_erased"
ERASURE_RECORD_TYPE = "vaara.record-erasure/v1"
_ERASURE_SIGNATURE_KEYS = frozenset({"type", "alg", "signature"})
#: Signature states that fail verification.
_ERASURE_FINDINGS = frozenset({"bad_signature", "malformed"})


def _erasure_pointer(rec: dict) -> str | None:
    """The erasure record id a tombstoned record points at, else None."""
    data = rec.get("data")
    if not isinstance(data, dict) or set(data) != {_ERASED_DATA_KEY}:
        return None
    inner = data[_ERASED_DATA_KEY]
    if not isinstance(inner, dict):
        return None
    pointer = inner.get("erasure_record_id")
    return pointer if isinstance(pointer, str) and pointer else None


def _jcs_string(value: str) -> str:
    # RFC 8785 escapes only the quote, the backslash and C0 controls, with
    # the same short forms json uses; everything else is raw UTF-8.
    return json.dumps(value, ensure_ascii=False)


def erasure_payload(data: dict) -> bytes:
    """The exact bytes an eraser's signature covers, from a ``record_erased``
    record's ``data``. RFC 8785 (JCS) over strings, one integer and null."""
    fields = {
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
    }
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


def _load_erasure_key(key):
    """An EC P-256 or RSA public key from PEM bytes, a path, or a key object."""
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


def _verify_erasure_signature(payload: bytes, alg: str, signature_hex: str, key) -> bool:
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


def _erasure_state(data: dict, key) -> str:
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


def check_erasures(records: list[dict], public_key=None) -> list[dict]:
    """Judge the eraser's signature on every ``record_erased`` record.

    ``public_key`` is the eraser's EC P-256 or RSA public key, held out of
    band; the record never carries one. Each erasure lands in one state:
    ``signed`` (verifies under the key), ``unsigned`` (no signature),
    ``unpinned`` (signed, no key given), ``bad_signature`` or ``malformed``.
    This judges who signed; whether the record was altered is the chain's job.
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


def _verify_chain(trail_bytes: bytes) -> tuple[str | None, list[dict]]:
    """Walk the hash chain. Returns ``(error, erased)``: error is None unless
    the chain is broken. A tombstoned record whose hash no longer recomputes is
    held until the walk has seen every ``record_erased`` record after it; one
    naming its id and stored hash makes it an erased slot (link verified,
    content not checkable), otherwise it is a hash mismatch."""
    held: list[tuple[int, dict]] = []
    erasures: dict[tuple[str, str], int] = {}
    prev_hash = ""
    for idx, raw in enumerate(ln for ln in trail_bytes.splitlines() if ln.strip()):
        try:
            rec = _loads(raw)
        except json.JSONDecodeError as e:
            return f"record {idx}: malformed JSON ({e})", []
        except _Refusal as e:
            return f"record {idx}: {e}", []
        if rec.get("previous_hash", "") != prev_hash:
            return f"record {idx}: previous_hash mismatch", []
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


def _trail_records(trail_bytes: bytes) -> list[dict]:
    out: list[dict] = []
    for raw in (ln for ln in trail_bytes.splitlines() if ln.strip()):
        try:
            out.append(_loads(raw))
        except _UNREADABLE:
            out.append({})
    return out


def _collect_erasures(zip_path: Path, erasure_key) -> tuple[list[dict], list[dict]]:
    """Erased slots and the judged erasure records, in chain order.

    Read after a passing verification, for the report. ``erasure_key`` is a
    loaded key from :func:`_load_erasure_key`, or None.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trail_bytes = zf.read("trail.jsonl")
    except (zipfile.BadZipFile, KeyError, FileNotFoundError):
        return [], []
    _, erased = _verify_chain(trail_bytes)
    return erased, check_erasures(_trail_records(trail_bytes), erasure_key)


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
            rec = _loads(raw)
        except _UNREADABLE:
            continue
        if rec.get("event_type") != "key_lifecycle":
            continue
        data = _data_of(rec)
        out.append({
            "action": data.get("action"),
            "fingerprint": data.get("fingerprint"),
            "threshold_k": data.get("threshold_k"),
            "signers_n": data.get("signers_n"),
            "reason": data.get("reason", ""),
        })
    return out


def _check_cage_lifecycle(
    lines: list[bytes],
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """Pair each cage stop with the start it names.

    Returns ``(records, open_starts, unpaired_stops, errors)``. Ids are read
    across every record in the export, so a stop naming some other record
    is told apart from a stop whose start lies outside the export. Mirrors
    ``vaara.audit.verify._check_cage_lifecycle``.
    """
    index_of: dict[str, int] = {}
    parsed: list[dict | None] = []
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


def describe_cage_lifecycle(
    records: list[dict], cage_open: list[dict], unpaired: list[dict],
) -> list[str]:
    """One plain line per cage start or stop, as ``vaara trail verify`` prints them."""
    open_ids = {e["record_id"] for e in cage_open}
    unpaired_ids = {e["record_id"] for e in unpaired}
    out = []
    for e in records:
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
        out.append(line)
    return out


def _collect_cage_lifecycle(zip_path: Path) -> list[str]:
    """The cage lines for the report, read after a passing verification."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trail_bytes = zf.read("trail.jsonl")
    except (zipfile.BadZipFile, KeyError, FileNotFoundError):
        return []
    lines = [ln for ln in trail_bytes.splitlines() if ln.strip()]
    records, cage_open, unpaired, _ = _check_cage_lifecycle(lines)
    return describe_cage_lifecycle(records, cage_open, unpaired)


_WINDOW_TEXT_FIELDS = ("first_failure_utc", "last_failure_utc", "resumed_record_id")
_WINDOW_CAUSE_LIMIT = 140


def _check_unrecorded_windows(lines: list[bytes]) -> tuple[list[dict], list[str]]:
    """Read each ``trail_not_recording`` record and place it on the chain.

    Returns ``(windows, errors)``. A record the trail did not write (a count
    that is not a whole number of at least 1, a missing time or record id, a
    record id that is not an earlier record) is an error and is not listed.
    A ``resumed_record_id`` outside the export is not an error: an export of a
    later range does not hold it. Mirrors
    ``vaara.audit.verify._check_unrecorded_windows``.
    """
    index_of: dict[str, int] = {}
    parsed: list[dict | None] = []
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


def describe_unrecorded_windows(windows: list[dict]) -> list[str]:
    """One plain line per unrecorded window, as ``vaara trail verify`` prints them."""
    out = []
    for w in windows:
        if w["resumed_index"] is None:
            where = "before this export"
        elif w["after_index"] is None:
            where = f"before record {w['resumed_index']}"
        else:
            where = f"between record {w['after_index']} and record {w['resumed_index']}"
        cause = ": ".join(x for x in (w["stage"], w["error"]) if x)
        if len(cause) > _WINDOW_CAUSE_LIMIT:
            cause = cause[:_WINDOW_CAUSE_LIMIT].rstrip() + "..."
        out.append(
            f"record {w['index']} trail not recording: the store refused at least "
            f"{w['unrecorded_writes']} write(s) {where}, {w['first_failure_utc']} to "
            f"{w['last_failure_utc']}{f' ({cause})' if cause else ''}; calls in that "
            "window ran with no record here"
        )
    return out


def _collect_unrecorded_windows(zip_path: Path) -> list[str]:
    """The unrecorded-window lines for the report, read after a passing verification."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            trail_bytes = zf.read("trail.jsonl")
    except (zipfile.BadZipFile, KeyError, FileNotFoundError):
        return []
    lines = [ln for ln in trail_bytes.splitlines() if ln.strip()]
    return describe_unrecorded_windows(_check_unrecorded_windows(lines)[0])


def verify(
    zip_path: Path, pubkey_path: Path | None, erasure_key=None,
) -> tuple[bool, list[str], dict | None]:
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
        manifest = _loads(manifest_bytes)
    except json.JSONDecodeError as e:
        return False, [f"manifest.json invalid JSON: {e}"], None
    except _Refusal as e:
        return False, [f"manifest.json: {e}"], None

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
        if rev_manifest is not None and not isinstance(rev_manifest, dict):
            errors.append("manifest revocation block is not an object")
        elif rev_manifest is None:
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

    if not isinstance(algorithm, str):
        errors.append(f"unknown signature_algorithm: {algorithm!r}")
    elif algorithm.startswith(THRESHOLD_PREFIX):
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

    chain_error, _ = _verify_chain(trail_bytes)
    if chain_error is not None:
        errors.append(f"hash chain check failed: {chain_error}")

    lines = [ln for ln in trail_bytes.splitlines() if ln.strip()]
    if manifest.get("record_count") != len(lines):
        errors.append(
            f"record_count mismatch: manifest={manifest.get('record_count')}, actual={len(lines)}"
        )
    if lines:
        try:
            first = _loads(lines[0]).get("record_hash", "")
            last = _loads(lines[-1]).get("record_hash", "")
            if manifest.get("first_hash") != first:
                errors.append("first_hash mismatch")
            if manifest.get("last_hash") != last:
                errors.append("last_hash mismatch")
        except _UNREADABLE:
            errors.append("could not parse first/last record")

    for e in check_erasures(_trail_records(trail_bytes), erasure_key):
        if e["state"] in _ERASURE_FINDINGS or (
                erasure_key is not None and e["state"] == "unsigned"):
            errors.append(f"record {e['index']}: erasure of "
                          f"{e['erased_record_id']} is {e['state']}")

    errors.extend(_check_cage_lifecycle(lines)[3])
    errors.extend(_check_unrecorded_windows(lines)[1])

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
    p.add_argument(
        "--erasure-key",
        default=None,
        help="Optional PEM public key (EC P-256 or RSA) of the person who signs "
        "erasures, received out-of-band. When given, each record_erased record "
        "must verify under it; without it a signed erasure is reported as not "
        "checked.",
    )
    args = p.parse_args(argv)

    erasure_key = None
    if args.erasure_key:
        try:
            erasure_key = _load_erasure_key(Path(args.erasure_key))
        except (OSError, ValueError) as exc:
            print(f"verify_vaara_trail: --erasure-key: {exc}", file=sys.stderr)
            return 2

    ok, errors, manifest = verify(
        Path(args.zip_path),
        Path(args.pubkey) if args.pubkey else None,
        erasure_key,
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
        print("  covers: signatures and chain interior; a dropped tail before export "
              "is not detected without an anchor or seal "
              "(docs/prove-what-an-ai-agent-did.md)")
        erased, erasures = _collect_erasures(Path(args.zip_path), erasure_key)
        for slot in erased:
            print(f"  record {slot['index']} ({slot['record_id']}) erased: link verified, "
                  f"content not checkable (erasure record {slot['erased_by_index']})")
        for e in erasures:
            ground = f", ground: {e['legal_ground']}" if e["legal_ground"] else ""
            if e["state"] == "signed":
                who = f"signed by {e['erased_by']} under the given key"
            elif e["state"] == "unpinned":
                who = (f"signed as {e['erased_by']}, not checked "
                       "(pass --erasure-key to check whose key)")
            else:
                who = "eraser not signed"
            print(f"  erasure record {e['index']}: {who}{ground}")
        for line in _collect_cage_lifecycle(Path(args.zip_path)):
            print(f"  {line}")
        for line in _collect_unrecorded_windows(Path(args.zip_path)):
            print(f"  {line}")
        return 0
    print("\nVerification: FAILED", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
