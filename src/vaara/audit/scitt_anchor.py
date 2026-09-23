"""``scitt`` receipt anchors: Merkle inclusion in an append-only log Vaara operates.

The method identifier ``scitt`` is registered in SPEC.md Section 4 and kept
for the receipts that already carry it. What the anchor is, exactly: the
receipt's signed-payload digest appended as a leaf to an RFC 6962-style
Merkle log, with the inclusion proof and the root at the time of append.

What it is not: registration with an IETF SCITT transparency service. No
signed statement is submitted anywhere and the anchor is not a COSE receipt.

The inclusion proof binds the digest to ``rootHash``. The root travels inside
the anchor, so on its own it proves only what the producer says. The anchor
becomes a witness when the verifier holds a tree head obtained independently
of the receipt (published by the log operator, or kept by a third party) and
checks the anchor's root against it, directly or through a consistency proof.
``verify_scitt_anchor`` reports which of the two the caller got.

The log persists as one line per leaf (64 lowercase hex characters, the
leaf data) in ``<log_dir>/<log_id>.leaves``. Appends take an exclusive file
lock and re-read the file first, so two processes anchoring at once agree on
leaf positions.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from vaara.attestation.transparency_log import (
    ConsistencyProof,
    ConsistencyVerdict,
    InProcessTransparencyLog,
    InclusionProof,
    verify_consistency,
    verify_inclusion,
)
from vaara.audit.timeanchor import TimeAnchorError, _signed_payload_digest

DEFAULT_LOG_DIR = Path("~/.vaara/anchor-log")
DEFAULT_LOG_ID = "vaara-scitt-log"

_LOG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LEAF_RE = re.compile(r"^[0-9a-f]{64}$")


class ScittAnchorError(RuntimeError):
    """Raised when a ``scitt`` anchor cannot be produced or verified."""


def _log_id_digest(log_id: str) -> str:
    return base64.b64encode(hashlib.sha256(log_id.encode()).digest()).decode("ascii")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``lock_path`` for the duration of the block."""
    with open(lock_path, "a+b") as fh:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read_leaves(path: Path) -> list[bytes]:
    if not path.exists():
        return []
    leaves: list[bytes] = []
    with open(path, encoding="ascii") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.rstrip("\n")
            if not _LEAF_RE.match(text) or not line.endswith("\n"):
                raise ScittAnchorError(
                    f"{path}: line {lineno} is not a complete leaf "
                    "(64 lowercase hex characters and a newline)")
            leaves.append(bytes.fromhex(text))
    return leaves


class ScittAnchor:
    """Produce ``scitt`` anchors over Vaara receipts.

    Without a path the log lives in memory and ends with the object, which is
    what tests and single-process callers that publish the head themselves
    want. ``load_or_create`` gives the file-backed log the CLI uses.
    """

    def __init__(self, log: Optional[InProcessTransparencyLog] = None,
                 log_id: str = DEFAULT_LOG_ID,
                 path: Optional[Path] = None) -> None:
        if not _LOG_ID_RE.match(log_id):
            raise ScittAnchorError(
                f"log id {log_id!r} must be 1-128 characters of "
                "letters, digits, '.', '_' or '-', starting with a letter or digit")
        if log is not None and path is not None:
            raise ScittAnchorError("pass a log or a path, not both")
        self._log_id = log_id
        self._path = path
        self._log = log or InProcessTransparencyLog()
        if path is not None:
            self._reload(path)

    @classmethod
    def load_or_create(cls, directory: Optional[Path] = None,
                       log_id: str = DEFAULT_LOG_ID) -> ScittAnchor:
        """Open the file-backed log ``<directory>/<log_id>.leaves``."""
        base = Path(directory if directory is not None else DEFAULT_LOG_DIR).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        return cls(log_id=log_id, path=base / f"{log_id}.leaves")

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def _reload(self, path: Path) -> None:
        log = InProcessTransparencyLog()
        for leaf in _read_leaves(path):
            log.append(leaf)
        self._log = log

    @staticmethod
    def _persist(path: Path, leaf: bytes) -> None:
        with open(path, "a", encoding="ascii") as fh:
            fh.write(leaf.hex() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def anchor_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = _signed_payload_digest(receipt)
        except TimeAnchorError as exc:
            raise ScittAnchorError(str(exc)) from exc
        path = self._path
        if path is None:
            return self._append(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(path.with_name(path.name + ".lock")):
            self._reload(path)
            self._persist(path, raw)
            return self._append(raw)

    def _append(self, raw: bytes) -> dict[str, Any]:
        entry = self._log.append(raw)
        proof = self._log.inclusion_proof(entry.log_index)
        return {
            "method": "scitt",
            "anchoredDigest": "sha256:" + raw.hex(),
            "logId": _log_id_digest(self._log_id),
            "leafIndex": entry.log_index,
            "treeSize": entry.tree_size_at_append,
            "inclusionProof": [_b64(s) for s in proof.siblings],
            "rootHash": _b64(entry.root_hash_at_append),
        }

    def head(self, consistency_from: Optional[int] = None) -> dict[str, Any]:
        """The current tree head, the value an independent party should hold.

        With ``consistency_from`` the head also carries the consistency proof
        from that earlier tree size, which is what a verifier needs to check
        an anchor made at that size against this head.
        """
        if self._path is not None:
            self._reload(self._path)
        size = self._log.tree_size
        out: dict[str, Any] = {
            "logId": _log_id_digest(self._log_id),
            "treeSize": size,
            "rootHash": _b64(self._log.root_hash),
        }
        if consistency_from is not None:
            if not 0 < consistency_from <= size:
                raise ScittAnchorError(
                    f"consistency_from {consistency_from} must be between 1 "
                    f"and the tree size {size}")
            proof = self._log.consistency_proof(consistency_from, size)
            out["consistency"] = {
                "firstSize": consistency_from,
                "hashes": [_b64(h) for h in proof.hashes],
            }
        return out


def _decode(value: Any, what: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ScittAnchorError(f"malformed {what}: {exc}") from exc


def _check_head(anchor: dict[str, Any], size: int, root: bytes,
                trusted_head: dict[str, Any]) -> tuple[bool, str]:
    """(witnessed, status) for the anchor's root against an independent head."""
    if trusted_head.get("logId") not in (None, anchor.get("logId")):
        return False, "INVALID: trusted head is from a different log"
    try:
        head_size = int(trusted_head["treeSize"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ScittAnchorError(f"malformed trusted head: {exc}") from exc
    head_root = _decode(trusted_head.get("rootHash"), "trusted head rootHash")
    if head_size == size:
        if head_root == root:
            return True, "root matches the trusted head"
        return False, "INVALID: root differs from the trusted head of the same size"
    if head_size < size:
        return False, "INVALID: trusted head is older than the anchor"
    consistency = trusted_head.get("consistency") or {}
    if not isinstance(consistency, dict) or consistency.get("firstSize") != size:
        return False, ("root not checked: the trusted head carries no "
                       f"consistency proof from tree size {size}")
    proof = ConsistencyProof(
        first_size=size, second_size=head_size,
        hashes=tuple(_decode(h, "consistency hash")
                     for h in consistency.get("hashes", [])),
    )
    verdict = verify_consistency(first_size=size, first_root=root,
                                 second_size=head_size, second_root=head_root,
                                 proof=proof)
    if verdict is ConsistencyVerdict.CONSISTENT:
        return True, f"root is a prefix of the trusted head (tree size {head_size})"
    return False, "INVALID: root is not consistent with the trusted head"


def verify_scitt_anchor(receipt: dict[str, Any], anchor: dict[str, Any], *,
                        trusted_head: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Verify a ``scitt`` anchor against its receipt.

    Always recomputes the Merkle root from the receipt digest and the
    inclusion proof, and compares it with the anchor's ``rootHash``. That
    root is the producer's own claim. ``trusted_head`` is a tree head the
    caller obtained independently (the output of ``ScittAnchor.head``, as
    published by the log operator); when given, the anchor's root is checked
    against it, and the anchor verifies only if it holds.

    Returns ``verified`` (bool), ``status`` (str), ``root_witnessed`` (bool),
    ``root`` (str, how the root was established), ``leaf_index`` and
    ``tree_size``. Raises ``ScittAnchorError`` if the anchor or the trusted
    head is malformed.
    """
    if anchor.get("method") != "scitt":
        raise ScittAnchorError(
            f"not a scitt anchor: method={anchor.get('method')!r}")

    leaf_index = anchor.get("leafIndex", -1)
    try:
        expected = "sha256:" + _signed_payload_digest(receipt).hex()
    except TimeAnchorError as exc:
        raise ScittAnchorError(str(exc)) from exc
    if anchor.get("anchoredDigest") != expected:
        return {"verified": False, "status": "anchoredDigest mismatch",
                "root_witnessed": False, "root": "not checked",
                "leaf_index": leaf_index}

    try:
        proof = InclusionProof(
            log_index=int(anchor["leafIndex"]),
            tree_size=int(anchor["treeSize"]),
            siblings=tuple(_decode(s, "inclusionProof entry")
                           for s in anchor.get("inclusionProof", [])),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ScittAnchorError(f"malformed scitt anchor: {exc}") from exc
    root_hash = _decode(anchor.get("rootHash"), "rootHash")

    leaf_data = bytes.fromhex(expected.split(":", 1)[1])
    if not verify_inclusion(leaf_data=leaf_data, proof=proof, expected_root=root_hash):
        return {"verified": False,
                "status": "INVALID: inclusion proof does not recompute to recorded root",
                "root_witnessed": False, "root": "not checked",
                "leaf_index": proof.log_index, "tree_size": proof.tree_size}

    if trusted_head is None:
        return {"verified": True, "status": "verified",
                "root_witnessed": False,
                "root": "carried in the anchor, not checked against an "
                        "independently held tree head",
                "leaf_index": proof.log_index, "tree_size": proof.tree_size}

    witnessed, root_status = _check_head(anchor, proof.tree_size, root_hash,
                                         trusted_head)
    return {"verified": witnessed,
            "status": "verified" if witnessed else root_status,
            "root_witnessed": witnessed, "root": root_status,
            "leaf_index": proof.log_index, "tree_size": proof.tree_size}


__all__ = [
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_ID",
    "ScittAnchor",
    "ScittAnchorError",
    "verify_scitt_anchor",
]
