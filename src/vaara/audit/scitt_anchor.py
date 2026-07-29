from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any, Optional

from vaara.attestation.transparency_log import (
    InProcessTransparencyLog,
    InclusionProof,
    verify_inclusion,
)
from vaara.audit.receipt_anchor import _signed_payload_digest


class ScittAnchorError(RuntimeError):
    """Raised when a SCITT anchor cannot be produced or verified."""


class ScittAnchor:
    """Produce ``scitt`` witness anchors over Vaara receipts.

    Each anchor commits the receipt's signed-payload digest to a Merkle
    transparency log and stores the inclusion proof. The log is append-only
    so a later verifier can recompute the root from (leaf, proof) alone.

    This is analogous to how the ``rfc3161`` anchor uses a self-hosted TSA:
    the log is local by default. A remote SCITT-compatible transparency log
    (e.g. Sigstore Rekor) can be wired in by subclassing or adapting the
    ``_append_and_prove`` interface.
    """

    def __init__(self, log: Optional[InProcessTransparencyLog] = None,
                 log_id: str = "vaara-scitt-log") -> None:
        self._log = log or InProcessTransparencyLog()
        self._log_id = log_id

    @classmethod
    def load_or_create(cls, directory: Optional[Path] = None,
                       log_id: str = "vaara-scitt-log") -> ScittAnchor:
        return cls(log_id=log_id)

    def anchor_receipt(self, receipt: dict) -> dict[str, Any]:
        raw = _signed_payload_digest(receipt)
        entry = self._log.append(raw)
        proof = self._log.inclusion_proof(entry.log_index)
        return _build_anchor_entry(entry.log_index, int(entry.tree_size_at_append),
                                   proof, self._log.root_hash, self._log_id, raw)


def _build_anchor_entry(
    log_index: int,
    tree_size: int,
    proof: InclusionProof,
    root_hash: bytes,
    log_id: str,
    leaf_data: bytes,
) -> dict[str, Any]:
    return {
        "method": "scitt",
        "anchoredDigest": "sha256:" + leaf_data.hex(),
        "logId": base64.b64encode(hashlib.sha256(log_id.encode()).digest()).decode("ascii"),
        "leafIndex": log_index,
        "treeSize": tree_size,
        "inclusionProof": [base64.b64encode(s).decode("ascii") for s in proof.siblings],
        "rootHash": base64.b64encode(root_hash).decode("ascii"),
    }


def verify_scitt_anchor(receipt: dict, anchor: dict) -> dict[str, Any]:
    """Verify a ``scitt`` anchor against its receipt by recomputing inclusion.

    Returns a dict with ``verified`` (bool), ``status`` (str), and
    ``leaf_index`` (int). Raises ``ScittAnchorError`` if the anchor's
    structure is malformed.
    """
    if anchor.get("method") != "scitt":
        raise ScittAnchorError(
            f"not a scitt anchor: method={anchor.get('method')!r}")

    expected = "sha256:" + _signed_payload_digest(receipt).hex()
    if anchor.get("anchoredDigest") != expected:
        return {"verified": False, "status": "anchoredDigest mismatch",
                "leaf_index": anchor.get("leafIndex", -1)}

    try:
        proof = InclusionProof(
            log_index=int(anchor["leafIndex"]),
            tree_size=int(anchor["treeSize"]),
            siblings=tuple(
                base64.b64decode(s) for s in anchor.get("inclusionProof", [])
            ),
        )
        root_hash = base64.b64decode(anchor["rootHash"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ScittAnchorError(f"malformed scitt anchor: {exc}") from exc

    leaf_data = bytes.fromhex(expected.split(":", 1)[1])
    ok = verify_inclusion(leaf_data=leaf_data, proof=proof, expected_root=root_hash)

    return {
        "verified": ok,
        "status": "verified" if ok else "INVALID: inclusion proof does not recompute to recorded root",
        "leaf_index": proof.log_index,
        "tree_size": proof.tree_size,
    }


__all__ = ["ScittAnchor", "ScittAnchorError", "verify_scitt_anchor"]
