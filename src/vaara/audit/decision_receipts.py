# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Signed decision receipts, emitted by the engine for every decision it records.

Every allow, escalate and deny the trail records also leaves a signed
``vaara.receipt/v1`` envelope (SPEC.md) next to the trail, one file per
decision under ``<trail dir>/receipts/<YYYY-MM-DD>/``. The file holds the
envelope and the evidence record it is addressed to, so a reader verifies it
with the issuer's public key and nothing else: the ES256 signature over the
envelope, then ``sha256(JCS(evidence)) == evidenceRef.digest``.

The evidence record is the ``vaara.trail-decision/v0`` profile: the decision
as the trail recorded it, plus the trail record's own hash. A reader holding
the trail looks the record up by ``recordId`` and confirms the hash, which
ties the receipt to its place in the hash chain. The record carries no tool
arguments, so a receipt can leave the machine without them.

The signing key is made on first use at ``<trail dir>/keys/receipt-es256.pem``
(mode 0600) and its public half is written to
``<trail dir>/receipts/issuer-es256.pub.pem`` for verifiers. Signing needs
``cryptography`` and ``rfc8785`` (``pip install 'vaara[attestation]'``);
without them the engine records decisions as before and emits no receipts.
``VAARA_RECEIPTS=0`` turns emission off.

Emission never blocks a decision. The decision is on the chain before its
receipt is written; a receipt that fails to write is logged and counted, and
the missing file is visible as a trail decision with no receipt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from vaara.audit.trail import AuditRecord, EventType

logger = logging.getLogger(__name__)

SCHEMA = "vaara.trail-decision/v0"
ISSUER = "vaara:engine"
POLICY_ID = "policy:vaara-engine/1"
RECEIPTS_DIRNAME = "receipts"
KEY_RELPATH = Path("keys") / "receipt-es256.pem"
PUBKEY_NAME = "issuer-es256.pub.pem"

# The trail's coarse decision word, as the envelope's verdict vocabulary.
_VERDICT = {"allow": "allow", "escalate": "escalate", "deny": "block"}
_DECISION_EVENTS = (EventType.DECISION_MADE, EventType.ACTION_BLOCKED)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def signing_available() -> bool:
    """True when ``cryptography`` and ``rfc8785`` are importable."""
    try:
        import cryptography  # noqa: F401
        import rfc8785  # noqa: F401
    except ImportError:
        return False
    return True


def _digest(obj: Any) -> str:
    from vaara.attestation._attest_canonical import canonical_json

    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


def _iso(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _decimal(value: Any) -> str:
    """A risk score as a decimal string: floats are banned on the wire."""
    try:
        return format(float(value), ".6f").rstrip("0").rstrip(".") or "0"
    except (TypeError, ValueError):
        return "0"


def _chain_digest(value: str) -> str:
    """A trail hash as a ``sha256:`` digest. The genesis link is empty."""
    if _HEX64.match(value or ""):
        return "sha256:" + value
    return "sha256:" + hashlib.sha256((value or "").encode()).hexdigest()


def build_evidence(record: AuditRecord) -> dict[str, Any]:
    """The ``vaara.trail-decision/v0`` evidence record for one decision record."""
    data = record.data or {}
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "recordId": record.record_id,
        "actionId": record.action_id,
        "eventType": record.event_type.value,
        "agentId": record.agent_id,
        "toolName": record.tool_name,
        "tenantId": record.tenant_id or "",
        "decision": str(data.get("decision", "")),
        "reason": str(data.get("reason", "")),
        "riskScore": _decimal(data.get("risk_score", 0)),
        "decidedAt": _iso(record.timestamp),
        "recordHash": _chain_digest(record.record_hash),
        "previousHash": _chain_digest(record.previous_hash),
    }
    if data.get("decision_detail"):
        evidence["decisionDetail"] = str(data["decision_detail"])
    if data.get("approver"):
        evidence["approver"] = str(data["approver"])
        evidence["humanDisposed"] = bool(data.get("human_disposed", False))
    return evidence


def _load_or_create_key(key_path: Path, pub_path: Path) -> Any:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    if key_path.exists():
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ValueError(f"{key_path} is not an EC private key")
    else:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(key_path.parent, 0o700)
        key = ec.generate_private_key(ec.SECP256R1())
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        # O_EXCL: two processes making the first key at once must not both
        # win; the loser reads the winner's key instead.
        try:
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return _load_or_create_key(key_path, pub_path)
        with os.fdopen(fd, "wb") as fh:
            fh.write(pem)
    pub_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if not pub_path.exists() or pub_path.read_bytes() != pub_pem:
        pub_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = pub_path.with_suffix(".tmp")
        tmp.write_bytes(pub_pem)
        os.replace(tmp, pub_path)
    return key


def key_id(public_key: Any) -> str:
    """``es256:`` plus the first 16 hex of sha256 over the SPKI DER."""
    from cryptography.hazmat.primitives import serialization

    der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return "es256:" + hashlib.sha256(der).hexdigest()[:16]


@dataclass
class DecisionReceiptSink:
    """Writes one signed receipt per decision record appended to a trail."""

    trail_dir: Path
    key_path: Optional[Path] = None
    written: int = 0
    failures: int = 0
    _key: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def receipts_dir(self) -> Path:
        return self.trail_dir / RECEIPTS_DIRNAME

    @property
    def public_key_path(self) -> Path:
        return self.receipts_dir / PUBKEY_NAME

    def _signing_key(self) -> Any:
        if self._key is None:
            self._key = _load_or_create_key(
                self.key_path or self.trail_dir / KEY_RELPATH, self.public_key_path
            )
        return self._key

    def mint(self, record: AuditRecord) -> dict[str, Any]:
        """Build and sign the receipt file contents for one decision record."""
        from vaara.attestation.decision import (
            BackLink,
            DecisionDerived,
            EvidenceRef,
            emit_decision_record,
        )

        evidence = build_evidence(record)
        key = self._signing_key()
        envelope = emit_decision_record(
            back_link=BackLink(
                attestation_digest=evidence["previousHash"],
                attestation_nonce=record.record_id,
            ),
            decision_derived=DecisionDerived(
                decision=_VERDICT.get(evidence["decision"], "block"),  # type: ignore[arg-type]
                decided_at=evidence["decidedAt"],
                reason=evidence["reason"] or None,
                risk_score=evidence["riskScore"],
                policy_id=POLICY_ID,
                evidence_ref=EvidenceRef(
                    digest=_digest(evidence),
                    canonicalization="jcs-rfc8785",
                    schema=SCHEMA,
                    ref=f"vaara:trail/{record.record_id}",
                ),
            ),
            iss=ISSUER,
            sub=record.agent_id or "unknown",
            secret_version=key_id(key.public_key()),
            alg="ES256",
            signing_material=key,
        )
        return {"receipt": envelope.to_dict(), "evidence": evidence}

    def __call__(self, record: AuditRecord) -> Optional[Path]:
        if record.event_type not in _DECISION_EVENTS:
            return None
        try:
            with self._lock:
                body = self.mint(record)
                day = body["evidence"]["decidedAt"][:10]
                out_dir = self.receipts_dir / day
                out_dir.mkdir(parents=True, exist_ok=True)
                name = _SAFE_NAME.sub("_", record.record_id) + ".json"
                path = out_dir / name
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
                os.replace(tmp, path)
                self.written += 1
                return path
        except Exception:
            self.failures += 1
            logger.exception(
                "decision receipt not written for record_id=%s; the decision "
                "stands on the trail",
                record.record_id,
            )
            return None


def default_sink(db_path: Any) -> Optional[DecisionReceiptSink]:
    """The sink for a trail database, or None when receipts are off.

    Off when ``VAARA_RECEIPTS`` is ``0``/``off``/``false``, when the trail is
    in memory, or when the signing libraries are not installed.
    """
    if os.environ.get("VAARA_RECEIPTS", "").strip().lower() in ("0", "off", "false", "no"):
        return None
    path = str(db_path)
    if not path or path == ":memory:" or path.startswith("file::memory:"):
        return None
    if not signing_available():
        return None
    return DecisionReceiptSink(trail_dir=Path(path).expanduser().resolve().parent)


@dataclass(frozen=True)
class ReceiptCheck:
    """What verifying one receipt file found."""

    path: str
    signature_ok: bool
    evidence_ok: bool
    # None when no trail was given to check against.
    trail_ok: Optional[bool]
    decision: str
    tool: str
    decided_at: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.signature_ok and self.evidence_ok and self.trail_ok is not False


def verify_receipt_file(
    path: Path,
    *,
    public_key_pem: Optional[bytes] = None,
    trail_hashes: Optional[dict[str, str]] = None,
) -> ReceiptCheck:
    """Verify one receipt file: signature, evidence digest, and trail record.

    ``public_key_pem`` defaults to ``issuer-es256.pub.pem`` in the receipts
    directory the file sits under. ``trail_hashes`` maps record id to trail
    record hash; when given, the evidence's ``recordHash`` must match.
    """
    from cryptography.hazmat.primitives import serialization

    from vaara.attestation.decision import parse_decision_record, verify_decision_signature

    body = json.loads(Path(path).read_text())
    envelope = body["receipt"]
    evidence = body["evidence"]
    if public_key_pem is None:
        root = Path(path).resolve().parent.parent
        public_key_pem = (root / PUBKEY_NAME).read_bytes()
    public_key = serialization.load_pem_public_key(public_key_pem)

    record = parse_decision_record(envelope)
    signature_ok = verify_decision_signature(record, verifying_material=public_key)
    ref = envelope.get("decisionDerived", {}).get("evidenceRef", {})
    evidence_ok = ref.get("digest") == _digest(evidence)
    trail_ok: Optional[bool] = None
    detail = ""
    if trail_hashes is not None:
        stored = trail_hashes.get(evidence.get("recordId", ""))
        if stored is None:
            trail_ok = False
            detail = "record not in trail"
        else:
            trail_ok = _chain_digest(stored) == evidence.get("recordHash")
            if not trail_ok:
                detail = "trail record hash differs"
    if not signature_ok:
        detail = "signature does not verify"
    elif not evidence_ok:
        detail = "evidence digest does not match"
    return ReceiptCheck(
        path=str(path),
        signature_ok=signature_ok,
        evidence_ok=evidence_ok,
        trail_ok=trail_ok,
        decision=str(envelope.get("decisionDerived", {}).get("decision", "")),
        tool=str(evidence.get("toolName", "")),
        decided_at=str(evidence.get("decidedAt", "")),
        detail=detail,
    )
