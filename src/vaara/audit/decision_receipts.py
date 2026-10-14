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
arguments, so a receipt can leave the machine without them. It carries
``argumentsCommitment`` instead when the trail recorded one for the request:
a salted digest of the arguments, which the trail's holder can open for one
decision by handing over its arguments and salt (``_arguments_commitment``).

The signing key is made on first use at ``<trail dir>/keys/receipt-es256.pem``
(mode 0600) and its public half is written to
``<trail dir>/receipts/issuer-es256.pub.pem`` for verifiers. Signing needs
``cryptography`` and ``rfc8785``, both in the base install. An install made
without them (``--no-deps``, a vendored tree) records decisions as before,
emits no receipts, and logs a warning once saying so.
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

# What a deny rests on, read from the ground the trail recorded. "evaluated":
# a check ran and the action failed it. "not_evaluated": a check the decision
# needed never completed (the scorer raised or could not be reached, its
# answer could not be read), so the action was refused without being judged.
# A ground in neither set, or none at all, is "undetermined": the receipt
# never claims a check ran when the trail does not say so.
_EVALUATED_GROUNDS = frozenset({
    "risk_threshold", "policy_rule", "privilege_attenuation",
    "perimeter_filter", "parameter_schema", "egress_not_allowed",
})
_NOT_EVALUATED_GROUNDS = frozenset({
    "scorer_failure", "scorer_unavailable", "invalid_decision",
})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMITMENT = re.compile(r"^sha256:[0-9a-f]{64}$")
# Requests whose decision has not been recorded yet, by action id. Bounded:
# an action decided in another process, or never decided, must not grow it.
_MAX_PENDING_COMMITMENTS = 4096
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

# One warning per process when the signing libraries are missing.
_warned_unsigned = False


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


def build_evidence(
    record: AuditRecord, arguments_commitment: Optional[str] = None,
) -> dict[str, Any]:
    """The ``vaara.trail-decision/v0`` evidence record for one decision record.

    ``arguments_commitment`` is the commitment the trail recorded on the
    action's request, when the caller has it.
    """
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
    if arguments_commitment and _COMMITMENT.match(arguments_commitment):
        evidence["argumentsCommitment"] = arguments_commitment
    if data.get("decision_detail"):
        evidence["decisionDetail"] = str(data["decision_detail"])
    if data.get("approver"):
        evidence["approver"] = str(data["approver"])
        evidence["humanDisposed"] = bool(data.get("human_disposed", False))
    if evidence["decision"] == "deny":
        evidence["refusal"] = _refusal_block(data)
    cage = data.get("cage")
    if isinstance(cage, dict) and cage.get("driver"):
        # The cage block as the trail recorded it: which cage the deciding
        # process ran in and whether the kernel confirmed it at that moment.
        # Camel-cased like the rest of the profile; a record written before
        # the cage layer has no block and the receipt says nothing.
        evidence["cage"] = _cage_block(cage, record.record_id)
    return evidence


def _refusal_block(data: dict[str, Any]) -> dict[str, Any]:
    """What a deny rests on: a failed check, a check that never completed, or unknown.

    A refusal word alone cannot tell a control that held from a control that
    was never reached. The block separates them, and where a check could not
    complete it names the input that was missing, as the trail recorded it.
    """
    ground = str(data.get("violation_type") or "")
    if ground in _EVALUATED_GROUNDS:
        basis = "evaluated"
    elif ground in _NOT_EVALUATED_GROUNDS:
        basis = "not_evaluated"
    else:
        basis = "undetermined"
    block: dict[str, Any] = {"basis": basis}
    if ground:
        block["violationType"] = ground
    if data.get("policy_id"):
        block["policyId"] = str(data["policy_id"])
    if basis == "not_evaluated" and data.get("unavailable_input"):
        block["unavailableInput"] = str(data["unavailable_input"])
    return block


def _cage_block(cage: dict[str, Any], record_id: str) -> dict[str, Any]:
    """The receipt's cage block, held to the rules a verifier applies.

    The trail keeps what it was given; the receipt never states more than
    that. A ``configDigest`` that is not ``sha256:`` and 64 lowercase hex is
    left out, and ``confirmed: true`` without a basis naming the fact that
    was read (or on ``driver: none``) is written as false. Both happen only
    when custom code passes a hand-built block to ``record_decision``.
    """
    driver = str(cage["driver"])
    # Only the boolean true confirms. A string "false", or any other value a
    # hand-built block carries, is unconfirmed; truthiness would sign it.
    confirmed = cage.get("confirmed", False) is True
    if driver == "none":
        if confirmed:
            logger.warning("cage block for record_id=%s claims confirmed with "
                           "driver none; writing confirmed false", record_id)
        return {"driver": driver, "confirmed": False}
    block: dict[str, Any] = {"driver": driver, "confirmed": confirmed}
    for src, dst in (("upstream", "upstream"), ("config_digest", "configDigest"),
                     ("basis", "basis"), ("name", "name")):
        if cage.get(src):
            block[dst] = str(cage[src])
    digest = block.get("configDigest")
    if digest is not None and not (digest.startswith("sha256:") and _HEX64.match(digest[7:])):
        logger.warning("cage block for record_id=%s has configDigest %r, not "
                       "sha256 hex; leaving it out", record_id, digest)
        del block["configDigest"]
    block.setdefault("basis", "declared")
    if confirmed and block["basis"] in ("none", "declared"):
        logger.warning("cage block for record_id=%s claims confirmed on basis "
                       "%r; writing confirmed false", record_id, block["basis"])
        block["confirmed"] = False
    return block


def _ignore_in_git(key_dir: Path) -> None:
    """Keep the key directory out of git wherever the trail lives.

    A trail can sit in a project folder (the MCP server and proxy default to
    one in the working directory), and the private key beside it must not be
    one ``git add .`` away from a commit.
    """
    ignore = key_dir / ".gitignore"
    if ignore.exists():
        return
    try:
        if not key_dir.exists():
            key_dir.mkdir(parents=True)
            os.chmod(key_dir, 0o700)
        # Only the key file: a keys/ folder that was already there may hold
        # files its owner tracks on purpose.
        ignore.write_text(f"{KEY_RELPATH.name}\n.gitignore\n", encoding="utf-8", newline="\n")
    except OSError:
        logger.warning("could not write %s; keep %s out of version control", ignore, key_dir)


def _load_or_create_key(key_path: Path, pub_path: Path) -> Any:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    _ignore_in_git(key_path.parent)
    if key_path.exists():
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ValueError(f"{key_path} is not an EC private key")
    else:
        if not key_path.parent.exists():
            key_path.parent.mkdir(parents=True)
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
    _pending: dict[str, str] = field(default_factory=dict, repr=False)

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

    def prepare(self) -> None:
        """Load the key and import the code that mint() needs, ahead of the first decision.

        For a caller that must not import anything once it is serving. The OS
        guard is one: it holds every open on a mount until one of its threads
        answers, and a compiled module imported in a worker thread is opened
        with the GIL held, so the thread that would answer never runs.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        from vaara.attestation import _attest_canonical, decision  # noqa: F401

        with self._lock:
            # One signature nobody keeps, so whatever the library sets up on
            # first use (providers, the random pool) is set up now.
            self._signing_key().sign(b"vaara", ec.ECDSA(hashes.SHA256()))

    def mint(self, record: AuditRecord) -> dict[str, Any]:
        """Build and sign the receipt file contents for one decision record."""
        from vaara.attestation.decision import (
            BackLink,
            DecisionDerived,
            EvidenceRef,
            emit_decision_record,
        )

        evidence = build_evidence(record, self._pending.pop(record.action_id, None))
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

    def _remember_commitment(self, record: AuditRecord) -> None:
        commitment = (record.data or {}).get("arguments_commitment")
        if not isinstance(commitment, str) or not record.action_id:
            return
        with self._lock:
            if len(self._pending) >= _MAX_PENDING_COMMITMENTS:
                for stale in list(self._pending)[:_MAX_PENDING_COMMITMENTS // 8]:
                    del self._pending[stale]
            self._pending[record.action_id] = commitment

    def __call__(self, record: AuditRecord) -> Optional[Path]:
        if record.event_type == EventType.ACTION_REQUESTED:
            self._remember_commitment(record)
            return None
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
                tmp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
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
        global _warned_unsigned
        if not _warned_unsigned:
            _warned_unsigned = True
            logger.warning(
                "decision receipts are OFF: cryptography and rfc8785 are not "
                "importable, so decisions are recorded without a signed receipt. "
                "Reinstall vaara with its dependencies to turn them on."
            )
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
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if public_key_pem is None:
        root = Path(path).resolve().parent.parent
        public_key_pem = (root / PUBKEY_NAME).read_bytes()
    return verify_receipt_body(
        body, public_key_pem=public_key_pem, trail_hashes=trail_hashes, path=str(path),
    )


def verify_receipt_body(
    body: dict[str, Any],
    *,
    public_key_pem: bytes,
    trail_hashes: Optional[dict[str, str]] = None,
    path: str = "",
) -> ReceiptCheck:
    """Verify a receipt already in memory: the checks verify_receipt_file runs.

    Raises ValueError, KeyError or TypeError when ``body`` is not a receipt
    or the key is not a PEM public key.
    """
    from cryptography.hazmat.primitives import serialization

    from vaara.attestation._attest_types import AttestationError
    from vaara.attestation.decision import parse_decision_record, verify_decision_signature

    if not isinstance(body, dict):
        raise ValueError("not a decision receipt: the file is not a JSON object")
    envelope = body["receipt"]
    evidence = body["evidence"]
    if not isinstance(envelope, dict) or not isinstance(evidence, dict):
        raise ValueError("not a decision receipt: receipt and evidence must be objects")
    public_key = serialization.load_pem_public_key(public_key_pem)
    try:
        record = parse_decision_record(envelope)
    except AttestationError as exc:
        raise ValueError(f"not a decision receipt: {exc}") from None
    signature_ok = verify_decision_signature(record, verifying_material=public_key)
    ref = envelope.get("decisionDerived", {}).get("evidenceRef", {})
    evidence_ok = ref.get("digest") == _digest(evidence)
    trail_ok: Optional[bool] = None
    detail = ""
    if trail_hashes is not None:
        record_id = evidence.get("recordId")
        stored = trail_hashes.get(record_id) if isinstance(record_id, str) else None
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


def check_disclosed_arguments(evidence: dict[str, Any], arguments: Any, salt: str) -> bool:
    """True when disclosed ``arguments`` and ``salt`` open the receipt's commitment.

    The trail's holder discloses both from the action's request record. A
    receipt with no ``argumentsCommitment`` opens to nothing and returns False.
    """
    from vaara.audit._arguments_commitment import verify

    commitment = evidence.get("argumentsCommitment")
    return isinstance(commitment, str) and verify(commitment, arguments, salt)


DISCLOSURE_SCHEMA = "vaara.arguments-disclosure/v0"


def disclose_arguments(evidence: dict[str, Any], db_path: Path) -> dict[str, Any]:
    """The disclosure that opens one receipt's ``argumentsCommitment``.

    Reads the action's request record from the trail database, read-only, and
    returns its recorded arguments and salt. Raises ``ValueError`` when the
    receipt carries no commitment, the request is not on the trail, or the
    trail's arguments do not open the receipt's commitment.
    """
    import sqlite3

    commitment = evidence.get("argumentsCommitment")
    if not isinstance(commitment, str):
        raise ValueError("the receipt carries no argumentsCommitment")
    action_id = str(evidence.get("actionId", ""))
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT data FROM audit_records WHERE action_id = ? AND event_type = ?",
            (action_id, EventType.ACTION_REQUESTED.value),
        ).fetchall()
    finally:
        con.close()
    for (raw,) in rows:
        data = json.loads(raw)
        disclosure = {
            "schema": DISCLOSURE_SCHEMA,
            "recordId": str(evidence.get("recordId", "")),
            "actionId": action_id,
            "arguments": data.get("parameters", {}),
            "salt": str(data.get("arguments_salt", "")),
        }
        if check_disclosure(evidence, disclosure):
            return disclosure
    raise ValueError(f"no request on the trail opens the commitment for action {action_id}")


def check_disclosure(evidence: dict[str, Any], disclosure: dict[str, Any]) -> bool:
    """True when a disclosure names this receipt's decision and opens its commitment."""
    if disclosure.get("schema") != DISCLOSURE_SCHEMA:
        return False
    if disclosure.get("actionId") != evidence.get("actionId"):
        return False
    return check_disclosed_arguments(
        evidence, disclosure.get("arguments"), str(disclosure.get("salt", "")))
