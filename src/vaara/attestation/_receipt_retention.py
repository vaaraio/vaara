"""Verify a retained execution record under a rotated or retired key.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt``.

The 7-year problem. EU AI Act Article 12 records must stay verifiable across
a retention window measured in years, but signing keys rotate inside that
window. A record signed in 2026 by ``#key-2026`` is audited in 2031, by which
time the issuer rotated to ``#key-2031`` and retired the old key. Three things
have to hold for the old record to still verify, and no single existing lens
checks all three:

1. **Binding.** The signature still verifies under the old key (level-2
   offline identity against the DID document the regulator *archived*, since
   the live document no longer lists the retired key).
2. **Validity window.** The claimed ``iat`` falls inside the key's
   ``[not_before, not_after)`` window (:mod:`_key_history`). A key already
   retired, or not yet activated, at the claimed time does not bind.
3. **Revocation.** The key was not revoked at or before the claimed signing
   time (the cross-stack :class:`RevocationRegistry` rule). Revocation
   overrides a graceful retirement.

These three settle ``verifiable``. But ``iat`` is self-asserted, and an
attacker who later steals a *retired* key could forge a record and backdate
``iat`` into the old window. The defence is the eIDAS RFC 3161 time anchor
(:mod:`vaara.audit.timeanchor`): a trusted authority attests the record
existed no later than ``anchored_time``. When that anchor predates the key's
retirement (and any revocation), the in-window claim cannot be a later forgery,
and the verdict is ``corroborated``. Without an anchor, the verdict rests on
the record's own clock and says so.

The window and revocation checks are pure standard library; binding needs the
``cryptography`` of the attestation extra, imported lazily through
``verify_receipt_identity``. See ``docs/design/key-rotation-retention-spec.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._key_history import KeyHistory
from vaara.attestation._receipt_identity import verify_receipt_identity
from vaara.attestation._receipt_types import ExecutionReceipt
from vaara.attestation._revocation import RevocationRegistry, _parse_iso


@dataclass(frozen=True)
class RetentionResult:
    """Verdict of a retained-record check under a rotated or retired key.

    ``verifiable`` is the offline verdict: the signature bound to a key the
    archived document lists, the claimed ``issued_at`` was inside that key's
    validity window, and the key was not revoked at or before issuance.

    ``corroborated`` is the stronger tier: ``verifiable`` *and* a verified time
    anchor proves the record existed before the key was retired and before any
    revocation. ``time_basis`` is ``"anchored"`` when an anchor was supplied,
    else ``"self_asserted"``.
    """

    bound: bool
    keyid: Optional[str]
    within_window: bool
    window_recorded: bool
    not_before: Optional[str]
    not_after: Optional[str]
    revoked: bool
    revoked_at: Optional[str]
    issued_at: Optional[str]
    time_basis: str
    anchored_time: Optional[str]
    anchored_before_retirement: bool
    anchored_before_revocation: bool
    verifiable: bool
    corroborated: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bound": self.bound,
            "keyid": self.keyid,
            "within_window": self.within_window,
            "window_recorded": self.window_recorded,
            "not_before": self.not_before,
            "not_after": self.not_after,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
            "issued_at": self.issued_at,
            "time_basis": self.time_basis,
            "anchored_time": self.anchored_time,
            "anchored_before_retirement": self.anchored_before_retirement,
            "anchored_before_revocation": self.anchored_before_revocation,
            "verifiable": self.verifiable,
            "corroborated": self.corroborated,
            "reason": self.reason,
        }


def _anchor_before(anchored_time: Optional[str], boundary: Optional[str]) -> bool:
    """Whether the anchor proves existence strictly before ``boundary``.

    True when there is no boundary to beat (the key never retired, or was
    never revoked). When a boundary exists, the anchor must be present and
    strictly earlier; an unparseable instant fails closed.
    """
    if boundary is None:
        return True
    if anchored_time is None:
        return False
    a = _parse_iso(anchored_time)
    b = _parse_iso(boundary)
    if a is None or b is None:
        return False
    return a < b


def verify_receipt_retained(
    receipt: ExecutionReceipt,
    did_document: dict[str, Any],
    *,
    key_history: Optional[KeyHistory] = None,
    revocations: Optional[RevocationRegistry] = None,
    anchored_time: Optional[str] = None,
    expected_keyid: Optional[str] = None,
) -> RetentionResult:
    """Verify a record under a key that may since have rotated out.

    Binds the receipt signature to a key in the archived ``did_document``
    (level-2, offline), then judges the bound key's validity window and
    revocation at the receipt's claimed ``iat``. ``key_history`` defaults to
    the windows the document itself records (``validFrom`` / ``validUntil`` per
    method); ``revocations`` defaults to the document's ``revoked`` markers.
    Pass ``anchored_time`` (the attested time from a verified
    :class:`TimeAnchor`) to upgrade a pass to ``corroborated``.
    """
    identity = verify_receipt_identity(
        receipt, did_document, expected_keyid=expected_keyid
    )
    issued_at = receipt.receipt_asserted.iat
    time_basis = "anchored" if anchored_time is not None else "self_asserted"

    if not identity.bound:
        return RetentionResult(
            bound=False, keyid=identity.keyid, within_window=False,
            window_recorded=False, not_before=None, not_after=None,
            revoked=False, revoked_at=None, issued_at=issued_at,
            time_basis=time_basis, anchored_time=anchored_time,
            anchored_before_retirement=False, anchored_before_revocation=False,
            verifiable=False, corroborated=False,
            reason=f"signature not bound to any archived key: {identity.reason}",
        )

    history = key_history or KeyHistory.from_did_document(did_document)
    registry = revocations or RevocationRegistry.from_did_document(
        did_document, receipt.receipt_asserted.iss
    )

    window = history.validity(identity.keyid, issued_at)
    revocation = registry.status(
        receipt.receipt_asserted.iss, issued_at, keyid=identity.keyid
    )

    verifiable = window.within and not revocation.revoked
    abr = _anchor_before(anchored_time, window.not_after)
    abrev = _anchor_before(anchored_time, revocation.revoked_at)
    has_anchor = anchored_time is not None
    corroborated = verifiable and has_anchor and abr and abrev

    return RetentionResult(
        bound=True, keyid=identity.keyid, within_window=window.within,
        window_recorded=window.recorded, not_before=window.not_before,
        not_after=window.not_after, revoked=revocation.revoked,
        revoked_at=revocation.revoked_at, issued_at=issued_at,
        time_basis=time_basis, anchored_time=anchored_time,
        anchored_before_retirement=abr if has_anchor else False,
        anchored_before_revocation=abrev if has_anchor else False,
        verifiable=verifiable, corroborated=corroborated,
        reason=_retention_reason(
            window, revocation, corroborated, anchored_time
        ),
    )


def _retention_reason(
    window: Any, revocation: Any, corroborated: bool, anchored_time: Optional[str],
) -> str:
    if not window.within:
        return str(window.reason)
    if revocation.revoked:
        return str(revocation.reason)
    if corroborated:
        return ("verifiable under a key valid at issuance, and a time anchor "
                "proves the record existed before the key's end of life")
    if anchored_time is not None:
        return ("verifiable under a key valid at issuance, but the time anchor "
                "does not predate the key's retirement or revocation")
    return ("verifiable under a key valid at issuance, on the record's "
            "self-asserted time; supply a time anchor to corroborate existence "
            "over the retention window")


__all__ = ["RetentionResult", "verify_receipt_retained"]
