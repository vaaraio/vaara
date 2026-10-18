# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-receipt clock trust: what ``receiptAsserted.timeSource`` lets a verifier say.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt``.

``iat`` is the issuer's wall clock, and a verifier holding only the receipt
cannot tell a synced host from one that booted at the epoch and never synced.
``timeSource`` is the issuer's signed declaration of where the time came from.
The check reads it and never upgrades it: an absent or ``unsynced`` source is
reported as such, an ``ntp`` or ``rtc`` source is the issuer's word, and only
``tsa`` backed by a token that verifies over the record is a trusted time.

A ``tsa`` claim with no token, or with a token that fails, is a finding: the
receipt asserts a witnessed time it cannot show. A token that is present but
cannot be checked because the ``timeanchor`` extra is missing is reported
``unchecked``, not trusted and not a finding, since the gap is the verifier's
install and not the receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from vaara.attestation._receipt_types import TIME_SOURCES

TimeSourceVerdict = Literal[
    "unknown", "unsynced", "declared", "attested", "unchecked",
    "unverifiable", "invalid",
]


@dataclass(frozen=True)
class TimeSourceResult:
    """Outcome of a time-source check on one execution record.

    ``verdict`` is one of:

    * ``"unknown"``: no ``timeSource``. ``iat`` is a wall time of unknown trust.
    * ``"unsynced"``: the issuer declared its clock unsynced.
    * ``"declared"``: ``ntp`` or ``rtc``. The issuer's signed word, unwitnessed.
    * ``"attested"``: ``tsa`` and the ``existenceProof`` token verifies over
      the record. The only trusted verdict.
    * ``"unchecked"``: ``tsa`` with a token, but the ``timeanchor`` extra is
      not installed here, so it was not checked.
    * ``"unverifiable"``: ``tsa`` with no token, or one that fails. A finding.
    * ``"invalid"``: ``timeSource`` is not one of the defined values, or the
      record has no ``receiptAsserted`` object. A finding.

    Only ``"attested"`` is truthy. ``attested_time`` is the token's time when
    attested; it bounds when the record existed, it does not replace ``iat``.
    """

    verdict: TimeSourceVerdict
    declared: Optional[str]
    trusted: bool
    finding: bool
    attested_time: Optional[str]
    reason: str

    def __bool__(self) -> bool:
        return self.trusted

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "declared": self.declared,
            "trusted": self.trusted,
            "finding": self.finding,
            "attestedTime": self.attested_time,
            "reason": self.reason,
        }


def _result(
    verdict: TimeSourceVerdict, declared: Optional[str], reason: str, *,
    trusted: bool = False, finding: bool = False,
    attested_time: Optional[str] = None,
) -> TimeSourceResult:
    return TimeSourceResult(
        verdict=verdict, declared=declared, trusted=trusted, finding=finding,
        attested_time=attested_time, reason=reason,
    )


def verify_receipt_time_source(
    record: dict[str, Any],
    *,
    trusted_issuer_cert: Optional[bytes] = None,
    trusted_signer_cert: Optional[bytes] = None,
) -> TimeSourceResult:
    """Grade the time a record's ``iat`` can be trusted for. Never raises.

    Takes the wire dict because a ``tsa`` claim is checked against the
    record's ``existenceProof``, which rides outside the signed preimage.
    Does not verify the receipt signature; run ``verify_receipt_signature``
    first. The trust pins are passed through to ``verify_existence_proof``
    and only decide whether an attested time is qualified, not whether it
    is attested.
    """
    ra = record.get("receiptAsserted") if isinstance(record, dict) else None
    if not isinstance(ra, dict):
        return _result("invalid", None, "record has no receiptAsserted object",
                       finding=True)
    if "timeSource" not in ra:
        return _result("unknown", None,
                       "receipt declares no timeSource; iat is a wall time of "
                       "unknown trust")
    source = ra["timeSource"]
    if not (isinstance(source, str) and source in TIME_SOURCES):
        return _result("invalid", None,
                       f"timeSource must be one of {sorted(TIME_SOURCES)}; "
                       f"got {source!r}", finding=True)
    if source == "unsynced":
        return _result("unsynced", source,
                       "issuer declared its clock unsynced; iat is not a "
                       "trusted time")
    if source in ("ntp", "rtc"):
        return _result("declared", source,
                       f"issuer declared a {source} clock; iat is its word, "
                       "not a witnessed time")

    # source == "tsa": the claim stands or falls on the token.
    if not isinstance(record.get("existenceProof"), dict):
        return _result("unverifiable", source,
                       "timeSource is tsa but the record carries no "
                       "existenceProof token", finding=True)
    from vaara.attestation._receipt_existence import verify_existence_proof

    try:
        ex = verify_existence_proof(
            record, trusted_issuer_cert=trusted_issuer_cert,
            trusted_signer_cert=trusted_signer_cert,
        )
    except ImportError:
        return _result("unchecked", source,
                       "timeSource is tsa and a token is present, but the "
                       "timeanchor extra is not installed to check it")
    if not ex.ok:
        return _result("unverifiable", source,
                       f"timeSource is tsa but the token does not check: "
                       f"{ex.reason}", finding=True)
    return _result("attested", source,
                   f"time attested by a {ex.basis} timestamp over the record",
                   trusted=True, attested_time=ex.attested_time)
