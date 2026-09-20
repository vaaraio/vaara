"""Audience binding for execution receipts, and the three-valued check.

The case this closes was described on audit@ietf.org on 2026-09-19 by an
operator of a payments authorization layer: every verdict signed, refusals on
the same path as approvals, and no claim inside the record naming the party
it was issued for. Integrity intact, association unsupported, so a verdict
issued for one counterparty verifies perfectly when replayed at another.

Two failures look alike to a boolean and are not alike:

* the record names a different audience than the one checking it. That is a
  positively identified conflict: refuse, and register the refusal.
* the record names no audience at all. That is missing linkage evidence: the
  relying party cannot detect misdirection, and no amount of checking the
  record will tell it so.

Collapsing both into "verification failed" loses the second case, because
there is nothing to fail on. So the check returns three values and the
unsupported one is decided before any comparison runs, for the same reason
``verify_consistency`` returns could-not-compare first: a state that only
exists in the model and never reaches the caller does not exist for the
party who has to rely on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._receipt_types import ExecutionReceipt

AudienceVerdict = Literal["bound", "conflict", "unsupported"]


@dataclass(frozen=True)
class AudienceResult:
    """Outcome of an audience check.

    ``verdict`` is one of:

    * ``"bound"``: the receipt carries ``aud`` and it equals the expected
      audience. The association is established by the signed bytes.
    * ``"conflict"``: the receipt carries ``aud`` and it differs. The
      association is contradicted. Refuse and record the refusal.
    * ``"unsupported"``: the receipt carries no ``aud``. The association is
      neither established nor contradicted; the record cannot answer.

    Only ``"bound"`` is truthy, so a caller written against a boolean refuses
    both of the other cases and cannot mistake silence for a match.
    """

    verdict: AudienceVerdict
    receipt_aud: str | None
    expected_aud: str
    reason: str

    def __bool__(self) -> bool:
        return self.verdict == "bound"


def verify_receipt_audience(
    receipt: ExecutionReceipt, expected_aud: str
) -> AudienceResult:
    """Three-valued audience check. Does not verify the signature.

    Run ``verify_receipt_signature`` first; this function reads the ``aud``
    field as the receipt carries it and says nothing about whether the bytes
    are authentic. ``expected_aud`` MUST be non-empty: an empty expectation
    would match nothing and read as a conflict, which misreports the caller's
    mistake as the issuer's.
    """
    if not expected_aud:
        raise ValueError("expected_aud must be a non-empty string")
    aud = receipt.receipt_asserted.aud
    if aud is None:
        return AudienceResult(
            "unsupported", None, expected_aud,
            "receipt carries no aud; association neither established nor contradicted",
        )
    if aud == expected_aud:
        return AudienceResult("bound", aud, expected_aud, "aud matches the expected audience")
    return AudienceResult(
        "conflict", aud, expected_aud,
        f"receipt was issued for {aud!r}, not {expected_aud!r}",
    )
