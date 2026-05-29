"""Back-link verification between a receipt and its request attestation.

Internal module. Public surface is in ``vaara.attestation.receipt``.

The back-link is the join that makes a request attestation (SEP-2787)
and an execution receipt one verifiable pair. A receipt with a valid
signature but a broken back-link proves an outcome that belongs to no
attested request, which is exactly what this check rejects.

Result-commitment verification is not duplicated here: a result
commitment is structurally an argument commitment, so callers reuse
``verify_args_commitment`` from the SEP-2787 verifier against the
runtime result object.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Literal, Optional

from vaara.attestation._receipt_types import BackLink, ExecutionReceipt
from vaara.attestation._sep2787_canonical import canonical_json
from vaara.attestation._sep2787_types import Attestation

BACK_LINK_MISMATCH: Literal["back_link_mismatch"] = "back_link_mismatch"


@dataclass(frozen=True)
class BackLinkResult:
    """Outcome of back-link verification.

    ``ok`` is True iff the receipt's ``backLink`` pins the given
    attestation: the digest matches the attestation's canonical wire
    bytes AND the nonce matches the attestation's issuer nonce.
    ``reason`` is None on success or ``"back_link_mismatch"`` on
    failure.
    """

    ok: bool
    reason: Optional[Literal["back_link_mismatch"]] = None


def attestation_digest(attestation: Attestation) -> str:
    """``sha256:<hex>`` over the JCS-canonical full attestation wire bytes.

    The signature is included: the digest pins the exact attestation
    instance the receipt answers, not just its unsigned body.
    """
    wire = canonical_json(attestation.to_dict())
    return f"sha256:{hashlib.sha256(wire).hexdigest()}"


def make_back_link(attestation: Attestation) -> BackLink:
    """Build the back-link that joins a receipt to ``attestation``."""
    return BackLink(
        attestation_digest=attestation_digest(attestation),
        attestation_nonce=attestation.issuer_asserted.nonce,
    )


def verify_back_link(
    receipt: ExecutionReceipt,
    *,
    attestation: Attestation,
) -> BackLinkResult:
    """Confirm the receipt's back-link pins ``attestation``.

    Recomputes the attestation digest and compares both it and the
    nonce against the receipt's ``backLink``. The digest is the binding
    check; the nonce is a fast-correlation field that must also agree
    so a receipt cannot carry one attestation's digest under another's
    nonce.
    """
    expected_digest = attestation_digest(attestation)
    if not hmac.compare_digest(
        receipt.back_link.attestation_digest, expected_digest
    ):
        return BackLinkResult(ok=False, reason=BACK_LINK_MISMATCH)
    if receipt.back_link.attestation_nonce != attestation.issuer_asserted.nonce:
        return BackLinkResult(ok=False, reason=BACK_LINK_MISMATCH)
    return BackLinkResult(ok=True)
