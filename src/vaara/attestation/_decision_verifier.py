"""Back-link verification and pairing for decision records.

Internal module. Public surface is in ``vaara.attestation.decision``.

The back-link is the join that makes a SEP-2787 attestation and a
decision record one verifiable pair, exactly as it does for an execution
receipt. Pairing then joins a decision record and the execution receipt
that answer the same governed call: both carry the same back-link, so a
verifier holding all three can reconstruct what was permitted, why, and
what the call did.

Result-commitment and signature checks are not duplicated here. The
attestation-digest computation (``attestation_digest``) and the
``BackLinkResult`` type are shared with the receipt verifier.
"""

from __future__ import annotations

import hmac

from vaara.attestation._decision_types import DecisionRecord
from vaara.attestation._receipt_types import ExecutionReceipt
from vaara.attestation._receipt_verifier import (
    BACK_LINK_MISMATCH,
    BackLinkResult,
    attestation_digest,
)
from vaara.attestation._sep2787_types import Attestation


def verify_decision_back_link(
    record: DecisionRecord,
    *,
    attestation: Attestation,
) -> BackLinkResult:
    """Confirm the decision record's back-link pins ``attestation``.

    Recomputes the attestation digest and compares both it and the nonce
    against the record's ``backLink``. The digest is the binding check;
    the nonce is a fast-correlation field that must also agree so a
    record cannot carry one attestation's digest under another's nonce.
    """
    expected_digest = attestation_digest(attestation)
    if not hmac.compare_digest(
        record.back_link.attestation_digest, expected_digest
    ):
        return BackLinkResult(ok=False, reason=BACK_LINK_MISMATCH)
    if record.back_link.attestation_nonce != attestation.issuer_asserted.nonce:
        return BackLinkResult(ok=False, reason=BACK_LINK_MISMATCH)
    return BackLinkResult(ok=True)


def records_paired(
    decision: DecisionRecord,
    receipt: ExecutionReceipt,
) -> bool:
    """True iff a decision record and an execution receipt describe one call.

    They pair when both carry the same back-link: the attestation digest
    (constant-time compared) and the attestation nonce both agree. This
    is instance-binding, not content-binding, so two byte-identical calls
    produce distinct attestations and therefore do not cross-pair.
    """
    if not hmac.compare_digest(
        decision.back_link.attestation_digest,
        receipt.back_link.attestation_digest,
    ):
        return False
    return (
        decision.back_link.attestation_nonce
        == receipt.back_link.attestation_nonce
    )
