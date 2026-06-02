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

import hashlib
import hmac
from collections.abc import Sequence

from vaara.attestation._decision_types import DecisionRecord
from vaara.attestation._receipt_types import ExecutionReceipt
from vaara.attestation._receipt_verifier import (
    BACK_LINK_MISMATCH,
    BackLinkResult,
    attestation_digest,
)
from vaara.attestation._sep2787_canonical import canonical_json
from vaara.attestation._sep2787_types import Attestation


def decision_digest(record: DecisionRecord) -> str:
    """``sha256:<hex>`` over the JCS-canonical full decision wire bytes.

    The signature is included, mirroring ``attestation_digest``: the
    digest pins the exact decision-record instance, so an outcome record
    that commits to it (SEP-2828 Check B) is bound to one decision's
    content, not merely to a record with the same fields.
    """
    wire = canonical_json(record.to_dict())
    return f"sha256:{hashlib.sha256(wire).hexdigest()}"


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

    Both SEP-2828 checks must hold:

    - **Check A (instance anchor).** Both records carry the same
      back-link: the attestation digest (constant-time compared) and the
      attestation nonce both agree. This is instance-binding, so two
      byte-identical calls produce distinct attestations and do not
      cross-pair. It also anchors the fallback case, where the back-link
      is over the request envelope rather than a SEP-2787 attestation.
    - **Check B (outcome-to-decision digest, normative pairing).** The
      receipt's ``outcomeDerived.decisionDigest`` equals the digest of
      *this* decision record. Check A alone admits a different decision
      made under the same attestation (e.g. a superseding verdict); Check
      B pins which decision's content the outcome answers. A receipt
      without ``decisionDigest`` does not pair: content binding is
      mandatory, not best-effort.
    """
    # Check A: same call instance.
    if not hmac.compare_digest(
        decision.back_link.attestation_digest,
        receipt.back_link.attestation_digest,
    ):
        return False
    if (
        decision.back_link.attestation_nonce
        != receipt.back_link.attestation_nonce
    ):
        return False
    # Check B: outcome commits to this decision's content.
    bound = receipt.outcome_derived.decision_digest
    if bound is None:
        return False
    return hmac.compare_digest(bound, decision_digest(decision))


def superseding_decision(
    decisions: Sequence[DecisionRecord],
) -> DecisionRecord:
    """Return the effective decision among records for one call.

    A superseding decision (for example a human resolving an
    ``escalate``) is a new decision record with the same back-link and a
    later ``decidedAt``; earlier records are retained as history. The
    record with the latest ``decidedAt`` is effective. When two share the
    same ``decidedAt``, the tie breaks deterministically on the issuer
    nonce, lowest lexicographic wins, so every verifier selects the same
    winner without a clock.

    The caller is responsible for passing records that share a back-link;
    this resolves ordering only. Raises ``ValueError`` on an empty input.
    """
    if not decisions:
        raise ValueError("superseding_decision requires at least one record")
    latest = max(d.decision_derived.decided_at for d in decisions)
    tied = [
        d for d in decisions if d.decision_derived.decided_at == latest
    ]
    return min(tied, key=lambda d: d.issuer_asserted.nonce)
