"""Execution receipts: the post-execution sibling of SEP-2787.

SEP-2787 attests a ``tools/call`` *request* before it runs: issuer,
subject, target, intent, nonce, time, argument commitment. It defers,
by design, any claim about whether the call executed or what came
back. The execution receipt covers that deferred half. It binds the
outcome of one attested request and links back to the attestation it
answers.

A receipt carries three parts plus a signature:

- ``backLink`` pins the SEP-2787 attestation by nonce and by a digest
  over its full wire bytes.
- ``receiptAsserted`` is the issuer block, set by whoever observed the
  outcome (the executing server, or an intermediary such as a
  governance proxy).
- ``outcomeDerived`` carries the status (``executed`` / ``refused`` /
  ``errored``), the completion time, and an optional commitment over
  the result, reusing the SEP-2787 argument-commitment shapes.

A receipt verifies in three composable checks: the signature
(``verify_receipt_signature``), the back-link to its attestation
(``verify_back_link``), and, when a result commitment is present, the
result binding (``verify_args_commitment`` against the runtime result
object). A receipt is a durable record rather than a time-bounded
capability, so there is no TTL.

Canonicalization (RFC 8785 JCS) and signing (HS256 / ES256 / RS256)
are shared with ``vaara.attestation.sep2787`` unchanged. A verifier
that already checks SEP-2787 signatures needs no new crypto to check
receipts.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

from vaara.attestation._receipt_emit import (
    emit_receipt,
    verify_receipt_signature,
)
from vaara.attestation._receipt_identity import (
    IdentityResult,
    did_web_to_url,
    verify_receipt_identity,
)
from vaara.attestation._receipt_types import (
    BackLink,
    ExecutionReceipt,
    OutcomeDerived,
    ReceiptAsserted,
    ReceiptStatus,
    ResultCommitment,
    receipt_from_dict as parse_receipt,
)
from vaara.attestation._receipt_verifier import (
    BackLinkResult,
    attestation_digest,
    make_back_link,
    verify_back_link,
)
from vaara.attestation._receipt_vc import (
    VAARA_RECEIPT_CONTEXT_URL,
    load_receipt_context,
    receipt_from_vc,
    receipt_to_vc,
)

# Result commitments reuse the SEP-2787 argument-commitment builders.
# Re-export under result-oriented names so call sites read naturally.
from vaara.attestation._sep2787_canonical import (
    make_args_digest as make_result_digest,
    make_args_projection as make_result_projection,
)

__all__ = [
    "VAARA_RECEIPT_CONTEXT_URL",
    "BackLink",
    "BackLinkResult",
    "ExecutionReceipt",
    "IdentityResult",
    "OutcomeDerived",
    "ReceiptAsserted",
    "ReceiptStatus",
    "ResultCommitment",
    "attestation_digest",
    "did_web_to_url",
    "emit_receipt",
    "load_receipt_context",
    "make_back_link",
    "make_result_digest",
    "make_result_projection",
    "parse_receipt",
    "receipt_from_vc",
    "receipt_to_vc",
    "verify_back_link",
    "verify_receipt_identity",
    "verify_receipt_signature",
]
