"""Decision records: the pre-execution sibling of SEP-2787.

SEP-2787 attests a ``tools/call`` *request* before it runs. The
execution receipt (``vaara.attestation.receipt``) binds the *outcome*
after it runs. The decision record covers the half between them: the
governing server's policy verdict and its risk basis, signed and
committed *before* the side effect. This is the commit-before-execute
property that lets a verifier prove the verdict was fixed before the
action ran.

A decision record carries three parts plus a signature:

- ``backLink`` pins the SEP-2787 attestation by nonce and by a digest
  over its full wire bytes, the same instance-binding the receipt uses.
- ``issuerAsserted`` is the governing server's issuer block.
- ``decisionDerived`` carries the verdict (``allow`` / ``block`` /
  ``escalate``), the risk basis (decimal-string risk score and
  thresholds, an optional policy id), and the decision time.

A decision record verifies in two composable checks: the signature
(``verify_decision_signature``) and the back-link to its attestation
(``verify_decision_back_link``). ``records_paired`` then joins it to the
execution receipt that answers the same call. A decision record is a
durable record rather than a time-bounded capability, so there is no TTL.

Canonicalization (RFC 8785 JCS) and signing (HS256 / ES256 / RS256) are
shared with ``vaara.attestation.sep2787`` and the receipt module
unchanged. A verifier that already checks SEP-2787 signatures needs no
new crypto to check decision records.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

from vaara.attestation._decision_emit import (
    emit_decision_record,
    verify_decision_signature,
)
from vaara.attestation._decision_types import (
    DecisionDerived,
    DecisionRecord,
    DecisionVerdict,
    IssuerAsserted,
    decision_record_from_dict as parse_decision_record,
)
from vaara.attestation._decision_verifier import (
    FALLBACK_PROJECTION_V1,
    AmbiguousSupersessionError,
    MalformedFallbackBindingError,
    decision_digest,
    fallback_projection,
    records_paired,
    request_envelope_digest,
    superseding_decision,
    verify_decision_back_link,
    verify_decision_fallback_binding,
)
from vaara.attestation._receipt_types import BackLink
from vaara.attestation._receipt_verifier import (
    BackLinkResult,
    attestation_digest,
    make_back_link,
)

__all__ = [
    "FALLBACK_PROJECTION_V1",
    "AmbiguousSupersessionError",
    "BackLink",
    "BackLinkResult",
    "DecisionDerived",
    "DecisionRecord",
    "DecisionVerdict",
    "IssuerAsserted",
    "MalformedFallbackBindingError",
    "attestation_digest",
    "decision_digest",
    "emit_decision_record",
    "fallback_projection",
    "make_back_link",
    "parse_decision_record",
    "records_paired",
    "request_envelope_digest",
    "superseding_decision",
    "verify_decision_back_link",
    "verify_decision_fallback_binding",
    "verify_decision_signature",
]
