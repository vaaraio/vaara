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
from collections.abc import Mapping, Sequence
from typing import Any

from vaara.attestation._decision_types import DecisionRecord
from vaara.attestation._receipt_types import ExecutionReceipt
from vaara.attestation._receipt_verifier import (
    BACK_LINK_MISMATCH,
    FALLBACK_BINDING_MALFORMED,
    BackLinkResult,
    attestation_digest,
)
from vaara.attestation._attest_canonical import canonical_json
from vaara.attestation._attest_types import Attestation


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


# The named projection version a no-SEP-2787 fallback digest commits to. The id
# is descriptive (it names what the preimage includes) and the signed record
# names which version it used (backLink.fallbackProjection), so a verifier
# reconstructs the same projection deterministically and a future projection rule
# is an explicit new version, not a silent reinterpretation. This is the id the
# non-normative SEP-2828 example note on modelcontextprotocol#2867 settled on.
FALLBACK_PROJECTION_V1 = "tools_call_params_plus_meta_authorization_binding_v1"
_SUPPORTED_FALLBACK_PROJECTIONS = frozenset({FALLBACK_PROJECTION_V1})

# The named binding block under ``_meta`` whose contents are authoritative for
# the fallback digest. The preimage is exactly the bound call params plus this
# block (an allowlist); everything else under ``_meta`` is excluded by
# construction, so the rule cannot drift as new _meta fields appear.
_AUTHORIZATION_BINDING = "authorization_binding"


class MalformedFallbackBindingError(ValueError):
    """A no-SEP-2787 fallback projection cannot be reconstructed: an
    unsupported projection version, or a request envelope whose named
    ``_meta.authorization_binding`` block is absent, not an object, or missing
    its ``nonce``, or whose ``tools/call`` ``name``/``arguments`` are missing.

    Per SEP-2828 the fallback case is then not conformant. Verifiers fail
    closed on this rather than widening the preimage to the whole ``_meta``,
    which would make observation-local material the authority boundary.
    """


def fallback_projection(
    request_envelope: Mapping[str, Any],
    *,
    version: str = FALLBACK_PROJECTION_V1,
) -> dict[str, Any]:
    """The named, versioned projection a no-SEP-2787 fallback digest commits to.

    The preimage is an allowlist: it contains exactly the fields that bind a
    decision to its originating ``tools/call`` and nothing else.

    - ``name`` and ``arguments`` (the canonical call params);
    - the named ``_meta.authorization_binding`` block (the server's per-call
      binding ``nonce``).

    Every other ``_meta`` field is excluded by construction (it is never read
    into the projection), so a gateway view and a provider view of the same
    call, differing only in observation-local ``_meta`` such as progress
    tokens, trace context, UI hints, or unrelated SEP blocks, project to the
    same bytes. The ``version`` is supplied by the caller, not inferred from the
    observed envelope: a verifier passes the version named on the signed record
    (``backLink.fallbackProjection``); an emitter passes the version it binds
    under. Raises ``MalformedFallbackBindingError`` on an unsupported version or
    an absent or malformed binding block, so the fallback never silently widens
    to the whole ``_meta``.
    """
    if version not in _SUPPORTED_FALLBACK_PROJECTIONS:
        raise MalformedFallbackBindingError(
            f"unsupported fallback projection version: {version!r}"
        )
    meta = request_envelope.get("_meta")
    if not isinstance(meta, Mapping):
        raise MalformedFallbackBindingError(
            "request envelope has no _meta object"
        )
    binding = meta.get(_AUTHORIZATION_BINDING)
    if not isinstance(binding, Mapping):
        raise MalformedFallbackBindingError(
            f"_meta.{_AUTHORIZATION_BINDING} is absent or not an object"
        )
    nonce = binding.get("nonce")
    if not isinstance(nonce, str) or not nonce:
        raise MalformedFallbackBindingError(
            f"_meta.{_AUTHORIZATION_BINDING}.nonce is absent or not a string"
        )
    if "name" not in request_envelope or "arguments" not in request_envelope:
        raise MalformedFallbackBindingError(
            "request envelope is missing tools/call name or arguments"
        )
    return {
        "projection": version,
        "name": request_envelope["name"],
        "arguments": request_envelope["arguments"],
        "authorizationBinding": dict(binding),
    }


def request_envelope_digest(
    request_envelope: Mapping[str, Any],
    *,
    version: str = FALLBACK_PROJECTION_V1,
) -> str:
    """``sha256:<hex>`` over the JCS canonicalization of the named fallback
    projection of a request envelope at ``version`` (see ``fallback_projection``).

    This is the no-SEP-2787 fallback preimage: when a deployment does not run
    2787, the decision back-link binds the request instance by this digest
    instead of an attestation digest (SEP-2828 backLink, fallback path). The
    digest commits to the bound call params and the named binding block only,
    so a re-presented envelope whose arguments or binding block differ
    recomputes to a different digest and does not bind, while transport-local
    ``_meta`` a gateway adds or strips does not change it. Raises
    ``MalformedFallbackBindingError`` if the projection cannot be reconstructed.
    """
    wire = canonical_json(fallback_projection(request_envelope, version=version))
    return f"sha256:{hashlib.sha256(wire).hexdigest()}"


def verify_decision_fallback_binding(
    record: DecisionRecord,
    *,
    request_envelope: Mapping[str, Any],
) -> BackLinkResult:
    """Confirm a decision record's back-link pins ``request_envelope`` under
    the no-SEP-2787 fallback projection.

    The no-SEP-2787 counterpart to ``verify_decision_back_link``. The
    projection version is taken from the signed record's
    ``backLink.fallbackProjection``, so reconstruction is deterministic from
    trusted data rather than inferred from the observed envelope. Recomputes the
    digest over the named projection at that version (bound call params plus the
    ``_meta.authorization_binding`` block), compares it constant-time to the
    record's ``backLink.attestationDigest``, then confirms the back-link's
    ``attestationNonce`` echoes the binding block's ``nonce``, mirroring the
    attestation path's nonce echo. A record naming no fallback projection or an
    unsupported one, or an envelope with no reconstructable binding block, fails
    closed (``ok=False``, ``fallback_binding_malformed``), never widening the
    preimage to the whole ``_meta``.
    """
    version = record.back_link.fallback_projection
    if version is None:
        return BackLinkResult(ok=False, reason=FALLBACK_BINDING_MALFORMED)
    try:
        expected_digest = request_envelope_digest(
            request_envelope, version=version
        )
    except MalformedFallbackBindingError:
        return BackLinkResult(ok=False, reason=FALLBACK_BINDING_MALFORMED)
    if not hmac.compare_digest(
        record.back_link.attestation_digest, expected_digest
    ):
        return BackLinkResult(ok=False, reason=BACK_LINK_MISMATCH)
    binding_nonce = request_envelope["_meta"][_AUTHORIZATION_BINDING]["nonce"]
    if record.back_link.attestation_nonce != binding_nonce:
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


class AmbiguousSupersessionError(ValueError):
    """Distinct decision records share the latest ``decidedAt`` and no
    deterministic ordering field separates them.

    For conformance the effective decision is undetermined. Reporting it
    is correct: resolving the tie by an incidental order (issuer nonce,
    file order, or arrival order) would name a "winner" that is not the
    genuinely-later decision, and would mask a producer that emitted two
    records which should never have tied.
    """


def superseding_decision(
    decisions: Sequence[DecisionRecord],
) -> DecisionRecord:
    """Return the effective decision among records for one call.

    A superseding decision (for example a human resolving an
    ``escalate``) is a new decision record with the same back-link and a
    later ``decidedAt``; earlier records are retained as history. The
    record with the latest ``decidedAt`` is effective.

    When distinct records share the latest ``decidedAt`` and carry no
    explicit ordering field to break the tie, the effective decision is
    ambiguous and this raises ``AmbiguousSupersessionError`` rather than
    guessing from nonce, file, or arrival order. Byte-identical records
    are one decision, not a tie. The caller is responsible for passing
    records that share a back-link; this resolves ordering only. Raises
    ``ValueError`` on an empty input.
    """
    if not decisions:
        raise ValueError("superseding_decision requires at least one record")
    latest = max(d.decision_derived.decided_at for d in decisions)
    tied = [
        d for d in decisions if d.decision_derived.decided_at == latest
    ]
    distinct = {canonical_json(d.to_dict()) for d in tied}
    if len(distinct) > 1:
        raise AmbiguousSupersessionError(
            "distinct decision records share the latest decidedAt with no "
            "deterministic ordering field; effective decision is ambiguous"
        )
    return tied[0]
