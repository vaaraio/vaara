"""One-call evidence-bundle verification: the 0.6 trust-plane capstone.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt`` and ``vaara.attestation``.

The 0.52 to 0.55 line added each verification lens on its own: resolvable
identity (level 2 and 3), the receipt signature, the back-link to the
request attestation, transparency-log inclusion, append-only consistency,
and cross-stack revocation. Each is a separate call returning a separate
verdict. A consumer holding a full evidence bundle had to invoke all six,
track which applied, and combine the answers itself.

This module is the single entrypoint that does that combination. Given an
:class:`EvidenceBundle` (one receipt plus whatever evidence the holder
has), :func:`verify_evidence_bundle` runs each lens whose evidence is
present, threads the identity-resolved keyid into the revocation lens, and
returns one :class:`BundleVerdict`: the per-lens results and a single
``ok``.

A lens with no evidence is *not applicable*, not a failure: a bundle
without a consistency proof is not rejected for lacking one. But ``ok`` is
fail-closed on authenticity. It is true only when the receipt signature is
actually established (the identity lens bound it to a resolved key, or the
signature lens verified it under supplied key material) AND every
applicable lens passed. A bundle that proves inclusion and non-revocation
but never verifies the signature is not ``ok``: an unauthenticated record
sitting in a log proves nothing about who issued it.

Purely additive. Composes the existing lens functions unchanged and touches
neither the receipt envelope nor any canonicalization, so every existing
conformance vector verifies exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._receipt_emit import verify_receipt_signature
from vaara.attestation._receipt_identity import verify_receipt_identity
from vaara.attestation._receipt_types import ExecutionReceipt
from vaara.attestation._receipt_verifier import verify_back_link
from vaara.attestation._revocation import (
    RevocationRegistry,
    check_receipt_revocation,
    receipt_leaf_bytes,
)
from vaara.attestation._sep2787_types import Attestation
from vaara.attestation.transparency_log import (
    ConsistencyProof,
    InclusionProof,
    verify_consistency,
    verify_inclusion,
)

# The six lenses, in the order the verdict reports them. Identity runs first
# so the keyid it resolves can sharpen the revocation lens.
LENS_NAMES: tuple[str, ...] = (
    "identity",
    "signature",
    "back_link",
    "inclusion",
    "consistency",
    "revocation",
)


@dataclass(frozen=True)
class LensResult:
    """Result of one lens over the bundle.

    ``applicable`` is True when the bundle carried the evidence the lens
    needs. ``ok`` is True only when the lens applied AND passed; a lens that
    did not apply has ``applicable=False`` and ``ok=False`` and does not
    count against the verdict. ``reason`` is a short human string for audit
    logs.
    """

    lens: str
    applicable: bool
    ok: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON output (CLI, logs)."""
        return {
            "lens": self.lens,
            "applicable": self.applicable,
            "ok": self.ok,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """One receipt plus the evidence available to verify it.

    Only ``receipt`` is required. Each remaining field feeds one lens; a
    field left ``None`` makes that lens not applicable. The inclusion lens
    needs both ``inclusion`` and ``log_root``; the consistency lens needs
    ``consistency`` and both tree-head roots. ``inclusion_leaf`` defaults to
    the full canonical receipt bytes when not given, matching what
    :func:`receipt_leaf_bytes` would append.
    """

    receipt: ExecutionReceipt
    did_document: Optional[dict[str, Any]] = None
    expected_keyid: Optional[str] = None
    verifying_material: Optional[Any] = None
    attestation: Optional[Attestation] = None
    inclusion: Optional[InclusionProof] = None
    log_root: Optional[bytes] = None
    inclusion_leaf: Optional[bytes] = None
    consistency: Optional[ConsistencyProof] = None
    consistency_first_root: Optional[bytes] = None
    consistency_second_root: Optional[bytes] = None
    registry: Optional[RevocationRegistry] = None


@dataclass(frozen=True)
class BundleVerdict:
    """Aggregate verdict over an evidence bundle.

    ``ok`` is the single answer: the receipt signature was established and
    every applicable lens passed. ``authenticity_established`` records
    whether the signature was verified at all (by identity or by the
    signature lens); without it ``ok`` is always False, however many other
    lenses passed. ``keyid`` is the identity-resolved key when one bound.
    ``lenses`` carries every lens result in :data:`LENS_NAMES` order.
    """

    ok: bool
    authenticity_established: bool
    keyid: Optional[str]
    lenses: tuple[LensResult, ...]
    reason: str

    def lens(self, name: str) -> LensResult:
        """Return the result for one lens by name."""
        for result in self.lenses:
            if result.lens == name:
                return result
        raise KeyError(f"no such lens: {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON output (CLI, logs)."""
        return {
            "ok": self.ok,
            "authenticity_established": self.authenticity_established,
            "keyid": self.keyid,
            "reason": self.reason,
            "lenses": [result.to_dict() for result in self.lenses],
        }


def _identity_lens(bundle: EvidenceBundle) -> tuple[LensResult, Optional[str]]:
    if bundle.did_document is None:
        return LensResult("identity", False, False, "no DID document supplied"), None
    result = verify_receipt_identity(
        bundle.receipt, bundle.did_document, expected_keyid=bundle.expected_keyid
    )
    ok = result.resolved and result.bound
    return LensResult("identity", True, ok, result.reason), result.keyid


def _signature_lens(bundle: EvidenceBundle) -> LensResult:
    if bundle.verifying_material is None:
        return LensResult(
            "signature", False, False, "no verifying key material supplied"
        )
    ok = verify_receipt_signature(
        bundle.receipt, verifying_material=bundle.verifying_material
    )
    reason = (
        "signature verifies under supplied key material"
        if ok
        else "signature does not verify under supplied key material"
    )
    return LensResult("signature", True, ok, reason)


def _back_link_lens(bundle: EvidenceBundle) -> LensResult:
    if bundle.attestation is None:
        return LensResult("back_link", False, False, "no request attestation supplied")
    result = verify_back_link(bundle.receipt, attestation=bundle.attestation)
    reason = (
        "back-link pins the supplied attestation"
        if result.ok
        else (result.reason or "back_link_mismatch")
    )
    return LensResult("back_link", True, result.ok, reason)


def _inclusion_lens(bundle: EvidenceBundle) -> LensResult:
    if bundle.inclusion is None or bundle.log_root is None:
        return LensResult(
            "inclusion", False, False, "no inclusion proof and log root supplied"
        )
    leaf = (
        bundle.inclusion_leaf
        if bundle.inclusion_leaf is not None
        else receipt_leaf_bytes(bundle.receipt)
    )
    ok = verify_inclusion(
        leaf_data=leaf, proof=bundle.inclusion, expected_root=bundle.log_root
    )
    reason = (
        "receipt is included under the supplied log root"
        if ok
        else "inclusion proof does not reproduce the supplied log root"
    )
    return LensResult("inclusion", True, ok, reason)


def _consistency_lens(bundle: EvidenceBundle) -> LensResult:
    if (
        bundle.consistency is None
        or bundle.consistency_first_root is None
        or bundle.consistency_second_root is None
    ):
        return LensResult(
            "consistency", False, False, "no consistency proof and tree heads supplied"
        )
    ok = verify_consistency(
        first_size=bundle.consistency.first_size,
        first_root=bundle.consistency_first_root,
        second_size=bundle.consistency.second_size,
        second_root=bundle.consistency_second_root,
        proof=bundle.consistency,
    )
    reason = (
        "log is append-only across the two supplied tree heads"
        if ok
        else "consistency proof does not reproduce the supplied tree heads"
    )
    return LensResult("consistency", True, ok, reason)


def _revocation_lens(bundle: EvidenceBundle, keyid: Optional[str]) -> LensResult:
    if bundle.registry is None:
        return LensResult("revocation", False, False, "no revocation registry supplied")
    status = check_receipt_revocation(bundle.receipt, bundle.registry, keyid=keyid)
    return LensResult("revocation", True, not status.revoked, status.reason)


def verify_evidence_bundle(bundle: EvidenceBundle) -> BundleVerdict:
    """Run every applicable lens over ``bundle`` and return one verdict.

    Runs identity, signature, back-link, inclusion, consistency, and
    revocation, each only when the bundle carries its evidence. The keyid
    the identity lens resolves (falling back to ``expected_keyid``) is
    threaded into the revocation lens so key-scope revocations apply. ``ok``
    is True only when the receipt signature was established by the identity
    or signature lens AND every applicable lens passed.
    """
    identity, resolved_keyid = _identity_lens(bundle)
    keyid = resolved_keyid if resolved_keyid is not None else bundle.expected_keyid
    signature = _signature_lens(bundle)
    back_link = _back_link_lens(bundle)
    inclusion = _inclusion_lens(bundle)
    consistency = _consistency_lens(bundle)
    revocation = _revocation_lens(bundle, keyid)

    lenses = (identity, signature, back_link, inclusion, consistency, revocation)

    authenticity = identity.ok or signature.ok
    applicable_failures = [r.lens for r in lenses if r.applicable and not r.ok]
    ok = authenticity and not applicable_failures

    if not authenticity:
        reason = (
            "authenticity not established: neither the identity lens nor the "
            "signature lens verified the receipt signature"
        )
    elif applicable_failures:
        reason = "applicable lenses failed: " + ", ".join(applicable_failures)
    else:
        applied = [r.lens for r in lenses if r.applicable]
        via = "identity" if identity.ok else "signature"
        reason = (
            f"all {len(applied)} applicable lenses passed; "
            f"authenticity established via {via}"
        )

    return BundleVerdict(
        ok=ok,
        authenticity_established=authenticity,
        keyid=resolved_keyid,
        lenses=lenses,
        reason=reason,
    )
