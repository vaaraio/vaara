"""Cross-org evidence handoff: hand one signed record to another org's regulator.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt`` and ``vaara.attestation``.

The cross-org problem. A provider (vendor A) produces and signs an execution
record. A deployer (customer B), a *different* organisation, runs vendor A's
system and is audited by B's own regulator. The regulator has no prior
relationship with vendor A, no live channel to A (A may be in another
jurisdiction, or out of business by audit time years later), and does not
inherently trust B, the party relaying the evidence. The regulator needs to
verify vendor A's record offline, from one self-contained file.

This module defines that file, the *handoff package*, and the two sides that
produce and check it:

- :func:`build_handoff` (vendor A / the relaying holder) stitches the record,
  the *archived* DID document that lists the now-retired signing key, the key
  history and revocations, and an optional eIDAS RFC 3161 time anchor into one
  document, and pins each component by a content digest.
- :func:`verify_handoff` (the regulator) recomputes every digest, routes the
  record through the retained-record lens (:func:`verify_receipt_retained`,
  the rotated-key / retention-window check), confirms an enclosed anchor binds
  to *this* record, and reports one verdict.

What trust this does and does not establish, stated plainly so a verdict is
never over-read:

- The record's authenticity rests on **vendor A's signature** verifying against
  **vendor A's genuine identity**, which the regulator must establish *out of
  band* (a live did:web resolution, an archived or notarised copy of A's key
  set, or a registry binding). This package encloses a DID document *claiming*
  to be vendor A's; it does not prove that claim. ``producer_identity_basis``
  stays ``self_asserted_unpinned`` until the caller supplies the DID document it
  independently trusts as A's.
- The content digests and the manifest only prove the package is *internally
  consistent* and that the holder handed over these exact bytes. The holder
  assembles the manifest and controls both the components and their pinned
  digests, so a green ``integrity_ok`` is **not** evidence of authenticity; it
  catches corruption and accidental drift, not a dishonest holder.
- The eIDAS time anchor is the one component the holder cannot forge: a trusted
  authority outside both A and B attests the record existed at a point in time.
  When it predates the signing key's retirement, it rules out a backdated
  forgery with a stolen retired key (the ``corroborated`` tier from
  :mod:`_receipt_retention`).

The digests, the anchor-to-record binding, and the retention-window arithmetic
are pure standard library. Binding the record signature needs the
``cryptography`` of the attestation extra (through
:func:`verify_receipt_retained`); verifying the RFC 3161 token and a holder
attestation need it too. See ``docs/design/cross-org-handoff-spec.md``. Purely
additive: no receipt-envelope or canonicalization change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._key_history import KeyHistory
from vaara.attestation._receipt_retention import (
    RetentionResult,
    verify_receipt_retained,
)
from vaara.attestation._receipt_types import receipt_from_dict
from vaara.attestation._revocation import RevocationRegistry
from vaara.attestation._sep2787_canonical import canonical_json
from vaara.attestation._sep2787_types import AttestationError

SCHEMA = "vaara.cross-org-handoff/v0"

# Holder attestation signs over an asymmetric key so a third party can verify it
# from the enclosed public JWK. A symmetric HS256 signature would need the shared
# secret and proves nothing cross-org, so it is not accepted.
_HOLDER_ALGS: tuple[str, ...] = ("ES256", "RS256")


def _jcs_digest(value: Any) -> str:
    """``sha256:<hex>`` over the RFC 8785 JCS bytes of ``value``."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _record_hash_hex(record: dict[str, Any]) -> str:
    """Bare hex sha256 over the JCS record bytes (the anchor's imprint)."""
    return hashlib.sha256(canonical_json(record)).hexdigest()


def _hex_equals(a: object, b: object) -> bool:
    """Whether two hex strings denote the same bytes; fail-closed on non-hex."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    try:
        return bytes.fromhex(a) == bytes.fromhex(b)
    except ValueError:
        return False


def _effective_key_history(
    evidence: dict[str, Any], did_document: dict[str, Any]
) -> KeyHistory:
    """The key history the verdict actually uses: an override, else the document.

    Mirrors :func:`verify_receipt_retained`'s default precedence, so the digest
    this module pins is over the exact windows the verdict rests on. The digest
    is a *model* digest (``KeyHistory.digest()`` canonicalises and re-sorts),
    never a raw hash of the supplied bytes, so a non-canonical override still
    pins the same value the verifier computes.
    """
    override = evidence.get("key_history")
    if override is not None:
        return KeyHistory.from_dict(override)
    return KeyHistory.from_did_document(did_document)


def _effective_revocations(
    evidence: dict[str, Any], did_document: dict[str, Any], iss: str
) -> RevocationRegistry:
    """The revocations the verdict actually uses: an override, else the document.

    The model-digest counterpart to :func:`_effective_key_history`. The document
    projection is keyed on ``iss`` exactly as :func:`verify_receipt_retained`
    builds it.
    """
    override = evidence.get("revocations")
    if override is not None:
        return RevocationRegistry.from_dict(override)
    return RevocationRegistry.from_did_document(did_document, iss)


@dataclass(frozen=True)
class ComponentDigest:
    """One component's content-addressing result.

    ``present`` is whether the package carried the component. ``expected`` is the
    digest the manifest pins; ``actual`` is the digest recomputed from the
    component bytes. ``ok`` is True only when the component is present and the
    two digests agree (or, for an absent component, when the manifest pins no
    digest for it).
    """

    name: str
    present: bool
    expected: Optional[str]
    actual: Optional[str]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "expected": self.expected,
            "actual": self.actual,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class HandoffVerdict:
    """Verdict over a cross-org handoff package.

    ``integrity_ok`` is internal consistency only (every pinned digest matched
    and the producer is coherent across record, document, and manifest); it is
    not a seal against a dishonest holder. ``verifiable`` and ``corroborated``
    are the record-level tiers from :class:`RetentionResult`, unchanged in
    meaning. ``producer_identity_basis`` records whether the enclosed identity
    was pinned against a document the caller trusts out of band. ``custody`` is
    the orthogonal holder-attestation result and never gates the record verdict.

    ``ok`` is the single overall answer for the chosen mode: integrity holds and
    the record is at least verifiable (default), or, under ``strict``, integrity
    holds and the record is corroborated with recorded windows, a consulted
    revocation source, and a pinned producer identity.
    """

    schema: str
    integrity_ok: bool
    components: tuple[ComponentDigest, ...]
    producer: Optional[str]
    holder: Optional[str]
    producer_identity_basis: str
    bound: bool
    keyid: Optional[str]
    window_recorded: bool
    revocation_source_present: bool
    revoked: bool
    anchor_present: bool
    anchor_verified: bool
    anchor_binds: bool
    verifiable: bool
    corroborated: bool
    custody: str
    holder_keyid: Optional[str]
    strict: bool
    ok: bool
    record: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ok": self.ok,
            "strict": self.strict,
            "integrity_ok": self.integrity_ok,
            "producer": self.producer,
            "holder": self.holder,
            "producer_identity_basis": self.producer_identity_basis,
            "bound": self.bound,
            "keyid": self.keyid,
            "window_recorded": self.window_recorded,
            "revocation_source_present": self.revocation_source_present,
            "revoked": self.revoked,
            "anchor_present": self.anchor_present,
            "anchor_verified": self.anchor_verified,
            "anchor_binds": self.anchor_binds,
            "verifiable": self.verifiable,
            "corroborated": self.corroborated,
            "custody": self.custody,
            "holder_keyid": self.holder_keyid,
            "components": [c.to_dict() for c in self.components],
            "record": self.record,
            "reason": self.reason,
        }


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field!r} must be a JSON object, got {type(value).__name__}")
    return value


def _bound_method_jwk(
    did_document: dict[str, Any], keyid: Optional[str]
) -> Optional[dict[str, Any]]:
    """The ``publicKeyJwk`` of the method whose id is ``keyid``, if any."""
    if keyid is None:
        return None
    methods = did_document.get("verificationMethod")
    if not isinstance(methods, list):
        return None
    for method in methods:
        if isinstance(method, dict) and method.get("id") == keyid:
            jwk = method.get("publicKeyJwk")
            if isinstance(jwk, dict):
                return jwk
    return None


def _producer_identity_basis(
    record: RetentionResult,
    did_document: dict[str, Any],
    trusted_did_document: Optional[dict[str, Any]],
) -> str:
    """Whether the bound key matches a key the caller trusts out of band.

    With no trusted reference the basis is ``self_asserted_unpinned``: the
    package's own DID document is the only identity on offer and nothing in the
    package can raise it. When the caller supplies a DID document it trusts as
    vendor A's (its retained key archive), the basis is ``pinned`` only if the
    bound key's id and public key material both appear, identically, in that
    trusted document, and ``pin_mismatch`` otherwise. This survives rotation:
    the trusted archive keeps retired keys, so an old signing key still pins.
    """
    if trusted_did_document is None:
        return "self_asserted_unpinned"
    if not record.bound or record.keyid is None:
        return "pin_mismatch"
    enclosed = _bound_method_jwk(did_document, record.keyid)
    trusted = _bound_method_jwk(trusted_did_document, record.keyid)
    if enclosed is not None and trusted is not None and enclosed == trusted:
        return "pinned"
    return "pin_mismatch"


def _verify_holder_attestation(
    attestation: dict[str, Any], manifest: dict[str, Any]
) -> bool:
    """Verify a holder attestation signature over the canonical manifest bytes.

    The holder signs ``canonical_json(manifest)`` (the same bytes hashed into
    ``manifest_digest``) with an asymmetric key, enclosing its public JWK. This
    proves only that whoever holds that key signed this exact package; with a
    self-supplied JWK it does not identify the holder and carries no
    non-repudiation weight unless the holder key is pinned out of band. Returns
    False on any malformed field, an unsupported algorithm, or a missing
    attestation extra; the caller maps False to ``holder_attestation_failed``.
    """
    alg = attestation.get("alg")
    signature = attestation.get("signature")
    jwk = attestation.get("verifying_jwk")
    if (
        alg not in _HOLDER_ALGS
        or not isinstance(signature, str)
        or not isinstance(jwk, dict)
    ):
        return False
    try:
        from vaara.attestation._receipt_identity import _jwk_to_public_key
        from vaara.attestation._sep2787_signing import verify_es256, verify_rs256
    except ImportError:
        return False
    try:
        public_key = _jwk_to_public_key(jwk)
    except Exception:  # noqa: BLE001 - any parse failure means unverifiable
        return False
    payload = canonical_json(manifest)
    if alg == "ES256":
        return verify_es256(payload, signature_hex=signature, public_key=public_key)
    return verify_rs256(payload, signature_hex=signature, public_key=public_key)


def verify_handoff(
    doc: dict[str, Any],
    *,
    anchor_attested_time: Optional[str] = None,
    trusted_did_document: Optional[dict[str, Any]] = None,
    strict: bool = False,
) -> HandoffVerdict:
    """Verify a cross-org handoff package and return one verdict.

    Runs four stages over ``doc`` (the on-disk handoff document):

    1. **Integrity.** Recompute each component digest from the enclosed bytes
       and compare it to the manifest's pinned value; recompute the manifest
       fingerprint; confirm the producer is coherent across the record issuer,
       the DID document id, and the manifest. Any drift sets ``integrity_ok``
       False and names the component. This is corruption checking, not a seal
       against the holder.
    2. **Record.** Route the record through :func:`verify_receipt_retained` with
       the *effective* key history and revocations (an enclosed override, else
       the document), yielding the unchanged ``verifiable`` / ``corroborated``
       tiers.
    3. **Anchor.** An enclosed anchor must bind to *this* record: its imprint
       (``chain_head_hash``) must equal ``sha256(jcs(record))`` and its hash
       algorithm must be sha256. Pass ``anchor_attested_time`` (the time the
       caller cryptographically verified from the RFC 3161 token) to let a bound
       anchor corroborate; without it the anchor is present-but-unverified.
    4. **Custody.** A holder attestation, if present, is verified separately and
       reported in ``custody``; it never affects the record verdict.

    ``trusted_did_document`` is the DID document the caller independently trusts
    as the producer's, used to pin ``producer_identity_basis``. ``strict`` makes
    ``ok`` require a corroborated record with recorded windows, a consulted
    revocation source, and a pinned producer identity. Raises :class:`ValueError`
    naming the offending field when the package shape is malformed.
    """
    if not isinstance(doc, dict):
        raise ValueError(f"handoff must be a JSON object, got {type(doc).__name__}")
    schema = doc.get("schema")
    evidence = _require_dict(doc.get("evidence"), "evidence")
    manifest = _require_dict(doc.get("manifest"), "manifest")
    if "record" not in evidence:
        raise ValueError("evidence is missing the required 'record' field")
    if "did_document" not in evidence:
        raise ValueError("evidence is missing the required 'did_document' field")
    record_doc = _require_dict(evidence["record"], "evidence.record")
    did_document = _require_dict(evidence["did_document"], "evidence.did_document")

    # A shape-valid but malformed receipt raises AttestationError (a
    # RuntimeError, not a ValueError); the effective-source overrides may carry
    # bad shapes. Normalise both to the ValueError this function documents, so a
    # hostile package fails closed with a named error rather than a traceback.
    try:
        receipt = receipt_from_dict(record_doc)
        iss = receipt.receipt_asserted.iss
        key_history = _effective_key_history(evidence, did_document)
        revocations = _effective_revocations(evidence, did_document, iss)
    except (AttestationError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed handoff evidence: {exc}") from exc

    # ── stage 1: integrity / content addressing ──────────────────────────────
    components: list[ComponentDigest] = []

    def _component(name: str, present: bool, actual: Optional[str]) -> bool:
        expected = manifest.get(f"{name}_digest")
        ok = present and expected is not None and actual == expected
        components.append(
            ComponentDigest(name, present, expected, actual if present else None, ok)
        )
        return ok

    rec_ok = _component("record", True, _jcs_digest(record_doc))
    did_ok = _component("did_document", True, _jcs_digest(did_document))
    kh_ok = _component("key_history", True, key_history.digest())
    rev_ok = _component("revocations", True, revocations.digest())

    anchor = evidence.get("anchor")
    anchor_present = anchor is not None
    if anchor_present:
        anchor_doc = _require_dict(anchor, "evidence.anchor")
        anc_ok = _component("anchor", True, _jcs_digest(anchor_doc))
    else:
        anchor_doc = {}
        # An absent anchor must not be pinned: a stray anchor_digest is drift.
        expected_anchor = manifest.get("anchor_digest")
        anc_ok = expected_anchor is None
        components.append(
            ComponentDigest("anchor", False, expected_anchor, None, anc_ok)
        )

    manifest_actual = _jcs_digest(manifest)
    manifest_expected = doc.get("manifest_digest")
    manifest_ok = manifest_actual == manifest_expected
    components.append(
        ComponentDigest("manifest", True, manifest_expected, manifest_actual, manifest_ok)
    )

    producer = manifest.get("producer")
    producer_coherent = (
        isinstance(producer, str)
        and producer == iss
        and did_document.get("id") == iss
    )

    integrity_ok = bool(
        rec_ok and did_ok and kh_ok and rev_ok and anc_ok
        and manifest_ok and producer_coherent
    )

    # ── stage 3 (prepared before the record call): anchor binding ────────────
    anchor_binds = False
    anchor_verified = False
    effective_anchored_time: Optional[str] = None
    if anchor_present:
        anchor_binds = bool(
            anchor_doc.get("hash_algorithm") == "sha256"
            and _hex_equals(
                anchor_doc.get("chain_head_hash"), _record_hash_hex(record_doc)
            )
        )
        anchor_verified = anchor_attested_time is not None
        if anchor_binds and anchor_verified:
            effective_anchored_time = anchor_attested_time

    # ── stage 2: the record verdict (C1, unchanged semantics) ────────────────
    record = verify_receipt_retained(
        receipt,
        did_document,
        key_history=key_history,
        revocations=revocations,
        anchored_time=effective_anchored_time,
    )

    revocation_source_present = (
        evidence.get("revocations") is not None or len(revocations) > 0
    )

    # ── stage 4: custody (orthogonal) ────────────────────────────────────────
    holder = manifest.get("holder")
    holder_attestation = doc.get("holder_attestation")
    custody = "unattested"
    holder_keyid: Optional[str] = None
    if holder_attestation is not None:
        att = _require_dict(holder_attestation, "holder_attestation")
        keyid = att.get("keyid")
        holder_keyid = keyid if isinstance(keyid, str) else None
        if _verify_holder_attestation(att, manifest):
            custody = "holder_attested_selfsupplied"
        else:
            custody = "holder_attestation_failed"

    producer_identity_basis = _producer_identity_basis(
        record, did_document, trusted_did_document
    )

    verifiable = record.verifiable
    corroborated = record.corroborated
    if strict:
        ok = bool(
            integrity_ok
            and corroborated
            and record.window_recorded
            and revocation_source_present
            and producer_identity_basis == "pinned"
        )
    else:
        ok = bool(integrity_ok and verifiable)

    reason = _handoff_reason(
        integrity_ok=integrity_ok,
        components=components,
        producer_coherent=producer_coherent,
        record=record,
        anchor_present=anchor_present,
        anchor_binds=anchor_binds,
        anchor_verified=anchor_verified,
        revocation_source_present=revocation_source_present,
        producer_identity_basis=producer_identity_basis,
        strict=strict,
        ok=ok,
    )

    return HandoffVerdict(
        schema=schema if isinstance(schema, str) else "",
        integrity_ok=integrity_ok,
        components=tuple(components),
        producer=producer if isinstance(producer, str) else None,
        holder=holder if isinstance(holder, str) else None,
        producer_identity_basis=producer_identity_basis,
        bound=record.bound,
        keyid=record.keyid,
        window_recorded=record.window_recorded,
        revocation_source_present=revocation_source_present,
        revoked=record.revoked,
        anchor_present=anchor_present,
        anchor_verified=anchor_verified,
        anchor_binds=anchor_binds,
        verifiable=verifiable,
        corroborated=corroborated,
        custody=custody,
        holder_keyid=holder_keyid,
        strict=strict,
        ok=ok,
        record=record.to_dict(),
        reason=reason,
    )


def _handoff_reason(
    *,
    integrity_ok: bool,
    components: list[ComponentDigest],
    producer_coherent: bool,
    record: RetentionResult,
    anchor_present: bool,
    anchor_binds: bool,
    anchor_verified: bool,
    revocation_source_present: bool,
    producer_identity_basis: str,
    strict: bool,
    ok: bool,
) -> str:
    if not integrity_ok:
        bad = [c.name for c in components if not c.ok]
        if not producer_coherent:
            bad.append("producer (record iss / document id / manifest disagree)")
        return "integrity failed: " + ", ".join(bad)
    if not record.verifiable:
        return (
            "package is internally consistent but the record is not verifiable: "
            f"{record.reason}"
        )
    if anchor_present and not anchor_binds:
        tail = "; the enclosed anchor does not bind to this record and was disregarded"
    elif anchor_present and not anchor_verified:
        tail = "; the enclosed anchor was not cryptographically verified (timeanchor extra)"
    else:
        tail = ""
    if strict and not ok:
        missing = []
        if not record.corroborated:
            missing.append("a verified anchor predating retirement")
        if not record.window_recorded:
            missing.append("a recorded validity window")
        if not revocation_source_present:
            missing.append("an affirmative revocation source")
        if producer_identity_basis != "pinned":
            missing.append("a pinned producer identity")
        return "strict mode unmet: needs " + ", ".join(missing) + tail
    tier = "corroborated" if record.corroborated else "verifiable"
    pinned = " (pinned)" if producer_identity_basis == "pinned" else ""
    note = (
        "; producer identity self-asserted, pin it out of band"
        if producer_identity_basis == "self_asserted_unpinned"
        else ""
    )
    return f"record is {tier} against the enclosed identity{pinned}{tail}{note}"


def build_handoff(
    *,
    record: dict[str, Any],
    did_document: dict[str, Any],
    key_history: Optional[dict[str, Any]] = None,
    revocations: Optional[dict[str, Any]] = None,
    anchor: Optional[dict[str, Any]] = None,
    producer: Optional[str] = None,
    holder: Optional[str] = None,
    cover: Optional[dict[str, Any]] = None,
    holder_attestation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble a cross-org handoff package from a producer's pieces.

    The issuer-side mirror of :func:`verify_handoff`. ``record`` and
    ``did_document`` are required; the rest fill the package's optional slots.
    ``producer`` defaults to the record's issuer and must equal it and the
    document id. An ``anchor`` must already bind to the record (its
    ``chain_head_hash`` is ``sha256(jcs(record))`` over sha256). ``cover`` is an
    opaque, caller-supplied plain-language block (system, action, period,
    provider, deployer, the obligation the evidence serves); Vaara carries it
    pinned but asserts no legal conclusion about it. ``holder_attestation`` is a
    pre-built custody signature block over the manifest (see
    :func:`sign_manifest`).

    Pins each component by digest (model digests for the key history and
    revocations, JCS digests for the rest), writes the manifest fingerprint, and
    validates by reconstructing: the assembled document is run back through
    :func:`verify_handoff` and must pass its integrity check, so a malformed
    piece fails here, named, rather than producing a package the verifier
    rejects. Returns the document as a plain dict for the caller to render.
    """
    receipt = receipt_from_dict(record)  # validates the record shape
    iss = receipt.receipt_asserted.iss
    producer = producer if producer is not None else iss
    if producer != iss:
        raise ValueError(
            f"producer {producer!r} does not match the record issuer {iss!r}"
        )
    if did_document.get("id") != iss:
        raise ValueError(
            f"did_document id {did_document.get('id')!r} does not match "
            f"the record issuer {iss!r}"
        )

    evidence: dict[str, Any] = {"record": record, "did_document": did_document}
    if key_history is not None:
        evidence["key_history"] = key_history
    if revocations is not None:
        evidence["revocations"] = revocations
    if anchor is not None:
        if anchor.get("hash_algorithm") != "sha256" or not _hex_equals(
            anchor.get("chain_head_hash"), _record_hash_hex(record)
        ):
            raise ValueError(
                "anchor does not bind to the record: chain_head_hash must be "
                "sha256(jcs(record)) and hash_algorithm must be 'sha256'"
            )
        evidence["anchor"] = anchor

    eff_kh = _effective_key_history(evidence, did_document)
    eff_rev = _effective_revocations(evidence, did_document, iss)

    manifest: dict[str, Any] = {
        "producer": producer,
        "record_digest": _jcs_digest(record),
        "did_document_digest": _jcs_digest(did_document),
        "key_history_digest": eff_kh.digest(),
        "revocations_digest": eff_rev.digest(),
    }
    if holder is not None:
        manifest["holder"] = holder
    if cover is not None:
        manifest["cover"] = cover
    if anchor is not None:
        manifest["anchor_digest"] = _jcs_digest(anchor)

    doc: dict[str, Any] = {"schema": SCHEMA, "evidence": evidence, "manifest": manifest}
    doc["manifest_digest"] = _jcs_digest(manifest)
    if holder_attestation is not None:
        doc["holder_attestation"] = holder_attestation

    verdict = verify_handoff(doc)
    if not verdict.integrity_ok:
        raise ValueError(
            f"assembled handoff failed its own integrity check: {verdict.reason}"
        )
    return doc


def sign_manifest(
    manifest: dict[str, Any],
    *,
    alg: str,
    keyid: str,
    signing_material: Any,
    verifying_jwk: dict[str, Any],
) -> dict[str, Any]:
    """Build a holder custody attestation over the canonical manifest bytes.

    Signs ``canonical_json(manifest)`` with an asymmetric key and encloses the
    public JWK so a third party can verify it. ``alg`` is ``ES256`` or ``RS256``
    (a symmetric secret cannot be verified cross-org). The result is the
    ``holder_attestation`` block :func:`build_handoff` accepts and
    :func:`verify_handoff` checks. The attestation is custody evidence only and
    never bears on the record verdict.
    """
    if alg not in _HOLDER_ALGS:
        raise ValueError(f"holder attestation alg must be one of {_HOLDER_ALGS}")
    from vaara.attestation._sep2787_signing import sign_es256, sign_rs256

    payload = canonical_json(manifest)
    if alg == "ES256":
        signature = sign_es256(payload, private_key=signing_material)
    else:
        signature = sign_rs256(payload, private_key=signing_material)
    return {
        "alg": alg,
        "keyid": keyid,
        "verifying_jwk": verifying_jwk,
        "signature": signature,
    }


__all__ = [
    "SCHEMA",
    "ComponentDigest",
    "HandoffVerdict",
    "build_handoff",
    "sign_manifest",
    "verify_handoff",
]
