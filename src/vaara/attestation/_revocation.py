"""Cross-stack revocation: one revocation-in-time rule for every lens.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt``.

Revocation began life inside the level-3 live identity check
(``_receipt_identity_live``): given a DID document, a signing key marked
``revoked`` at or before a receipt's ``iat`` no longer yields a trusted
verdict, while a key revoked afterwards still binds (revocation is not
retroactive). That rule lived in exactly one place, so the same receipt
checked through the receipt verifier, the transparency log, or an
Article-12 export ignored revocation entirely.

This module lifts the rule out into a source-agnostic ``RevocationRegistry``
so every lens consults the same predicate. The registry holds revocation
*facts*, each one a ``(scope, subject, revoked_at)`` triple:

- ``scope="key"``: a specific signing key, named by its keyid.
- ``scope="identity"``: a whole agent identity, named by its ``did:web``
  issuer.

The single predicate is :meth:`RevocationRegistry.status`. A receipt issued
at ``issued_at`` by ``iss`` (optionally bound to ``keyid``) is
revoked-in-time iff a matching entry's ``revoked_at`` is at or before
``issued_at``. An unparseable revocation or issuance instant fails closed.

See ``docs/design/cross-stack-revocation-spec.md``. Purely additive: the
receipt envelope and canonicalization are untouched, and the registry pulls
in only the standard library (``rfc8785`` is needed only for the export
digest, imported lazily).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Literal, Optional

from vaara.attestation._receipt_types import ExecutionReceipt

RevocationScope = Literal["key", "identity"]


def _parse_iso(value: object) -> Optional[datetime]:
    """Parse an ISO 8601 instant to an aware UTC datetime, or None.

    Accepts a trailing ``Z``. A naive timestamp is read as UTC so two
    instants are comparable regardless of which form an emitter used. The
    single shared parser used by both the registry and the level-3 live
    identity check, so revocation comparisons cannot drift between lenses.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def revoked_in_time(revoked_at: object, issued_at: object) -> bool:
    """Whether a revocation at ``revoked_at`` binds a receipt issued at ``issued_at``.

    True iff the revocation is at or before issuance. An unparseable
    revocation or issuance instant fails closed (returns True): if the
    verifier cannot establish that revocation came after issuance, it does
    not grant the benefit of the doubt. This is the exact comparison the
    level-3 live identity check applies, shared so the rule has one home.
    """
    revoked_dt = _parse_iso(revoked_at)
    issued_dt = _parse_iso(issued_at)
    if revoked_dt is None or issued_dt is None:
        return True
    return revoked_dt <= issued_dt


@dataclass(frozen=True)
class RevocationEntry:
    """One revocation fact.

    ``scope`` is ``"key"`` (``subject`` is a keyid) or ``"identity"``
    (``subject`` is a ``did:web`` issuer). ``revoked_at`` is an ISO 8601
    instant; the entry binds any receipt that the subject issued at or after
    that instant.
    """

    scope: RevocationScope
    subject: str
    revoked_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "scope": self.scope,
            "subject": self.subject,
            "revoked_at": self.revoked_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RevocationEntry":
        if not isinstance(data, dict):
            raise ValueError("each revocation entry must be an object")
        scope = data.get("scope")
        if scope not in ("key", "identity"):
            raise ValueError(
                f"revocation entry scope must be 'key' or 'identity', got {scope!r}"
            )
        subject = data.get("subject")
        revoked_at = data.get("revoked_at")
        if not isinstance(subject, str) or not subject:
            raise ValueError("revocation entry 'subject' must be a non-empty string")
        if not isinstance(revoked_at, str) or not revoked_at:
            raise ValueError("revocation entry 'revoked_at' must be a non-empty string")
        return cls(scope=scope, subject=subject, revoked_at=revoked_at)


@dataclass(frozen=True)
class RevocationStatus:
    """Verdict of a revocation check.

    ``revoked`` is the single answer: was the receipt's issuer (or its bound
    key) revoked at or before the receipt was issued. ``revoked_at`` and
    ``matched_by`` name the binding entry when one matched. ``issued_at`` is
    surfaced so a verifier with a stronger time anchor than the receipt's
    self-asserted ``iat`` can re-decide.
    """

    revoked: bool
    matched_by: Optional[RevocationScope]
    revoked_at: Optional[str]
    issued_at: Optional[str]
    reason: str


class RevocationRegistry:
    """A set of revocation facts with one revocation-in-time predicate.

    Source-agnostic: entries may come from a DID document
    (:meth:`from_did_document`), an operator's out-of-band revocation list
    (:meth:`from_dict` / constructor), or revocations published in the
    transparency log. Every verification lens consults the same registry, so
    a receipt gets the same revoked verdict whichever lens looks.
    """

    def __init__(self, entries: Iterable[RevocationEntry] = ()) -> None:
        self._entries: tuple[RevocationEntry, ...] = tuple(entries)

    @property
    def entries(self) -> tuple[RevocationEntry, ...]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def status(
        self,
        iss: str,
        issued_at: str,
        *,
        keyid: Optional[str] = None,
    ) -> RevocationStatus:
        """Whether the issuer (or bound key) was revoked at or before issuance.

        Matches identity-scope entries on ``iss`` and key-scope entries on
        ``keyid`` (only when ``keyid`` is given). When several entries match,
        the earliest revocation wins, since the strictest fact governs. A
        key-scope match is reported in preference to an identity-scope one at
        the same instant, because it is the more specific statement.
        """
        best: Optional[tuple[datetime, RevocationEntry]] = None
        unparseable: Optional[RevocationEntry] = None
        for entry in self._entries:
            if entry.scope == "key":
                if keyid is None or entry.subject != keyid:
                    continue
            else:  # identity
                if entry.subject != iss:
                    continue
            if not revoked_in_time(entry.revoked_at, issued_at):
                continue
            revoked_dt = _parse_iso(entry.revoked_at)
            issued_dt = _parse_iso(issued_at)
            if revoked_dt is None or issued_dt is None:
                # Fail-closed match with no usable instant to rank by; keep
                # it as a fallback if nothing rankable matches.
                if unparseable is None:
                    unparseable = entry
                continue
            if best is None or _ranks_before(revoked_dt, entry, best):
                best = (revoked_dt, entry)

        if best is not None:
            _, entry = best
            return RevocationStatus(
                revoked=True,
                matched_by=entry.scope,
                revoked_at=entry.revoked_at,
                issued_at=issued_at,
                reason=(
                    f"{entry.scope} {entry.subject!r} was revoked at or before "
                    f"issuance (revoked_at={entry.revoked_at}, iat={issued_at})"
                ),
            )
        if unparseable is not None:
            return RevocationStatus(
                revoked=True,
                matched_by=unparseable.scope,
                revoked_at=unparseable.revoked_at,
                issued_at=issued_at,
                reason=(
                    f"{unparseable.scope} {unparseable.subject!r} revocation or "
                    f"issuance instant is unparseable; failing closed"
                ),
            )
        return RevocationStatus(
            revoked=False,
            matched_by=None,
            revoked_at=None,
            issued_at=issued_at,
            reason="no matching revocation at or before issuance",
        )

    def to_dict(self) -> dict[str, object]:
        """Canonical, sorted dict form. Stable across constructions."""
        entries = sorted(
            (e.to_dict() for e in self._entries),
            key=lambda d: (d["scope"], d["subject"], d["revoked_at"]),
        )
        return {"version": 1, "entries": entries}

    def canonical_bytes(self) -> bytes:
        """RFC 8785 JCS bytes over :meth:`to_dict`, for a stable digest."""
        from vaara.attestation._attest_canonical import canonical_json

        return canonical_json(self.to_dict())

    def digest(self) -> str:
        """``sha256:<hex>`` over :meth:`canonical_bytes`.

        Pins the exact revocation state into a signed export so a regulator
        recomputes every receipt's verdict against the registry the exporter
        actually used.
        """
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RevocationRegistry":
        if not isinstance(data, dict):
            raise ValueError("revocation registry must be a JSON object")
        raw = data.get("entries", [])
        if not isinstance(raw, list):
            raise ValueError("revocation registry 'entries' must be a list")
        return cls(RevocationEntry.from_dict(e) for e in raw)

    @classmethod
    def from_did_document(
        cls, did_document: dict[str, object], iss: str
    ) -> "RevocationRegistry":
        """Project a DID document's revoked verification methods into entries.

        Each ``verificationMethod`` carrying a ``revoked`` instant becomes a
        key-scope entry keyed by the method id. This is the same data the
        level-3 live identity check reads, so the registry and that check
        agree on the document by construction. Deactivation
        (``deactivated: true``) is identity existence, not time-scoped
        revocation, so it stays a level-3 concern and is not projected here.
        ``iss`` is accepted for symmetry with identity-scope sources and to
        document which issuer the document describes.
        """
        methods = did_document.get("verificationMethod")
        entries: list[RevocationEntry] = []
        if isinstance(methods, list):
            for method in methods:
                if not isinstance(method, dict):
                    continue
                revoked = method.get("revoked")
                mid = method.get("id")
                if (
                    isinstance(revoked, str)
                    and revoked
                    and isinstance(mid, str)
                    and mid
                ):
                    entries.append(
                        RevocationEntry(scope="key", subject=mid, revoked_at=revoked)
                    )
        return cls(entries)


def _ranks_before(
    revoked_dt: datetime,
    entry: RevocationEntry,
    best: tuple[datetime, RevocationEntry],
) -> bool:
    """True if ``entry`` should replace ``best`` as the governing revocation.

    Earlier revocation wins; on a tie a key-scope entry beats an
    identity-scope one because it is the more specific statement.
    """
    best_dt, best_entry = best
    if revoked_dt != best_dt:
        return revoked_dt < best_dt
    return entry.scope == "key" and best_entry.scope != "key"


def check_receipt_revocation(
    receipt: ExecutionReceipt,
    registry: RevocationRegistry,
    *,
    keyid: Optional[str] = None,
) -> RevocationStatus:
    """Receipt-verifier-side revocation check.

    Reads the receipt's ``iss`` and ``iat`` and consults ``registry``. The
    offline counterpart of the level-3 revocation rule: no DID fetch, no
    network. Pass ``keyid`` (resolved from a level-2/3 identity check) for
    key-level granularity; without it, only identity-scope revocations
    apply.
    """
    asserted = receipt.receipt_asserted
    return registry.status(asserted.iss, asserted.iat, keyid=keyid)


def receipt_leaf_bytes(receipt: ExecutionReceipt) -> bytes:
    """JCS-canonical bytes of the full receipt (signature included).

    The natural transparency-log leaf for a receipt: the complete signed
    record, so the logged bytes pin the exact receipt instance.
    """
    from vaara.attestation._attest_canonical import canonical_json

    return canonical_json(receipt.to_dict())


@dataclass(frozen=True)
class LoggedReceiptVerdict:
    """Transparency-log-lens verdict for a logged receipt.

    ``included`` is the inclusion-proof result; ``revocation`` is the same
    revocation status the receipt verifier produces. ``ok`` is the single
    answer a monitor wants: the receipt is in the log *and* its issuer was
    not revoked-in-time. A receipt with a valid inclusion proof but a
    revoked-in-time issuer is ``included`` yet not ``ok``, the same way that
    receipt fails the identity lens.
    """

    included: bool
    revocation: RevocationStatus
    ok: bool


def verify_logged_receipt(
    *,
    receipt: ExecutionReceipt,
    proof: object,
    expected_root: bytes,
    registry: RevocationRegistry,
    leaf_data: Optional[bytes] = None,
    keyid: Optional[str] = None,
) -> LoggedReceiptVerdict:
    """Check transparency-log inclusion and revocation in one call.

    Verifies the inclusion ``proof`` against ``expected_root`` (using the
    full canonical receipt bytes as the leaf unless ``leaf_data`` is given to
    match exactly what was appended), then applies the same revocation rule
    the receipt verifier uses. ``ok`` is True only when the receipt is both
    included and not revoked-in-time, so a monitor reconstructing a registry
    from logged revocations reaches the same conclusion as every other lens.
    """
    from vaara.attestation.transparency_log import InclusionProof, verify_inclusion

    if not isinstance(proof, InclusionProof):
        raise TypeError("proof must be an InclusionProof")
    leaf = leaf_data if leaf_data is not None else receipt_leaf_bytes(receipt)
    included = verify_inclusion(
        leaf_data=leaf, proof=proof, expected_root=expected_root
    )
    revocation = check_receipt_revocation(receipt, registry, keyid=keyid)
    return LoggedReceiptVerdict(
        included=included,
        revocation=revocation,
        ok=included and not revocation.revoked,
    )
