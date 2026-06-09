"""Key validity windows: verify a record under a rotated or retired key.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt``.

The cross-stack revocation registry answers "was this key *revoked* at or
before the record was signed?". It does not answer the other half of the
key-lifecycle question that an EU AI Act Article 12 retention window forces:
keys rotate. A record signed in 2026 by ``#key-2026`` is audited in 2031,
by which time the issuer has rotated to ``#key-2031`` and the old key is
retired. A naive identity check against the *current* DID document either
fails (the document dropped the old key) or wrongly rejects it (the key is
present but marked end-of-life). Neither outcome is correct: a signature a
key made while it was valid stays valid forever, exactly as a paper signature
does not become void when the signer later changes pens.

This module models a key's *validity window*. Each key has an optional
``not_before`` (activation) and ``not_after`` (retirement) instant. A
signature is within the window when it was made at or after ``not_before``
and strictly before ``not_after``, the same half-open convention the
revocation rule uses (the lifecycle-change instant belongs to the state that
follows it). A key with neither bound is unbounded: this keeps every existing
DID document working unchanged, since a document that records no lifecycle
markers places no window constraint.

The model is source-agnostic, mirroring ``RevocationRegistry``: a
:class:`KeyHistory` is built from a DID document's per-method ``validFrom`` /
``validUntil`` markers (:meth:`KeyHistory.from_did_document`), from an
operator's out-of-band key-history list (:meth:`KeyHistory.from_dict`), or
constructed directly. It carries no key material and no signature logic, so it
runs in the base install with only the standard library; ``rfc8785`` is needed
only for the canonical digest and is imported lazily.

See ``docs/design/key-rotation-retention-spec.md``. Purely additive: the
receipt envelope and canonicalization are untouched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional

from vaara.attestation._revocation import _parse_iso

# Per-method validity markers we read from a DID document, in precedence
# order. ``validFrom`` / ``validUntil`` is the W3C VC Data Model 2.0 spelling
# and the canonical one; ``notBefore`` / ``notAfter`` is accepted as an alias
# so a document that borrows the X.509 vocabulary still resolves.
_NOT_BEFORE_KEYS = ("validFrom", "notBefore")
_NOT_AFTER_KEYS = ("validUntil", "notAfter")


def within_validity(
    at_time: object,
    not_before: Optional[str],
    not_after: Optional[str],
) -> bool:
    """Whether ``at_time`` falls inside the half-open window ``[nb, na)``.

    True iff ``at_time`` is at or after ``not_before`` (when set) and strictly
    before ``not_after`` (when set). A signature made exactly at ``not_after``
    is outside the window: the key retired at that instant, mirroring the
    revocation rule where a revocation at the issuance instant binds. An
    unparseable ``at_time``, or a bound that is present but unparseable, fails
    closed (returns False): the verifier cannot establish the signature was in
    window, so it does not grant the benefit of the doubt.
    """
    at_dt = _parse_iso(at_time)
    if at_dt is None:
        return False
    if not_before is not None:
        nb_dt = _parse_iso(not_before)
        if nb_dt is None or at_dt < nb_dt:
            return False
    if not_after is not None:
        na_dt = _parse_iso(not_after)
        if na_dt is None or at_dt >= na_dt:
            return False
    return True


@dataclass(frozen=True)
class KeyValidity:
    """One key's validity window.

    ``keyid`` is the verification method id. ``not_before`` and ``not_after``
    are optional ISO 8601 instants; ``None`` means that side is unbounded. The
    key is valid for signatures made in ``[not_before, not_after)``.
    """

    keyid: str
    not_before: Optional[str] = None
    not_after: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        out: dict[str, str] = {"keyid": self.keyid}
        if self.not_before is not None:
            out["not_before"] = self.not_before
        if self.not_after is not None:
            out["not_after"] = self.not_after
        return out

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "KeyValidity":
        keyid = data.get("keyid")
        if not isinstance(keyid, str) or not keyid:
            raise ValueError("key-validity entry 'keyid' must be a non-empty string")
        not_before = data.get("not_before")
        not_after = data.get("not_after")
        if not_before is not None and not isinstance(not_before, str):
            raise ValueError("key-validity entry 'not_before' must be a string or absent")
        if not_after is not None and not isinstance(not_after, str):
            raise ValueError("key-validity entry 'not_after' must be a string or absent")
        return cls(keyid=keyid, not_before=not_before, not_after=not_after)


@dataclass(frozen=True)
class KeyValidityStatus:
    """Verdict of a key-validity check at one instant.

    ``within`` is the single answer: was the key valid for a signature made at
    the checked instant. ``recorded`` is True when the history actually knows
    a window for the key; when False the key is treated as unbounded (``within``
    is True) and a verifier learns the judgment rested on no recorded lifecycle.
    ``not_before`` and ``not_after`` echo the governing window for the report.
    """

    within: bool
    recorded: bool
    not_before: Optional[str]
    not_after: Optional[str]
    reason: str


class KeyHistory:
    """A set of key validity windows with one in-window predicate.

    Source-agnostic, like :class:`RevocationRegistry`: windows may come from a
    DID document's per-method ``validFrom`` / ``validUntil`` markers
    (:meth:`from_did_document`), an operator's out-of-band key-history list
    (:meth:`from_dict` / the constructor), or be built directly. Every lens
    that needs the rotation judgment consults the same history, so a key gets
    the same in-window verdict whoever looks.
    """

    def __init__(self, entries: Iterable[KeyValidity] = ()) -> None:
        self._entries: tuple[KeyValidity, ...] = tuple(entries)

    @property
    def entries(self) -> tuple[KeyValidity, ...]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def validity(self, keyid: Optional[str], at_time: object) -> KeyValidityStatus:
        """Whether ``keyid`` was valid for a signature made at ``at_time``.

        A key with no recorded window is unbounded: ``within`` is True and
        ``recorded`` is False, so an existing document that carries no
        lifecycle markers verifies exactly as before. When more than one window
        is recorded for the same key (a key reactivated after a gap), the key
        is in window if ``at_time`` falls inside any of them; the governing
        window is the one that admitted it, or the first recorded otherwise.
        """
        if keyid is None:
            return KeyValidityStatus(
                True, False, None, None,
                "no keyid bound; key validity window not checked",
            )
        matching = [e for e in self._entries if e.keyid == keyid]
        if not matching:
            return KeyValidityStatus(
                True, False, None, None,
                f"no validity window recorded for {keyid!r}; treated as unbounded",
            )
        for entry in matching:
            if within_validity(at_time, entry.not_before, entry.not_after):
                return KeyValidityStatus(
                    True, True, entry.not_before, entry.not_after,
                    f"signature within the validity window of {keyid!r}",
                )
        first = matching[0]
        return KeyValidityStatus(
            False, True, first.not_before, first.not_after,
            f"signature outside the validity window of {keyid!r} "
            f"(not_before={first.not_before}, not_after={first.not_after})",
        )

    def to_dict(self) -> dict[str, object]:
        """Canonical, sorted dict form. Stable across constructions."""
        keys = sorted(
            (e.to_dict() for e in self._entries),
            key=lambda d: (d["keyid"], d.get("not_before", ""), d.get("not_after", "")),
        )
        return {"version": 1, "keys": keys}

    def canonical_bytes(self) -> bytes:
        """RFC 8785 JCS bytes over :meth:`to_dict`, for a stable digest."""
        from vaara.attestation._sep2787_canonical import canonical_json

        return canonical_json(self.to_dict())

    def digest(self) -> str:
        """``sha256:<hex>`` over :meth:`canonical_bytes`.

        Pins the exact key-history state into a signed export so a regulator
        recomputes every record's verdict against the windows the exporter used.
        """
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "KeyHistory":
        if not isinstance(data, dict):
            raise ValueError("key history must be a JSON object")
        raw = data.get("keys", [])
        if not isinstance(raw, list):
            raise ValueError("key-history 'keys' must be a list")
        entries: list[KeyValidity] = []
        for e in raw:
            if not isinstance(e, dict):
                raise ValueError("each key-history entry must be an object")
            entries.append(KeyValidity.from_dict(e))
        return cls(entries)

    @classmethod
    def from_did_document(cls, did_document: dict[str, object]) -> "KeyHistory":
        """Project a DID document's per-method validity markers into windows.

        Each ``verificationMethod`` that carries a ``validFrom`` / ``validUntil``
        (or the ``notBefore`` / ``notAfter`` alias) instant becomes a window
        keyed by the method id. A method with neither marker contributes no
        entry and so stays unbounded. This is the document a regulator archives
        for the retention window, retired keys retained and marked, so the
        rotation judgment is reproducible offline years later.
        """
        methods = did_document.get("verificationMethod")
        entries: list[KeyValidity] = []
        if isinstance(methods, list):
            for method in methods:
                if not isinstance(method, dict):
                    continue
                mid = method.get("id")
                if not isinstance(mid, str) or not mid:
                    continue
                not_before = _first_str(method, _NOT_BEFORE_KEYS)
                not_after = _first_str(method, _NOT_AFTER_KEYS)
                if not_before is None and not_after is None:
                    continue
                entries.append(
                    KeyValidity(keyid=mid, not_before=not_before, not_after=not_after)
                )
        return cls(entries)


def _first_str(method: dict[str, object], keys: tuple[str, ...]) -> Optional[str]:
    """Return the first present non-empty string value among ``keys``."""
    for key in keys:
        value = method.get(key)
        if isinstance(value, str) and value:
            return value
    return None
