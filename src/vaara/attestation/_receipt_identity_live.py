"""Live-resolvable agent identity (did:web) for execution receipts.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt``.

Level-3 verification, per ``docs/design/resolvable-agent-identity-spec.md``:
given a receipt whose ``receiptAsserted.iss`` is a ``did:web`` identifier,
fetch the DID document over HTTPS at audit time, then run the level-2
pinned-resolvable check against the freshly fetched document. Two properties
distinguish level 3 from level 2:

1. **Auditable resolution.** The fetch is recorded (URL, fetch time, and a
   digest over the exact document bytes) so the resolution itself is
   reproducible: a second auditor handed the same recorded document
   reproduces the level-2 verdict offline, and the digest pins which document
   was seen.
2. **Revocation in time.** A DID document may mark a verification method
   ``revoked`` (an ISO 8601 instant) or mark the whole identity
   ``deactivated``. A key revoked at or before the receipt was issued does
   not bind the receipt, even when the signature math still checks out. A key
   revoked *after* issuance still binds: revocation is not retroactive, the
   same "revoked before compromise" reasoning the threshold-signing
   key-lifecycle markers use.

The revocation comparison uses ``receiptAsserted.iat``, which is
self-asserted. The verdict exposes both instants (``issued_at`` and
``revoked_at``) so a verifier holding a stronger time anchor (the audit-trail
hash chain) can re-decide rather than trust the receipt's own clock.

This is purely additive. It composes ``verify_receipt_identity`` (level 2)
and ``did_web_to_url`` unchanged, touches neither the receipt envelope nor
the canonicalization, and pulls in no dependency beyond the standard library
for the default fetch. The fetcher is injectable, so the unit tests and
conformance vectors run with no network.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from vaara.attestation._receipt_identity import (
    IdentityResult,
    did_web_to_url,
    verify_receipt_identity,
)
from vaara.attestation._receipt_types import ExecutionReceipt
from vaara.attestation._revocation import _parse_iso, revoked_in_time
from vaara.attestation._attest_types import AttestationError

# A DID document is a small JSON file. Cap the read so a misbehaving or
# hostile host cannot stream an unbounded body into the verifier.
_DEFAULT_MAX_BYTES = 1 << 20  # 1 MiB
_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_CACHE_TTL_S = 3600.0

# Fetcher signature: takes the resolved HTTPS URL, returns the raw document
# bytes. Injectable so tests and conformance vectors run offline.
Fetcher = Callable[[str], bytes]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_epoch(iso: str) -> float:
    dt = _parse_iso(iso)
    return 0.0 if dt is None else dt.timestamp()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects: a did:web document resolves at a fixed URL.

    Following redirects from a hostile host is an SSRF vector (a 3xx to an
    internal address or to ``http://``). Returning None makes urllib raise
    on the 3xx, which the caller treats as a resolution failure: fail closed
    rather than chase the redirect.
    """

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def https_fetch(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> bytes:
    """Default DID-document fetcher: a size-capped HTTPS GET, stdlib only.

    HTTPS only, since ``did:web`` documents resolve over the web PKI, and
    redirects are refused (an SSRF vector). The body is read up to
    ``max_bytes`` and rejected if it exceeds the cap. Deployers needing host
    allowlisting, pinning, or proxy egress inject their own fetcher instead.
    """
    if not url.startswith("https://"):
        raise AttestationError(f"DID document URL is not HTTPS: {url!r}")
    request = urllib.request.Request(
        url, headers={"Accept": "application/did+json, application/json"}
    )
    with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:  # noqa: S310
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise AttestationError(f"DID document exceeds {max_bytes} bytes: {url!r}")
    return body


@dataclass(frozen=True)
class ResolutionMeta:
    """Auditable record of a DID-document resolution.

    ``document_digest`` is ``sha256:<hex>`` over the exact bytes fetched, so
    an auditor handed the same document reproduces the verdict and can
    confirm it is the document that was seen. ``from_cache`` is True when the
    document was served from a prior fetch within its TTL.
    """

    did: str
    url: str
    fetched_at: str
    document_digest: str
    from_cache: bool


@dataclass(frozen=True)
class LiveIdentityResult:
    """Verdict of a level-3 live-resolvable identity check.

    Extends the level-2 verdict with revocation, deactivation, and the
    resolution record. ``trusted`` is the single overall verdict: the
    issuer resolved, the signature bound to a key the document lists, the
    identity is not deactivated, and the bound key was not revoked at or
    before issuance.

    ``revoked_at`` and ``issued_at`` are surfaced even on a pass so a
    verifier with a stronger time anchor than the receipt's self-asserted
    ``iat`` can re-decide.
    """

    resolved: bool
    bound: bool
    keyid: Optional[str]
    revoked: bool
    deactivated: bool
    trusted: bool
    reason: str
    resolution: Optional[ResolutionMeta]
    issued_at: Optional[str] = None
    revoked_at: Optional[str] = None

    @property
    def identity(self) -> IdentityResult:
        """The level-2 verdict embedded in this level-3 result."""
        return IdentityResult(self.resolved, self.bound, self.keyid, self.reason)


class DidDocumentCache:
    """In-memory TTL cache of resolved DID documents, keyed by DID.

    A verifier checking many receipts from the same issuer resolves the
    document once per TTL window. The cache stores the parsed document and
    the resolution metadata of the fetch that filled it; a cache hit reuses
    that metadata with ``from_cache=True``. Time is supplied by the caller
    (``now`` epoch seconds) so the cache is deterministic under test.
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_CACHE_TTL_S) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[float, dict[str, Any], ResolutionMeta]] = {}

    def get(
        self, did: str, *, now: float
    ) -> Optional[tuple[dict[str, Any], ResolutionMeta]]:
        entry = self._entries.get(did)
        if entry is None:
            return None
        stored_at, document, meta = entry
        if now - stored_at > self._ttl:
            del self._entries[did]
            return None
        cached_meta = ResolutionMeta(
            did=meta.did, url=meta.url, fetched_at=meta.fetched_at,
            document_digest=meta.document_digest, from_cache=True,
        )
        return document, cached_meta

    def put(
        self, did: str, document: dict[str, Any], meta: ResolutionMeta, *, now: float
    ) -> None:
        self._entries[did] = (now, document, meta)


def _revocation_instant(
    did_document: dict[str, Any], keyid: Optional[str]
) -> Optional[str]:
    """Return the ``revoked`` instant of the named verification method, if any."""
    if keyid is None:
        return None
    methods = did_document.get("verificationMethod")
    if not isinstance(methods, list):
        return None
    for method in methods:
        if isinstance(method, dict) and method.get("id") == keyid:
            revoked = method.get("revoked")
            return revoked if isinstance(revoked, str) else None
    return None


def _live_reason(
    identity: IdentityResult,
    deactivated: bool,
    revoked: bool,
    revoked_at: Optional[str],
) -> str:
    if deactivated:
        return "DID document is deactivated"
    if revoked:
        return f"signing key was revoked at or before issuance ({revoked_at})"
    return identity.reason


def _fail(reason: str) -> "LiveIdentityResult":
    return LiveIdentityResult(
        False, False, None, False, False, False, reason, None
    )


def _resolve(
    iss: str,
    fetcher: Optional[Fetcher],
    cache: Optional[DidDocumentCache],
    fetched_at: str,
    cache_now: float,
) -> "tuple[dict[str, Any], ResolutionMeta] | LiveIdentityResult":
    """Return ``(document, meta)`` or a failed ``LiveIdentityResult``."""
    if cache is not None:
        hit = cache.get(iss, now=cache_now)
        if hit is not None:
            return hit
    try:
        url = did_web_to_url(iss)
    except AttestationError as exc:
        return _fail(f"cannot map did:web to URL: {exc}")
    fetch = fetcher if fetcher is not None else https_fetch
    try:
        raw = fetch(url)
    except Exception as exc:  # noqa: BLE001 - any fetch failure is a resolution failure
        return _fail(f"DID document fetch failed: {exc}")
    raw_bytes = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
    try:
        parsed = json.loads(raw_bytes)
    except (ValueError, TypeError) as exc:
        return _fail(f"DID document is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        return _fail("DID document is not a JSON object")
    meta = ResolutionMeta(
        did=iss, url=url, fetched_at=fetched_at,
        document_digest="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        from_cache=False,
    )
    if cache is not None:
        cache.put(iss, parsed, meta, now=cache_now)
    return parsed, meta


def verify_receipt_identity_live(
    receipt: ExecutionReceipt,
    *,
    fetcher: Optional[Fetcher] = None,
    cache: Optional[DidDocumentCache] = None,
    now: Optional[str] = None,
    now_epoch: Optional[float] = None,
    expected_keyid: Optional[str] = None,
) -> LiveIdentityResult:
    """Level-3 live-resolvable identity check.

    Resolves the receipt's ``did:web`` ``iss`` to a DID document over HTTPS
    (or from ``cache``), records the resolution, runs the level-2 check, and
    then applies deactivation and revocation-in-time. ``fetcher`` defaults to
    a size-capped stdlib HTTPS GET; pass your own for allowlisting, pinning,
    or proxy egress, or to verify offline against a captured document.

    ``now`` (ISO 8601) stamps the resolution record and is the default clock
    for revocation; ``now_epoch`` drives the cache TTL. Both default to the
    current UTC time. A plain-string ``iss`` is never failed for lack of a
    DID: it returns ``resolved=False`` with no fetch attempted.
    """
    iss = receipt.receipt_asserted.iss
    if not iss.startswith("did:web:"):
        return _fail("iss is not a did:web identifier")

    fetched_at = now if now is not None else _utc_now_iso()
    cache_now = now_epoch if now_epoch is not None else _parse_epoch(fetched_at)

    resolved = _resolve(iss, fetcher, cache, fetched_at, cache_now)
    if isinstance(resolved, LiveIdentityResult):
        return resolved
    document, meta = resolved

    identity = verify_receipt_identity(
        receipt, document, expected_keyid=expected_keyid
    )

    deactivated = document.get("deactivated") is True
    issued_at = receipt.receipt_asserted.iat

    revoked = False
    revoked_at: Optional[str] = None
    if identity.bound:
        revoked_at = _revocation_instant(document, identity.keyid)
        # Revoked binds only when revocation is at or before issuance; an
        # unparseable revocation or issuance instant fails closed. Shared
        # with the cross-stack registry so every lens applies one rule.
        if revoked_at is not None and revoked_in_time(revoked_at, issued_at):
            revoked = True

    trusted = (
        identity.resolved and identity.bound and not deactivated and not revoked
    )
    reason = _live_reason(identity, deactivated, revoked, revoked_at)

    return LiveIdentityResult(
        resolved=identity.resolved,
        bound=identity.bound,
        keyid=identity.keyid,
        revoked=revoked,
        deactivated=deactivated,
        trusted=trusted,
        reason=reason,
        resolution=meta,
        issued_at=issued_at,
        revoked_at=revoked_at,
    )
