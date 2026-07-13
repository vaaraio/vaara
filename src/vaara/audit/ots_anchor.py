"""OpenTimestamps witness anchors for Vaara receipts (SPEC.md Section 4).

The ``opentimestamps`` timestampAnchors method commits a receipt's
signed-payload sha256 digest (the same ``anchoredDigest`` the ``rfc3161``
method pins) to public OpenTimestamps calendar servers, which aggregate it
into a Bitcoin transaction. It stacks on the RFC 3161 anchor rather than
replacing it: the TSA token is instant and legal-grade; the OTS proof is a
trust-minimized public witness with zero recurring cost, final once the
calendar's commitment is buried in a Bitcoin block (typically 1-6 hours).

Pending never blocks. :func:`ots_anchor_receipt` returns the anchor entry
immediately with the calendars' pending proof; :func:`upgrade_ots_anchor`
later folds the Bitcoin attestation into the same entry in place.

The ``proof`` field is base64 of a standard detached ``.ots`` file, readable
by the reference ``ots`` client. :func:`verify_ots_anchor` checks offline that
the proof commits exactly the receipt's signed payload and reports the proof's
attestations; it does NOT verify Bitcoin block headers. For that final step
use the standard tool: write :func:`signed_payload_bytes` to ``payload``, the
decoded proof to ``payload.ots``, then run ``ots verify payload.ots``.

Optional dependency (the ``ots`` extra): the ``opentimestamps`` library. The
.ots serialization and commitment-operation format is what makes the proof
interoperable, so it is not hand-rolled here.
"""

from __future__ import annotations

import base64
import copy
import os
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

import rfc8785

from vaara.audit.receipt_anchor import _SIGNED_BLOCKS, _signed_payload_digest
from vaara.audit.timeanchor import TimeAnchorError

try:  # optional: the 'ots' extra
    from opentimestamps.core.notary import (  # type: ignore
        BitcoinBlockHeaderAttestation,
        PendingAttestation,
    )
    from opentimestamps.core.op import OpAppend, OpSHA256  # type: ignore
    from opentimestamps.core.serialize import (  # type: ignore
        BytesDeserializationContext,
        BytesSerializationContext,
    )
    from opentimestamps.core.timestamp import (  # type: ignore
        DetachedTimestampFile,
        Timestamp,
    )

    _HAS_DEPS = True
except ImportError:  # pragma: no cover - exercised via the install-hint path
    _HAS_DEPS = False

_INSTALL_HINT = (
    "OpenTimestamps anchoring requires the 'ots' extra. "
    "Install with: pip install \"vaara[ots]\" "
    "(provides the opentimestamps library)."
)

# Public aggregation calendars run by the OpenTimestamps project and
# Eternity Wall. Free, no account; each accepts the digest in milliseconds
# and commits it to Bitcoin within hours.
DEFAULT_CALENDARS = (
    "https://alice.btc.calendar.opentimestamps.org",
    "https://bob.btc.calendar.opentimestamps.org",
    "https://finney.calendar.eternitywall.com",
)

# A transport takes (url, post_body_or_None, timeout) and returns response
# bytes. POST when a body is given, GET otherwise. Injectable for tests and
# offline deployments.
Transport = Callable[[str, Optional[bytes], float], bytes]


def _require_deps() -> None:
    if not _HAS_DEPS:
        raise TimeAnchorError(_INSTALL_HINT)


def _default_transport(url: str, data: Optional[bytes], timeout: float) -> bytes:
    headers = {"Accept": "application/vnd.opentimestamps.v1",
               "User-Agent": "vaara-ots"}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    # URL is operator-supplied calendar configuration, not attacker-controlled.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def signed_payload_bytes(receipt: dict) -> bytes:
    """The JCS signed-payload bytes; ``sha256`` of this is the anchoredDigest.

    Write these bytes to a file next to the decoded proof (``payload`` and
    ``payload.ots``) to verify the anchor with the standard ``ots`` client.
    """
    try:
        payload = {k: receipt[k] for k in _SIGNED_BLOCKS}
    except KeyError as exc:
        raise TimeAnchorError(f"receipt missing signed-payload block: {exc}") from exc
    return rfc8785.dumps(payload)


def parse_ots_proof(proof: bytes) -> "DetachedTimestampFile":
    """Parse raw ``.ots`` bytes into a ``DetachedTimestampFile``."""
    _require_deps()
    try:
        return DetachedTimestampFile.deserialize(BytesDeserializationContext(proof))
    except Exception as exc:
        raise TimeAnchorError(f"could not parse .ots proof: {exc}") from exc


def _serialize_proof(detached: "DetachedTimestampFile") -> bytes:
    ctx = BytesSerializationContext()
    detached.serialize(ctx)
    return ctx.getbytes()


def _classify(timestamp: "Timestamp") -> tuple[list[int], list[str]]:
    """Return (bitcoin block heights, pending calendar URIs) in the proof."""
    heights: list[int] = []
    pending: list[str] = []
    for _msg, attestation in timestamp.all_attestations():
        if isinstance(attestation, BitcoinBlockHeaderAttestation):
            heights.append(attestation.height)
        elif isinstance(attestation, PendingAttestation):
            pending.append(attestation.uri)
    return sorted(set(heights)), sorted(set(pending))


def ots_anchor_receipt(
    receipt: dict,
    calendars: tuple[str, ...] | list[str] = DEFAULT_CALENDARS,
    *,
    timeout: float = 10.0,
    transport: Optional[Transport] = None,
) -> dict[str, Any]:
    """Submit the receipt's signed-payload digest to OTS calendars.

    Returns a SPEC.md Section 4 ``timestampAnchors`` entry immediately: the
    calendars answer in milliseconds with a *pending* proof, and the entry
    never blocks on Bitcoin finality. Fold the Bitcoin attestation in later
    with :func:`upgrade_ots_anchor`. At least one calendar must accept; per-
    calendar failures are tolerated as long as one succeeds.
    """
    _require_deps()
    send = transport or _default_transport
    digest = _signed_payload_digest(receipt)

    file_ts = Timestamp(digest)
    # Blind the digest with a per-anchor nonce before it leaves the host, the
    # same privacy step the reference client takes.
    nonced = file_ts.ops.add(OpAppend(os.urandom(16)))
    commitment = nonced.ops.add(OpSHA256())

    used: list[str] = []
    failures: list[str] = []
    for calendar in calendars:
        try:
            body = send(calendar.rstrip("/") + "/digest", commitment.msg, timeout)
            reply = Timestamp.deserialize(
                BytesDeserializationContext(body), commitment.msg)
            commitment.merge(reply)
            used.append(calendar)
        except Exception as exc:  # network, HTTP, malformed reply
            failures.append(f"{calendar}: {exc}")
    if not used:
        raise TimeAnchorError(
            "no OTS calendar accepted the digest: " + "; ".join(failures))

    return {
        "method": "opentimestamps",
        "anchoredDigest": "sha256:" + digest.hex(),
        "proof": base64.b64encode(
            _serialize_proof(DetachedTimestampFile(OpSHA256(), file_ts))
        ).decode("ascii"),
        "status": "pending",
        "calendars": used,
    }


def upgrade_ots_anchor(
    anchor: dict,
    *,
    timeout: float = 10.0,
    transport: Optional[Transport] = None,
) -> dict[str, Any]:
    """Fetch calendar upgrades and fold Bitcoin attestations into ``anchor``.

    Idempotent: an anchor whose proof already carries a Bitcoin attestation is
    returned confirmed without any network call. On network failure (or when
    no calendar has the Bitcoin attestation yet) the anchor is returned
    unchanged and still pending; call again later. Returns a new dict, never
    mutates the input.

    Pending-attestation URIs live inside the proof bytes, which may come from
    an untrusted party; following them blindly would let a crafted proof point
    the upgrade fetch anywhere (SSRF). Only URIs whose https origin matches the
    anchor's recorded ``calendars`` or the package defaults are fetched; any
    other attestation is left pending, untouched.
    """
    _require_deps()
    if anchor.get("method") != "opentimestamps":
        raise TimeAnchorError(
            f"not an opentimestamps anchor: method={anchor.get('method')!r}")
    send = transport or _default_transport
    try:
        detached = parse_ots_proof(base64.b64decode(anchor["proof"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise TimeAnchorError(f"anchor proof is not valid base64: {exc}") from exc

    heights, _pending = _classify(detached.timestamp)
    if heights:
        return dict(anchor, status="confirmed")

    trusted = {
        _https_origin(cal)
        for cal in tuple(anchor.get("calendars") or ()) + tuple(DEFAULT_CALENDARS)
        if _https_origin(cal) is not None
    }
    upgraded = False
    for msg, attestation in list(detached.timestamp.all_attestations()):
        if not isinstance(attestation, PendingAttestation):
            continue
        if _https_origin(attestation.uri) not in trusted:
            continue  # proof-supplied URI outside the trusted calendars
        url = attestation.uri.rstrip("/") + "/timestamp/" + msg.hex()
        try:
            body = send(url, None, timeout)
            reply = Timestamp.deserialize(BytesDeserializationContext(body), msg)
        except Exception:
            continue  # calendar unreachable or not yet Bitcoin-final
        # Merge into the sub-timestamp whose msg this attestation covers.
        for sub in _walk(detached.timestamp):
            if sub.msg == msg:
                sub.merge(reply)
                upgraded = True
                break
    if not upgraded:
        return copy.deepcopy(anchor)  # nothing final yet; try again later

    heights, _pending = _classify(detached.timestamp)
    return dict(
        anchor,
        proof=base64.b64encode(_serialize_proof(detached)).decode("ascii"),
        status="confirmed" if heights else "pending",
    )


def _https_origin(uri: str) -> Optional[str]:
    """``scheme://netloc`` for an https URI, ``None`` for anything else."""
    try:
        parsed = urllib.parse.urlsplit(uri.strip())
    except ValueError:
        return None
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    return f"https://{parsed.netloc.lower()}"


def _walk(timestamp: "Timestamp"):
    yield timestamp
    for _op, sub in timestamp.ops.items():
        yield from _walk(sub)


def verify_ots_anchor(receipt: dict, anchor: dict) -> dict[str, Any]:
    """Verify an ``opentimestamps`` anchor against its receipt, offline.

    Pins anchoredDigest == sha256 of the JCS signed payload (SPEC.md Section 6
    rule 3, the same rule ``verify_receipt_anchor`` applies), then parses the
    proof and confirms it commits exactly that digest. Returns a dict with
    ``status`` (``confirmed`` when a Bitcoin block-header attestation is
    present, else ``pending``), ``bitcoin_block_heights``, and
    ``pending_calendars``.

    This does NOT verify Bitcoin block headers; a ``confirmed`` status here
    means the proof *claims* Bitcoin attestations at the reported heights. For
    full verification use the standard ``ots`` client against a Bitcoin node:
    write :func:`signed_payload_bytes` to ``payload`` and the decoded proof to
    ``payload.ots``, then ``ots verify payload.ots``.
    """
    _require_deps()
    if anchor.get("method") != "opentimestamps":
        raise TimeAnchorError(
            f"not an opentimestamps anchor: method={anchor.get('method')!r}")
    digest = _signed_payload_digest(receipt)
    if anchor.get("anchoredDigest") != "sha256:" + digest.hex():
        raise TimeAnchorError(
            "anchoredDigest does not match the receipt's signed payload")
    try:
        detached = parse_ots_proof(base64.b64decode(anchor["proof"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise TimeAnchorError(f"anchor proof is not valid base64: {exc}") from exc
    if not isinstance(detached.file_hash_op, OpSHA256):
        raise TimeAnchorError("proof file-hash operation is not sha256")
    if detached.timestamp.msg != digest:
        raise TimeAnchorError(
            "proof does not commit the receipt's signed-payload digest "
            "(it timestamped a different value)")
    heights, pending = _classify(detached.timestamp)
    if not heights and not pending:
        raise TimeAnchorError("proof carries no attestations")
    return {
        "status": "confirmed" if heights else "pending",
        "bitcoin_block_heights": heights,
        "pending_calendars": pending,
    }


__all__ = [
    "DEFAULT_CALENDARS",
    "ots_anchor_receipt",
    "parse_ots_proof",
    "signed_payload_bytes",
    "upgrade_ots_anchor",
    "verify_ots_anchor",
]
