# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Registration with an IETF SCITT Transparency Service, and the offline check.

``scitt_anchor`` appends to a Merkle log Vaara operates. This module talks to
a Transparency Service somebody else operates, over SCRAPI
(draft-ietf-scitt-scrapi):

1. ``signed_statement`` builds a SCITT Signed Statement: a tagged COSE_Sign1
   whose protected header carries ``alg``, ``content type``, CWT claims (15)
   with ``iss`` and ``sub``, and an ``x5chain`` (33). The issuer is a
   ``did:x509`` identifier (``did_x509``) bound to the chain's root, which is
   what the Microsoft SCITT CCF ledger requires of a statement it accepts.
2. ``register`` POSTs it to ``/entries`` and follows the service to the
   receipt: a 303 to ``/entries/{id}``, polled while it answers 302, until a
   200 carries the COSE receipt. A 201 with the receipt in the body is taken
   as is.
3. ``service_keys`` reads ``/.well-known/scitt-keys`` once. From then on
   ``verify_receipt`` needs no network: it recomputes the ledger root from the
   statement bytes and the receipt's inclusion proof and checks the service's
   signature over that root.

Receipts use the CCF_LEDGER_SHA256 verifiable data structure (``vds`` 2,
draft-birkholz-cose-receipts-ccf-profile), which is what the CCF ledger emits.
The leaf commits to ``sha256(signed statement)``, so a receipt checked here
covers exactly the bytes that were registered.

Needs ``cbor2`` and ``cryptography`` (base install). Uses the standard
library for HTTP.
"""
from __future__ import annotations

import base64
import hashlib
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

#: The SCRAPI version the CCF ledger serves the 303 flow under.
API_VERSION = "2026-03-26"
CT_COSE = "application/cose"

COSE_SIGN1_TAG = 18
_HDR_ALG = 1
_HDR_CONTENT_TYPE = 3
_HDR_KID = 4
_HDR_CWT_CLAIMS = 15
_HDR_X5CHAIN = 33
_HDR_VDS = 395
_HDR_VDP = 396
_VDP_INCLUSION = -1
_CWT_ISS, _CWT_SUB, _CWT_IAT = 1, 2, 6
VDS_CCF_LEDGER_SHA256 = 2
_CCF_LEAF, _CCF_PATH = 1, 2

#: COSE alg id -> (hash, coordinate size). CCF service identities are P-384
#: by default, so ES384 is as likely as ES256.
_ECDSA = {-7: ("sha256", 32), -35: ("sha384", 48), -36: ("sha512", 66)}
#: COSE_Key crv -> curve name in ``cryptography``.
_CURVES = {1: "SECP256R1", 2: "SECP384R1", 3: "SECP521R1"}


class ScittServiceError(RuntimeError):
    """Raised when a statement cannot be built, registered or checked."""


def _cbor() -> Any:
    try:
        import cbor2
    except ImportError as exc:  # pragma: no cover - base dependency
        raise ScittServiceError("cbor2 is required") from exc
    return cbor2


# -- the statement ------------------------------------------------------------

def did_x509(root_der: bytes, subject_cn: str) -> str:
    """``did:x509`` naming the chain's root by fingerprint and the leaf by CN."""
    fp = base64.urlsafe_b64encode(hashlib.sha256(root_der).digest()).rstrip(b"=")
    return (f"did:x509:0:sha256:{fp.decode('ascii')}::subject:CN:"
            + urllib.parse.quote(subject_cn, safe=""))


def signed_statement(
    payload: bytes,
    *,
    private_key: Any,
    chain: Sequence[bytes],
    issuer: str,
    subject: str,
    content_type: str = "application/json",
    issued_at: Optional[int] = None,
) -> bytes:
    """A tagged COSE_Sign1 Signed Statement, ES256, payload attached.

    ``chain`` is DER certificates leaf first; ``private_key`` is the leaf's
    P-256 key. ``issuer`` is normally ``did_x509(chain[-1], <leaf CN>)``.
    """
    from vaara.attestation._receipt_cose import _es256_sign

    if not chain:
        raise ScittServiceError("an x5chain needs at least the leaf certificate")
    cbor2 = _cbor()
    claims = {_CWT_ISS: issuer, _CWT_SUB: subject,
              _CWT_IAT: int(time.time() if issued_at is None else issued_at)}
    header = {_HDR_ALG: -7, _HDR_CONTENT_TYPE: content_type,
              _HDR_CWT_CLAIMS: claims,
              _HDR_X5CHAIN: [bytes(c) for c in chain] if len(chain) > 1 else bytes(chain[0])}
    protected = cbor2.dumps(header, canonical=True)
    to_sign = cbor2.dumps(["Signature1", protected, b"", bytes(payload)], canonical=True)
    signature = _es256_sign(private_key, to_sign)
    return bytes(cbor2.dumps(cbor2.CBORTag(
        COSE_SIGN1_TAG, [protected, {}, bytes(payload), signature]), canonical=True))


# -- talking to the service ---------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """SCRAPI puts meaning in 302 and 303, so redirects are read, not followed."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


@dataclass
class _Response:
    status: int
    headers: Mapping[str, str]
    body: bytes


def _context(cafile: Optional[str], insecure: bool) -> ssl.SSLContext:
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context(cafile=cafile)


def _request(method: str, url: str, *, body: Optional[bytes], ctx: ssl.SSLContext,
             timeout: float, headers: Optional[dict[str, str]] = None) -> _Response:
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=ctx))
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with opener.open(req, timeout=timeout) as resp:
            return _Response(resp.status, dict(resp.headers), resp.read())
    except urllib.error.HTTPError as exc:
        return _Response(exc.code, dict(exc.headers or {}), exc.read() or b"")


def _with_version(url: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}api-version={API_VERSION}"


def _header(resp: _Response, name: str) -> str:
    for k, v in resp.headers.items():
        if k.lower() == name:
            return v
    return ""


@dataclass
class Registration:
    """What a completed registration returned."""

    entry_id: str
    receipt: bytes


def register(
    service_url: str,
    statement: bytes,
    *,
    cafile: Optional[str] = None,
    insecure: bool = False,
    timeout: float = 120.0,
    request_timeout: float = 30.0,
) -> Registration:
    """Register ``statement`` and wait for its receipt. Raises on any failure."""
    base = service_url.rstrip("/")
    ctx = _context(cafile, insecure)
    resp = _request("POST", _with_version(base + "/entries"), body=statement, ctx=ctx,
                    timeout=request_timeout, headers={"Content-Type": CT_COSE})
    if resp.status == 201 and resp.body:
        location = _header(resp, "location")
        return Registration(location.rsplit("/entries/", 1)[-1], resp.body)
    if resp.status != 303:
        raise ScittServiceError(
            f"POST /entries answered {resp.status}: {resp.body[:300]!r}")
    location = _header(resp, "location")
    if "/entries/" not in location:
        raise ScittServiceError(f"303 without an /entries/ Location: {location!r}")
    entry_id = location.rsplit("/entries/", 1)[1].split("?", 1)[0]
    entry_url = urllib.parse.urljoin(base + "/", location)

    deadline = time.monotonic() + timeout
    while True:
        resp = _request("GET", _with_version(entry_url.split("?", 1)[0]), body=None,
                        ctx=ctx, timeout=request_timeout)
        if resp.status == 200 and resp.body:
            return Registration(entry_id, resp.body)
        if resp.status not in (202, 302, 404, 503):
            raise ScittServiceError(
                f"GET /entries/{entry_id} answered {resp.status}: {resp.body[:300]!r}")
        if time.monotonic() > deadline:
            raise ScittServiceError(f"no receipt for {entry_id} within {timeout:.0f}s")
        try:
            wait = float(_header(resp, "retry-after") or 1)
        except ValueError:
            wait = 1.0
        time.sleep(min(max(wait, 0.2), 5.0))


def service_keys(
    service_url: str, *, cafile: Optional[str] = None, insecure: bool = False,
    request_timeout: float = 30.0,
) -> bytes:
    """The raw COSE_Key_Set from ``/.well-known/scitt-keys``. Keep it to verify offline."""
    resp = _request("GET", service_url.rstrip("/") + "/.well-known/scitt-keys", body=None,
                    ctx=_context(cafile, insecure), timeout=request_timeout)
    if resp.status != 200:
        raise ScittServiceError(f"scitt-keys answered {resp.status}")
    return resp.body


# -- offline ------------------------------------------------------------------

def load_key_set(key_set: bytes) -> dict[bytes, Any]:
    """COSE_Key_Set (EC2 keys) -> {kid: public key}."""
    from cryptography.hazmat.primitives.asymmetric import ec

    keys: dict[bytes, Any] = {}
    for k in _cbor().loads(key_set):
        if not isinstance(k, Mapping) or k.get(1) != 2 or k.get(-1) not in _CURVES:
            continue
        curve = getattr(ec, _CURVES[k[-1]])()
        pub = ec.EllipticCurvePublicNumbers(
            int.from_bytes(k[-2], "big"), int.from_bytes(k[-3], "big"), curve).public_key()
        kid = k.get(2)
        if isinstance(kid, str):
            kid = kid.encode()
        if isinstance(kid, (bytes, bytearray)):
            keys[bytes(kid)] = pub
    return keys


def _ecdsa_verify(alg: int, public_key: Any, signature: bytes, data: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    if alg not in _ECDSA:
        return False
    hname, size = _ECDSA[alg]
    if len(signature) != 2 * size:
        return False
    der = utils.encode_dss_signature(int.from_bytes(signature[:size], "big"),
                                     int.from_bytes(signature[size:], "big"))
    try:
        public_key.verify(der, data, ec.ECDSA(getattr(hashes, hname.upper())()))
    except (InvalidSignature, TypeError, ValueError):
        return False
    return True


@dataclass
class ReceiptCheck:
    ok: bool
    detail: str
    kid: Optional[bytes] = None
    root: Optional[bytes] = None


def _seq(v: Any) -> bool:
    return isinstance(v, Sequence) and not isinstance(v, (bytes, bytearray, str))


def verify_receipt(receipt: bytes, *, statement: bytes, keys: Mapping[bytes, Any]) -> ReceiptCheck:
    """Check a CCF-profile COSE receipt against the registered statement bytes.

    Every inclusion proof must commit to ``sha256(statement)`` and lead to the
    same root, and the service's signature must verify over that root under
    the key the receipt's ``kid`` names. No network.
    """
    cbor2 = _cbor()
    try:
        msg = cbor2.loads(receipt)
    except Exception as exc:  # cbor2 raises assorted decode errors
        return ReceiptCheck(False, f"receipt is not CBOR: {exc}")
    if not isinstance(msg, cbor2.CBORTag) or msg.tag != COSE_SIGN1_TAG:
        return ReceiptCheck(False, "receipt is not a tagged COSE_Sign1")
    parts = msg.value
    if not _seq(parts) or len(parts) != 4:
        return ReceiptCheck(False, "receipt is not a four-element COSE_Sign1")
    protected_bytes, unprotected, payload, signature = parts
    if payload is not None or not isinstance(signature, bytes):
        return ReceiptCheck(False, "receipt payload must be detached")
    try:
        protected = cbor2.loads(protected_bytes)
    except Exception:
        return ReceiptCheck(False, "protected header is not CBOR")
    if not isinstance(protected, Mapping) or not isinstance(unprotected, Mapping):
        return ReceiptCheck(False, "headers are not maps")
    if protected.get(_HDR_VDS) != VDS_CCF_LEDGER_SHA256:
        return ReceiptCheck(False, f"vds is {protected.get(_HDR_VDS)!r}, not CCF_LEDGER_SHA256 (2)")
    kid = protected.get(_HDR_KID)
    kid = kid.encode() if isinstance(kid, str) else kid
    key = keys.get(kid) if isinstance(kid, bytes) else None
    if key is None:
        return ReceiptCheck(False, "no service key for the receipt's kid", kid=kid)
    vdp = unprotected.get(_HDR_VDP)
    proofs = vdp.get(_VDP_INCLUSION) if isinstance(vdp, Mapping) else None
    if not _seq(proofs) or not proofs:
        return ReceiptCheck(False, "no inclusion proof", kid=kid)

    claims_digest = hashlib.sha256(statement).digest()
    root: Optional[bytes] = None
    for encoded in proofs:
        try:
            proof = cbor2.loads(encoded)
            leaf, path = proof[_CCF_LEAF], proof[_CCF_PATH]
            internal_hash, evidence, data_hash = leaf
            if data_hash != claims_digest:
                return ReceiptCheck(False, "the proof commits to different statement bytes", kid=kid)
            ev = evidence.encode() if isinstance(evidence, str) else bytes(evidence)
            acc = hashlib.sha256(internal_hash + hashlib.sha256(ev).digest() + data_hash).digest()
            for left, sibling in path:
                acc = hashlib.sha256(sibling + acc if left else acc + sibling).digest()
        except Exception as exc:
            return ReceiptCheck(False, f"malformed inclusion proof: {exc}", kid=kid)
        if root is not None and acc != root:
            return ReceiptCheck(False, "inclusion proofs disagree on the root", kid=kid)
        root = acc

    to_verify = cbor2.dumps(["Signature1", protected_bytes, b"", root], canonical=True)
    alg = protected.get(_HDR_ALG)
    if not isinstance(alg, int) or not _ecdsa_verify(alg, key, signature, to_verify):
        return ReceiptCheck(False, "service signature does not verify over the root",
                            kid=kid, root=root)
    return ReceiptCheck(True, "ok", kid=kid, root=root)
