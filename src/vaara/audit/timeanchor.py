"""External time anchoring for the audit hash chain (v0.48).

The hash chain proves order and integrity, but not *when* it existed: every
timestamp comes from the runtime's own clock and is signed on export by the
runtime's own key. If that key is later compromised, an attacker who also
controls the clock can forge a backdated alternate chain that nothing internal
to Vaara can distinguish from the original.

An external time anchor closes that gap. At intervals Vaara takes the current
chain head (the last record's ``record_hash``, itself a SHA-256 digest) and
obtains a trusted timestamp over it from an authority it does not control. The
authority signs ``(digest, time)``, proving the chain head existed no later
than the attested time. A forger who later steals the signing key still cannot
date records before a genuine anchor without forging the authority's signature,
which lives outside Vaara's trust boundary.

The default authority is an RFC 3161 Time-Stamp Authority. RFC 3161 underpins
eIDAS qualified electronic timestamps, recognised EU-wide, so a qualified TSA
makes this regulator-grade evidence under EU AI Act Article 12. Verification is
offline. This is the anti-backdating mechanism cited by the server-side signed
execution-record SEP (``docs/sep/sep-server-execution-record.md``); the
execution records sit inside the chain, so anchoring the head anchors them.

Optional dependencies (the ``timeanchor`` extra): ``asn1crypto`` and
``cryptography``. The TSA HTTP round trip uses the standard library.
"""

from __future__ import annotations

import secrets
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

try:  # optional: the 'timeanchor' extra
    from asn1crypto import algos, cms, core, tsp  # type: ignore
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding
    from cryptography.x509 import load_der_x509_certificate

    _HAS_DEPS = True
except ImportError:  # pragma: no cover - exercised via the install-hint path
    _HAS_DEPS = False

_INSTALL_HINT = (
    "External time anchoring requires the 'timeanchor' extra. "
    "Install with: pip install 'vaara[timeanchor]' "
    "(provides asn1crypto and cryptography)."
)

# Message-imprint hash algorithms we accept. The chain head is a SHA-256
# digest, so sha256 is the natural default.
_DIGEST_LEN = {"sha256": 32, "sha384": 48, "sha512": 64}


class TimeAnchorError(Exception):
    """Anchoring or anchor verification failed."""


def _require_deps() -> None:
    if not _HAS_DEPS:
        raise TimeAnchorError(_INSTALL_HINT)


@dataclass
class TimeAnchor:
    """A trusted-timestamp anchor over one chain-head digest.

    ``chain_position`` is the index of the anchored record in the trail and
    ``chain_head_hash`` is that record's ``record_hash`` (hex). ``token_b64``
    is the base64 DER of the RFC 3161 ``TimeStampToken`` and is the actual
    evidence, verified offline. ``anchored_time`` is the attested time parsed
    from the token for display; the verifier re-derives it from the token.
    """

    chain_position: int
    chain_head_hash: str
    backend: str
    tsa_url: str
    hash_algorithm: str
    token_b64: str
    anchored_time: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimeAnchor":
        fields = {
            "chain_position", "chain_head_hash", "backend", "tsa_url",
            "hash_algorithm", "token_b64", "anchored_time",
        }
        return cls(**{k: d[k] for k in fields})


def build_timestamp_request(
    digest: bytes,
    *,
    hash_algorithm: str = "sha256",
    nonce: Optional[int] = None,
    cert_req: bool = True,
) -> bytes:
    """Build a DER-encoded RFC 3161 ``TimeStampReq`` over ``digest``."""
    _require_deps()
    expected = _DIGEST_LEN.get(hash_algorithm)
    if expected is None:
        raise TimeAnchorError(f"unsupported hash algorithm: {hash_algorithm!r}")
    if len(digest) != expected:
        raise TimeAnchorError(
            f"digest length {len(digest)} does not match {hash_algorithm} "
            f"({expected} bytes)"
        )
    req = tsp.TimeStampReq({
        "version": 1,
        "message_imprint": tsp.MessageImprint({
            "hash_algorithm": algos.DigestAlgorithm({"algorithm": hash_algorithm}),
            "hashed_message": digest,
        }),
        "nonce": nonce if nonce is not None else secrets.randbits(64),
        "cert_req": cert_req,
    })
    return req.dump()


def extract_token_from_response(der: bytes) -> bytes:
    """Pull the DER ``TimeStampToken`` out of a ``TimeStampResp``.

    Raises if the TSA did not grant the request.
    """
    _require_deps()
    try:
        resp = tsp.TimeStampResp.load(der)
    except Exception as exc:  # malformed reply from an untrusted endpoint
        raise TimeAnchorError(f"could not parse TimeStampResp: {exc}") from exc
    status = resp["status"]["status"].native
    if status not in ("granted", "granted_with_mods"):
        try:
            fail = resp["status"]["fail_info"].native
        except (KeyError, ValueError):
            fail = None  # fail_info is optional and may be absent
        raise TimeAnchorError(
            f"TSA refused the request: status={status} fail_info={fail}"
        )
    token = resp["time_stamp_token"]
    if token is None or token.native is None:
        raise TimeAnchorError("TSA reply granted but carried no time_stamp_token")
    return token.dump()


def _hash_for(name: str) -> Any:
    # Instantiate the concrete hash directly so type checkers do not see an
    # attempt to construct the abstract HashAlgorithm base.
    if name == "sha256":
        return hashes.SHA256()
    if name == "sha384":
        return hashes.SHA384()
    if name == "sha512":
        return hashes.SHA512()
    raise TimeAnchorError(f"unsupported digest algorithm in token: {name!r}")


def _signer_cert(signed_data: Any) -> Any:
    """Return the cryptography certificate that signed ``signed_data``.

    Matches the SignerInfo ``sid`` (issuer + serial) against the embedded
    certificate set, falling back to the sole certificate when only one is
    present (the common TSA case).
    """
    certs = [c for c in signed_data["certificates"] if c.name == "certificate"]
    if not certs:
        raise TimeAnchorError("token carries no signer certificate")
    sid = signed_data["signer_infos"][0]["sid"]
    if sid.name == "issuer_and_serial_number":
        want_serial = sid.chosen["serial_number"].native
        want_issuer = sid.chosen["issuer"]
        for choice in certs:
            tbs = choice.chosen["tbs_certificate"]
            if tbs["serial_number"].native == want_serial and tbs["issuer"] == want_issuer:
                return load_der_x509_certificate(choice.chosen.dump())
        raise TimeAnchorError("no embedded certificate matches the signer id")
    if len(certs) == 1:
        return load_der_x509_certificate(certs[0].chosen.dump())
    raise TimeAnchorError("cannot disambiguate signer certificate by key id")


def _verify_cms_signature(signed_data: Any, signer_cert: Any) -> None:
    """Verify the TSA's signature over the SignedAttributes."""
    signer_info = signed_data["signer_infos"][0]
    signed_attrs = signer_info["signed_attrs"]
    if not signed_attrs or signed_attrs.native is None:
        raise TimeAnchorError("token has no signed attributes")

    # The signature covers the SignedAttributes encoded as an explicit SET OF
    # (tag 0x31), not the [0] IMPLICIT tag used inside the SignerInfo
    # (RFC 5652 section 5.4). untag() drops the implicit tag for re-encoding.
    signed_attrs_der = signed_attrs.untag().dump()

    digest_alg = signer_info["digest_algorithm"]["algorithm"].native
    sig = signer_info["signature"].native
    sig_alg = signer_info["signature_algorithm"]["algorithm"].native
    pub = signer_cert.public_key()
    try:
        if sig_alg in ("rsassa_pkcs1v15", "sha256_rsa", "sha384_rsa", "sha512_rsa", "rsa"):
            pub.verify(sig, signed_attrs_der, padding.PKCS1v15(), _hash_for(digest_alg))
        elif sig_alg in ("ecdsa", "sha256_ecdsa", "sha384_ecdsa", "sha512_ecdsa"):
            pub.verify(sig, signed_attrs_der, ec.ECDSA(_hash_for(digest_alg)))
        else:
            raise TimeAnchorError(f"unsupported TSA signature algorithm: {sig_alg!r}")
    except InvalidSignature as exc:
        raise TimeAnchorError(
            "TSA signature did not verify (token forged or corrupt)"
        ) from exc


def verify_timestamp_token(
    token_der: bytes,
    expected_digest: bytes,
    *,
    hash_algorithm: str = "sha256",
) -> datetime:
    """Verify an RFC 3161 token and return the attested UTC time.

    Checks, in order: the token is CMS SignedData wrapping a TSTInfo; the
    TSTInfo message imprint equals ``(hash_algorithm, expected_digest)``; the
    SignedAttributes message-digest equals the hash of the TSTInfo content; and
    the TSA's signature over the SignedAttributes verifies under the embedded
    signer certificate. Raises :class:`TimeAnchorError` on any failure.

    Trust note: this proves the token is internally consistent and signed by
    the certificate it carries. Establishing that the certificate is a trusted
    (e.g. eIDAS-qualified) TSA is the deployer's policy; pin the TSA
    certificate out of band to enforce that.
    """
    _require_deps()
    try:
        content_info = cms.ContentInfo.load(token_der)
    except Exception as exc:
        raise TimeAnchorError(f"could not parse TimeStampToken: {exc}") from exc
    if content_info["content_type"].native != "signed_data":
        raise TimeAnchorError("token is not CMS SignedData")
    signed_data = content_info["content"]

    encap = signed_data["encap_content_info"]
    if encap["content_type"].native != "tst_info":
        raise TimeAnchorError("token does not encapsulate a TSTInfo")
    econtent = encap["content"]
    if econtent is None:
        raise TimeAnchorError("token has no eContent")
    # Read the exact eContent octets. asn1crypto may auto-parse the tst_info
    # spec, so cast to a plain OctetString to recover the raw bytes the
    # message_digest attribute was computed over (re-encoding could differ).
    tst_bytes = econtent.cast(core.OctetString).native
    tst_info = tsp.TSTInfo.load(tst_bytes)

    imprint = tst_info["message_imprint"]
    token_alg = imprint["hash_algorithm"]["algorithm"].native
    token_digest = imprint["hashed_message"].native
    if token_alg != hash_algorithm:
        raise TimeAnchorError(
            f"token imprint algorithm {token_alg!r} != expected {hash_algorithm!r}"
        )
    if token_digest != expected_digest:
        raise TimeAnchorError(
            "token does not attest the expected digest "
            "(it timestamped a different value)"
        )

    # The SignedAttributes message-digest must equal the hash of the TSTInfo
    # eContent, or the signature would cover something other than this token.
    signer_info = signed_data["signer_infos"][0]
    digest_alg = signer_info["digest_algorithm"]["algorithm"].native
    md_attr = None
    for attr in signer_info["signed_attrs"]:
        if attr["type"].native == "message_digest":
            md_attr = attr["values"][0].native
            break
    if md_attr is None:
        raise TimeAnchorError("token signed attributes lack a message_digest")
    h = hashes.Hash(_hash_for(digest_alg))
    h.update(bytes(tst_bytes))
    if h.finalize() != md_attr:
        raise TimeAnchorError("signed message_digest does not match the TSTInfo content")

    _verify_cms_signature(signed_data, _signer_cert(signed_data))

    gen_time = tst_info["gen_time"].native
    if gen_time.tzinfo is None:
        gen_time = gen_time.replace(tzinfo=timezone.utc)
    return gen_time.astimezone(timezone.utc)


# A transport takes (url, der_request, timeout) and returns the der response.
# Injectable so tests and offline deployments can supply a local TSA.
Transport = Callable[[str, bytes, float], bytes]


def _urllib_transport(url: str, der_request: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(
        url,
        data=der_request,
        headers={
            "Content-Type": "application/timestamp-query",
            "Accept": "application/timestamp-reply",
        },
        method="POST",
    )
    # URL is operator-supplied TSA configuration, not attacker-controlled.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


class RFC3161TimeAnchorClient:
    """Anchors chain-head digests to an RFC 3161 Time-Stamp Authority."""

    backend = "rfc3161"

    def __init__(
        self,
        tsa_url: str,
        *,
        hash_algorithm: str = "sha256",
        timeout: float = 10.0,
        transport: Optional[Transport] = None,
    ) -> None:
        _require_deps()
        if hash_algorithm not in _DIGEST_LEN:
            raise TimeAnchorError(f"unsupported hash algorithm: {hash_algorithm!r}")
        self.tsa_url = tsa_url
        self.hash_algorithm = hash_algorithm
        self.timeout = timeout
        self._transport = transport or _urllib_transport

    def anchor(self, chain_position: int, chain_head_hash: str) -> TimeAnchor:
        """Obtain a trusted timestamp over a chain-head digest.

        ``chain_head_hash`` is the record's ``record_hash`` (a SHA-256 hex
        digest), anchored directly as the message imprint so the TSA attests
        this exact chain head existed at the returned time.
        """
        import base64
        digest = _digest_bytes(chain_head_hash, self.hash_algorithm)
        request = build_timestamp_request(digest, hash_algorithm=self.hash_algorithm)
        try:
            response = self._transport(self.tsa_url, request, self.timeout)
        except Exception as exc:  # network, TLS, HTTP error
            raise TimeAnchorError(
                f"TSA request to {self.tsa_url} failed: {exc}"
            ) from exc
        token_der = extract_token_from_response(response)
        attested = verify_timestamp_token(
            token_der, digest, hash_algorithm=self.hash_algorithm
        )
        return TimeAnchor(
            chain_position=chain_position,
            chain_head_hash=chain_head_hash,
            backend=self.backend,
            tsa_url=self.tsa_url,
            hash_algorithm=self.hash_algorithm,
            token_b64=base64.b64encode(token_der).decode("ascii"),
            anchored_time=attested.isoformat(),
        )


def _digest_bytes(chain_head_hash: str, hash_algorithm: str) -> bytes:
    try:
        digest = bytes.fromhex(chain_head_hash)
    except ValueError as exc:
        raise TimeAnchorError(
            f"chain_head_hash is not hex: {chain_head_hash!r}"
        ) from exc
    if len(digest) != _DIGEST_LEN[hash_algorithm]:
        raise TimeAnchorError(
            f"chain_head_hash is {len(digest)} bytes, not a {hash_algorithm} digest"
        )
    return digest


def verify_anchor(anchor: TimeAnchor) -> datetime:
    """Verify an anchor's token attests its own ``chain_head_hash``.

    Returns the attested UTC time. Does not check the hash belongs to a given
    chain; use :func:`verify_anchor_over_records` for that.
    """
    _require_deps()
    import base64
    try:
        token_der = base64.b64decode(anchor.token_b64)
    except Exception as exc:
        raise TimeAnchorError(f"anchor token_b64 is not valid base64: {exc}") from exc
    digest = _digest_bytes(anchor.chain_head_hash, anchor.hash_algorithm)
    return verify_timestamp_token(token_der, digest, hash_algorithm=anchor.hash_algorithm)


def verify_anchor_over_records(anchor: TimeAnchor, record_hashes: list[str]) -> datetime:
    """Verify an anchor binds to an actual record in a chain.

    ``record_hashes`` is the ordered list of ``record_hash`` values from the
    trail. Confirms the record at ``anchor.chain_position`` has exactly
    ``anchor.chain_head_hash``, then verifies the token over it. This proves
    the chain (and every execution record up to that position) existed at the
    attested time. Returns the attested UTC time.
    """
    pos = anchor.chain_position
    if pos < 0 or pos >= len(record_hashes):
        raise TimeAnchorError(
            f"anchor chain_position {pos} is outside the trail "
            f"(len={len(record_hashes)})"
        )
    if record_hashes[pos] != anchor.chain_head_hash:
        raise TimeAnchorError(
            f"anchor chain_position {pos} does not match the trail "
            "(anchor refers to a different or rewritten chain)"
        )
    return verify_anchor(anchor)


__all__ = [
    "TimeAnchor",
    "TimeAnchorError",
    "RFC3161TimeAnchorClient",
    "build_timestamp_request",
    "extract_token_from_response",
    "verify_timestamp_token",
    "verify_anchor",
    "verify_anchor_over_records",
]
