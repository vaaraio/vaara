"""Self-hosted RFC 3161 timestamp anchors for Vaara receipts (SPEC.md Section 4).

A receipt's signature proves who decided and what it bound to, not *when*. A
``timestampAnchors`` entry adds a time attestation over the receipt from an
authority separate from the signing key. This is the ``rfc3161`` method's
*producer*: it stands up a local Time-Stamping Authority (EC keypair +
self-signed cert with the timeStamping EKU) and issues RFC 3161 tokens over a
receipt's signed-payload digest, fully offline, no third party. The
``rfc3161-eidas-qualified`` method is the same token from a qualified TSA and
adds only legal weight; it is swappable.

Tokens verify offline via ``vaara.audit.timeanchor.verify_timestamp_token`` so
producer and verifier share one trust check. ``verify_receipt_anchor`` also pins
anchoredDigest == sha256 of the JCS signed payload (SPEC.md Section 6 rule 3).
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import rfc8785
from asn1crypto import algos, cms, core, tsp
from asn1crypto import x509 as asn1_x509
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from vaara.audit.timeanchor import (
    TimeAnchorError,
    Transport,
    _urllib_transport,
    build_timestamp_request,
    extract_token_from_response,
    verify_timestamp_token,
)

# Signed payload (SPEC.md 2.1): signature + timestampAnchors excluded, so a
# receipt can gain anchors after signing without invalidating it.
_SIGNED_BLOCKS = ("version", "alg", "backLink", "decisionDerived",
                  "issuerAsserted")
# Unregistered policy OID: marks "self-hosted, not qualified" by construction.
_TSA_POLICY_OID = "1.3.6.1.4.1.58530.3161.1"


def _signed_payload_digest(receipt: dict) -> bytes:
    try:
        payload = {k: receipt[k] for k in _SIGNED_BLOCKS}
    except KeyError as exc:
        raise TimeAnchorError(f"receipt missing signed-payload block: {exc}") from exc
    return hashlib.sha256(rfc8785.dumps(payload)).digest()


def anchored_digest(receipt: dict) -> str:
    """The ``anchoredDigest`` a conforming anchor must carry for this receipt."""
    return "sha256:" + _signed_payload_digest(receipt).hex()


class SelfHostedTSA:
    """A local RFC 3161 TSA: EC P-256 key + self-signed timeStamping cert."""

    def __init__(self, key: ec.EllipticCurvePrivateKey, cert: x509.Certificate) -> None:
        self._key = key
        self._cert = cert
        self._cert_asn1 = asn1_x509.Certificate.load(
            cert.public_bytes(serialization.Encoding.DER))

    @property
    def authority(self) -> str:
        cn = self._cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        return cn[0].value if cn else "vaara-self-hosted-tsa"

    @classmethod
    def create(cls, common_name: str = "vaara-self-hosted-tsa",
               *, serial: int = 1) -> "SelfHostedTSA":
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        # ponytail: the token's gen_time carries the attested instant; the wide
        # cert window just keeps the authority from expiring under old receipts.
        cert = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(serial)
            .not_valid_before(datetime(2020, 1, 1, tzinfo=timezone.utc))
            .not_valid_after(datetime(2100, 1, 1, tzinfo=timezone.utc))
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]),
                critical=True)
            .sign(key, hashes.SHA256())
        )
        return cls(key, cert)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tsa_key.pem").write_bytes(self._key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
        (directory / "tsa_cert.pem").write_bytes(
            self._cert.public_bytes(serialization.Encoding.PEM))

    @classmethod
    def load(cls, directory: Path) -> "SelfHostedTSA":
        key = serialization.load_pem_private_key(
            (directory / "tsa_key.pem").read_bytes(), password=None)
        cert = x509.load_pem_x509_certificate(
            (directory / "tsa_cert.pem").read_bytes())
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise TimeAnchorError("TSA key is not an EC private key")
        return cls(key, cert)

    @classmethod
    def load_or_create(cls, directory: Path,
                       common_name: str = "vaara-self-hosted-tsa") -> "SelfHostedTSA":
        if (directory / "tsa_key.pem").exists():
            return cls.load(directory)
        tsa = cls.create(common_name)
        tsa.save(directory)
        return tsa

    def issue_token(self, digest: bytes, *, serial: int = 1,
                    gen_time: Optional[datetime] = None) -> bytes:
        """Issue a DER ``TimeStampToken`` over a 32-byte sha256 ``digest``."""
        if len(digest) != 32:
            raise TimeAnchorError(
                f"expected a 32-byte sha256 digest, got {len(digest)} bytes")
        tst_info = tsp.TSTInfo({
            "version": "v1",
            "policy": _TSA_POLICY_OID,
            "message_imprint": tsp.MessageImprint({
                "hash_algorithm": algos.DigestAlgorithm({"algorithm": "sha256"}),
                "hashed_message": digest,
            }),
            "serial_number": serial,
            "gen_time": gen_time or datetime.now(timezone.utc),
        })
        tst_der = tst_info.dump()
        signed_attrs = cms.CMSAttributes([
            cms.CMSAttribute({"type": "content_type", "values": ["tst_info"]}),
            cms.CMSAttribute({"type": "message_digest",
                              "values": [hashlib.sha256(tst_der).digest()]}),
        ])
        # Signature covers the attrs as a SET OF; the verifier rebuilds the same
        # bytes via signed_attrs.untag().dump().
        signature = self._key.sign(signed_attrs.dump(), ec.ECDSA(hashes.SHA256()))
        tbs = self._cert_asn1["tbs_certificate"]
        signer_info = cms.SignerInfo({
            "version": "v1",
            "sid": cms.SignerIdentifier({
                "issuer_and_serial_number": cms.IssuerAndSerialNumber({
                    "issuer": tbs["issuer"], "serial_number": tbs["serial_number"],
                })}),
            "digest_algorithm": algos.DigestAlgorithm({"algorithm": "sha256"}),
            "signed_attrs": signed_attrs,
            "signature_algorithm": algos.SignedDigestAlgorithm(
                {"algorithm": "sha256_ecdsa"}),
            "signature": signature,
        })
        signed_data = cms.SignedData({
            "version": "v3",
            "digest_algorithms": cms.DigestAlgorithms(
                [algos.DigestAlgorithm({"algorithm": "sha256"})]),
            "encap_content_info": cms.EncapsulatedContentInfo({
                "content_type": "tst_info",
                "content": core.ParsableOctetString(tst_der)}),
            "certificates": cms.CertificateSet(
                [cms.CertificateChoices({"certificate": self._cert_asn1})]),
            "signer_infos": cms.SignerInfos([signer_info]),
        })
        return cms.ContentInfo(
            {"content_type": "signed_data", "content": signed_data}).dump()

    def anchor_receipt(self, receipt: dict) -> dict[str, Any]:
        """Produce a SPEC.md Section 4 ``timestampAnchors`` entry for ``receipt``."""
        raw = _signed_payload_digest(receipt)
        return {
            "method": "rfc3161",
            "anchoredDigest": "sha256:" + raw.hex(),
            "token": base64.b64encode(self.issue_token(raw)).decode("ascii"),
            "authority": self.authority,
        }


class QualifiedTSA:
    """Anchors receipts to a remote (eIDAS-qualified) RFC 3161 TSA, cert-pinned.

    Produces ``rfc3161-eidas-qualified`` timestampAnchors entries: the same
    token shape as :class:`SelfHostedTSA` but issued by a QTSP the operator
    does not control, so the attested time carries eIDAS Article 41's
    presumption of accuracy. The pin is mandatory — a "qualified" anchor whose
    signer is whoever answered the URL would be a self-asserted claim; the
    token is verified against ``trusted_signer_cert`` before it is recorded.

    ``trusted_signer_cert`` pins the QTSP's TSU signing certificate exactly;
    ``trusted_issuer_cert`` pins the CA that issued it, which is the stable
    trusted-list entry and survives signer rotation. Both are PEM or DER,
    obtained out of band and checkable against the EU trusted list; at least
    one is required. ``authority`` defaults to the token signer's subject CN
    (issuer pin) or the pinned certificate's subject CN (signer pin).
    """

    def __init__(
        self,
        tsa_url: str,
        *,
        trusted_signer_cert: Optional[bytes] = None,
        trusted_issuer_cert: Optional[bytes] = None,
        authority: Optional[str] = None,
        timeout: float = 10.0,
        transport: Optional[Transport] = None,
    ) -> None:
        if not trusted_signer_cert and not trusted_issuer_cert:
            raise TimeAnchorError(
                "a qualified anchor requires a certificate to pin "
                "(trusted_signer_cert or trusted_issuer_cert); without one "
                "the 'qualified' claim is self-asserted")
        if authority is None and trusted_signer_cert:
            try:
                cert = x509.load_pem_x509_certificate(trusted_signer_cert)
            except ValueError:
                cert = x509.load_der_x509_certificate(trusted_signer_cert)
            cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            authority = cn[0].value if cn else tsa_url  # type: ignore[assignment]
        self.tsa_url = tsa_url
        self.authority = authority
        self.timeout = timeout
        self._pin = trusted_signer_cert
        self._issuer_pin = trusted_issuer_cert
        self._transport = transport or _urllib_transport

    def anchor_receipt(self, receipt: dict) -> dict[str, Any]:
        """Fetch, pin-verify, and return a qualified anchor for ``receipt``."""
        raw = _signed_payload_digest(receipt)
        request = build_timestamp_request(raw)
        try:
            response = self._transport(self.tsa_url, request, self.timeout)
        except Exception as exc:  # network, TLS, HTTP error
            raise TimeAnchorError(
                f"QTSP request to {self.tsa_url} failed: {exc}") from exc
        token = extract_token_from_response(response)
        # Refuse before recording: the anchor must never carry a token whose
        # signer fails the pin (exact signer certificate and/or issuing CA).
        verify_timestamp_token(token, raw, trusted_signer_cert=self._pin,
                               trusted_issuer_cert=self._issuer_pin)
        authority = self.authority or self._token_signer_cn(token)
        return {
            "method": "rfc3161-eidas-qualified",
            "anchoredDigest": "sha256:" + raw.hex(),
            "token": base64.b64encode(token).decode("ascii"),
            "authority": authority,
            "tsaUrl": self.tsa_url,
        }

    def _token_signer_cn(self, token: bytes) -> str:
        from asn1crypto import cms as _cms

        from vaara.audit.timeanchor import _signer_cert

        cert = _signer_cert(_cms.ContentInfo.load(token)["content"])
        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        return cn[0].value if cn else self.tsa_url


def verify_receipt_anchor(
    receipt: dict, anchor: dict, *,
    trusted_signer_cert: Optional[bytes] = None,
    trusted_issuer_cert: Optional[bytes] = None,
) -> datetime:
    """Verify a ``timestampAnchors`` entry against its receipt. Returns UTC time.

    Pins anchoredDigest == sha256 of the JCS signed payload (rule 3), then
    verifies the token attests that digest via the shared offline verifier.

    Without ``trusted_signer_cert`` the returned time is self-asserted by
    whoever signed the token (the ``method`` string, including
    ``rfc3161-eidas-qualified``, is not proof on its own). Pass the TSA
    certificate you hold out of band to require the token's signer to be that
    exact certificate before the time is treated as independently anchored.
    """
    if anchor.get("method") not in ("rfc3161", "rfc3161-eidas-qualified"):
        raise TimeAnchorError(f"not an rfc3161 anchor: method={anchor.get('method')!r}")
    expected = "sha256:" + _signed_payload_digest(receipt).hex()
    if anchor.get("anchoredDigest") != expected:
        raise TimeAnchorError(
            "anchoredDigest does not match the receipt's signed payload")
    try:
        token_der = base64.b64decode(anchor["token"])
    except Exception as exc:
        raise TimeAnchorError(f"anchor token is not valid base64: {exc}") from exc
    return verify_timestamp_token(
        token_der, bytes.fromhex(expected.split(":", 1)[1]),
        hash_algorithm="sha256", trusted_signer_cert=trusted_signer_cert,
        trusted_issuer_cert=trusted_issuer_cert)


__all__ = ["QualifiedTSA", "SelfHostedTSA", "anchored_digest",
           "verify_receipt_anchor"]
