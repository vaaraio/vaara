# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sigstore Rekor backing for transparency-log anchoring.

``transparency_log.InProcessTransparencyLog`` keeps the protocol demonstrable
with no external dependency, and its docstring has always said a Rekor-backed
adapter could drop into the same call site. This is that adapter.

Why it matters: a log the emitter operates proves the chain is internally
consistent. It cannot prove the emitter kept one history. Publishing the trail
head to a log somebody else runs fixes the shape of the trail at a moment the
emitter does not control, which is the property a relying party actually wants
and the one an in-process log cannot offer at any level of cryptographic care.

What leaves the machine is one digest and a signature over it. No record, no
payload, no count of entries, no identifiers.

Rekor is chosen over a blockchain anchor deliberately: it is the witness class
named in Vaara's anchoring model (RFC 3161 / eIDAS plus Rekor or SCITT), it is
append-only and publicly auditable, and a ``hashedrekord`` entry needs no
account, no OIDC and no key ceremony. The signing key is generated locally on
first use and its public half travels with every entry, so a verifier can tie
all published heads to one signer without any identity provider.

Offline verification is unaffected. A Rekor entry is corroboration a verifier
MAY fetch, never a dependency: ``verify_publication`` is the only function here
that touches the network, and nothing in the receipt path calls it.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

DEFAULT_REKOR = "https://rekor.sigstore.dev"
_TIMEOUT = 30.0


class RekorError(RuntimeError):
    """Raised when a Rekor publication or verification cannot be completed."""


@dataclass(frozen=True)
class RekorPublication:
    """A trail head recorded in a public append-only log.

    ``uuid`` and ``log_index`` locate the entry. ``digest`` is the value the
    log attests, and ``integrated_time`` is when the log incorporated it,
    which is the number a relying party cares about: the head cannot have
    been constructed after it.
    """

    digest: str
    uuid: str
    log_index: int
    integrated_time: int
    log_url: str

    @property
    def integrated_at(self) -> datetime:
        return datetime.fromtimestamp(self.integrated_time, timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "rekor",
            "digest": self.digest,
            "uuid": self.uuid,
            "log_index": self.log_index,
            "integrated_time": self.integrated_time,
            "log_url": self.log_url,
            "entry_url": f"{self.log_url}/api/v1/log/entries/{self.uuid}",
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RekorPublication":
        try:
            return cls(
                digest=d["digest"],
                uuid=d["uuid"],
                log_index=int(d["log_index"]),
                integrated_time=int(d["integrated_time"]),
                log_url=d.get("log_url", DEFAULT_REKOR),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RekorError(f"malformed rekor publication: {exc}") from exc


def _post(log_url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        log_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:300]
        raise RekorError(f"rekor rejected the entry ({exc.code}): {detail}") from exc
    except Exception as exc:
        raise RekorError(f"rekor unreachable at {log_url}: {exc}") from exc


def _get(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        raise RekorError(f"could not fetch {url}: {exc}") from exc


def publish_head(
    digest_hex: str,
    signer: Any,
    *,
    log_url: str = DEFAULT_REKOR,
) -> RekorPublication:
    """Record ``digest_hex`` in a public log, signed by ``signer``.

    ``signer`` is an ``EllipticCurvePrivateKey``. Rekor's ``hashedrekord``
    treats the artifact hash as a pre-hashed digest, so the signature is over
    the raw digest bytes rather than over a re-hash of them.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils

    try:
        digest = bytes.fromhex(digest_hex)
    except ValueError as exc:
        raise RekorError(f"digest is not hex: {digest_hex!r}") from exc
    if len(digest) != 32:
        raise RekorError(f"digest is {len(digest)} bytes, not a sha256")

    signature = signer.sign(digest, ec.ECDSA(asym_utils.Prehashed(hashes.SHA256())))
    pub_pem = signer.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    body = {
        "apiVersion": "0.0.1",
        "kind": "hashedrekord",
        "spec": {
            "data": {"hash": {"algorithm": "sha256", "value": digest_hex}},
            "signature": {
                "content": base64.b64encode(signature).decode(),
                "publicKey": {"content": base64.b64encode(pub_pem).decode()},
            },
        },
    }
    result = _post(log_url, "/api/v1/log/entries", body)
    uuid = next(iter(result))
    entry = result[uuid]
    return RekorPublication(
        digest=digest_hex,
        uuid=uuid,
        log_index=int(entry["logIndex"]),
        integrated_time=int(entry["integratedTime"]),
        log_url=log_url,
    )


def verify_publication(
    publication: RekorPublication,
    *,
    expected_digest: Optional[str] = None,
) -> datetime:
    """Re-fetch the entry and confirm the log holds the digest claimed.

    Returns the integrated time as recorded by the log, which may differ from
    the value stored locally if the stored copy was edited. The log's answer
    wins; that is the entire point of publishing there.
    """
    url = f"{publication.log_url}/api/v1/log/entries/{publication.uuid}"
    fetched = _get(url)
    entry = fetched.get(publication.uuid)
    if not entry:
        raise RekorError(f"log has no entry {publication.uuid}")
    try:
        decoded = json.loads(base64.b64decode(entry["body"]).decode())
        logged = decoded["spec"]["data"]["hash"]["value"]
    except Exception as exc:
        raise RekorError(f"entry body is not a readable hashedrekord: {exc}") from exc

    if logged != publication.digest:
        raise RekorError(
            f"log holds {logged}, publication claims {publication.digest}"
        )
    if expected_digest is not None and logged != expected_digest:
        raise RekorError(
            f"log holds {logged}, caller expected {expected_digest}"
        )
    return datetime.fromtimestamp(int(entry["integratedTime"]), timezone.utc)
