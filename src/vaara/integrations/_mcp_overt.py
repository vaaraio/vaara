"""OVERT 1.0 Base Envelope emission for the Vaara MCP proxy.

Internal helper that turns each governed MCP interaction (``tools/call``,
``resources/read``, ``prompts/get`` — both allowed and perimeter-blocked)
into a signed Protocol Profile 1.0 Annex B.6 Base Envelope written to a
receipts directory. ``vaara overt verify`` consumes those files one by one.

Off unless the operator wires the proxy with a signing key, operator HMAC
key, and receipts directory. With none of those set, the proxy behaves as
it did pre-v0.24.0 — no envelopes, no extra crypto, no extra disk writes.

The receipts directory layout, per envelope:
    {nanosecond_timestamp}-{counter:010d}.cbor
plus a one-time ``pubkey.bin`` (32 raw Ed25519 public-key bytes) so the
auditor knows which key to verify against.

Internal module. Public surface is the ``--overt-*`` flags on ``vaara-mcp-proxy``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from vaara import __version__ as _VAARA_VERSION

logger = logging.getLogger(__name__)


class OVERTConfigError(RuntimeError):
    """Operator-side OVERT configuration is incomplete or unusable."""


class OVERTReceiptEmitter:
    """Per-proxy OVERT 1.0 Base Envelope emitter.

    One instance per proxy process. Owns the signing key, operator HMAC
    key, monotonic counter, and arbiter-instance identifier. Thread-safe:
    emission acquires a lock around counter increment and file write.
    """

    def __init__(
        self,
        *,
        signing_key: Any,
        operator_key: bytes,
        receipts_dir: Path,
        policy_hash: bytes,
        arbiter_version: str = f"vaara/{_VAARA_VERSION}",
    ) -> None:
        from vaara.attestation.overt import encoder_binary_identity

        self._signing_key = signing_key
        self._operator_key = bytes(operator_key)
        self._receipts_dir = Path(receipts_dir)
        self._receipts_dir.mkdir(parents=True, exist_ok=True)
        self._encoder_identity = encoder_binary_identity(
            arbiter_version=arbiter_version, policy_hash=policy_hash,
        )
        self._arbiter_instance = uuid.uuid4().bytes
        self._counter = 0
        self._lock = threading.Lock()
        self._write_pubkey_pin()

    @property
    def receipts_dir(self) -> Path:
        return self._receipts_dir

    @property
    def arbiter_instance_identifier(self) -> bytes:
        return self._arbiter_instance

    def emit(self, *, request_payload: bytes, non_content_metadata: dict) -> Any:
        """Build, sign, and persist one Base Envelope.

        Returns the BaseEnvelope. The on-disk artifact is a canonical-CBOR
        file inside ``receipts_dir`` named for its timestamp and counter.
        """
        from vaara.attestation import envelope_to_canonical_cbor
        from vaara.attestation.overt import (
            emit_base_envelope,
            make_request_commitment,
        )

        commitment = make_request_commitment(
            request_payload, operator_key=self._operator_key,
        )
        with self._lock:
            self._counter += 1
            counter = self._counter
            envelope = emit_base_envelope(
                signing_key=self._signing_key,
                request_commitment=commitment,
                encoder_binary_identity=self._encoder_identity,
                non_content_metadata=non_content_metadata,
                monotonic_counter=counter,
                arbiter_instance_identifier=self._arbiter_instance,
            )
            cbor_bytes = envelope_to_canonical_cbor(envelope)
            filename = (
                f"{envelope.nanosecond_timestamp}-"
                f"{envelope.monotonic_counter:010d}.cbor"
            )
            (self._receipts_dir / filename).write_bytes(cbor_bytes)
        return envelope

    def _write_pubkey_pin(self) -> None:
        from cryptography.hazmat.primitives import serialization
        pub = self._signing_key.public_key()
        try:
            pub_raw = pub.public_bytes_raw()
        except AttributeError:
            pub_raw = pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        (self._receipts_dir / "pubkey.bin").write_bytes(pub_raw)


def policy_hash_from_perimeter(
    *,
    tool_allow: Optional[set[str]],
    tool_deny: set[str],
    resource_allow: Optional[set[str]],
    resource_deny: set[str],
    prompt_allow: Optional[set[str]],
    prompt_deny: set[str],
    policy_source: Optional[bytes] = None,
) -> bytes:
    """SHA-256 over canonical JSON of the operator perimeter configuration.

    Stable across processes given identical perimeter config. The hash
    flows into ``encoder_binary_identity`` so a regulator can detect any
    silent perimeter change between two envelope batches signed by the
    same arbiter instance.
    """
    config = {
        "tool_allow": sorted(tool_allow) if tool_allow else None,
        "tool_deny": sorted(tool_deny),
        "resource_allow": sorted(resource_allow) if resource_allow else None,
        "resource_deny": sorted(resource_deny),
        "prompt_allow": sorted(prompt_allow) if prompt_allow else None,
        "prompt_deny": sorted(prompt_deny),
    }
    if policy_source is not None:
        # A --policy file governs decisions, so it must be covered by the
        # attested hash. The key is added only when a policy is present, so
        # perimeter-only deployments keep their historical hash.
        config["policy_sha256"] = hashlib.sha256(policy_source).hexdigest()
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).digest()


def build_emitter(
    *,
    signing_key_path: Path,
    operator_key_path: Optional[Path],
    operator_key_hex: Optional[str],
    receipts_dir: Path,
    policy_hash: bytes,
) -> OVERTReceiptEmitter:
    """Load the Ed25519 PEM private key and HMAC operator key, return an emitter.

    Raises OVERTConfigError if either key is missing or unusable, or if
    the attestation extra is not installed.
    """
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:
        raise OVERTConfigError(
            "vaara-mcp-proxy --overt-* flags require the attestation extra. "
            "Install with: pip install 'vaara[attestation]'"
        ) from exc

    signing_key_path = Path(signing_key_path).expanduser()
    if not signing_key_path.is_file():
        raise OVERTConfigError(
            f"--overt-signing-key file not found: {signing_key_path}"
        )
    try:
        key = serialization.load_pem_private_key(
            signing_key_path.read_bytes(), password=None,
        )
    except Exception as exc:
        raise OVERTConfigError(
            f"--overt-signing-key is not a usable PEM-encoded private key: {exc}"
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise OVERTConfigError(
            "--overt-signing-key must be an Ed25519 private key "
            "(vaara keygen --dev produces one)"
        )

    operator_key = _load_operator_key(operator_key_path, operator_key_hex)

    return OVERTReceiptEmitter(
        signing_key=key,
        operator_key=operator_key,
        receipts_dir=Path(receipts_dir).expanduser(),
        policy_hash=policy_hash,
    )


def _load_operator_key(
    path: Optional[Path], hex_value: Optional[str],
) -> bytes:
    if path is not None:
        p = Path(path).expanduser()
        if not p.is_file():
            raise OVERTConfigError(
                f"--overt-operator-key file not found: {p}"
            )
        data = p.read_bytes()
    elif hex_value is not None:
        try:
            data = bytes.fromhex(hex_value.strip())
        except ValueError as exc:
            raise OVERTConfigError(
                f"VAARA_OVERT_OPERATOR_KEY_HEX is not valid hex: {exc}"
            ) from exc
    else:
        raise OVERTConfigError(
            "OVERT emission requires an operator HMAC key. Pass "
            "--overt-operator-key PATH (raw key bytes) or set "
            "VAARA_OVERT_OPERATOR_KEY_HEX in the environment."
        )
    if len(data) < 16:
        raise OVERTConfigError(
            f"operator HMAC key must be at least 16 bytes; got {len(data)}"
        )
    return data
