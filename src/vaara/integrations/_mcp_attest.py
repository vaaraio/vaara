"""SEP-2787 attestation + execution-receipt pairing for the Vaara MCP proxy.

Internal helper. Off unless the operator wires the proxy with a signing key
and receipts directory. When active, every allowed ``tools/call`` writes two
JSON files to the receipts directory:

    {counter:010d}-{nonce[:8]}-attest.json   -- SEP-2787 request attestation
    {counter:010d}-{nonce[:8]}-receipt.json  -- execution receipt via backLink

Signing modes and key auto-detection:

- PEM file with EC P-256 key -> ES256 (recommended)
- PEM file with RSA key -> RS256
- Raw bytes file (no PEM header) -> HS256

``iss`` is always ``"vaara-mcp-proxy"``. ``sub`` is
``"{tenant_id}/{upstream_name}"`` when a tenant is present, else just
``"{upstream_name}"``.

``serverFingerprint`` starts as ``cmd:sha256:{hex}`` (SHA-256 of the upstream
command string, computed at construction). The first ``tools/list`` response
per upstream upgrades it to ``manifest:sha256:{hex}`` (SHA-256 of the
canonical JSON of the effective tools array), binding the exact capability set
the proxy presented to the agent.

Intent defaults to ``tools/call/{tool_name}``. Operators can supply a richer
label via the ``X-Vaara-Intent`` HTTP request header; stdio transport uses the
derived default.

For ES256 / RS256, a ``pubkey.pem`` (SubjectPublicKeyInfo) is written to the
receipts directory so external verifiers need only the public key.

Internal module. Public surface: ``--attest-signing-key`` /
``--attest-receipts-dir`` on ``vaara-mcp-proxy``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ISS = "vaara-mcp-proxy"


class AttestConfigError(RuntimeError):
    """Operator-side attestation config is incomplete or unusable."""


class AttestPairEmitter:
    """Per-proxy SEP-2787 attestation + execution-receipt emitter.

    One instance per proxy process. Owns the signing key, fingerprint
    table, and monotonic counter. Thread-safe.
    """

    def __init__(
        self,
        *,
        signing_key: Any,
        alg: str,
        receipts_dir: Path,
        secret_version: str,
        upstream_commands: dict[str, list[str]],
        exp_seconds: int = 300,
    ) -> None:
        from vaara.attestation._sep2787_types import VALID_ALGS
        if alg not in VALID_ALGS:
            raise AttestConfigError(f"unsupported alg: {alg!r}; use HS256, ES256, or RS256")
        self._signing_key = signing_key
        self._alg = alg
        self._receipts_dir = Path(receipts_dir)
        self._receipts_dir.mkdir(parents=True, exist_ok=True)
        self._secret_version = secret_version
        self._exp_seconds = exp_seconds
        self._counter = 0
        self._lock = threading.Lock()
        # cmd-hash fingerprints pre-computed at construction; upgraded to
        # manifest-hash on first tools/list response per upstream.
        self._fingerprints: dict[str, str] = {
            name: _cmd_hash(cmd) for name, cmd in upstream_commands.items()
        }
        self._write_pubkey_pin()

    @property
    def receipts_dir(self) -> Path:
        return self._receipts_dir

    def update_manifest_fingerprint(
        self, upstream_name: str, tools_list_response: dict
    ) -> None:
        """Upgrade from cmd-hash to manifest-hash on first tools/list response.

        Hashes the canonical JSON of the tools array from the response (which
        may be post-operator-filter, binding the effective capability set the
        proxy presents to agents). Idempotent after the first upgrade.
        """
        with self._lock:
            current = self._fingerprints.get(upstream_name, "")
        if current.startswith("manifest:sha256:"):
            return
        try:
            from vaara.attestation._sep2787_canonical import canonical_json
            result = tools_list_response.get("result") or {}
            tools = result.get("tools") or []
            manifest_bytes = canonical_json({"tools": tools})
            h = hashlib.sha256(manifest_bytes).hexdigest()
            with self._lock:
                self._fingerprints[upstream_name] = f"manifest:sha256:{h}"
            logger.debug(
                "Manifest fingerprint for upstream %r: sha256:...%s", upstream_name, h[-8:]
            )
        except Exception:
            logger.exception(
                "Failed to capture manifest fingerprint for upstream %r", upstream_name
            )

    def fingerprint_for(self, upstream_name: str) -> str:
        """Best available fingerprint: manifest if captured, cmd-hash otherwise."""
        with self._lock:
            return self._fingerprints.get(
                upstream_name, f"cmd:sha256:unknown-{upstream_name}"
            )

    def emit_attestation(
        self,
        *,
        tool_name: str,
        arguments: dict,
        upstream_name: str,
        tenant_id: str,
        intent_override: str = "",
    ) -> "Optional[tuple[Any, int]]":
        """Build, sign, and persist a SEP-2787 attestation.

        Returns ``(Attestation, counter)`` on success, ``None`` on failure.
        Failures are logged and swallowed: attestation must not block traffic.
        The counter is passed to ``emit_receipt`` for paired filenames.
        """
        try:
            from vaara.attestation.sep2787 import emit_attestation as _emit, make_args_digest
            from vaara.attestation._sep2787_types import (
                PayloadDerived, PlannerDeclared, ToolCallBinding,
            )

            intent = intent_override.strip() if intent_override else f"tools/call/{tool_name}"
            sub = f"{tenant_id}/{upstream_name}" if tenant_id else upstream_name
            fingerprint = self.fingerprint_for(upstream_name)
            args_commitment = make_args_digest(arguments)

            planner = PlannerDeclared(intent=intent)
            payload = PayloadDerived(
                tool_calls=(ToolCallBinding(
                    name=tool_name,
                    server_fingerprint=fingerprint,
                    args=args_commitment,
                ),),
            )

            with self._lock:
                self._counter += 1
                counter = self._counter

            attestation = _emit(
                planner_declared=planner,
                payload_derived=payload,
                iss=_ISS,
                sub=sub,
                secret_version=self._secret_version,
                alg=self._alg,
                signing_material=self._signing_key,
                exp_seconds=self._exp_seconds,
            )

            nonce_tag = attestation.issuer_asserted.nonce[:8]
            path = self._receipts_dir / f"{counter:010d}-{nonce_tag}-attest.json"
            path.write_text(json.dumps(attestation.to_dict(), indent=2), encoding="utf-8")
            logger.debug("Attestation %s tool=%r upstream=%r", path.name, tool_name, upstream_name)
            return (attestation, counter)
        except Exception:
            logger.exception("SEP-2787 attestation emission failed for tool=%r", tool_name)
            return None

    def emit_receipt(
        self,
        *,
        attestation: Any,
        counter: int,
        outcome_severity: float,
        upstream_name: str,
        tenant_id: str,
    ) -> None:
        """Build, sign, and persist an execution receipt paired to the attestation.

        ``outcome_severity == 0.0`` maps to ``executed``; anything above maps
        to ``errored``. Failures are logged and swallowed.
        """
        try:
            from vaara.attestation.receipt import emit_receipt as _emit_receipt, make_back_link
            from vaara.attestation._receipt_types import OutcomeDerived
            from vaara.attestation._sep2787_canonical import now_iso8601

            status: str = "errored" if outcome_severity > 0.0 else "executed"
            back_link = make_back_link(attestation)
            outcome = OutcomeDerived(
                status=status,
                completed_at=now_iso8601(),
            )
            sub = f"{tenant_id}/{upstream_name}" if tenant_id else upstream_name

            receipt = _emit_receipt(
                back_link=back_link,
                outcome_derived=outcome,
                iss=_ISS,
                sub=sub,
                secret_version=self._secret_version,
                alg=self._alg,
                signing_material=self._signing_key,
            )

            nonce_tag = attestation.issuer_asserted.nonce[:8]
            path = self._receipts_dir / f"{counter:010d}-{nonce_tag}-receipt.json"
            path.write_text(json.dumps(receipt.to_dict(), indent=2), encoding="utf-8")
            logger.debug("Receipt %s status=%s", path.name, status)
        except Exception:
            logger.exception("Execution receipt emission failed")

    def _write_pubkey_pin(self) -> None:
        if self._alg == "HS256":
            return
        try:
            from cryptography.hazmat.primitives import serialization
            pub = self._signing_key.public_key()
            pem_bytes = pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            (self._receipts_dir / "pubkey.pem").write_bytes(pem_bytes)
        except Exception:
            logger.warning("Failed to write pubkey.pem to receipts directory")


def _cmd_hash(command: list[str]) -> str:
    cmd_str = " ".join(command)
    return "cmd:sha256:" + hashlib.sha256(cmd_str.encode("utf-8")).hexdigest()


def build_attest_emitter(
    *,
    signing_key_path: Path,
    receipts_dir: Path,
    upstream_commands: dict[str, list[str]],
    secret_version: Optional[str] = None,
    exp_seconds: int = 300,
) -> AttestPairEmitter:
    """Load signing key from path and return an ``AttestPairEmitter``.

    Key type auto-detection:
    - PEM file (EC P-256) -> ES256
    - PEM file (RSA) -> RS256
    - Raw bytes file (no PEM header) -> HS256

    ``secret_version`` defaults to the first 8 hex chars of the SHA-256 of
    the public-key DER (PEM keys) or raw bytes (HS256), so key rotation is
    automatically reflected.

    Raises ``AttestConfigError`` if the key is missing, unusable, or if the
    ``attestation`` extra is not installed.
    """
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
    except ImportError as exc:
        raise AttestConfigError(
            "vaara-mcp-proxy --attest-* flags require the attestation extra. "
            "Install with: pip install 'vaara[attestation]'"
        ) from exc

    key_path = Path(signing_key_path).expanduser()
    if not key_path.is_file():
        raise AttestConfigError(f"--attest-signing-key file not found: {key_path}")

    raw = key_path.read_bytes()

    if raw.lstrip().startswith(b"-----BEGIN"):
        try:
            key = serialization.load_pem_private_key(raw, password=None)
        except Exception as exc:
            raise AttestConfigError(
                f"--attest-signing-key is not a usable PEM private key: {exc}"
            ) from exc
        if isinstance(key, EllipticCurvePrivateKey):
            alg: str = "ES256"
        elif isinstance(key, RSAPrivateKey):
            alg = "RS256"
        else:
            raise AttestConfigError(
                "--attest-signing-key must be EC P-256 (ES256) or RSA (RS256). "
                "Generate: openssl ecparam -genkey -name prime256v1 | "
                "openssl pkcs8 -topk8 -nocrypt -out attest_key.pem"
            )
        signing_material: Any = key
        if secret_version is None:
            pub_der = key.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            secret_version = hashlib.sha256(pub_der).hexdigest()[:8]
    else:
        if len(raw) < 16:
            raise AttestConfigError(
                f"--attest-signing-key raw bytes must be at least 16 bytes; got {len(raw)}"
            )
        alg = "HS256"
        signing_material = raw
        if secret_version is None:
            secret_version = hashlib.sha256(raw).hexdigest()[:8]

    return AttestPairEmitter(
        signing_key=signing_material,
        alg=alg,
        receipts_dir=Path(receipts_dir).expanduser(),
        secret_version=secret_version,
        upstream_commands=upstream_commands,
        exp_seconds=exp_seconds,
    )
