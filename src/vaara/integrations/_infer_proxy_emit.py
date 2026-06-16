# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Inference attestation/receipt emitter for the inference proxy.

Mirrors ``_mcp_attest.AttestPairEmitter`` one layer down:
signs an ``InferenceAttestation`` + ``InferenceReceipt`` pair per chat call.
Public surface is ``infer_proxy``.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from vaara.integrations._infer_proxy_sign import InferProxyConfigError, _ISS

logger = logging.getLogger("vaara.infer_proxy")


class InferenceAttestEmitter:
    """Per-proxy InferenceAttestation + InferenceReceipt emitter.

    One instance per process. Owns the signing key and a monotonic counter.
    Thread-safe. Emission failures are logged and swallowed: signing must
    never block inference traffic.
    """

    def __init__(
        self,
        *,
        signing_key: Any,
        alg: str,
        receipts_dir: Path,
        secret_version: str,
        exp_seconds: int = 300,
    ) -> None:
        from vaara.attestation._sep2787_types import VALID_ALGS

        if alg not in VALID_ALGS:
            raise InferProxyConfigError(
                f"unsupported alg: {alg!r}; use HS256, ES256, or RS256"
            )
        self._signing_key = signing_key
        self._alg = alg
        self._receipts_dir = Path(receipts_dir)
        self._receipts_dir.mkdir(parents=True, exist_ok=True)
        self._secret_version = secret_version
        self._exp_seconds = exp_seconds
        self._counter = 0
        self._lock = threading.Lock()
        self._write_pubkey_pin()

    @property
    def receipts_dir(self) -> Path:
        return self._receipts_dir

    def emit_attestation(
        self, *, model_ref: str, model_derived: Any, messages: Any,
        sampling: dict[str, Any],
    ) -> "Optional[tuple[Any, int]]":
        """Build, sign, and persist an InferenceAttestation.

        Returns ``(attestation, counter)`` or ``None`` on failure. The counter
        is reused for the paired receipt filename.
        """
        try:
            from vaara.attestation._inference_types import RequestDeclared
            from vaara.attestation.inference import (
                emit_inference_attestation, make_request_commitment,
            )

            rd = RequestDeclared(
                intent=f"inference/chat/{model_ref}",
                request_commitment=make_request_commitment(
                    messages=messages, sampling=sampling
                ),
            )
            with self._lock:
                self._counter += 1
                counter = self._counter

            attestation = emit_inference_attestation(
                request_declared=rd, model_derived=model_derived, iss=_ISS,
                sub=model_ref, secret_version=self._secret_version, alg=self._alg,
                signing_material=self._signing_key, exp_seconds=self._exp_seconds,
            )
            nonce_tag = attestation.issuer_asserted.nonce[:8]
            path = self._receipts_dir / f"{counter:010d}-{nonce_tag}-infer-attest.json"
            path.write_text(
                json.dumps(attestation.to_dict(), indent=2), encoding="utf-8"
            )
            logger.debug("InferenceAttestation %s model=%r", path.name, model_ref)
            return (attestation, counter)
        except Exception:
            logger.exception(
                "Inference attestation emission failed (model=%r)", model_ref
            )
            return None

    def emit_receipt(
        self, *, attestation: Any, counter: int, status: str, output: Any,
        eval_stats: Optional[dict[str, int]], tier: str = "integrity",
    ) -> None:
        """Build, sign, and persist the back-linked InferenceReceipt."""
        try:
            from vaara.attestation._inference_types import InferenceOutcome
            from vaara.attestation._sep2787_canonical import now_iso8601
            from vaara.attestation.inference import (
                emit_inference_receipt, make_inference_back_link,
                make_output_commitment,
            )

            back_link = make_inference_back_link(attestation)
            output_commitment = (
                make_output_commitment(output)
                if output is not None and status != "refused"
                else None
            )
            outcome = InferenceOutcome(
                status=status, completed_at=now_iso8601(), tier=tier,
                output_commitment=output_commitment, eval_stats=(eval_stats or None),
            )
            receipt = emit_inference_receipt(
                back_link=back_link, outcome_derived=outcome, iss=_ISS,
                sub=attestation.issuer_asserted.sub,
                secret_version=self._secret_version, alg=self._alg,
                signing_material=self._signing_key,
            )
            nonce_tag = attestation.issuer_asserted.nonce[:8]
            path = self._receipts_dir / f"{counter:010d}-{nonce_tag}-infer-receipt.json"
            path.write_text(json.dumps(receipt.to_dict(), indent=2), encoding="utf-8")
            logger.debug("InferenceReceipt %s status=%s", path.name, status)
        except Exception:
            logger.exception("Inference receipt emission failed")

    def _write_pubkey_pin(self) -> None:
        if self._alg == "HS256":
            return
        try:
            from cryptography.hazmat.primitives import serialization

            pub = self._signing_key.public_key()
            pem = pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            (self._receipts_dir / "pubkey.pem").write_bytes(pem)
        except Exception:
            logger.warning("Failed to write pubkey.pem to receipts directory")
