# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Model-identity resolution for the inference proxy.

Resolves a served model name to a ``ModelDerived`` by asking
the inference server (``/api/show`` + ``/api/tags``), cached per model.
Public surface is ``infer_proxy``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("vaara.infer_proxy")

# Chat endpoints the proxy intercepts and signs; everything else passes
# through. ``/v1/chat/completions`` is OpenAI-compatible, ``/api/chat`` is
# ollama-native (Goose's ollama provider).
CHAT_PATHS = frozenset({"/v1/chat/completions", "/api/chat", "/v1/messages"})


def stable_hash(obj: Any) -> str:
    """``sha256:<hex>`` over a stable JSON encoding of ``obj``.

    Plain sorted ``json.dumps`` (not JCS) because this digest is a local
    identity pin we compute ourselves, and the GGUF metadata block can carry
    floats that JCS deliberately rejects. ``default=str`` keeps it from
    throwing on anything exotic.
    """
    encoded = json.dumps(
        obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def fallback_model_derived(model_name: str) -> Any:
    """A ``ModelDerived`` from the name alone, when the server won't tell us.

    The schema requires ``sha256:`` digests, so we derive deterministic
    placeholders rather than block the attestation. A verifier sees a model
    that could not be pinned to weights, which is the honest signal.
    """
    from vaara.attestation._inference_types import ModelDerived

    name_digest = stable_hash({"unresolvedModel": model_name})
    return ModelDerived(
        model_ref=model_name,
        manifest_digest=name_digest,
        gguf_metadata_hash=name_digest,
        quantization="unresolved",
        param_count="unresolved",
    )


class ModelResolver:
    """Resolves a model name to a ``ModelDerived`` via the upstream, cached."""

    def __init__(self, client: Any, upstream: str) -> None:
        self._client = client
        self._upstream = upstream.rstrip("/")
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    async def resolve(self, model_name: str) -> Any:
        with self._lock:
            cached = self._cache.get(model_name)
        if cached is not None:
            return cached
        try:
            derived = await self._resolve_uncached(model_name)
        except Exception:
            logger.warning("Model resolution failed for %r; using fallback", model_name)
            derived = fallback_model_derived(model_name)
        with self._lock:
            self._cache[model_name] = derived
        return derived

    async def _resolve_uncached(self, model_name: str) -> Any:
        from vaara.attestation._inference_types import ModelDerived

        show_resp = await self._client.post(
            f"{self._upstream}/api/show", json={"model": model_name}
        )
        show_resp.raise_for_status()
        show = show_resp.json()
        model_info = show.get("model_info") or {}
        details = show.get("details") or {}
        gguf_hash = stable_hash(model_info) if model_info else stable_hash(details)

        manifest_digest = await self._manifest_digest(model_name)
        if manifest_digest is None:
            # No tag match: pin to the weights metadata we did resolve, so the
            # digest binds the served model rather than just its name.
            manifest_digest = stable_hash(
                {"model": model_name, "ggufMetadataHash": gguf_hash}
            )

        return ModelDerived(
            model_ref=model_name,
            manifest_digest=manifest_digest,
            gguf_metadata_hash=gguf_hash,
            quantization=details.get("quantization_level"),
            param_count=details.get("parameter_size"),
        )

    async def _manifest_digest(self, model_name: str) -> Optional[str]:
        """The ollama manifest digest for ``model_name`` from ``/api/tags``."""
        try:
            tags_resp = await self._client.get(f"{self._upstream}/api/tags")
            tags_resp.raise_for_status()
            models = tags_resp.json().get("models") or []
        except Exception:
            return None
        base = model_name.split(":")[0]
        candidates = {model_name, f"{model_name}:latest"}
        for entry in models:
            name = entry.get("name") or entry.get("model") or ""
            digest = entry.get("digest")
            if not digest:
                continue
            if name in candidates or name.split(":")[0] == base:
                text = str(digest)
                return text if text.startswith("sha256:") else f"sha256:{text}"
        return None
