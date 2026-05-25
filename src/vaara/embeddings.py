"""Sentence-transformer embeddings for the adversarial classifier (v0.32+).

Loads ``sentence-transformers/all-MiniLM-L6-v2`` lazily on first call and
caches it as a module-level singleton. 384-dim L2-normalized float32 vectors.

Used by ``scripts/train_adversarial_classifier.py`` at training time and by
``vaara.adversarial_classifier`` at inference time to concatenate semantic
features after the 236 hand-features.

Requires sentence-transformers + torch. Both ship with ``pip install vaara[ml]``
once the v0.32 extras are declared.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "embeddings require sentence-transformers: pip install vaara[ml]"
            ) from exc
        _model = SentenceTransformer(EMBED_MODEL_ID, device="cpu")
    return _model


def embed(text: str) -> np.ndarray:
    """Encode a single string to a 384-dim L2-normalized float32 vector."""
    return _get_model().encode(
        text, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)


def embed_batch(texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
    """Encode many strings; returns (N, 384) float32 array, row-normalized."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return _get_model().encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


__all__ = ["embed", "embed_batch", "EMBED_DIM", "EMBED_MODEL_ID"]
