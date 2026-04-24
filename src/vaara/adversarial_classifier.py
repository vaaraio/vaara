"""Adversarial tool-call classifier — opt-in ML scorer for v0.5.0.

Ships with a pre-trained XGBoost model (~300 KB bundle) trained on:
    - 200 hand-crafted adversarial seeds across 8 attack categories
    - 1945 LLM-generated variants on MI300X (Qwen2.5-3B)
    - 487 benign variants for class balance

Held-out test (by-seed, no leakage): 61% attack recall, 13% benign FPR at threshold 0.8.
Compare to shipped heuristic: 44% recall, 25% FPR.

Usage:
    pip install vaara[ml]
    from vaara.adversarial_classifier import AdversarialClassifier
    clf = AdversarialClassifier()
    prob = clf.score(tool_name="...", parameters={...}, context={...})
    if prob >= clf.threshold:
        # block or escalate
        ...

Requires xgboost, scikit-learn, joblib, numpy. All stay out of the default install.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_URL_SCHEMES = ("http", "https", "file", "dict", "gopher", "data", "ftp")
_IP_PATTERNS = [
    ("meta_169", re.compile(r"169\.254\.169\.254")),
    ("meta_gcp", re.compile(r"metadata\.google\.internal")),
    ("meta_ecs", re.compile(r"169\.254\.170\.2")),
    ("loopback_v4", re.compile(r"\b127\.0\.0\.1\b|\blocalhost\b")),
    ("loopback_v6", re.compile(r"\[::1\]|\bfd00::\b")),
    ("rfc1918_10", re.compile(r"\b10\.\d+\.\d+\.\d+\b")),
    ("rfc1918_192", re.compile(r"\b192\.168\.\d+\.\d+\b")),
    ("internal_tld", re.compile(r"\.(local|internal|corp|intranet)\b", re.I)),
]
_CRED_PATTERNS = [
    ("secret_word", re.compile(r"\b(secret|password|credentials|token|api_key)\b", re.I)),
    ("env_var", re.compile(r"(AWS_|GITHUB_|STRIPE_|SLACK_|DB_|JWT_)[A-Z_]+")),
    ("ssh_path", re.compile(r"\.ssh(/|$)")),
    ("aws_creds", re.compile(r"\.aws/credentials|AKIA[A-Z0-9]{16}")),
    ("shadow_etc", re.compile(r"/etc/(shadow|passwd|sudoers)")),
    ("proc_environ", re.compile(r"/proc/.*/environ")),
]
_SQL_PATTERNS = [
    ("sql_drop", re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", re.I)),
    ("sql_truncate", re.compile(r"\bTRUNCATE\b", re.I)),
    ("sql_delete_all", re.compile(r"\bDELETE\s+FROM\s+\w+\s*(;|$)", re.I)),
    ("sql_union", re.compile(r"\bUNION\s+SELECT\b", re.I)),
    ("sql_update_all", re.compile(r"\bUPDATE\s+\w+\s+SET\b(?!.*WHERE)", re.I | re.S)),
]
_SHELL_PATTERNS = [
    ("shell_rm", re.compile(r"\brm\s+-rf\b")),
    ("shell_dd", re.compile(r"\bdd\s+if=")),
    ("shell_pipe_sh", re.compile(r"\|\s*(sh|bash|zsh)\b")),
    ("shell_curl_pipe", re.compile(r"\bcurl\b[^|]*\|\s*(sh|bash)")),
    ("shell_fork_bomb", re.compile(r":\(\)\s*\{\s*:\|")),
]

_DEFAULT_BUNDLE = Path(__file__).parent / "data" / "adversarial_classifier_v1.joblib"

_STATIC_FEATURES = [f"ip__{n}" for n,_ in _IP_PATTERNS] + [f"cred__{n}" for n,_ in _CRED_PATTERNS] + [f"sql__{n}" for n,_ in _SQL_PATTERNS] + [f"shell__{n}" for n,_ in _SHELL_PATTERNS] + [f"scheme__{s}" for s in _URL_SCHEMES] + ["ctx_source_injected", "param_blob_len", "has_wildcard_star", "has_all_keyword", "has_recursive_flag"]


def _param_blob(entry: dict) -> str:
    try:
        return json.dumps({
            "tool_name": entry.get("tool_name", ""),
            "parameters": entry.get("parameters", {}),
            "context": entry.get("context", {}),
        }, default=str)
    except Exception:
        return str(entry)


def _build_features_one(entry: dict, vocab: dict):
    import numpy as np
    top_keys = vocab["top_keys"]
    top_ngrams = vocab["top_ngrams"]
    n_features = (
        len(top_ngrams) + len(top_keys) + len(_IP_PATTERNS) + len(_CRED_PATTERNS)
        + len(_SQL_PATTERNS) + len(_SHELL_PATTERNS) + len(_URL_SCHEMES) + 5
    )
    X = np.zeros((1, n_features), dtype=np.float32)
    tn = (entry.get("tool_name") or "").lower()
    params = entry.get("parameters") or {}
    if not isinstance(params, dict):
        params = {"_raw": str(params)}
    ctx = entry.get("context") or {}
    if not isinstance(ctx, dict):
        ctx = {"original_task": str(ctx)}
    blob = _param_blob({"tool_name": tn, "parameters": params, "context": ctx})
    col = 0
    for g in top_ngrams:
        X[0, col] = float(g in tn); col += 1
    for k in top_keys:
        X[0, col] = float(k in params); col += 1
    for _, pat in _IP_PATTERNS:
        X[0, col] = float(bool(pat.search(blob))); col += 1
    for _, pat in _CRED_PATTERNS:
        X[0, col] = float(bool(pat.search(blob))); col += 1
    for _, pat in _SQL_PATTERNS:
        X[0, col] = float(bool(pat.search(blob))); col += 1
    for _, pat in _SHELL_PATTERNS:
        X[0, col] = float(bool(pat.search(blob))); col += 1
    for s in _URL_SCHEMES:
        X[0, col] = float(f"{s}://" in blob.lower()); col += 1
    X[0, col] = float("source" in ctx and "injected" in str(ctx.get("source", ""))); col += 1
    X[0, col] = float(min(len(blob), 2000) / 2000.0); col += 1
    X[0, col] = float("*" in json.dumps(params, default=str)); col += 1
    X[0, col] = float(" all " in json.dumps(params, default=str) or '"all"' in json.dumps(params, default=str)); col += 1
    X[0, col] = float(re.search(r"--force|--recursive|-rf\b", blob) is not None); col += 1
    return X


class AdversarialClassifier:
    """Opt-in ML classifier for adversarial tool-call detection.

    Loads a pre-trained XGBoost bundle shipped with the package.
    Raises ImportError at construction time if xgboost/joblib aren't installed —
    install with ``pip install vaara[ml]``.
    """

    def __init__(self, bundle_path: Optional[str] = None, threshold: Optional[float] = None) -> None:
        try:
            import joblib  # noqa: F401
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AdversarialClassifier requires extras: pip install vaara[ml]"
            ) from exc
        import joblib
        path = Path(bundle_path) if bundle_path else _DEFAULT_BUNDLE
        if not path.exists():
            raise FileNotFoundError(f"bundle not found: {path}")
        bundle = joblib.load(path)
        self._model = bundle["model"]
        self._vocab = bundle["vocab"]
        self.threshold: float = threshold if threshold is not None else bundle["default_threshold"]
        self.bundle_version: str = bundle.get("version", "unknown")
        tail = (bundle.get("feature_names") or [])[-len(_STATIC_FEATURES):]
        if tail != _STATIC_FEATURES:
            diff = next((i for i, (a, b) in enumerate(zip(_STATIC_FEATURES, tail)) if a != b), -1)
            raise ValueError(f"bundle feature schema drift at static-feature index {diff}: runtime={_STATIC_FEATURES[diff] if diff>=0 else '?'!r} bundle={tail[diff] if diff>=0 else '?'!r} (len runtime={len(_STATIC_FEATURES)} bundle={len(tail)})")

    def score(self, tool_name: str, parameters: Optional[dict] = None, context: Optional[dict] = None) -> float:
        """Return adversarial probability in [0, 1]."""
        entry = {"tool_name": tool_name, "parameters": parameters or {}, "context": context or {}}
        X = _build_features_one(entry, self._vocab)
        return float(self._model.predict_proba(X)[0, 1])

    def is_malicious(self, tool_name: str, parameters: Optional[dict] = None, context: Optional[dict] = None, threshold: Optional[float] = None) -> bool:
        """Return True if score >= threshold (default: bundle's default_threshold)."""
        th = threshold if threshold is not None else self.threshold
        return self.score(tool_name, parameters, context) >= th


__all__ = ["AdversarialClassifier"]
