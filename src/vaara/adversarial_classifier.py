"""Adversarial tool-call classifier (opt-in ML scorer).

Ships with a pre-trained XGBoost model trained on the v0.31 split's
TRAIN fold (5,563 entries from a 7,955-entry corpus). v0.32 bundles
concatenate 384-dim ``all-MiniLM-L6-v2`` embeddings of the parameter
blob after the 236 hand-features.

Headline numbers and methodology live in README.md and
``bench/vaara-bench-v0.31.md``. Per-source breakdown and adaptive-attacker
calibration live in COMPLIANCE.md. Numbers intentionally not duplicated
here so this docstring does not go stale on every release.

Usage:
    pip install vaara[ml]
    from vaara.adversarial_classifier import AdversarialClassifier
    clf = AdversarialClassifier()
    prob = clf.score(tool_name="...", parameters={...}, context={...})
    if prob >= clf.threshold:
        # block or escalate
        ...

Requires xgboost, scikit-learn, joblib, numpy, sentence-transformers.
All stay out of the default install.
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
# Cloud-metadata endpoints (AWS IMDS, GCP metadata server, ECS task-role) are
# unambiguous SSRF / credential-theft destinations with no legitimate benign
# use inside an agent tool call. The learned scorer underweights a bare
# http_post to these, so a deterministic floor guarantees the security
# property the model alone does not. Defense-in-depth: the model still scores
# every input; this only lifts the floor when a known-bad destination matches.
# Floored above the calibrated default threshold so these are actually flagged
# at runtime, not merely scored above the class-balance midpoint.
#
# The list covers the canonical dotted-decimal endpoints AND the parser-
# confusion encodings that resolve to the same AWS IMDS address but slip past
# a literal-string check: the dotless 32-bit integer (2852039166) and hex
# (0xa9fea9fe) forms of 169.254.169.254, and the IPv6 link-local address AWS
# serves IMDS on (fd00:ec2::254). Each is still zero-benign: no legitimate tool
# call encodes a host this way. Not exhaustive on purpose; IMDSv2 token flows,
# DNS rebinding, and arbitrary octal/mixed encodings still fall back to the
# model. The defensible public claim is "flags the well-known cloud
# instance-metadata endpoints," not "detects SSRF" in general.
_METADATA_SSRF_PATTERNS = (
    re.compile(r"169\.254\.169\.254"),
    re.compile(r"metadata\.google\.internal"),
    re.compile(r"169\.254\.170\.2"),
    # AWS IMDS over IPv6 link-local.
    re.compile(r"fd00:ec2::254", re.I),
    # 169.254.169.254 as a dotless 32-bit decimal integer, bounded so it is not
    # a slice of a longer digit run.
    re.compile(r"(?<!\d)2852039166(?!\d)"),
    # 169.254.169.254 as a dotless hex integer; the 0x prefix anchors it.
    re.compile(r"0xa9fea9fe(?![0-9a-f])", re.I),
)
_METADATA_SSRF_FLOOR = 0.95
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
# v0.36 destination-aware features. Must stay byte-for-byte identical with
# scripts/train_adversarial_classifier.py:_DST_PATTERNS so training and inference
# produce the same feature vector. Bundles trained without dst features still
# load; the schema check below adapts to the bundle's feature_names.
_DST_PATTERNS = [
    ("personal_email", re.compile(
        r"@(gmail|yahoo|outlook|hotmail|icloud|protonmail|proton\.me|aol|"
        r"mail\.ru|qq\.com|ya\.ru|gmx|fastmail|tutanota|zoho|posteo|hushmail)"
        r"(\.[a-z]{2,4})?\b", re.I)),
    ("webhook_relay", re.compile(
        r"\b(webhook\.site|requestbin|hookbin|beeceptor|mockbin|pipedream|"
        r"ngrok\.io|ngrok-free\.app|trycloudflare|loca\.lt|serveo)", re.I)),
    ("paste_service", re.compile(
        r"\b(pastebin|paste\.ee|paste\.org|hastebin|gist\.github|"
        r"controlc|justpaste|paste\.rs|privatebin|0bin)", re.I)),
    ("free_filehost", re.compile(
        r"\b(transfer\.sh|file\.io|wetransfer|sendspace|mega\.nz|"
        r"mediafire|filebin|0x0\.st|catbox\.moe|anonfile|gofile\.io)", re.I)),
    ("public_bucket", re.compile(
        r"(public[-_]?(temp|share|storage|bucket|read)|public-read|"
        r"\"acl\"\s*:\s*\"public|\"visibility\"\s*:\s*\"public|"
        r"\"private\"\s*:\s*\"?(false|no))", re.I)),
    ("never_expires", re.compile(
        r"\"expir(y|ation|es)\"\s*:\s*\"?(never|none|null|0|-1)|"
        r"\"ttl\"\s*:\s*\"?(0|-1|none|null|never)", re.I)),
    ("ext_share_scheme", re.compile(
        r"\b(smb|ftp|sftp|webdav)://", re.I)),
    ("ext_bucket_name", re.compile(
        r"(temp|external|staging|public)[-_]?(storage|bucket|backup|share)", re.I)),
    ("pii_columns", re.compile(
        r"\b(ssn|social[-_]?security|date[-_]?of[-_]?birth|credit[-_]?card|"
        r"cvv|passport|driver[-_]?license|medical[-_]?record|payroll|salary|"
        r"tax[-_]?id|bank[-_]?account|routing[-_]?number|phi|pii)\b", re.I)),
    ("bulk_select_limit", re.compile(
        r"\bSELECT\b.{0,200}\bLIMIT\s+([1-9]\d{3,})\b", re.I | re.S)),
    ("select_star_from", re.compile(r"\bSELECT\s+\*\s+FROM\s+\w+", re.I)),
    ("export_dump_to", re.compile(
        r"\b(export|copy|backup|dump)[-_]?(url|to|destination|target)\b", re.I)),
    ("susp_tld", re.compile(
        r"\.(tk|ml|ga|cf|gq|xyz|top|click|download|stream|loan|date|win|"
        r"work|men|review)/", re.I)),
    ("webhook_param_key", re.compile(
        r"\"(webhook_url|callback_url|hook|destination_url|external_url|"
        r"forward_to|relay_to)\"", re.I)),
    ("tmp_or_public_path", re.compile(r"/(tmp|public|shared)/", re.I)),
    ("share_platform", re.compile(
        r"\b(slack\.com/files|discord\.com/channels|telegram\.me|t\.me/|"
        r"signal\.org|whatsapp\.com|wa\.me/|m\.me/)", re.I)),
    ("recipient_to_personal", re.compile(
        r"\"to\"\s*:\s*\"[^\"]*@(gmail|yahoo|outlook|hotmail|icloud)", re.I)),
    ("attachment_with_external_to", re.compile(
        r"\"attachments?\"\s*:\s*\[.*\].*\"to\"\s*:\s*\"[^\"]*@(?!.*(corp|internal|\.local))",
        re.I | re.S)),
]

_DEFAULT_BUNDLE = Path(__file__).parent / "data" / "adversarial_classifier_v9.joblib"

_STATIC_FEATURES = [f"ip__{n}" for n,_ in _IP_PATTERNS] + [f"cred__{n}" for n,_ in _CRED_PATTERNS] + [f"sql__{n}" for n,_ in _SQL_PATTERNS] + [f"shell__{n}" for n,_ in _SHELL_PATTERNS] + [f"scheme__{s}" for s in _URL_SCHEMES] + ["ctx_source_injected", "param_blob_len", "has_wildcard_star", "has_all_keyword", "has_recursive_flag"]
_DST_STATIC = [f"dst__{n}" for n, _ in _DST_PATTERNS]


def _param_blob(entry: dict) -> str:
    try:
        return json.dumps({
            "tool_name": entry.get("tool_name", ""),
            "parameters": entry.get("parameters", {}),
            "context": entry.get("context", {}),
        }, default=str)
    except Exception:
        return str(entry)


def _build_features_one(entry: dict, vocab: dict, use_embeddings: bool = False, use_dst: bool = False):
    import numpy as np
    top_keys = vocab["top_keys"]
    top_ngrams = vocab["top_ngrams"]
    n_hand = (
        len(top_ngrams) + len(top_keys) + len(_IP_PATTERNS) + len(_CRED_PATTERNS)
        + len(_SQL_PATTERNS) + len(_SHELL_PATTERNS) + len(_URL_SCHEMES) + 5
        + (len(_DST_PATTERNS) if use_dst else 0)
    )
    if use_embeddings:
        from vaara.embeddings import EMBED_DIM
        n_features = n_hand + EMBED_DIM
    else:
        n_features = n_hand
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
    if use_dst:
        for _, pat in _DST_PATTERNS:
            X[0, col] = float(bool(pat.search(blob))); col += 1
    if use_embeddings:
        from vaara.embeddings import embed, EMBED_DIM
        X[0, col:col + EMBED_DIM] = embed(blob)
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
        feat_names = bundle.get("feature_names") or []
        self._use_embeddings = any(n.startswith("embed__") for n in feat_names)
        self._use_dst = any(n.startswith("dst__") for n in feat_names)
        expected_static = list(_STATIC_FEATURES) + (list(_DST_STATIC) if self._use_dst else [])
        if self._use_embeddings:
            tail_end = next((i for i, n in enumerate(feat_names) if n.startswith("embed__")), len(feat_names))
            tail = feat_names[tail_end - len(expected_static):tail_end]
        else:
            tail = feat_names[-len(expected_static):]
        if tail != expected_static:
            diff = next((i for i, (a, b) in enumerate(zip(expected_static, tail)) if a != b), -1)
            raise ValueError(f"bundle feature schema drift at static-feature index {diff}: runtime={expected_static[diff] if diff>=0 else '?'!r} bundle={tail[diff] if diff>=0 else '?'!r} (len runtime={len(expected_static)} bundle={len(tail)})")

    def score(self, tool_name: str, parameters: Optional[dict] = None, context: Optional[dict] = None) -> float:
        """Return adversarial probability in [0, 1]."""
        entry = {"tool_name": tool_name, "parameters": parameters or {}, "context": context or {}}
        X = _build_features_one(entry, self._vocab, use_embeddings=self._use_embeddings, use_dst=self._use_dst)
        prob = float(self._model.predict_proba(X)[0, 1])
        blob = json.dumps(entry, default=str)
        if any(p.search(blob) for p in _METADATA_SSRF_PATTERNS):
            return max(prob, _METADATA_SSRF_FLOOR)
        return prob

    def is_malicious(self, tool_name: str, parameters: Optional[dict] = None, context: Optional[dict] = None, threshold: Optional[float] = None) -> bool:
        """Return True if score >= threshold (default: bundle's default_threshold)."""
        th = threshold if threshold is not None else self.threshold
        return self.score(tool_name, parameters, context) >= th


__all__ = ["AdversarialClassifier"]
