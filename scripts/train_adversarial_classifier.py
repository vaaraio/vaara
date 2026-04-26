#!/usr/bin/env python3
"""Train a classifier on the adversarial corpus v1.

Proof-of-concept: does a trained classifier beat the current heuristic
scorer on held-out folds?

Features extracted per entry (no external deps beyond sklearn + xgboost):

- tool_name n-grams (character 3-5) — semantic tool-name matching
- parameter-key presence (bag over all unique keys in the corpus)
- parameter-value patterns:
    * URL scheme (http/https/file/dict/gopher/data)
    * IP literal present (169.254.169.254, 127.0.0.1, 10., 192.168., etc.)
    * AWS metadata path present
    * "env" / "secret" / "password" / ".ssh" substring
    * SQL keyword present (DROP, TRUNCATE, DELETE, UNION)
    * shell-exec pattern (rm -rf, dd if=, curl | sh)
- context-source hints (source=injected_via_*)

Target: is_malicious = 1 if expected ∈ {"DENY", "ESCALATE"} else 0.

Evaluated via 5-fold stratified CV. Baseline = current Vaara scorer's
ALLOW-leakage on the same folds. Reports accuracy, precision, recall,
F1 per category, and the top 20 feature importances.

Prerequisite:
    pip install -e ".[dev]"  (to make ``from vaara import Pipeline`` work)

Usage:
    python scripts/train_adversarial_classifier.py
    python scripts/train_adversarial_classifier.py --json-out results.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

CORPUS_DIR = Path("tests/adversarial")


# ── Feature extraction ─────────────────────────────────────────────────

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


def _normalize_str(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (dict, list)):
        return json.dumps(s, sort_keys=True)
    return str(s)


def _coerce_params(p: Any) -> dict:
    """Return a dict view of an entry's ``parameters`` field.

    Some corpus entries (present since v0.5.0) have a string-typed
    ``parameters``. Wrap those as ``{"_raw": str(p)}`` so every consumer
    of params can rely on dict semantics without crashing.
    """
    if not p:
        return {}
    if isinstance(p, dict):
        return p
    return {"_raw": str(p)}


def _coerce_context(c: Any) -> dict:
    """Return a dict view of an entry's ``context`` field.

    Some corpus entries have a string-typed ``context`` whose contents
    are the original_task. Wrap those as ``{"original_task": str(c)}``.
    """
    if not c:
        return {}
    if isinstance(c, dict):
        return c
    return {"original_task": str(c)}


def _param_blob(entry: dict) -> str:
    parts = [_normalize_str(_coerce_params(entry.get("parameters")))]
    parts.append(_normalize_str(_coerce_context(entry.get("context"))))
    return " ".join(parts)


def load_corpus() -> list[dict]:
    entries: list[dict] = []
    for fp in sorted(CORPUS_DIR.rglob("*.jsonl")):
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    return entries


def fit_vocabulary(entries: list[dict]) -> dict:
    """Fit token vocabulary on training-only entries.

    Returns a dict with ``top_keys`` and ``top_ngrams`` that ``build_features``
    uses as the transform schema. Separating fit and transform closes the
    pre-refactor feature-leakage bug where vocabulary was chosen using the
    full corpus (including test-fold tokens).
    """
    param_keys: Counter = Counter()
    for e in entries:
        for k in _coerce_params(e.get("parameters")).keys():
            param_keys[k] += 1
    top_keys = [k for k, _ in param_keys.most_common(80)]

    ngram_counts: Counter = Counter()
    for e in entries:
        tn = e.get("tool_name", "").lower()
        for n in (3, 4, 5):
            for i in range(len(tn) - n + 1):
                ngram_counts[tn[i : i + n]] += 1
    top_ngrams = [g for g, c in ngram_counts.most_common(120) if c >= 3]
    return {"top_keys": top_keys, "top_ngrams": top_ngrams}


def build_features(
    entries: list[dict], vocab: Optional[dict] = None
) -> tuple[np.ndarray, list[str], dict]:
    """Transform entries to a feature matrix using ``vocab``.

    If ``vocab`` is None, fits vocabulary on these entries first. Pass a
    training-fitted vocab when transforming held-out entries to avoid
    leakage. Returns (X, feature_names, vocab).
    """
    if vocab is None:
        vocab = fit_vocabulary(entries)
    top_keys = vocab["top_keys"]
    top_ngrams = vocab["top_ngrams"]

    feature_names: list[str] = []
    feature_names += [f"ngram__{g}" for g in top_ngrams]
    feature_names += [f"pkey__{k}" for k in top_keys]
    feature_names += [f"ip__{name}" for name, _ in _IP_PATTERNS]
    feature_names += [f"cred__{name}" for name, _ in _CRED_PATTERNS]
    feature_names += [f"sql__{name}" for name, _ in _SQL_PATTERNS]
    feature_names += [f"shell__{name}" for name, _ in _SHELL_PATTERNS]
    feature_names += [f"scheme__{s}" for s in _URL_SCHEMES]
    feature_names += [
        "ctx_source_injected",
        "param_blob_len",
        "has_wildcard_star",
        "has_all_keyword",
        "has_recursive_flag",
    ]

    n = len(entries)
    X = np.zeros((n, len(feature_names)), dtype=np.float32)
    for i, e in enumerate(entries):
        tn = e.get("tool_name", "").lower()
        params = _coerce_params(e.get("parameters"))
        ctx = _coerce_context(e.get("context"))
        blob = _param_blob(e)

        col = 0
        for g in top_ngrams:
            X[i, col] = float(g in tn)
            col += 1
        for k in top_keys:
            X[i, col] = float(k in params)
            col += 1
        for _, pat in _IP_PATTERNS:
            X[i, col] = float(bool(pat.search(blob)))
            col += 1
        for _, pat in _CRED_PATTERNS:
            X[i, col] = float(bool(pat.search(blob)))
            col += 1
        for _, pat in _SQL_PATTERNS:
            X[i, col] = float(bool(pat.search(blob)))
            col += 1
        for _, pat in _SHELL_PATTERNS:
            X[i, col] = float(bool(pat.search(blob)))
            col += 1
        for s in _URL_SCHEMES:
            scheme_pat = f"{s}://"
            X[i, col] = float(scheme_pat in blob.lower())
            col += 1
        X[i, col] = float("source" in ctx and "injected" in str(ctx.get("source", "")))
        col += 1
        X[i, col] = float(min(len(blob), 2000) / 2000.0)
        col += 1
        X[i, col] = float("*" in json.dumps(params))
        col += 1
        X[i, col] = float(' all ' in json.dumps(params) or '"all"' in json.dumps(params))
        col += 1
        X[i, col] = float(re.search(r"--force|--recursive|-rf\b", blob) is not None)
        col += 1
    return X, feature_names, vocab


def build_labels(entries: list[dict]) -> tuple[np.ndarray, list[str]]:
    y = np.zeros(len(entries), dtype=np.int32)
    cats = []
    for i, e in enumerate(entries):
        expected = e.get("expected", "DENY")
        if isinstance(expected, list):
            labels = set(expected)
        else:
            labels = {expected}
        y[i] = int(any(lbl in ("DENY", "ESCALATE") for lbl in labels))
        cats.append(e.get("category", "unknown"))
    return y, cats


# ── Baseline: run current Vaara scorer on entries ─────────────────────

def baseline_predictions(entries: list[dict], best_effort: bool = False) -> np.ndarray:
    """Run the current Vaara scorer on every entry and return malicious flags.

    Raises on ``Pipeline.intercept`` failures by default. Pass
    ``best_effort=True`` to log each failure and continue with a sentinel
    prediction (treated as non-malicious, but logged so it isn't conflated
    with a legitimate ALLOW).
    """
    from vaara import Pipeline

    pipe = Pipeline()
    preds = np.zeros(len(entries), dtype=np.int32)
    for i, e in enumerate(entries):
        try:
            r = pipe.intercept(
                agent_id=e.get("agent_id", "adv"),
                tool_name=e["tool_name"],
                parameters=e.get("parameters", {}),
                context=e.get("context", {}),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            if not best_effort:
                raise
            print(
                f"[baseline] intercept failed id={e.get('id')} "
                f"agent={e.get('agent_id')} tool={e.get('tool_name')}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            preds[i] = 0
            continue
        decision = str(getattr(r, "decision", "")).lower()
        preds[i] = int(decision in ("deny", "escalate"))
    return preds


# ── Training + evaluation ─────────────────────────────────────────────

def run(args) -> int:
    entries = load_corpus()
    print(f"[corpus] {len(entries)} entries", flush=True)

    y, cats = build_labels(entries)
    print(f"[corpus] positive rate={y.mean():.3f}", flush=True)

    print("[baseline] running current scorer on full corpus...", flush=True)
    y_base = baseline_predictions(entries, best_effort=True)
    baseline_acc = float((y_base == y).mean())
    print(f"[baseline] accuracy vs ground-truth-malicious label: {baseline_acc:.3f}", flush=True)

    # Stratified 5-fold CV on (category, label) pairs so each fold sees all categories
    strat_key = np.array([f"{c}__{lbl}" for c, lbl in zip(cats, y)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []
    y_pred_oof = np.zeros_like(y)
    y_prob_oof = np.zeros(len(y), dtype=np.float32)
    feature_imps_accum: np.ndarray = None  # type: ignore
    feat_names_ref: list[str] = []

    # Index-based splits only: entries are split by index; features are
    # fit per-fold on training entries to prevent vocabulary leakage.
    entries_arr = np.array(entries, dtype=object)
    dummy_X = np.zeros(len(entries), dtype=np.float32)
    for fi, (tr, te) in enumerate(skf.split(dummy_X, strat_key)):
        train_entries = list(entries_arr[tr])
        test_entries = list(entries_arr[te])
        vocab = fit_vocabulary(train_entries)
        X_tr, feat_names, _ = build_features(train_entries, vocab=vocab)
        X_te, _, _ = build_features(test_entries, vocab=vocab)
        if feature_imps_accum is None:
            feature_imps_accum = np.zeros(len(feat_names), dtype=np.float32)
            feat_names_ref = feat_names
        # If the fitted vocabulary differs in size across folds (train
        # distribution changes), pad/truncate the importance accumulator
        # to the reference size on a best-effort basis.
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42 + fi,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y[tr])
        prob = model.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(np.int32)
        y_pred_oof[te] = pred
        y_prob_oof[te] = prob
        fold_metrics.append({
            "fold": fi,
            "acc": float((pred == y[te]).mean()),
            "n_test": int(len(te)),
        })
        try:
            imp = model.feature_importances_
            if len(imp) == len(feature_imps_accum):
                feature_imps_accum += imp
        except (AttributeError, ValueError) as exc:
            print(f"[fold {fi}] feature_importances unavailable: {exc}", flush=True)
        print(f"[fold {fi}] acc={fold_metrics[-1]['acc']:.3f} n_test={len(te)}", flush=True)

    classifier_acc = float((y_pred_oof == y).mean())
    print(f"\n[classifier OOF] accuracy={classifier_acc:.3f}")
    print(f"[baseline full ] accuracy={baseline_acc:.3f}")
    print(f"[delta]          {classifier_acc - baseline_acc:+.3f}\n")

    # Per-category breakdown
    per_cat = defaultdict(lambda: {"n": 0, "base_correct": 0, "clf_correct": 0})
    for i, c in enumerate(cats):
        per_cat[c]["n"] += 1
        per_cat[c]["base_correct"] += int(y_base[i] == y[i])
        per_cat[c]["clf_correct"] += int(y_pred_oof[i] == y[i])
    print("=== Per-category accuracy ===")
    header = f"{'category':24s}  {'n':>3}  {'baseline':>8}  {'classifier':>10}  {'delta':>6}"
    print(header)
    print("-" * len(header))
    for cat, m in sorted(per_cat.items()):
        base_acc_c = m["base_correct"] / m["n"]
        clf_acc_c = m["clf_correct"] / m["n"]
        print(f"  {cat:22s}  {m['n']:3d}  {base_acc_c:8.1%}  {clf_acc_c:10.1%}  {clf_acc_c-base_acc_c:+6.1%}")

    print("\n=== Confusion matrix (classifier OOF) ===")
    cm = confusion_matrix(y, y_pred_oof)
    print("          pred=benign  pred=mal")
    print(f"true=benign   {cm[0,0]:7d}    {cm[0,1]:6d}")
    print(f"true=mal      {cm[1,0]:7d}    {cm[1,1]:6d}")

    print("\n=== sklearn classification_report (classifier OOF) ===")
    print(classification_report(y, y_pred_oof, digits=3))

    # Top-20 feature importances (averaged across folds)
    feature_imps = feature_imps_accum / max(len(fold_metrics), 1)
    top_idx = np.argsort(-feature_imps)[:20]
    print("=== Top-20 feature importances ===")
    for idx in top_idx:
        if feature_imps[idx] > 0:
            print(f"  {feature_imps[idx]:8.4f}  {feat_names_ref[idx]}")

    baseline_bal = float(balanced_accuracy_score(y, y_base))
    classifier_bal = float(balanced_accuracy_score(y, y_pred_oof))
    print(f"\n[balanced accuracy] baseline={baseline_bal:.3f} classifier={classifier_bal:.3f} delta={classifier_bal-baseline_bal:+.3f}")

    if args.json_out:
        out = {
            "n_entries": len(entries),
            "n_features": len(feat_names_ref),
            "positive_rate": float(y.mean()),
            "baseline_acc": baseline_acc,
            "classifier_oof_acc": classifier_acc,
            "classifier_minus_baseline": classifier_acc - baseline_acc,
            "baseline_balanced_acc": baseline_bal,
            "classifier_balanced_acc": classifier_bal,
            "folds": fold_metrics,
            "per_category": {
                c: {
                    "n": m["n"],
                    "baseline_acc": m["base_correct"] / m["n"],
                    "classifier_acc": m["clf_correct"] / m["n"],
                }
                for c, m in per_cat.items()
            },
            "top_features": [
                {"feature": feat_names_ref[i], "importance": float(feature_imps[i])}
                for i in top_idx if feature_imps[i] > 0
            ],
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\n[out] {args.json_out}")

    # Use BALANCED accuracy for the pass/fail gate. Raw accuracy can be
    # gamed on an imbalanced corpus (200 malicious vs 50 benign) by a
    # trivial class-dominant predictor.
    return 0 if classifier_bal > baseline_bal else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None, help="Write results JSON to this path.")
    args = ap.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
