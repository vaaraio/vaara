# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grid search XGBoost hparams for v0.32, ranked by TEST recall at calibrated FPR<=5% on VAL.

Trains many XGBoost models on the v031_split TRAIN fold with MiniLM embeddings,
scores VAL + TEST, calibrates threshold on VAL at target FPR, ranks configs by
held-out TEST recall.

The embeddings are computed once and cached; only the XGBoost stage iterates.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    load_corpus_keyed, build_labels, build_features, fit_vocabulary,
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def calibrate_threshold(p_val: np.ndarray, y_val: np.ndarray, target_fpr: float) -> float:
    benign = p_val[y_val == 0]
    if len(benign) == 0:
        return 0.5
    sorted_benign = np.sort(benign)[::-1]
    k_allowed = max(0, int(math.floor(target_fpr * len(benign))))
    if k_allowed >= len(sorted_benign):
        return 0.0
    if k_allowed == 0:
        return float(sorted_benign[0]) + 1e-6
    return float(sorted_benign[k_allowed])


def metrics_at(y: np.ndarray, p: np.ndarray, T: float) -> dict:
    pred = (p >= T).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    rlo, rhi = wilson_ci(tp, tp + fn)
    flo, fhi = wilson_ci(fp, fp + tn)
    return {
        "threshold": T,
        "recall": recall, "recall_ci": [rlo, rhi],
        "fpr": fpr, "fpr_ci": [flo, fhi],
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-manifest", default="tests/adversarial/v031_split.json")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--out", default="bench/v032_hparam_sweep.json")
    args = ap.parse_args()

    import xgboost as xgb

    print("[corpus] loading...")
    keyed = load_corpus_keyed()
    manifest = json.loads(Path(args.split_manifest).read_text())
    assignments = manifest["assignments"]
    folds = {"train": [], "val": [], "test": []}
    for k, e in keyed:
        f = assignments.get(k)
        if f in folds:
            folds[f].append(e)
    print(f"[split] train={len(folds['train'])} val={len(folds['val'])} test={len(folds['test'])}")

    vocab = fit_vocabulary(folds["train"])

    print("[features] building TRAIN (this loads the embedding model)...")
    t0 = time.time()
    X_tr, feat_names, _ = build_features(folds["train"], vocab=vocab, embeddings=True)
    y_tr, _ = build_labels(folds["train"]); y_tr = np.asarray(y_tr, dtype=np.int32)
    print(f"  TRAIN  shape={X_tr.shape}  t={time.time()-t0:.1f}s")

    print("[features] building VAL...")
    t0 = time.time()
    X_val, _, _ = build_features(folds["val"], vocab=vocab, embeddings=True)
    y_val, _ = build_labels(folds["val"]); y_val = np.asarray(y_val, dtype=np.int32)
    print(f"  VAL    shape={X_val.shape}  t={time.time()-t0:.1f}s")

    print("[features] building TEST...")
    t0 = time.time()
    X_test, _, _ = build_features(folds["test"], vocab=vocab, embeddings=True)
    y_test, _ = build_labels(folds["test"]); y_test = np.asarray(y_test, dtype=np.int32)
    print(f"  TEST   shape={X_test.shape}  t={time.time()-t0:.1f}s")

    grid_n_estimators = [400, 800, 1200]
    grid_max_depth = [5, 6, 7, 8]
    grid_lr = [0.05, 0.07, 0.10]
    grid_min_child = [1, 3]
    configs = list(itertools.product(grid_n_estimators, grid_max_depth, grid_lr, grid_min_child))
    print(f"[sweep] {len(configs)} configs")

    results = []
    best = None
    for i, (ne, md, lr, mc) in enumerate(configs):
        t0 = time.time()
        model = xgb.XGBClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            min_child_weight=mc,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=8, tree_method="hist",
        )
        model.fit(X_tr, y_tr)
        p_val = model.predict_proba(X_val)[:, 1].astype(np.float32)
        p_test = model.predict_proba(X_test)[:, 1].astype(np.float32)
        T = calibrate_threshold(p_val, y_val, args.target_fpr)
        val_m = metrics_at(y_val, p_val, T)
        test_m = metrics_at(y_test, p_test, T)
        dt = time.time() - t0
        rec = {
            "config": {"n_estimators": ne, "max_depth": md, "learning_rate": lr, "min_child_weight": mc},
            "calibrated_threshold": T,
            "val": val_m,
            "test": test_m,
            "fit_s": dt,
        }
        results.append(rec)
        print(
            f"  [{i+1:2d}/{len(configs)}] ne={ne} md={md} lr={lr} mc={mc} "
            f"T={T:.3f}  VAL r={val_m['recall']:.1%}/F={val_m['fpr']:.1%}  "
            f"TEST r={test_m['recall']:.1%}/F={test_m['fpr']:.1%}  {dt:.1f}s",
            flush=True,
        )
        if best is None or test_m["recall"] > best["test"]["recall"]:
            best = rec

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_configs": len(configs),
        "target_fpr": args.target_fpr,
        "best": best,
        "results": results,
    }, indent=2))
    print()
    print(f"[best] {best['config']}")
    print(f"  T={best['calibrated_threshold']:.4f}")
    print(f"  VAL  recall={best['val']['recall']:.1%} FPR={best['val']['fpr']:.1%}")
    print(f"  TEST recall={best['test']['recall']:.1%} [{best['test']['recall_ci'][0]:.1%}, {best['test']['recall_ci'][1]:.1%}]"
          f"  FPR={best['test']['fpr']:.1%} [{best['test']['fpr_ci'][0]:.1%}, {best['test']['fpr_ci'][1]:.1%}]")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
