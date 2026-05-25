"""v0.33 step 2 experiment: train v4 bundle with BAAI/bge-base-en-v1.5 and evaluate.

Standalone A/B against v0.32's MiniLM-backed v3 bundle. Does NOT touch
``vaara.embeddings`` (production singleton stays MiniLM until ship-or-skip lands).

Pipeline: corpus + v031_split → 236 hand-features → bge-base 768d embed →
concat (1004 features) → XGBoost (ne=400 md=6 lr=0.10, matching v0.32) →
VAL FPR=5% calibration → TEST recall/FPR with Wilson CIs.

Ship-or-skip: bge-base wins iff TEST recall lifts >= 2pp over v0.32 (84.3%)
at <= 5% FPR.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    load_corpus_keyed, build_labels, build_features, fit_vocabulary, _param_blob,
)

BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
BGE_DIM = 768


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def calibrate_threshold(p: np.ndarray, y: np.ndarray, target_fpr: float) -> float:
    benign = p[y == 0]
    if len(benign) == 0:
        return 0.5
    sorted_benign = np.sort(benign)[::-1]
    k = max(0, int(math.floor(target_fpr * len(benign))))
    if k >= len(sorted_benign):
        return 0.0
    if k == 0:
        return float(sorted_benign[0]) + 1e-6
    return float(sorted_benign[k])


def metrics_at(y: np.ndarray, p: np.ndarray, T: float) -> dict:
    pred = (p >= T).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    rlo, rhi = wilson_ci(tp, tp + fn)
    flo, fhi = wilson_ci(fp, fp + tn)
    return {
        "threshold": T, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / max(tp + fn, 1), "recall_ci": [rlo, rhi],
        "fpr": fp / max(fp + tn, 1), "fpr_ci": [flo, fhi],
        "precision": tp / max(tp + fp, 1),
    }


def embed_blobs(blobs: list[str], device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL_ID, revision=BGE_REVISION, device=device)
    return model.encode(
        blobs, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-manifest", default="tests/adversarial/v031_split.json")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--json-out", default="bench/v033_bge_base_eval.json")
    args = ap.parse_args()

    import xgboost as xgb

    print(f"[bge] {BGE_MODEL_ID} rev={BGE_REVISION[:12]} dim={BGE_DIM} device={args.device}")
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

    def hand(entries):
        X, names, _ = build_features(entries, vocab=vocab, embeddings=False)
        return X, names

    X_tr_h, hand_names = hand(folds["train"])
    X_val_h, _ = hand(folds["val"])
    X_test_h, _ = hand(folds["test"])
    y_tr = np.asarray(build_labels(folds["train"])[0], dtype=np.int32)
    y_val = np.asarray(build_labels(folds["val"])[0], dtype=np.int32)
    y_test = np.asarray(build_labels(folds["test"])[0], dtype=np.int32)
    print(f"[hand] TRAIN={X_tr_h.shape} VAL={X_val_h.shape} TEST={X_test_h.shape}")

    print(f"[embed] loading bge-base (~440MB first-time download)...")
    t0 = time.time()
    blobs = (
        [_param_blob(e) for e in folds["train"]]
        + [_param_blob(e) for e in folds["val"]]
        + [_param_blob(e) for e in folds["test"]]
    )
    E = embed_blobs(blobs, device=args.device, batch_size=args.batch_size)
    n_tr, n_val = len(folds["train"]), len(folds["val"])
    E_tr, E_val, E_test = E[:n_tr], E[n_tr:n_tr + n_val], E[n_tr + n_val:]
    print(f"[embed] done in {time.time()-t0:.1f}s")

    X_tr = np.hstack([X_tr_h, E_tr]).astype(np.float32)
    X_val = np.hstack([X_val_h, E_val]).astype(np.float32)
    X_test = np.hstack([X_test_h, E_test]).astype(np.float32)
    print(f"[stack] X_tr={X_tr.shape}")

    print(f"[fit] ne={args.n_estimators} md={args.max_depth} lr={args.learning_rate}")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, min_child_weight=1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, n_jobs=8, tree_method="hist",
    )
    model.fit(X_tr, y_tr)
    print(f"[fit] done in {time.time()-t0:.1f}s")

    p_val = model.predict_proba(X_val)[:, 1].astype(np.float32)
    p_test = model.predict_proba(X_test)[:, 1].astype(np.float32)
    T = calibrate_threshold(p_val, y_val, args.target_fpr)
    val_m = metrics_at(y_val, p_val, T)
    test_m = metrics_at(y_test, p_test, T)

    print()
    print(f"=== T={T:.4f} (VAL FPR<={args.target_fpr:.0%}) ===")
    print(f"VAL  recall {val_m['recall']:.1%} [{val_m['recall_ci'][0]:.1%}, {val_m['recall_ci'][1]:.1%}]"
          f"  FPR {val_m['fpr']:.1%} [{val_m['fpr_ci'][0]:.1%}, {val_m['fpr_ci'][1]:.1%}]")
    print(f"TEST recall {test_m['recall']:.1%} [{test_m['recall_ci'][0]:.1%}, {test_m['recall_ci'][1]:.1%}]"
          f"  FPR {test_m['fpr']:.1%} [{test_m['fpr_ci'][0]:.1%}, {test_m['fpr_ci'][1]:.1%}]")

    delta = test_m["recall"] - 0.843
    ship = delta >= 0.02 and test_m["fpr"] <= 0.05
    print()
    print(f"v0.32 baseline: recall 84.3% FPR 4.6% @ T=0.9226")
    print(f"Δ recall = {delta:+.1%} (ship gate: +2.0pp at FPR<=5%)")
    print(f"SHIP: {ship}")

    out = {
        "experiment": "v0.33_bge_base_ab",
        "embedding_model_id": BGE_MODEL_ID,
        "embedding_model_revision": BGE_REVISION,
        "embedding_dim": BGE_DIM,
        "split_manifest": args.split_manifest,
        "target_fpr": args.target_fpr,
        "calibrated_threshold": T,
        "hparams": {"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate},
        "val": val_m, "test": test_m,
        "v032_baseline": {"recall": 0.843, "fpr": 0.046, "threshold": 0.9226},
        "delta_recall_pp": delta, "ship_or_skip": "ship" if ship else "skip",
        "trained_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
