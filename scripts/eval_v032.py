"""Evaluate a classifier bundle on the v031_split VAL and TEST folds.

Calibrates the decision threshold on VAL at target FPR (default 5%), then
reports recall and FPR on TEST with Wilson confidence intervals. Used to
gate v0.32 release against the v0.31 baseline (recall 53.9%, FPR 4.6%).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    load_corpus_keyed, build_labels, build_features,
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--split-manifest", default="tests/adversarial/v031_split.json")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import joblib
    bundle = joblib.load(args.bundle)
    model = bundle["model"]
    vocab = bundle["vocab"]
    feat_names = bundle["feature_names"]
    uses_embeddings = bundle.get("uses_embeddings", False) or any(
        n.startswith("embed__") for n in feat_names
    )
    print(f"[bundle] version={bundle.get('version')} "
          f"n_features={len(feat_names)} uses_embeddings={uses_embeddings}")

    manifest = json.loads(Path(args.split_manifest).read_text())
    assignments = manifest["assignments"]
    keyed = load_corpus_keyed()

    folds = {"train": [], "val": [], "test": []}
    for k, e in keyed:
        f = assignments.get(k)
        if f in folds:
            folds[f].append(e)
    print(f"[split] train={len(folds['train'])} val={len(folds['val'])} test={len(folds['test'])}")

    def score(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        y, _ = build_labels(entries)
        y = np.asarray(y, dtype=np.int32)
        X, names, _ = build_features(entries, vocab=vocab, embeddings=uses_embeddings)
        if names != feat_names:
            raise ValueError(
                f"feature schema mismatch: bundle has {len(feat_names)}, "
                f"build_features produced {len(names)}"
            )
        prob = model.predict_proba(X)[:, 1].astype(np.float32)
        return y, prob

    print("[score] val...")
    y_val, p_val = score(folds["val"])
    print("[score] test...")
    y_test, p_test = score(folds["test"])

    pos_val = (y_val == 1).sum()
    neg_val = (y_val == 0).sum()
    pos_test = (y_test == 1).sum()
    neg_test = (y_test == 0).sum()
    print(f"[val] n={len(y_val)} pos={pos_val} neg={neg_val}")
    print(f"[test] n={len(y_test)} pos={pos_test} neg={neg_test}")

    # Calibrate threshold on val: smallest T such that FPR_val <= target.
    benign_val = p_val[y_val == 0]
    if len(benign_val) == 0:
        print("[error] no benign in val")
        return 2
    target_fpr = args.target_fpr
    sorted_benign = np.sort(benign_val)[::-1]  # descending
    k_allowed = max(0, int(math.floor(target_fpr * len(benign_val))))
    if k_allowed >= len(sorted_benign):
        T_cal = 0.0
    elif k_allowed == 0:
        T_cal = float(sorted_benign[0]) + 1e-6
    else:
        T_cal = float(sorted_benign[k_allowed])

    print(f"[calibration] target FPR={target_fpr:.3f} on val -> threshold={T_cal:.4f}")

    def metrics_at(y: np.ndarray, p: np.ndarray, T: float):
        pred = (p >= T).astype(np.int32)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        rlo, rhi = wilson_ci(tp, tp + fn)
        flo, fhi = wilson_ci(fp, fp + tn)
        return {
            "threshold": T,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "recall_ci": [rlo, rhi],
            "fpr": fpr, "fpr_ci": [flo, fhi],
            "precision": precision,
        }

    val_m = metrics_at(y_val, p_val, T_cal)
    test_m = metrics_at(y_test, p_test, T_cal)

    print()
    print(f"=== Calibrated threshold T={T_cal:.4f} ===")
    print(f"VAL  recall {val_m['recall']:.1%} [{val_m['recall_ci'][0]:.1%}, {val_m['recall_ci'][1]:.1%}]"
          f"  FPR {val_m['fpr']:.1%} [{val_m['fpr_ci'][0]:.1%}, {val_m['fpr_ci'][1]:.1%}]")
    print(f"TEST recall {test_m['recall']:.1%} [{test_m['recall_ci'][0]:.1%}, {test_m['recall_ci'][1]:.1%}]"
          f"  FPR {test_m['fpr']:.1%} [{test_m['fpr_ci'][0]:.1%}, {test_m['fpr_ci'][1]:.1%}]")

    # Also report at a few canonical thresholds for context
    print()
    print("=== TEST metrics at canonical thresholds ===")
    for T in (0.5, 0.7, 0.8, 0.9, 0.95):
        m = metrics_at(y_test, p_test, T)
        print(f"  T={T:.2f}  recall={m['recall']:.1%}  FPR={m['fpr']:.1%}  precision={m['precision']:.1%}")

    out = {
        "bundle": args.bundle,
        "bundle_version": bundle.get("version"),
        "uses_embeddings": uses_embeddings,
        "n_features": len(feat_names),
        "split_manifest": args.split_manifest,
        "target_fpr": target_fpr,
        "calibrated_threshold": T_cal,
        "val": {**val_m, "n": len(y_val), "pos": int(pos_val), "neg": int(neg_val)},
        "test": {**test_m, "n": len(y_test), "pos": int(pos_test), "neg": int(neg_test)},
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\n[out] {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
