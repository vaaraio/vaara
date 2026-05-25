"""Per-category recall + FPR for a classifier bundle on the v031_split TEST fold.

Calibrates the decision threshold on VAL at target FPR (default 5%), then
reports recall and FPR per ``entry["category"]`` on TEST. Sorted ascending by
recall so the weakest categories surface first. Feeds v0.34 targeted corpus
generation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--split-manifest", default="tests/adversarial/v031_split.json")
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="Used only if --threshold is not provided (calibrates on VAL).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override calibration; use this exact threshold on TEST.")
    ap.add_argument("--json-out", required=True)
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

    def score(entries: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        y, cats = build_labels(entries)
        y = np.asarray(y, dtype=np.int32)
        X, names, _ = build_features(entries, vocab=vocab, embeddings=uses_embeddings)
        if names != feat_names:
            raise ValueError(
                f"feature schema mismatch: bundle has {len(feat_names)}, "
                f"build_features produced {len(names)}"
            )
        prob = model.predict_proba(X)[:, 1].astype(np.float32)
        return y, prob, cats

    if args.threshold is None:
        print("[score] val...")
        y_val, p_val, _ = score(folds["val"])
        T = calibrate_threshold(p_val, y_val, args.target_fpr)
        print(f"[calibration] target FPR={args.target_fpr:.3f} on val -> T={T:.4f}")
    else:
        T = float(args.threshold)
        print(f"[threshold] using --threshold={T:.4f} (no VAL calibration)")

    print("[score] test...")
    y_test, p_test, cats_test = score(folds["test"])
    pred_test = (p_test >= T).astype(np.int32)

    # Per-category buckets, separating positives (recall) from negatives (FPR).
    pos_tp: dict[str, int] = defaultdict(int)
    pos_n: dict[str, int] = defaultdict(int)
    neg_fp: dict[str, int] = defaultdict(int)
    neg_n: dict[str, int] = defaultdict(int)
    for yi, pi, ci in zip(y_test, pred_test, cats_test):
        if yi == 1:
            pos_n[ci] += 1
            if pi == 1:
                pos_tp[ci] += 1
        else:
            neg_n[ci] += 1
            if pi == 1:
                neg_fp[ci] += 1

    categories = sorted(set(cats_test))
    rows = []
    for cat in categories:
        tp = pos_tp[cat]
        fn = pos_n[cat] - tp
        fp = neg_fp[cat]
        tn = neg_n[cat] - fp
        recall = tp / pos_n[cat] if pos_n[cat] else None
        fpr = fp / neg_n[cat] if neg_n[cat] else None
        rlo, rhi = wilson_ci(tp, pos_n[cat]) if pos_n[cat] else (None, None)
        flo, fhi = wilson_ci(fp, neg_n[cat]) if neg_n[cat] else (None, None)
        rows.append({
            "category": cat,
            "n_pos": pos_n[cat], "tp": tp, "fn": fn,
            "n_neg": neg_n[cat], "fp": fp, "tn": tn,
            "recall": recall, "recall_ci": [rlo, rhi],
            "fpr": fpr, "fpr_ci": [flo, fhi],
        })

    # Sort ascending by recall (None last). Positives-only categories show on top of weakest list.
    rows.sort(key=lambda r: (r["recall"] is None, r["recall"] if r["recall"] is not None else 1.0))

    print()
    header = f"{'category':28s}  {'n_pos':>5}  {'recall':>16}  {'n_neg':>5}  {'fpr':>16}"
    print(header)
    print("-" * len(header))
    for r in rows:
        rec_s = (
            f"{r['recall']:.1%} [{r['recall_ci'][0]:.0%},{r['recall_ci'][1]:.0%}]"
            if r["recall"] is not None else "  n/a"
        )
        fpr_s = (
            f"{r['fpr']:.1%} [{r['fpr_ci'][0]:.0%},{r['fpr_ci'][1]:.0%}]"
            if r["fpr"] is not None else "  n/a"
        )
        print(f"{r['category']:28s}  {r['n_pos']:>5}  {rec_s:>16}  {r['n_neg']:>5}  {fpr_s:>16}")

    out = {
        "bundle": args.bundle,
        "bundle_version": bundle.get("version"),
        "split_manifest": args.split_manifest,
        "threshold": T,
        "threshold_source": "argument" if args.threshold is not None else f"val_target_fpr_{args.target_fpr}",
        "n_test": int(len(y_test)),
        "n_test_pos": int((y_test == 1).sum()),
        "n_test_neg": int((y_test == 0).sum()),
        "categories": rows,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
