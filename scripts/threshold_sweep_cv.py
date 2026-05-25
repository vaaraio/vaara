"""5-fold cross-validated threshold sweep on the adversarial corpus.

For each fold, fits XGBoost on the training partition, collects out-of-fold
(OOF) predicted probabilities on the held-out partition. After all folds, sweeps
candidate thresholds against the union of OOF predictions and reports
balanced_acc / attack_recall / benign_FPR / jailbreak_recall / benign_control_acc.

Use this to pick a defensible default_threshold for the classifier bundle: the
sweep is on held-out predictions, so the chosen threshold is not overfit to the
training data.

Output JSON includes raw OOF probabilities and labels so a downstream
confidence-interval analysis (bootstrap, etc.) can use them.

Usage:
    .venv/bin/python scripts/threshold_sweep_cv.py \\
        --thresholds 0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \\
        --out tests/adversarial/v031/threshold_sweep_cv.json \\
        --random-seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    load_corpus, build_labels, fit_vocabulary, build_features,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
                    help="Comma-separated candidate thresholds.")
    ap.add_argument("--out", default=str(REPO / "tests/adversarial/v031/threshold_sweep_cv.json"))
    ap.add_argument("--random-seed", type=int, default=42,
                    help="Seed for both StratifiedKFold shuffle and XGBoost random_state.")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb

    entries = load_corpus()
    print(f"[corpus] {len(entries)} entries")
    y, cats = build_labels(entries)
    y = np.array(y)
    cats = np.array(cats)

    oof = np.zeros(len(entries), dtype=float)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        tr = [entries[i] for i in tr_idx]
        te = [entries[i] for i in te_idx]
        vocab = fit_vocabulary(tr)
        X_tr, _, _ = build_features(tr, vocab=vocab)
        X_te, _, _ = build_features(te, vocab=vocab)
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=args.random_seed, n_jobs=8, tree_method="hist",
        )
        model.fit(X_tr, y[tr_idx])
        oof[te_idx] = model.predict_proba(X_te)[:, 1]
        print(f"[fold {fold}] done")

    attack_mask = (y == 1)
    benign_mask = (y == 0)
    benign_control_mask = (cats == "benign_control")
    jailbreak_mask = (cats == "jailbreak")

    thresholds = [float(x) for x in args.thresholds.split(",")]
    rows = []
    print()
    print(f"{'th':>5} {'bal_acc':>8} {'attack_recall':>14} {'benign_FPR':>11} "
          f"{'jb_recall':>10} {'bc_acc':>7}")
    print("-" * 60)
    for th in thresholds:
        pred_mal = (oof >= th)
        attack_recall = float(pred_mal[attack_mask].mean()) if attack_mask.any() else 0.0
        benign_fpr = float(pred_mal[benign_mask].mean()) if benign_mask.any() else 0.0
        bal_acc = 0.5 * (attack_recall + (1 - benign_fpr))
        jb_recall = float(pred_mal[jailbreak_mask].mean()) if jailbreak_mask.any() else 0.0
        bc_acc = float((~pred_mal[benign_control_mask]).mean()) if benign_control_mask.any() else 0.0
        print(f"{th:>5.2f} {bal_acc:>8.3f} {attack_recall:>14.3f} {benign_fpr:>11.3f} "
              f"{jb_recall:>10.3f} {bc_acc:>7.3f}")
        rows.append({
            "threshold": th, "balanced_acc": bal_acc,
            "attack_recall": attack_recall, "benign_fpr": benign_fpr,
            "jailbreak_recall": jb_recall, "benign_control_acc": bc_acc,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "random_seed": args.random_seed,
        "n_splits": args.n_splits,
        "oof_prob": oof.tolist(),
        "labels": y.tolist(),
        "categories": cats.tolist(),
        "sweep": rows,
    }, indent=2))
    print(f"\n[json] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
