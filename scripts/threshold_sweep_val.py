# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classifier threshold sweep on VAL fold for v0.31 item 7.

Reads the attribution JSON and sweeps classifier thresholds on the VAL
fold only (no TEST contact). Reports recall, FPR, and a few common
single-number summaries (F1, balanced accuracy, Youden's J) per threshold.
Picks the threshold that maximises Youden's J as the data-driven default.

A reviewer asking "why this threshold?" gets a Pareto curve + a named
selection rule. The picked value gets recorded back into the bundle by
re-running save_classifier_bundle.py with --threshold <picked>.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def file_sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else None


def is_malicious(r):
    exp = r["expected"]
    labels = set(exp if isinstance(exp, list) else [exp])
    return any(lbl in ("DENY", "ESCALATE") for lbl in labels)


def sweep_one(rows, th):
    tp = fn_c = fp = tn = 0
    for r in rows:
        s = r.get("classifier", {}).get("score")
        if s is None:
            continue
        y = is_malicious(r)
        yhat = s >= th
        if y and yhat: tp += 1
        elif y and not yhat: fn_c += 1
        elif not y and yhat: fp += 1
        else: tn += 1
    pos = tp + fn_c
    neg = fp + tn
    rec = tp / pos if pos else 0.0
    fpr = fp / neg if neg else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal = 0.5 * (rec + (1 - fpr))
    youden = rec - fpr
    return {
        "threshold": round(th, 4),
        "TP": tp, "FN": fn_c, "FP": fp, "TN": tn,
        "recall": round(rec, 4),
        "FPR": round(fpr, 4),
        "precision": round(prec, 4),
        "F1": round(f1, 4),
        "balanced_acc": round(bal, 4),
        "youden_J": round(youden, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution",
        default="tests/adversarial/v031/pipeline_rule_attribution_v031.json")
    ap.add_argument("--fold", default="val", choices=("val", "train"))
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--out",
        default="tests/adversarial/v031/threshold_sweep_val_v031.json")
    args = ap.parse_args()

    attr_path = Path(args.attribution)
    attr = json.loads(attr_path.read_text())
    rows = [r for r in attr["rows"] if r["fold"] == args.fold]
    print(f"[input] {attr_path}  fold={args.fold}  n={len(rows)}")

    thresholds = []
    t = 0.0
    while t <= 1.0 + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    sweep = [sweep_one(rows, th) for th in thresholds]

    best_youden = max(sweep, key=lambda r: r["youden_J"])
    best_f1 = max(sweep, key=lambda r: r["F1"])
    best_bal = max(sweep, key=lambda r: r["balanced_acc"])

    fpr_5 = min((r for r in sweep if r["FPR"] <= 0.05),
                key=lambda r: -r["recall"], default=None)
    fpr_10 = min((r for r in sweep if r["FPR"] <= 0.10),
                 key=lambda r: -r["recall"], default=None)

    print()
    print("[picks]")
    print(f"  max Youden's J   th={best_youden['threshold']:.2f}  "
          f"rec={best_youden['recall']:.3f}  FPR={best_youden['FPR']:.3f}  "
          f"J={best_youden['youden_J']:.3f}")
    print(f"  max F1           th={best_f1['threshold']:.2f}  "
          f"rec={best_f1['recall']:.3f}  FPR={best_f1['FPR']:.3f}  "
          f"F1={best_f1['F1']:.3f}")
    print(f"  max balanced acc th={best_bal['threshold']:.2f}  "
          f"rec={best_bal['recall']:.3f}  FPR={best_bal['FPR']:.3f}  "
          f"bal={best_bal['balanced_acc']:.3f}")
    if fpr_5:
        print(f"  max rec @ FPR<=5%   th={fpr_5['threshold']:.2f}  "
              f"rec={fpr_5['recall']:.3f}  FPR={fpr_5['FPR']:.3f}")
    if fpr_10:
        print(f"  max rec @ FPR<=10%  th={fpr_10['threshold']:.2f}  "
              f"rec={fpr_10['recall']:.3f}  FPR={fpr_10['FPR']:.3f}")

    out = {
        "metadata": {
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "attribution_path":
                str(attr_path.resolve().relative_to(REPO)),
            "attribution_sha256": file_sha256(attr_path),
            "fold": args.fold,
            "n_entries": len(rows),
            "step": args.step,
            "vaara_commit": attr["metadata"].get("vaara_commit"),
            "classifier_bundle_version":
                attr["metadata"].get("classifier_bundle_version"),
            "classifier_bundle_sha256":
                attr["metadata"].get("classifier_bundle_sha256"),
            "split_manifest_sha256":
                attr["metadata"].get("split_manifest_sha256"),
        },
        "picks": {
            "max_youden": best_youden,
            "max_f1": best_f1,
            "max_balanced_acc": best_bal,
            "max_recall_at_fpr_5pct": fpr_5,
            "max_recall_at_fpr_10pct": fpr_10,
        },
        "sweep": sweep,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
