"""Three-way variant comparison for v0.31 item 6.

Reads the attribution JSON produced by eval_pipeline_attribution.py and
computes recall / FPR / FNR / coverage for three predictor variants on
the VAL fold (by default):

    rules-only       intercept decision in {deny, escalate}
    classifier-only  classifier score >= --classifier-threshold
    both             rules-only OR classifier-only

Output: tests/adversarial/v031/three_way_variants_v031.json with per-
variant headline metrics, per-category breakdown, per-source breakdown,
and provenance pointers back to the attribution JSON.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def file_sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else None


def is_malicious_truth(r):
    exp = r["expected"]
    labels = set(exp if isinstance(exp, list) else [exp])
    return any(l in ("DENY", "ESCALATE") for l in labels)


def predict_rules(r):
    dec = r.get("intercept", {}).get("decision")
    return dec in ("deny", "escalate")


def predict_classifier(r, threshold):
    c = r.get("classifier", {})
    if "score" not in c:
        return False
    return c["score"] >= threshold


def metrics(rows, predict_fn):
    tp = fn_c = fp = tn = 0
    for r in rows:
        y = is_malicious_truth(r)
        yhat = predict_fn(r)
        if y and yhat: tp += 1
        elif y and not yhat: fn_c += 1
        elif not y and yhat: fp += 1
        else: tn += 1
    n = tp + fn_c + fp + tn
    rec = tp / (tp + fn_c) if (tp + fn_c) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = 1 - rec if (tp + fn_c) else 0.0
    return {"n": n, "TP": tp, "FN": fn_c, "FP": fp, "TN": tn,
            "recall": round(rec, 4), "FPR": round(fpr, 4),
            "FNR": round(fnr, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution",
        default="tests/adversarial/v031/pipeline_rule_attribution_v031.json")
    ap.add_argument("--fold", default="val",
                    choices=("train", "val", "test", "all"))
    ap.add_argument("--classifier-threshold", type=float, default=0.5)
    ap.add_argument("--out",
        default="tests/adversarial/v031/three_way_variants_v031.json")
    args = ap.parse_args()

    attr_path = Path(args.attribution)
    attr = json.loads(attr_path.read_text())
    rows = attr["rows"]
    if args.fold != "all":
        rows = [r for r in rows if r["fold"] == args.fold]
    print(f"[input] {attr_path}  fold={args.fold}  n={len(rows)}")

    th = args.classifier_threshold
    variants = {
        "rules_only": predict_rules,
        "classifier_only": lambda r: predict_classifier(r, th),
        "both": lambda r: predict_rules(r) or predict_classifier(r, th),
    }

    headline = {name: metrics(rows, fn) for name, fn in variants.items()}
    print("[headline]")
    for k, v in headline.items():
        print(f"  {k:18s} rec={v['recall']:.3f} FPR={v['FPR']:.3f}  "
              f"TP={v['TP']:4d} FN={v['FN']:4d} FP={v['FP']:4d} TN={v['TN']:4d}")

    per_cat = defaultdict(dict)
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        for name, fn in variants.items():
            per_cat[cat][name] = metrics(sub, fn)

    per_src = defaultdict(dict)
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        for name, fn in variants.items():
            per_src[src][name] = metrics(sub, fn)

    out = {
        "metadata": {
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "attribution_path":
                str(attr_path.resolve().relative_to(REPO)),
            "attribution_sha256": file_sha256(attr_path),
            "fold": args.fold,
            "n_entries": len(rows),
            "classifier_threshold": th,
            "vaara_commit": attr["metadata"].get("vaara_commit"),
            "classifier_bundle_version":
                attr["metadata"].get("classifier_bundle_version"),
            "classifier_bundle_sha256":
                attr["metadata"].get("classifier_bundle_sha256"),
            "split_manifest_sha256":
                attr["metadata"].get("split_manifest_sha256"),
        },
        "headline": headline,
        "per_category": per_cat,
        "per_source": per_src,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
