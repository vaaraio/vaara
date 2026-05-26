"""Evaluate a classifier bundle on the v037 holdout fold.

Holdout composition (from v037_split.json): v036 Mixtral DE + v036 Claude DE
+ v037 Llama-3.3 TM/PE/DE. Three-leg breakdown (mixtral, claude, llama33).
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
    load_corpus_keyed,
    build_labels,
    build_features,
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def leg_of(key: str) -> str:
    if "mixtral" in key:
        return "mixtral"
    if "claude" in key:
        return "claude"
    if "llama33" in key:
        return "llama33"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--split-manifest", default="tests/adversarial/v037_split.json")
    ap.add_argument("--threshold", type=float, default=0.9006)
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
    print(f"[bundle] version={bundle.get('version')} n_features={len(feat_names)} "
          f"uses_embeddings={uses_embeddings}")

    assignments = json.loads(Path(args.split_manifest).read_text())["assignments"]
    keyed = load_corpus_keyed()
    holdout = [(k, e) for k, e in keyed if assignments.get(k) == "holdout"]
    print(f"[split] holdout entries: {len(holdout)}")
    if not holdout:
        print("[error] no entries matched assignments")
        return 2

    entries = [e for _, e in holdout]
    y, _ = build_labels(entries)
    y = np.asarray(y, dtype=np.int32)
    X, names, _ = build_features(entries, vocab=vocab, embeddings=uses_embeddings)
    if names != feat_names:
        raise ValueError(
            f"feature schema mismatch: bundle has {len(feat_names)}, "
            f"build_features produced {len(names)}"
        )
    prob = model.predict_proba(X)[:, 1].astype(np.float32)
    T = args.threshold
    pred = (prob >= T).astype(np.int32)

    pos = int((y == 1).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / max(tp + fn, 1)
    rlo, rhi = wilson_ci(tp, tp + fn)
    print(f"\n=== v037 held-out eval at T={T:.4f} ===")
    print(f"holdout n={len(y)} pos(DENY)={pos}")
    print(f"recall {recall:.1%} [{rlo:.1%}, {rhi:.1%}]  ({tp}/{tp + fn} DENY entries caught)")

    per_cat: dict[str, dict[str, int]] = {}
    per_leg: dict[str, dict[str, int]] = {}
    per_cat_leg: dict[str, dict[str, int]] = {}

    for (k, e), pr in zip(holdout, pred):
        cat = e.get("category", "?")
        leg = leg_of(k)
        cat_leg = f"{cat}__{leg}"
        for bucket, d in [(cat, per_cat), (leg, per_leg), (cat_leg, per_cat_leg)]:
            d.setdefault(bucket, {"n": 0, "tp": 0})
            d[bucket]["n"] += 1
            if pr == 1 and e.get("expected") == "DENY":
                d[bucket]["tp"] += 1

    def render(label: str, m: dict[str, dict[str, int]]):
        print(f"\n--- recall by {label} ---")
        for k in sorted(m):
            n, tp = m[k]["n"], m[k]["tp"]
            r = tp / max(n, 1)
            lo, hi = wilson_ci(tp, n)
            print(f"  {k:38s} n={n:4d} tp={tp:4d} recall={r:.1%} [{lo:.1%}, {hi:.1%}]")

    render("category", per_cat)
    render("leg", per_leg)
    render("category x leg", per_cat_leg)

    out = {
        "bundle": args.bundle,
        "bundle_version": bundle.get("version"),
        "threshold": T,
        "split_manifest": args.split_manifest,
        "n": len(y),
        "pos": pos,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "recall_ci": [rlo, rhi],
        "per_category": {k: {**v, "recall": v["tp"] / max(v["n"], 1)} for k, v in per_cat.items()},
        "per_leg": {k: {**v, "recall": v["tp"] / max(v["n"], 1)} for k, v in per_leg.items()},
        "per_category_per_leg": {
            k: {**v, "recall": v["tp"] / max(v["n"], 1)} for k, v in per_cat_leg.items()
        },
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\n[out] {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
