"""Retrain v9 with BIPIA follows upweighted via XGBoost sample_weight.

Reuses build_features/build_labels/load_corpus_keyed from
train_adversarial_classifier so feature schema matches v8. Loads the
v039 split, filters to TRAIN entries, builds X/y, and applies a per-row
weight: 1.0 default, --follow-weight for entries where the v039 BIPIA
metadata flags `reflects_injection`. Trains the same XGBoost
hyperparameters v8 shipped with and saves a v9 bundle.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    build_features, build_labels, fit_vocabulary, load_corpus_keyed,
)

V039_SPLIT = REPO / "tests/adversarial/v039_split.json"
DEFAULT_OUT = REPO / "src/vaara/data/adversarial_classifier_v9.joblib"
MANIFEST = REPO / "tests/adversarial/MANIFEST.sha256"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow-weight", type=float, default=8.0)
    ap.add_argument("--max-bipia-benign", type=int, default=0,
                    help="If >0, subsample BIPIA benign-under-pressure train rows to "
                         "this many (seed=43). Default 0 = keep all.")
    ap.add_argument("--threshold", type=float, default=0.9006)
    ap.add_argument("--bundle-out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.07)
    args = ap.parse_args()

    import joblib
    import xgboost as xgb

    assignments = json.loads(V039_SPLIT.read_text())["assignments"]
    keyed = load_corpus_keyed()
    train_keyed = [(k, e) for k, e in keyed if assignments.get(k) == "train"]
    if not train_keyed:
        print("[error] no train entries")
        return 2

    if args.max_bipia_benign > 0:
        import random
        keep_idx: list[int] = []
        bipia_benign_idx: list[int] = []
        for i, (_, e) in enumerate(train_keyed):
            meta = e.get("_v039_bipia")
            if meta and not meta.get("reflects_injection"):
                bipia_benign_idx.append(i)
            else:
                keep_idx.append(i)
        rng = random.Random(43)
        rng.shuffle(bipia_benign_idx)
        keep_idx.extend(bipia_benign_idx[: args.max_bipia_benign])
        keep_idx.sort()
        before = len(train_keyed)
        train_keyed = [train_keyed[i] for i in keep_idx]
        print(f"[subsample] BIPIA benign-under-pressure {len(bipia_benign_idx)} -> "
              f"{min(args.max_bipia_benign, len(bipia_benign_idx))}; "
              f"train fold {before} -> {len(train_keyed)}")

    entries = [e for _, e in train_keyed]
    y, _ = build_labels(entries)
    y = np.asarray(y, dtype=np.int32)
    print(f"[corpus] {len(keyed)} entries; train fold n={len(entries)} pos_rate={y.mean():.3f}")

    # Per-row weights: upweight rows whose v039 BIPIA metadata flags follow.
    w = np.ones(len(entries), dtype=np.float32)
    n_follow_train = 0
    for i, e in enumerate(entries):
        meta = e.get("_v039_bipia")
        if meta and meta.get("reflects_injection"):
            w[i] = args.follow_weight
            n_follow_train += 1
    print(f"[weights] {n_follow_train} BIPIA train follows upweighted {args.follow_weight:.1f}x; "
          f"effective n_eff={w.sum():.1f}")

    vocab = fit_vocabulary(entries)
    X, feat_names, _ = build_features(entries, vocab=vocab, embeddings=True)
    print(f"[features] {X.shape}")

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, min_child_weight=1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, n_jobs=8, tree_method="hist",
    )
    model.fit(X, y, sample_weight=w)
    print(f"[fit] done  ne={args.n_estimators} md={args.max_depth} lr={args.learning_rate}")

    from vaara.embeddings import EMBED_MODEL_ID, EMBED_MODEL_REVISION

    split_sha = hashlib.sha256(V039_SPLIT.read_bytes()).hexdigest()
    manifest_sha = (
        hashlib.sha256(MANIFEST.read_bytes()).hexdigest() if MANIFEST.exists() else None
    )

    bundle = {
        "model": model, "vocab": vocab, "default_threshold": args.threshold,
        "version": "v0.39", "feature_names": feat_names,
        "trained_at": dt.datetime.now(dt.UTC).isoformat(),
        "training_commit": git_commit(),
        "n_entries": int(len(entries)), "positive_rate": float(y.mean()),
        "training_corpus_manifest_sha256": manifest_sha,
        "split_manifest_path": "tests/adversarial/v039_split.json",
        "split_manifest_sha256": split_sha, "training_fold": "train",
        "uses_embeddings": True, "embedding_model_id": EMBED_MODEL_ID,
        "embedding_model_revision": EMBED_MODEL_REVISION,
        "hparams": {
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "learning_rate": args.learning_rate, "min_child_weight": 1,
            "follow_weight": args.follow_weight,
        },
        "training_notes": (
            f"BIPIA follows ({n_follow_train} entries) upweighted "
            f"{args.follow_weight:.1f}x via sample_weight."
        ),
    }
    out_path = Path(args.bundle_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"[saved] {out_path.relative_to(REPO)}  size={out_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
