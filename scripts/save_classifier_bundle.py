"""Train an XGBoost adversarial classifier and save a deployable joblib bundle.

Loads the corpus via ``train_adversarial_classifier.load_corpus_keyed``,
optionally restricts to the TRAIN fold of a split manifest, and writes a
bundle with provenance metadata to the default bundle path under
``src/vaara/data/`` (or a custom path).

Provenance fields written to the bundle:
    - model, vocab, feature_names, default_threshold, version
    - trained_at (ISO UTC), training_commit (git SHA at runtime)
    - n_entries, positive_rate
    - training_corpus_manifest_sha256 (SHA256 of tests/adversarial/MANIFEST.sha256
      if present; else None)
    - split_manifest_path, split_manifest_sha256, training_fold
      (recorded only when ``--split-manifest`` is passed)

Backs up the existing bundle to ``<path>.<oldver>.bak`` before overwriting.

Usage:
    .venv/bin/python scripts/save_classifier_bundle.py \\
        --version v0.31 --threshold 0.55

    .venv/bin/python scripts/save_classifier_bundle.py \\
        --version v0.31 --threshold 0.5 \\
        --split-manifest tests/adversarial/v031_split.json \\
        --bundle-out src/vaara/data/adversarial_classifier_v2.joblib
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    load_corpus_keyed, build_labels, fit_vocabulary, build_features,
)

DEFAULT_BUNDLE_PATH = REPO / "src/vaara/data/adversarial_classifier_v1.joblib"
DEFAULT_MANIFEST = REPO / "tests/adversarial/MANIFEST.sha256"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def manifest_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help='Bundle version string, e.g. "v0.31".')
    ap.add_argument("--threshold", type=float, required=True,
                    help="Default decision threshold to embed in the bundle.")
    ap.add_argument("--bundle-out", default=str(DEFAULT_BUNDLE_PATH),
                    help="Output path. Existing file is backed up to <path>.<old_version>.bak.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="Path to corpus MANIFEST.sha256 (hashed and recorded for provenance).")
    ap.add_argument("--split-manifest", default=None,
                    help="If set, train only on entries whose fold == --train-fold in this split JSON "
                         "(produced by scripts/build_train_val_test_split.py).")
    ap.add_argument("--train-fold", default="train", choices=("train", "val", "test"),
                    help='Which fold of the split manifest to train on (default: "train"). '
                         'Use anything other than "train" only deliberately; leaking val/test '
                         'into training invalidates the held-out metrics.')
    ap.add_argument("--embeddings", action="store_true",
                    help="Concatenate 384-dim MiniLM embeddings to the hand-features (v0.32+).")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.07)
    ap.add_argument("--min-child-weight", type=int, default=1)
    args = ap.parse_args()

    import numpy as np
    import xgboost as xgb
    import joblib

    bundle_path = Path(args.bundle_out)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    if bundle_path.exists():
        try:
            existing = joblib.load(bundle_path)
            old_ver = existing.get("version", "unknown")
        except Exception:
            old_ver = "unknown"
        backup_path = bundle_path.with_suffix(f".joblib.{old_ver}.bak")
        if not backup_path.exists():
            shutil.copy2(bundle_path, backup_path)
            print(f"[backup] {backup_path}")

    keyed = load_corpus_keyed()
    print(f"[corpus] {len(keyed)} entries loaded")

    split_manifest_sha: str | None = None
    split_manifest_rel: str | None = None
    if args.split_manifest:
        split_path = Path(args.split_manifest)
        if not split_path.exists():
            print(f"[error] split manifest not found: {split_path}", file=sys.stderr)
            return 2
        split_manifest_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
        try:
            split_manifest_rel = str(split_path.resolve().relative_to(REPO))
        except ValueError:
            split_manifest_rel = str(split_path)
        assignments = json.loads(split_path.read_text())["assignments"]
        before = len(keyed)
        keyed = [(k, e) for k, e in keyed if assignments.get(k) == args.train_fold]
        print(
            f"[split] {split_manifest_rel}  sha={split_manifest_sha[:12]}  "
            f"fold={args.train_fold}  kept={len(keyed)}/{before}"
        )
        if not keyed:
            print(f"[error] no entries matched fold={args.train_fold} in split manifest",
                  file=sys.stderr)
            return 2

    entries = [e for _, e in keyed]
    y, _cats = build_labels(entries)
    y = np.array(y)
    print(f"[labels] n={len(entries)} positive_rate={y.mean():.3f}")

    vocab = fit_vocabulary(entries)
    X, feat_names, _ = build_features(entries, vocab=vocab, embeddings=args.embeddings)
    print(f"[features] {X.shape}  embeddings={args.embeddings}")

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, min_child_weight=args.min_child_weight,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, n_jobs=8, tree_method="hist",
    )
    model.fit(X, y)
    print(f"[fit] done  ne={args.n_estimators} md={args.max_depth} "
          f"lr={args.learning_rate} mc={args.min_child_weight}")

    embedding_model_id = None
    embedding_model_revision = None
    if args.embeddings:
        from vaara.embeddings import EMBED_MODEL_ID, EMBED_MODEL_REVISION
        embedding_model_id = EMBED_MODEL_ID
        embedding_model_revision = EMBED_MODEL_REVISION

    bundle = {
        "model": model,
        "vocab": vocab,
        "default_threshold": args.threshold,
        "version": args.version,
        "feature_names": feat_names,
        "trained_at": dt.datetime.now(dt.UTC).isoformat(),
        "training_commit": git_commit(),
        "n_entries": int(len(entries)),
        "positive_rate": float(y.mean()),
        "training_corpus_manifest_sha256": manifest_sha256(Path(args.manifest)),
        "split_manifest_path": split_manifest_rel,
        "split_manifest_sha256": split_manifest_sha,
        "training_fold": args.train_fold if args.split_manifest else None,
        "uses_embeddings": bool(args.embeddings),
        "embedding_model_id": embedding_model_id,
        "embedding_model_revision": embedding_model_revision,
        "hparams": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_child_weight": args.min_child_weight,
        },
    }
    joblib.dump(bundle, bundle_path)
    sz = bundle_path.stat().st_size
    print(f"[saved] {bundle_path}  size={sz} bytes")
    print(f"[meta]  version={args.version}  threshold={args.threshold}  "
          f"n_features={len(feat_names)}  commit={bundle['training_commit'][:8]}  "
          f"corpus_manifest_sha={(bundle['training_corpus_manifest_sha256'] or '<absent>')[:12]}")
    if args.embeddings:
        print(f"[embed] {bundle['embedding_model_id']} "
              f"revision={(bundle['embedding_model_revision'] or '<absent>')[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
