"""Train an XGBoost adversarial classifier and save a deployable joblib bundle.

Loads the corpus via train_adversarial_classifier.load_corpus(), fits on the
full corpus by default, and writes a bundle with provenance metadata to the
default bundle path under src/vaara/data/ (or a custom path).

Provenance fields written to the bundle:
    - model, vocab, feature_names, default_threshold, version
    - trained_at (ISO UTC), training_commit (git SHA at runtime)
    - n_entries, positive_rate
    - training_corpus_manifest_sha256 (SHA256 of tests/adversarial/MANIFEST.sha256
      if present; else None)

Backs up the existing bundle to <path>.<oldver>.bak before overwriting.

Usage:
    .venv/bin/python scripts/save_classifier_bundle.py \\
        --version v0.31 --threshold 0.55

    .venv/bin/python scripts/save_classifier_bundle.py \\
        --version v0.31 --threshold 0.70 \\
        --bundle-out src/vaara/data/adversarial_classifier_v2.joblib
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    load_corpus, build_labels, fit_vocabulary, build_features,
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

    entries = load_corpus()
    print(f"[corpus] {len(entries)} entries")
    y, _cats = build_labels(entries)
    y = np.array(y)
    print(f"[labels] positive rate={y.mean():.3f}")

    vocab = fit_vocabulary(entries)
    X, feat_names, _ = build_features(entries, vocab=vocab)
    print(f"[features] {X.shape}")

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.07,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, n_jobs=8, tree_method="hist",
    )
    model.fit(X, y)
    print("[fit] done")

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
    }
    joblib.dump(bundle, bundle_path)
    sz = bundle_path.stat().st_size
    print(f"[saved] {bundle_path}  size={sz} bytes")
    print(f"[meta]  version={args.version}  threshold={args.threshold}  "
          f"n_features={len(feat_names)}  commit={bundle['training_commit'][:8]}  "
          f"corpus_manifest_sha={(bundle['training_corpus_manifest_sha256'] or '<absent>')[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
