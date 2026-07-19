# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic stratified train/val/test split over the adversarial corpus.

Ratio:        70 / 15 / 15
Stratify by:  (category, source)
              source = "hand_curated" if the JSONL sits at the top of
              ``tests/adversarial/``, else "llm_generated"
Random state: 42 (Python ``random.Random``)
Key:          ``<relative_path>#L<line_index>`` because raw ``id`` is not
              unique across files (1,855 cross-file collisions, all
              between v0.31 LLM-generated regenerations and earlier sets)

Each stratum is shuffled with the seeded RNG and partitioned into
contiguous train/val/test slices. Counts per stratum: ``n_test`` and
``n_val`` use ``round(0.15 * n)`` with a floor of 1 when ``n >= 7``;
strata with ``n < 7`` raise an error so the split never silently drops a
class from val or test.

Why per-source stratification: keeps per-source recall computable on
the test set. A reviewer who asks "what is your recall on hand-curated
attacks vs LLM-generated attacks?" needs both subsets represented in
TEST. Without source stratification the small hand-curated subset (250
entries vs 7,705) can collapse into a noisy or zero-cell partition.

Output: ``tests/adversarial/v031_split.json`` with metadata, per-stratum
counts, and an ``assignments`` dict mapping every entry-key to one of
``train``, ``val``, ``test``.

Usage:
    .venv/bin/python scripts/build_train_val_test_split.py
    .venv/bin/python scripts/build_train_val_test_split.py --out custom.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import load_corpus_keyed  # noqa: E402

DEFAULT_OUT = REPO / "tests/adversarial/v031_split.json"
DEFAULT_MANIFEST = REPO / "tests/adversarial/MANIFEST.sha256"

RATIO_TRAIN = 0.70
RATIO_VAL = 0.15
RATIO_TEST = 0.15
RANDOM_STATE = 42
MIN_STRATUM_SIZE = 7  # ensures >=1 entry per fold under 0.15 rounding


def source_of(rel_path: str) -> str:
    """``hand_curated`` for top-level JSONLs, ``llm_generated`` otherwise."""
    return "hand_curated" if "/" not in rel_path else "llm_generated"


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_stratum(keys: list[str], rng: random.Random) -> dict[str, list[str]]:
    n = len(keys)
    if n < MIN_STRATUM_SIZE:
        raise ValueError(
            f"stratum has {n} entries, below MIN_STRATUM_SIZE={MIN_STRATUM_SIZE}; "
            f"a 70/15/15 split cannot guarantee one entry per fold"
        )
    shuffled = list(keys)
    rng.shuffle(shuffled)
    n_test = max(1, round(RATIO_TEST * n))
    n_val = max(1, round(RATIO_VAL * n))
    n_train = n - n_test - n_val
    if n_train < 1:
        raise ValueError(f"stratum split would produce {n_train} train entries (n={n})")
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSON path.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="Corpus integrity manifest hashed into split provenance.")
    args = ap.parse_args()

    keyed = load_corpus_keyed()
    if not keyed:
        print("[error] no entries loaded from tests/adversarial/", file=sys.stderr)
        return 2
    print(f"[corpus] {len(keyed)} entries")

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    entry_meta: dict[str, dict] = {}
    for key, e in keyed:
        rel = key.split("#L", 1)[0]
        src = source_of(rel)
        cat = e.get("category", "UNKNOWN")
        strata[(cat, src)].append(key)
        entry_meta[key] = {
            "id": e.get("id", ""),
            "category": cat,
            "source": src,
            "expected": e.get("expected", ""),
        }

    print(f"[strata] {len(strata)} non-empty (category, source) buckets")

    # Sorted strata + a fresh RNG per stratum keeps the split insensitive
    # to corpus growth order. Adding a new file later only perturbs the
    # stratum it joins.
    assignments: dict[str, str] = {}
    per_stratum: dict[str, dict] = {}
    for (cat, src) in sorted(strata.keys()):
        keys = sorted(strata[(cat, src)])
        rng = random.Random(f"{RANDOM_STATE}::{cat}::{src}")
        folds = split_stratum(keys, rng)
        for fold_name, fold_keys in folds.items():
            for k in fold_keys:
                assignments[k] = fold_name
        stratum_key = f"{cat}__{src}"
        per_stratum[stratum_key] = {
            "total": len(keys),
            "train": len(folds["train"]),
            "val": len(folds["val"]),
            "test": len(folds["test"]),
        }
        print(
            f"  {stratum_key:48s} n={len(keys):5d}  "
            f"train={len(folds['train']):5d}  val={len(folds['val']):4d}  "
            f"test={len(folds['test']):4d}"
        )

    counts = {
        "train": sum(1 for v in assignments.values() if v == "train"),
        "val": sum(1 for v in assignments.values() if v == "val"),
        "test": sum(1 for v in assignments.values() if v == "test"),
    }
    counts["total"] = sum(counts.values())
    assert counts["total"] == len(keyed), \
        f"assignment count {counts['total']} != corpus {len(keyed)}"

    manifest = {
        "metadata": {
            "version": "v0.31",
            "ratios": {"train": RATIO_TRAIN, "val": RATIO_VAL, "test": RATIO_TEST},
            "stratify_by": ["category", "source"],
            "random_state": RANDOM_STATE,
            "rng_seed_format": "f'{random_state}::{category}::{source}'",
            "key_format": "<relative_path>#L<line_index>",
            "min_stratum_size": MIN_STRATUM_SIZE,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "corpus_manifest_path": str(Path(args.manifest).relative_to(REPO))
                if Path(args.manifest).exists() else None,
            "corpus_manifest_sha256": file_sha256(Path(args.manifest)),
        },
        "counts": counts,
        "strata": per_stratum,
        "assignments": dict(sorted(assignments.items())),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sz = out_path.stat().st_size
    print(
        f"\n[totals] train={counts['train']}  val={counts['val']}  "
        f"test={counts['test']}  total={counts['total']}"
    )
    print(f"[saved] {out_path}  size={sz} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
