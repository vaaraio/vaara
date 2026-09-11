# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sweep one bundle's operating point across every eval surface at once.

eval_v039_v9.py picks a threshold to hit a target FPR on val and then reports
every surface at that single point. That answers "how does this model do at the
calibrated point" and not "is the calibrated point the right one", which is a
different question and the one that decides a ship.

This scores each surface once and re-thresholds in memory, so a full sweep costs
one embedding pass rather than one per threshold. Surfaces are the same four
eval_v039_v9.py uses, plus the inherited/added split of the test fold when the
manifest declares inherits_from.

Usage:
  python scripts/threshold_surface_sweep.py \
      --bundle bench/candidates/adversarial_classifier_v11_candidate.joblib \
      --split tests/adversarial/v040_split.json \
      --compare src/vaara/data/adversarial_classifier_v9.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    build_features, build_labels, load_corpus_keyed,
)
from eval_v039_v9 import (  # noqa: E402
    V038_FILES, fold_keyed, load_b, parent_keys, sc, v38,
)


def surfaces(split: Path):
    """(name, entries, labels) for every surface, test split by origin."""
    out = []
    for fold in ("val", "test", "holdout"):
        keyed = fold_keyed(fold, split)
        ents = [e for _, e in keyed]
        y = np.asarray(build_labels(ents)[0], dtype=np.int32)
        out.append((fold, ents, y))
        if fold != "test":
            continue
        pk = parent_keys(split)
        if pk is None:
            continue
        inh = np.array([k in pk for k, _ in keyed])
        for lbl, m in (("test:inherited", inh), ("test:added", ~inh)):
            if m.any():
                out.append((lbl, [e for e, keep in zip(ents, m) if keep], y[m]))
    v38e = v38()
    out.append(("v038_phase1", v38e,
                np.asarray(build_labels(v38e)[0], dtype=np.int32)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--compare", default=None,
                    help="held at its own stored threshold, printed as a ref")
    ap.add_argument("--lo", type=float, default=0.80)
    ap.add_argument("--hi", type=float, default=0.99)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    split = Path(args.split)
    b = load_b(args.bundle)
    surf = surfaces(split)
    scored = [(name, sc(b, ents), y) for name, ents, y in surf]

    ref = {}
    if args.compare:
        rb = load_b(args.compare)
        for name, ents, y in surf:
            p = sc(rb, ents)
            pred = p >= rb["T"]
            pos, neg = int((y == 1).sum()), int((y == 0).sum())
            ref[name] = (int((pred & (y == 1)).sum()) / max(pos, 1),
                         int((pred & (y == 0)).sum()) / max(neg, 1))
        print(f"[ref] {Path(args.compare).stem} at its stored T={rb['T']:.4f}")
        for name, _, _ in surf:
            print(f"  {name:<16} recall={ref[name][0]:6.1%} FPR={ref[name][1]:6.1%}")

    names = [n for n, _, _ in surf]
    print(f"\n[sweep] {Path(args.bundle).stem}")
    print("  " + "T".ljust(8) + "".join(n[:14].ljust(16) for n in names))
    rows = []
    for T in np.linspace(args.lo, args.hi, args.steps):
        cells, row = [], {"threshold": float(T)}
        for name, p, y in scored:
            pred = p >= T
            pos, neg = int((y == 1).sum()), int((y == 0).sum())
            r = int((pred & (y == 1)).sum()) / max(pos, 1)
            f = int((pred & (y == 0)).sum()) / max(neg, 1)
            row[name] = {"recall": r, "fpr": f}
            cells.append(f"{r:5.1%}/{f:5.1%}".ljust(16))
        rows.append(row)
        print(f"  {T:<8.4f}" + "".join(cells))
    print("\n  cells are recall/FPR")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"bundle": args.bundle, "split": args.split,
             "reference": {k: {"recall": v[0], "fpr": v[1]} for k, v in ref.items()},
             "sweep": rows}, indent=2))
        print(f"\n[out] {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
