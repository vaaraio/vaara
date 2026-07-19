# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wilson 95% intervals on v0.31 headline numbers.

The Wilson score interval is the conventional small-sample CI for a
binomial proportion. Reviewer-friendly because it stays inside [0, 1]
even at p=0 or p=1, unlike the normal approximation.

Run after eval_pipeline_attribution.py + three_way_variants.py have
produced their JSONs. Writes a single JSON with intervals on every
headline metric so the v0.31 release notes can cite (recall, lower,
upper) instead of bare point estimates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"k": k, "n": n, "p": 0.0, "lower": 0.0, "upper": 0.0}
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {
        "k": k, "n": n,
        "p": round(p, 4),
        "lower": round(max(0.0, (centre - half) / denom), 4),
        "upper": round(min(1.0, (centre + half) / denom), 4),
    }


def metric_block(m: dict) -> dict:
    pos = m["TP"] + m["FN"]
    neg = m["FP"] + m["TN"]
    return {
        "recall": wilson(m["TP"], pos),
        "FPR": wilson(m["FP"], neg),
        "n_total": m["n"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants",
        default="tests/adversarial/v031/test_final_eval_v031.json")
    ap.add_argument("--pair-n", type=int, default=25,
        help="Per-attacker PAIR seed count (default 25).")
    ap.add_argument("--pair-successes", type=int, default=0,
        help="Per-attacker successful jailbreak count (default 0).")
    ap.add_argument("--out",
        default="tests/adversarial/v031/wilson_intervals_v031.json")
    args = ap.parse_args()

    var_path = Path(args.variants)
    var = json.loads(var_path.read_text())
    fold = var["metadata"]["fold"]
    th = var["metadata"]["classifier_threshold"]

    headline_ci = {name: metric_block(m) for name, m in var["headline"].items()}
    print(f"[fold={fold} threshold={th}]")
    for name, b in headline_ci.items():
        r = b["recall"]; f = b["FPR"]
        print(f"  {name:18s}  rec={r['p']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}]"
              f"   FPR={f['p']:.3f} [{f['lower']:.3f}, {f['upper']:.3f}]"
              f"   n={b['n_total']}")

    pair_ci = {
        f"per_attacker_n_{args.pair_n}": wilson(args.pair_successes, args.pair_n),
    }
    print()
    print("[PAIR ASR (per-attacker)]")
    p = pair_ci[f"per_attacker_n_{args.pair_n}"]
    print(f"  successes={p['k']}/{p['n']}  p={p['p']:.3f}  "
          f"95% Wilson [{p['lower']:.3f}, {p['upper']:.3f}]")

    per_cat_ci = {}
    for cat, variants in var["per_category"].items():
        per_cat_ci[cat] = {name: metric_block(m) for name, m in variants.items()}
    per_src_ci = {}
    for src, variants in var["per_source"].items():
        per_src_ci[src] = {name: metric_block(m) for name, m in variants.items()}

    out = {
        "metadata": {
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "variants_path": str(var_path),
            "fold": fold,
            "classifier_threshold": th,
            "z": 1.96,
            "confidence": 0.95,
            "vaara_commit": var["metadata"].get("vaara_commit"),
        },
        "headline": headline_ci,
        "pair_asr": pair_ci,
        "per_category": per_cat_ci,
        "per_source": per_src_ci,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
