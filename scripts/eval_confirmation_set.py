# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score bundles on a confirmation set: files in, no split manifest.

A confirmation set exists to test a decision that was already made, so every
entry in it is held out by construction. That means it needs no train/val/test
manifest, and it should not get one. Building a manifest would reintroduce the
exact machinery that orphaned 2800 matched benigns for a whole build: a file on
disk, correctly labelled, assigned to no fold and therefore invisible in every
number computed afterwards. Files in, scores out, nothing to silently skip.

Thresholds are passed in rather than calibrated. Calibrating here would defeat
the point: the operating point is the thing under test, so it has to arrive
fixed, from the decision that this set was generated to confirm.

Usage:
  python scripts/eval_confirmation_set.py \
      --glob 'tests/adversarial/generated/*-v041-llama33-*.jsonl' \
      --glob 'tests/adversarial/benign_generated/BT-v041-*.jsonl' \
      --bundle src/vaara/data/adversarial_classifier_v9.joblib:0.9150 \
      --bundle bench/candidates/adversarial_classifier_v11_candidate.joblib:0.8800
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob as globmod
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import build_labels  # noqa: E402
from eval_v039_v9 import load_b, mx, sc, sha256_of  # noqa: E402


def load_files(patterns: list[str]) -> list[tuple[str, dict]]:
    """(source_file, entry) for every line, with the source kept.

    Every file that matches is reported, including the ones that contribute
    nothing, because a zero-line file is a generation that died and looks
    identical to one that was never started unless something says so.
    """
    out: list[tuple[str, dict]] = []
    seen_files = 0
    for pat in patterns:
        matched = sorted(globmod.glob(str(REPO / pat)))
        if not matched:
            print(f"[warn] no files match {pat!r}")
            continue
        for fp in matched:
            seen_files += 1
            n0 = len(out)
            with open(fp) as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        out.append((Path(fp).name, json.loads(line)))
                    except json.JSONDecodeError:
                        # A generator killed mid-write leaves a partial last
                        # line. Skipping it is right; being quiet about it is
                        # not, because a truncated file is also how a run fails.
                        print(f"[warn] {Path(fp).name}: undecodable line skipped")
            print(f"[load] {Path(fp).name}: {len(out) - n0} entries")
    print(f"[load] {seen_files} file(s), {len(out)} entries total")
    return out


#: Matched-benign files are BT-<tag>-<PREFIX>.jsonl. The ENTRIES inside carry
#: category "benign_control" regardless of what they were matched to, so
#: grouping on the entry alone collapses every negative into one bucket and the
#: per-category false-positive rate stays invisible. That is the same shape of
#: blindness that hid the orphaned benigns: the information exists, nothing reads
#: it. The filename is where the matching is recorded, so read it from there.
_PREFIX_CATEGORY = {
    "TM": "tool_misuse", "PE": "privilege_escalation", "DE": "data_exfil",
    "PI": "prompt_injection", "SR": "ssrf_via_tools",
    "DA": "destructive_actions", "CE": "credential_exfil",
}


def category_of(source_file: str, entry: dict) -> str:
    """Category for grouping, resolved from the filename for matched benigns."""
    cat = entry.get("category", "?")
    if cat != "benign_control":
        return cat
    parts = Path(source_file).stem.split("-")
    if parts and parts[0] == "BT" and len(parts) > 2:
        matched = _PREFIX_CATEGORY.get(parts[2])
        if matched:
            return matched
    # An unmatched benign belongs to no category and must not be silently
    # folded into one, or it inflates somebody's denominator.
    return "benign_unmatched"


def report(name: str, y: np.ndarray, scored: dict[str, np.ndarray],
           thresholds: dict[str, float]) -> dict:
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    row = {"n": int(len(y)), "pos": pos, "neg": neg, "models": {}}
    print(f"\n[{name}] n={len(y)} pos={pos} neg={neg}")
    if neg == 0:
        print("  [warn] no benign entries here, FPR is UNMEASURED at any threshold")
    for label, p in scored.items():
        m = mx(p, y, thresholds[label])
        row["models"][label] = m
        print(f"  {label:<24} recall={m['recall']:6.1%} "
              f"[{m['recall_ci'][0]:.1%},{m['recall_ci'][1]:.1%}] "
              f"FPR={m['fpr']:6.1%} @T={thresholds[label]:.4f}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True,
                    help="repo-relative glob; repeat for attacks and benigns")
    ap.add_argument("--bundle", action="append", required=True,
                    help="path:threshold, repeat per model")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    keyed = load_files(args.glob)
    if not keyed:
        print("[error] nothing loaded; generation may not have produced files yet",
              file=sys.stderr)
        return 2
    ents = [e for _, e in keyed]
    y = np.asarray(build_labels(ents)[0], dtype=np.int32)

    bundles, thresholds, prov = {}, {}, {}
    for spec in args.bundle:
        if ":" not in spec:
            print(f"[error] --bundle needs path:threshold, got {spec!r}", file=sys.stderr)
            return 2
        path, t = spec.rsplit(":", 1)
        label = Path(path).stem.replace("adversarial_classifier_", "")
        bundles[label] = load_b(str(REPO / path) if not Path(path).is_absolute() else path)
        thresholds[label] = float(t)
        prov[label] = {"path": path, "sha256": sha256_of(REPO / path), "threshold": float(t)}

    scored = {label: sc(b, ents) for label, b in bundles.items()}

    out = {"evaluated_at": dt.datetime.now(dt.UTC).isoformat(),
           "globs": args.glob, "bundles": prov, "surfaces": {}}

    out["surfaces"]["ALL"] = report("ALL", y, scored, thresholds)

    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, (src, e) in enumerate(keyed):
        by_cat[category_of(src, e)].append(i)
    for cat, idx in sorted(by_cat.items()):
        m = np.zeros(len(ents), bool)
        m[idx] = True
        out["surfaces"][cat] = report(cat, y[m],
                                      {k: v[m] for k, v in scored.items()},
                                      thresholds)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\n[out] {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
