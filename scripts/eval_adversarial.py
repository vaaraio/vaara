#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evaluate Vaara scorer against the adversarial corpus.

Usage:
    python scripts/eval_adversarial.py
    python scripts/eval_adversarial.py --corpus-dir tests/adversarial
    python scripts/eval_adversarial.py --only-category credential_exfil

Reports per-category:
    - n entries
    - true_positive_rate (model correctly DENY/ESCALATE when expected is DENY/ESCALATE)
    - escalation_agreement (model ESCALATE aligning with expected ESCALATE)
    - allow_leakage (model ALLOW for an attack that should have been DENY/ESCALATE)
    - mean risk score
    - coverage (fraction of entries where the conformal interval contains the
      ground-truth risk label; should track 1 - alpha for a calibrated scorer)
    - mean_interval_width (average width of the conformal prediction interval)
Writes results to tests/adversarial/results_<UTC>.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

from vaara import Pipeline


def load_corpus(corpus_dir: Path, only_category: str | None) -> list[dict]:
    entries: list[dict] = []
    files = sorted(corpus_dir.glob("*.jsonl"))
    for fp in files:
        cat = fp.stem
        if only_category and cat != only_category:
            continue
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    return entries


def _actual_risk_from_expected(expected_set: set[str]) -> float:
    """Map the corpus's discrete expected-decision label to the [0, 1] risk axis.

    The scorer's conformal interval lives on the [0, 1] risk axis, so to check
    coverage we need a numeric ground-truth on that same axis. Attack categories
    (expected DENY or ESCALATE) carry true_risk = 1.0; benign control entries
    (expected ALLOW only) carry true_risk = 0.0.
    """
    if "DENY" in expected_set or "ESCALATE" in expected_set:
        return 1.0
    return 0.0


def _interval_from_result(result) -> tuple[float, float]:
    """Pull a (lower, upper) conformal interval off a Pipeline result.

    Falls back to the trivial [0.0, 1.0] interval when the field is missing or
    malformed so coverage arithmetic stays well-defined on error paths.
    """
    interval = getattr(result, "risk_interval", None)
    if interval is None:
        return (0.0, 1.0)
    try:
        lower = float(interval[0])
        upper = float(interval[1])
    except (TypeError, ValueError, IndexError):
        return (0.0, 1.0)
    if lower > upper:
        lower, upper = upper, lower
    return (max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper)))


def decide(entry: dict, pipe: Pipeline, classifier=None) -> dict:
    """Run heuristic pipeline; optionally stack with AdversarialClassifier.

    Stacking rule (per v0.5.2 CHANGELOG recommendation):
        - heuristic DENY  to keep DENY
        - heuristic ESCALATE to keep ESCALATE
        - heuristic ALLOW + classifier prob ≥ threshold to ESCALATE
        - heuristic ALLOW + classifier prob <  threshold to ALLOW

    The conformal interval reported here is the pipeline's, not the classifier's.
    The classifier can override the action but does not produce a calibrated
    interval of its own, so coverage is always measured against the pipeline.
    """
    result = pipe.intercept(
        agent_id=entry.get("agent_id", "adv"),
        tool_name=entry["tool_name"],
        parameters=entry.get("parameters", {}),
        context=entry.get("context", {}),
    )
    decision_str = getattr(result, "decision", None)
    decision_str = str(decision_str).upper() if decision_str is not None else "UNKNOWN"
    risk = float(getattr(result, "risk_score", 0.0) or 0.0)
    lower, upper = _interval_from_result(result)

    if classifier is not None and decision_str == "ALLOW":
        prob = classifier.score(
            tool_name=entry["tool_name"],
            parameters=entry.get("parameters", {}),
            context=entry.get("context", {}),
        )
        if prob >= classifier.threshold:
            decision_str = "ESCALATE"
            risk = max(risk, prob)
    return {"decision": decision_str, "risk": risk, "lower": lower, "upper": upper}


def summarize(rows: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    summary = {}
    for cat, items in by_cat.items():
        n = len(items)
        denied = sum(1 for x in items if x["actual"] == "DENY")
        escalated = sum(1 for x in items if x["actual"] == "ESCALATE")
        allowed = sum(1 for x in items if x["actual"] == "ALLOW")
        allow_leakage = sum(1 for x in items if x["actual"] == "ALLOW" and ("DENY" in x["expected_set"] or "ESCALATE" in x["expected_set"]))
        mean_risk = round(sum(x["risk"] for x in items) / max(n, 1), 4)
        covered = sum(
            1 for x in items
            if x["lower"] <= x["actual_risk"] <= x["upper"]
        )
        mean_width = sum(x["upper"] - x["lower"] for x in items) / max(n, 1)
        summary[cat] = {
            "n": n,
            "denied": denied,
            "escalated": escalated,
            "allowed": allowed,
            "allow_leakage": allow_leakage,
            "allow_leakage_rate": round(allow_leakage / max(n, 1), 4),
            "mean_risk": mean_risk,
            "coverage": round(covered / max(n, 1), 4),
            "mean_interval_width": round(mean_width, 4),
        }
    return summary


def overall(rows: list[dict]) -> dict:
    """Aggregate coverage and mean width across all categories.

    Per-category coverage can hide imbalance (a 90% marginal headline can sit on
    top of 60% on credential_exfil and 99% on benign_control), so the overall
    block stays a single line and the per-category table is where regressions
    are read.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0, "coverage": 0.0, "mean_interval_width": 0.0}
    covered = sum(1 for x in rows if x["lower"] <= x["actual_risk"] <= x["upper"])
    width = sum(x["upper"] - x["lower"] for x in rows) / n
    return {
        "n": n,
        "coverage": round(covered / n, 4),
        "mean_interval_width": round(width, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="tests/adversarial")
    ap.add_argument("--only-category", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--with-classifier",
        action="store_true",
        help="Stack AdversarialClassifier on top of the heuristic pipeline. "
             "Requires vaara[ml] extras. Heuristic DENY/ESCALATE are preserved; "
             "heuristic ALLOW is upgraded to ESCALATE when classifier prob >= threshold.",
    )
    ap.add_argument(
        "--mondrian",
        action="store_true",
        help="Run the AdaptiveScorer in Mondrian (class-conditional) conformal "
             "mode so coverage holds per action category instead of only "
             "marginally. Run with and without to compare per-category coverage "
             "and surface class-conditional miscoverage.",
    )
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    entries = load_corpus(corpus_dir, args.only_category)
    if not entries:
        raise SystemExit(f"No entries found under {corpus_dir}")
    print(f"[corpus] {len(entries)} entries across {len({e['category'] for e in entries})} categories")

    if args.mondrian:
        from vaara.scorer.adaptive import AdaptiveScorer
        pipe = Pipeline(scorer=AdaptiveScorer(mondrian_categories=True))
        print("[scorer] AdaptiveScorer in Mondrian (class-conditional) mode")
    else:
        pipe = Pipeline()
        print("[scorer] AdaptiveScorer in marginal mode (default)")
    classifier = None
    if args.with_classifier:
        try:
            from vaara.adversarial_classifier import AdversarialClassifier
            classifier = AdversarialClassifier()
            print(f"[classifier] engaged: bundle v{classifier.bundle_version}, threshold={classifier.threshold}")
        except ImportError as exc:
            raise SystemExit(f"--with-classifier needs vaara[ml]: {exc}")
    else:
        print("[classifier] disabled (heuristic only). Pass --with-classifier for stacked eval.")
    rows = []
    for entry in entries:
        try:
            out = decide(entry, pipe, classifier=classifier)
        except Exception as exc:
            out = {
                "decision": f"ERROR:{type(exc).__name__}",
                "risk": 0.0,
                "lower": 0.0,
                "upper": 1.0,
            }
        expected = entry.get("expected", "DENY")
        expected_set = set(expected) if isinstance(expected, list) else {expected}
        rows.append({
            "id": entry["id"],
            "category": entry["category"],
            "tool_name": entry["tool_name"],
            "expected_set": expected_set,
            "actual": out["decision"],
            "risk": out["risk"],
            "lower": out["lower"],
            "upper": out["upper"],
            "actual_risk": _actual_risk_from_expected(expected_set),
        })
    summary = summarize(rows)
    overall_block = overall(rows)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else corpus_dir / f"results_{timestamp}.json"
    out_path.write_text(json.dumps({
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "n_entries": len(entries),
        "overall": overall_block,
        "summary": summary,
        "rows": [{**r, "expected_set": sorted(r["expected_set"])} for r in rows],
    }, indent=2))
    print(f"[results] {out_path}")
    print("\n=== Per-category summary ===")
    for cat, s in sorted(summary.items()):
        leak_pct = s["allow_leakage_rate"] * 100
        cov_pct = s["coverage"] * 100
        print(
            f"  {cat:24s} n={s['n']:3d} deny={s['denied']:3d} esc={s['escalated']:3d} "
            f"allow={s['allowed']:3d} allow_leakage={leak_pct:5.1f}% "
            f"mean_risk={s['mean_risk']:.3f} coverage={cov_pct:5.1f}% "
            f"width={s['mean_interval_width']:.3f}"
        )
    print(
        f"\n[overall] n={overall_block['n']} "
        f"coverage={overall_block['coverage'] * 100:5.1f}% "
        f"width={overall_block['mean_interval_width']:.3f}"
    )


if __name__ == "__main__":
    main()
