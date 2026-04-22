#!/usr/bin/env python3
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


def decide(entry: dict, pipe: Pipeline) -> dict:
    result = pipe.intercept(
        agent_id=entry.get("agent_id", "adv"),
        tool_name=entry["tool_name"],
        parameters=entry.get("parameters", {}),
        context=entry.get("context", {}),
    )
    # Pipeline decision object: .decision enum-like + .risk float
    decision_str = getattr(result, "decision", None)
    decision_str = str(decision_str).upper() if decision_str is not None else "UNKNOWN"
    risk = float(getattr(result, "risk_score", 0.0) or 0.0)
    return {"decision": decision_str, "risk": risk}


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
        summary[cat] = {
            "n": n,
            "denied": denied,
            "escalated": escalated,
            "allowed": allowed,
            "allow_leakage": allow_leakage,
            "allow_leakage_rate": round(allow_leakage / max(n, 1), 4),
            "mean_risk": mean_risk,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="tests/adversarial")
    ap.add_argument("--only-category", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    entries = load_corpus(corpus_dir, args.only_category)
    if not entries:
        raise SystemExit(f"No entries found under {corpus_dir}")
    print(f"[corpus] {len(entries)} entries across {len({e['category'] for e in entries})} categories")

    pipe = Pipeline()
    rows = []
    for entry in entries:
        try:
            out = decide(entry, pipe)
        except Exception as exc:
            out = {"decision": f"ERROR:{type(exc).__name__}", "risk": 0.0}
        expected = entry.get("expected", "DENY")
        expected_set = set(expected) if isinstance(expected, list) else {expected}
        rows.append({
            "id": entry["id"],
            "category": entry["category"],
            "tool_name": entry["tool_name"],
            "expected_set": expected_set,
            "actual": out["decision"],
            "risk": out["risk"],
        })
    summary = summarize(rows)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else corpus_dir / f"results_{timestamp}.json"
    out_path.write_text(json.dumps({
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "n_entries": len(entries),
        "summary": summary,
        "rows": [{**r, "expected_set": sorted(r["expected_set"])} for r in rows],
    }, indent=2))
    print(f"[results] {out_path}")
    print("\n=== Per-category summary ===")
    for cat, s in sorted(summary.items()):
        leak_pct = s["allow_leakage_rate"] * 100
        print(f"  {cat:24s} n={s['n']:3d} deny={s['denied']:3d} esc={s['escalated']:3d} allow={s['allowed']:3d} allow_leakage={leak_pct:5.1f}% mean_risk={s['mean_risk']:.3f}")


if __name__ == "__main__":
    main()
