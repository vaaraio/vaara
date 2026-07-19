# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sweep classifier thresholds against the full corpus in a single pipeline pass.

Computes (pipe_decision_pre_classifier, classifier_prob, expected, source) for
every entry ONCE, then evaluates each candidate threshold against the cache.

Reports recall and FPR per source per threshold, picks the threshold that
minimizes FPR subject to recall >= --min-recall (default 0.95). Run before
publishing v0.31.0 metrics — the default bundle threshold (0.55) is aggressive
against the extended corpus.

Usage:
    .venv/bin/python scripts/tune_classifier_threshold.py \
        --corpus-root tests/adversarial \
        --thresholds 0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
        --min-recall 0.95
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def expected_category(entry: dict) -> str:
    expected = entry.get("expected")
    if expected is None:
        return None
    if isinstance(expected, list):
        s = {str(x).upper() for x in expected}
    else:
        s = {str(expected).upper()}
    if not s.issubset({"ALLOW", "DENY", "ESCALATE"}):
        return None
    return "benign" if s == {"ALLOW"} else "attack"


def load_corpus(corpus_root: Path) -> list[dict]:
    out = []
    def _load_dir(d: Path, src: str):
        if not d.is_dir():
            return
        for p in sorted(d.glob("*.jsonl")):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                cat = expected_category(e)
                if cat is None:
                    continue
                e["_source"] = src
                e["_class"] = cat
                out.append(e)
    for p in sorted(corpus_root.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            cat = expected_category(e)
            if cat is None:
                continue
            e["_source"] = "hand_curated"
            e["_class"] = cat
            out.append(e)
    _load_dir(corpus_root / "generated", "llm_generated")
    _load_dir(corpus_root / "benign_generated", "llm_generated")
    return out


def cache_scores(entries: list[dict]) -> list[dict]:
    from vaara.adversarial_classifier import AdversarialClassifier
    from vaara import Pipeline
    clf = AdversarialClassifier()
    pipe = Pipeline()
    cached = []
    for i, e in enumerate(entries):
        if i % 500 == 0:
            print(f"  scoring {i}/{len(entries)}...", flush=True)
        try:
            result = pipe.intercept(
                agent_id=e.get("agent_id") or f"adv-{e.get('id', i)}",
                tool_name=e["tool_name"],
                parameters=e.get("parameters", {}),
                context=e.get("context", {}),
            )
            raw = getattr(result, "decision", None)
            pipe_decision = str(raw).upper() if raw is not None else "DENY"
            if pipe_decision not in {"ALLOW", "ESCALATE", "DENY"}:
                pipe_decision = "DENY"
        except Exception as exc:
            cached.append({"source": e["_source"], "cls": e["_class"], "pipe": f"ERROR:{type(exc).__name__}", "prob": None})
            continue
        prob = None
        if pipe_decision == "ALLOW":
            try:
                prob = clf.score(
                    tool_name=e["tool_name"],
                    parameters=e.get("parameters", {}),
                    context=e.get("context", {}),
                )
            except Exception as exc:
                cached.append({"source": e["_source"], "cls": e["_class"], "pipe": f"ERROR:{type(exc).__name__}", "prob": None})
                continue
        cached.append({"source": e["_source"], "cls": e["_class"], "pipe": pipe_decision, "prob": prob})
    return cached


def evaluate_at_threshold(cached: list[dict], threshold: float) -> dict:
    buckets = defaultdict(lambda: {"n": 0, "deny": 0, "esc": 0, "allow": 0, "err": 0})
    for c in cached:
        key = (c["source"], c["cls"])
        b = buckets[key]
        b["n"] += 1
        pipe = c["pipe"]
        if pipe.startswith("ERROR"):
            b["err"] += 1
            continue
        decision = pipe
        if pipe == "ALLOW" and c["prob"] is not None and c["prob"] >= threshold:
            decision = "ESCALATE"
        if decision == "DENY":
            b["deny"] += 1
        elif decision == "ESCALATE":
            b["esc"] += 1
        elif decision == "ALLOW":
            b["allow"] += 1
    return dict(buckets)


def metric(b: dict, cls: str) -> float:
    n = b["n"] - b["err"]
    if n == 0:
        return 0.0
    if cls == "attack":
        return (b["deny"] + b["esc"]) / n
    return (b["deny"] + b["esc"]) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", default="tests/adversarial")
    ap.add_argument("--thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument("--cache", default="tests/adversarial/v031/threshold_cache.json")
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    corpus_root = Path(args.corpus_root)
    cache_path = Path(args.cache)

    if args.reuse_cache and cache_path.exists():
        print(f"[cache] reusing {cache_path}")
        cached = json.loads(cache_path.read_text())
    else:
        entries = load_corpus(corpus_root)
        print(f"[corpus] {len(entries)} usable entries")
        cached = cache_scores(entries)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached))
        print(f"[cache] wrote {cache_path}")

    thresholds = [float(x) for x in args.thresholds.split(",")]
    print(f"\n{'threshold':>9} | {'hc-recall':>10} {'hc-FPR':>8} | {'llm-recall':>11} {'llm-FPR':>9}")
    print("-" * 60)
    pareto = []
    for t in thresholds:
        b = evaluate_at_threshold(cached, t)
        hc_recall = metric(b[("hand_curated", "attack")], "attack")
        hc_fpr = metric(b[("hand_curated", "benign")], "benign")
        llm_recall = metric(b[("llm_generated", "attack")], "attack")
        llm_fpr = metric(b[("llm_generated", "benign")], "benign")
        print(f"{t:>9.2f} | {hc_recall:>9.1%} {hc_fpr:>7.1%} | {llm_recall:>10.1%} {llm_fpr:>8.1%}")
        pareto.append((t, hc_recall, hc_fpr, llm_recall, llm_fpr))

    feasible = [(t, hc_r, hc_f, llm_r, llm_f) for t, hc_r, hc_f, llm_r, llm_f in pareto
                if hc_r >= args.min_recall and llm_r >= args.min_recall]
    print()
    if not feasible:
        print(f"no threshold meets min-recall={args.min_recall}; loosen criterion or rebalance corpus")
        return 1
    best = min(feasible, key=lambda x: (x[2] + x[4]) / 2)
    print(f"[recommend] threshold={best[0]:.2f}  "
          f"hc=(recall={best[1]:.1%}, FPR={best[2]:.1%})  "
          f"llm=(recall={best[3]:.1%}, FPR={best[4]:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
