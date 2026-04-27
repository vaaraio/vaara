#!/usr/bin/env python3
"""Distribution-shift split — hand-curated vs LLM-generated recall.

v0.5.3's adversarial classifier was trained on a corpus that mixes:
  - Hand-curated entries          (tests/adversarial/*.jsonl, ~250 items)
  - LLM-generated attack variants (tests/adversarial/generated/*.jsonl)
  - LLM-generated benign variants (tests/adversarial/benign_generated/*.jsonl)

CHANGELOG quoted balanced-accuracy numbers don't reveal the gap between
LLM-generated and hand-curated recall. v0.6 owes that split.

This script runs the full Vaara stack (heuristic + classifier) on each
source separately and reports:
  - attack recall = (DENY + ESCALATE) / N for entries with expected != ALLOW
  - benign FPR    = (DENY + ESCALATE) / N for entries with expected == ALLOW

NOTE on asymmetry: hand-curated entries are held-out (not in training).
LLM-generated entries WERE in training, so their numbers are in-sample
fit, not generalization. The gap (hand-curated < LLM-generated) is the
distribution-shift signal. Re-running with proper out-of-fold predictions
on the LLM-generated set is a v0.7 follow-up if the gap demands it.

Requires `vaara[ml]` for the classifier.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vaara import Pipeline


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_source(corpus_root: Path, *, source: str) -> list[dict]:
    """Load entries tagged with a source label.

    source = 'hand_curated' → top-level *.jsonl (excluding subdirs)
    source = 'llm_generated' → generated/ + benign_generated/ subdirs
    """
    entries: list[dict] = []
    if source == "hand_curated":
        for fp in sorted(corpus_root.glob("*.jsonl")):
            entries.extend(_load_jsonl(fp))
    elif source == "llm_generated":
        for sub in ("generated", "benign_generated"):
            for fp in sorted((corpus_root / sub).glob("*.jsonl")):
                entries.extend(_load_jsonl(fp))
    else:
        raise ValueError(f"unknown source: {source}")
    for e in entries:
        e["_source"] = source
    return entries


def decide(entry: dict, pipe: Pipeline, classifier) -> str:
    """Run the full Vaara stack. Returns DENY / ESCALATE / ALLOW / ERROR:..."""
    try:
        result = pipe.intercept(
            agent_id=entry.get("agent_id", "adv"),
            tool_name=entry["tool_name"],
            parameters=entry.get("parameters", {}),
            context=entry.get("context", {}),
        )
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}"
    # Fail closed: a missing or unrecognised decision must NOT be silently
    # dropped — evaluate() only buckets {DENY, ESCALATE, ALLOW, ERROR:*},
    # so any other value would inflate `n` without incrementing any outcome
    # counter and quietly skew recall / FPR.
    raw = getattr(result, "decision", None)
    decision = str(raw).upper() if raw is not None else "DENY"
    if decision not in {"ALLOW", "ESCALATE", "DENY"}:
        decision = "DENY"
    if classifier is not None and decision == "ALLOW":
        try:
            prob = classifier.score(
                tool_name=entry["tool_name"],
                parameters=entry.get("parameters", {}),
                context=entry.get("context", {}),
            )
            if prob >= classifier.threshold:
                decision = "ESCALATE"
        except Exception as exc:
            return f"ERROR:{type(exc).__name__}"
    return decision


def expected_category(entry: dict) -> str:
    """Return 'attack' or 'benign' from entry expected field."""
    expected = entry.get("expected", "DENY")
    if isinstance(expected, list):
        expected_set = {str(x).upper() for x in expected}
    else:
        expected_set = {str(expected).upper()}
    return "benign" if expected_set == {"ALLOW"} else "attack"


def evaluate(entries: list[dict], pipe: Pipeline, classifier) -> dict:
    """Run pipeline + classifier on entries; return per-source / per-class counts."""
    buckets: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "deny": 0, "escalate": 0, "allow": 0, "error": 0},
    )
    for e in entries:
        src = e["_source"]
        cls = expected_category(e)  # 'attack' | 'benign'
        decision = decide(e, pipe, classifier)
        b = buckets[(src, cls)]
        b["n"] += 1
        if decision.startswith("ERROR:"):
            b["error"] += 1
        elif decision == "DENY":
            b["deny"] += 1
        elif decision == "ESCALATE":
            b["escalate"] += 1
        elif decision == "ALLOW":
            b["allow"] += 1
    return {f"{src}/{cls}": {**counts} for (src, cls), counts in buckets.items()}


def report(buckets: dict, *, json_out: Path | None) -> None:
    """Print recall/FPR per source/class. Optionally write JSON."""
    print("\n=== Distribution-shift split ===")
    print(f"{'source/class':<28} {'n':>6} {'deny':>6} {'esc':>6} {'allow':>6} {'err':>5} {'metric':>20}")
    rows = []
    for key, c in sorted(buckets.items()):
        n = c["n"]
        if n == 0:
            continue
        gated = c["deny"] + c["escalate"]
        cls = key.split("/")[1]
        if cls == "attack":
            metric_name = "recall"
            metric_val = gated / n
        else:
            metric_name = "FPR"
            metric_val = gated / n
        print(
            f"{key:<28} {n:>6d} {c['deny']:>6d} {c['escalate']:>6d} "
            f"{c['allow']:>6d} {c['error']:>5d} {metric_name}={metric_val*100:>5.1f}%"
        )
        rows.append({"key": key, **c, "metric": metric_name, "value": metric_val})
    if json_out is not None:
        json_out.write_text(json.dumps({"buckets": rows}, indent=2))
        print(f"\n[json] {json_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus-root", default="tests/adversarial")
    ap.add_argument("--out", default=None, help="Write JSON to this path")
    args = ap.parse_args()

    corpus_root = Path(args.corpus_root)
    if not corpus_root.exists():
        raise SystemExit(f"corpus root not found: {corpus_root}")

    try:
        from vaara.adversarial_classifier import AdversarialClassifier
        classifier = AdversarialClassifier()
    except ImportError as exc:
        raise SystemExit(f"this script needs vaara[ml]: {exc}")
    print(f"[classifier] bundle v{classifier.bundle_version}, threshold={classifier.threshold}")

    pipe = Pipeline()

    entries: list[dict] = []
    for source in ("hand_curated", "llm_generated"):
        loaded = load_source(corpus_root, source=source)
        print(f"[corpus] {source}: {len(loaded)} entries")
        entries.extend(loaded)

    buckets = evaluate(entries, pipe, classifier)
    report(buckets, json_out=Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
