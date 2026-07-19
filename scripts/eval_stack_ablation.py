#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stack ablation — heuristic-only / classifier-only / full-stack comparison.

v0.5.3 ships two stacking layers:
  1. AdaptiveScorer (heuristic) — taxonomy base + sequence + agent history
     + MWU expert weighting + conformal prior
  2. AdversarialClassifier (XGBoost, opt-in via vaara[ml]) — upgrades
     heuristic ALLOW -> ESCALATE when prob >= threshold

This script measures each layer's independent contribution. Three
configurations evaluated against the same corpus (hand-curated +
LLM-generated, source-tagged):

  heuristic_only   pipe.intercept() with classifier=None
  classifier_only  classifier.score() + threshold, no heuristic
  full_stack       pipe.intercept() + classifier upgrade

Per-source recall (attacks) and FPR (benigns) reported for each.

Defends the architecture story for grant audiences ("why both layers?")
and quantifies what item 5 already hinted at: most full-stack benign
escalations come from the heuristic, not the classifier.

Requires vaara[ml].
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


def load_corpus(corpus_root: Path) -> list[dict]:
    """Load both hand-curated and LLM-generated entries with source tags."""
    entries: list[dict] = []
    for fp in sorted(corpus_root.glob("*.jsonl")):
        for e in _load_jsonl(fp):
            e["_source"] = "hand_curated"
            entries.append(e)
    for sub in ("generated", "benign_generated"):
        for fp in sorted((corpus_root / sub).glob("*.jsonl")):
            for e in _load_jsonl(fp):
                e["_source"] = "llm_generated"
                entries.append(e)
    return entries


def expected_class(entry: dict) -> str:
    expected = entry.get("expected", "DENY")
    s = {str(x).upper() for x in (expected if isinstance(expected, list) else [expected])}
    return "benign" if s == {"ALLOW"} else "attack"


def heuristic_only(entry: dict, pipe: Pipeline) -> str:
    """AdaptiveScorer-only via pipe.intercept (no classifier upgrade)."""
    # Per-entry agent_id keeps sequence/agent-history state isolated across
    # samples (parallel to the same fix in eval_distribution_shift.py).
    agent_id = entry.get("agent_id") or f"adv-{entry.get('id', id(entry))}"
    try:
        result = pipe.intercept(
            agent_id=agent_id,
            tool_name=entry["tool_name"],
            parameters=entry.get("parameters", {}),
            context=entry.get("context", {}),
        )
        return str(getattr(result, "decision", "UNKNOWN")).upper()
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}"


def classifier_only(entry: dict, classifier) -> str:
    """Classifier-only: ALLOW if prob<threshold else ESCALATE. No heuristic."""
    try:
        prob = classifier.score(
            tool_name=entry["tool_name"],
            parameters=entry.get("parameters", {}),
            context=entry.get("context", {}),
        )
        return "ESCALATE" if prob >= classifier.threshold else "ALLOW"
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}"


def full_stack(heuristic_result: str, classifier_result: str) -> str:
    """Heuristic + classifier upgrade rule.

    Takes the precomputed per-config decisions instead of re-running
    pipe.intercept(). Re-running would advance sequence / agent-history state
    on the second pass and make `heuristic_only` and `full_stack` not
    directly comparable (the published v0.6 ablation table assumes one
    intercept call per entry).
    """
    if heuristic_result == "ALLOW" and classifier_result == "ESCALATE":
        return "ESCALATE"
    return heuristic_result


def metric(decision: str, expected_cls: str) -> str:
    """Classify decision against expected as tp / fn / fp / tn."""
    gated = decision in ("DENY", "ESCALATE")
    if expected_cls == "attack":
        return "tp" if gated else "fn"
    return "fp" if gated else "tn"


def evaluate(entries, pipe, classifier) -> dict:
    """Run all three configs, return nested counts: config -> source/class -> counts."""
    counts: dict = defaultdict(
        lambda: defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "err": 0})
    )
    for e in entries:
        src = e["_source"]
        cls = expected_class(e)
        # Compute each pipeline path exactly once per entry. full_stack is
        # derived from the two decisions; running pipe.intercept() a second
        # time would advance sequence/agent-history state.
        h = heuristic_only(e, pipe)
        c = classifier_only(e, classifier)
        for config_name, decision in (
            ("heuristic_only", h),
            ("classifier_only", c),
            ("full_stack", full_stack(h, c)),
        ):
            if decision.startswith("ERROR:"):
                counts[config_name][f"{src}/{cls}"]["err"] += 1
            else:
                counts[config_name][f"{src}/{cls}"][metric(decision, cls)] += 1
    return counts


def report(counts, json_out: Path | None) -> None:
    """Print per-config / per-source recall + FPR table."""
    rows = []
    print("\n=== Stack ablation (v0.5.3) ===")
    print(f"{'config':<18} {'source/class':<28} {'tp':>5} {'fn':>5} {'fp':>5} {'tn':>5} {'err':>4} {'metric':>20}")
    for config_name in ("heuristic_only", "classifier_only", "full_stack"):
        for key, c in sorted(counts[config_name].items()):
            cls = key.split("/")[1]
            if cls == "attack":
                metric_name = "recall"
                metric_val = c["tp"] / max(c["tp"] + c["fn"], 1)
            else:
                metric_name = "FPR"
                metric_val = c["fp"] / max(c["fp"] + c["tn"], 1)
            print(
                f"{config_name:<18} {key:<28} {c['tp']:>5d} {c['fn']:>5d} "
                f"{c['fp']:>5d} {c['tn']:>5d} {c['err']:>4d} "
                f"{metric_name}={metric_val*100:>5.1f}%"
            )
            rows.append({"config": config_name, "key": key, **c, "metric": metric_name, "value": metric_val})
    if json_out is not None:
        json_out.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\n[json] {json_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus-root", default="tests/adversarial")
    ap.add_argument("--out", default=None)
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
    entries = load_corpus(corpus_root)
    print(f"[corpus] {len(entries)} entries")

    counts = evaluate(entries, pipe, classifier)
    report(counts, json_out=Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
