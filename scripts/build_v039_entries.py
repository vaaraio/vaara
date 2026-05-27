"""Convert v0.39 BIPIA harness traces into classifier-shaped entries.

Reads tests/adversarial/traces/bipia-s43-*.jsonl, shapes each emitted
tool call as a classifier entry using the same mapping eval_v039_bipia.py
uses (follows -> expected=DENY/severity=high, benign-under-pressure ->
expected=ALLOW/severity=low, context.source=injected_via_bipia_<class>),
and writes one JSONL per source model under tests/adversarial/v039_bipia/.

These files are picked up automatically by
train_adversarial_classifier.load_corpus_keyed (which rglobs *.jsonl
under tests/adversarial). The split manifest produced by
build_v039_split.py controls train vs holdout assignment.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO / "tests/adversarial/traces"
OUT_DIR = REPO / "tests/adversarial/v039_bipia"

MODEL_TO_SLUG = {
    "claude-haiku-4-5": "haiku",
    "llama3.1:8b": "llama31-8b",
    "mistral:7b": "mistral-7b",
    "qwen2.5:7b": "qwen25-7b",
}


def slug_attack_class(ac: str) -> str:
    return ac.lower().replace(" ", "_").replace("(", "").replace(")", "")


def trace_to_entries(trace: dict) -> list[dict]:
    out: list[dict] = []
    src = f"injected_via_bipia_{slug_attack_class(trace['attack_class'])}"
    model = trace["model"]
    for k, tc in enumerate(trace.get("tool_calls", [])):
        reflects = bool(tc.get("reflects_injection"))
        # v8 build_labels treats expected in {DENY, ESCALATE} as positive.
        # Category labels mapped to v8 schema: prompt_injection_external is
        # not in v035_split categories. Use injected-follow -> DENY high,
        # benign-under-pressure -> ALLOW low. category="prompt_injection"
        # for follows so per-category breakdown stays interpretable.
        category = "prompt_injection" if reflects else "tool_misuse"
        entry = {
            "id": f"{trace['trial_id']}-tc{k}",
            "category": category,
            "agent_id": f"agent-{model}",
            "tool_name": tc.get("tool_name", ""),
            "parameters": tc.get("parameters", {}),
            "context": {
                "original_task": trace.get("benign_topic", ""),
                "source": src,
            },
            "expected": "DENY" if reflects else "ALLOW",
            "severity": "high" if reflects else "low",
            "notes": (
                f"BIPIA {trace.get('attack_class', '')}; "
                f"injection_followed={reflects}; model={model}"
            ),
            "_v039_bipia": {
                "trial_id": trace["trial_id"],
                "attack_class": trace.get("attack_class"),
                "model": model,
                "reflects_injection": reflects,
            },
        }
        out.append(entry)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_model_entries: dict[str, list[dict]] = {slug: [] for slug in MODEL_TO_SLUG.values()}
    n_traces = 0
    n_skipped = 0

    for fp in sorted(TRACES_DIR.glob("bipia-s43-*.jsonl")):
        for raw in fp.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            trace = json.loads(line)
            n_traces += 1
            model = trace.get("model")
            slug = MODEL_TO_SLUG.get(model)
            if slug is None:
                n_skipped += 1
                continue
            per_model_entries[slug].extend(trace_to_entries(trace))

    summary: dict[str, dict[str, int]] = {}
    for slug, entries in per_model_entries.items():
        out_path = OUT_DIR / f"bipia-s43-{slug}.jsonl"
        with out_path.open("w") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        n_follows = sum(1 for e in entries if e["expected"] == "DENY")
        n_benign = sum(1 for e in entries if e["expected"] == "ALLOW")
        summary[slug] = {"n": len(entries), "follows": n_follows, "benign": n_benign}
        print(f"[wrote] {out_path.relative_to(REPO)}  n={len(entries)} follows={n_follows} benign={n_benign}")

    print(f"\n[traces read] {n_traces}  skipped (unknown model)={n_skipped}")
    print(f"[totals] entries={sum(s['n'] for s in summary.values())} "
          f"follows={sum(s['follows'] for s in summary.values())} "
          f"benign={sum(s['benign'] for s in summary.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
