# vaara-bench-v1

A versioned, reproducible adversarial-detection benchmark for the Vaara
governance kernel. v1 freezes the corpus, the scoring methodology, and
the headline numbers under a specific Vaara release so that external
auditors, reviewers, and competing implementations can verify the same
numbers and compare like-for-like.

vaara-bench-v1 is the **first published open-source adversarial
benchmark for an AI-agent runtime governance kernel** with reproducible
numbers, a permissive license, and a fixed methodology. No comparable
benchmark has been published by any of the verified peer projects.

## Corpus

- File: `bench/adversarial_corpus.jsonl`
- SHA-256: `7a3219776e1c93a5127ab3b63832d73ba75f32fa044cabdbaa4e5d7088b33ff2`
- Traces: 77 (25 benign synthetic, 52 malicious synthetic)
- Generator: `bench/build_corpus.py` (deterministic; re-running produces
  byte-identical output)
- Generator SHA-256: `d28bc41ef9ffc064da9c57105a09e96743c421381399bdad4868180ffa6ea829`

Each trace is one agent's ordered sequence of proposed tool calls.
Categories covered:

| category               | label    | count |
| ---------------------- | -------- | ----- |
| benign_read            | benign   | 5     |
| benign_read_write      | benign   | 5     |
| benign_swap            | benign   | 5     |
| benign_api_sync        | benign   | 5     |
| benign_long_history    | benign   | 5     |
| data_exfiltration      | attack   | 8     |
| data_destruction       | attack   | 8     |
| privilege_escalation   | attack   | 8     |
| financial_drain        | attack   | 8     |
| governance_takeover    | attack   | 5     |
| safety_override        | attack   | 5     |
| burst_attack           | attack   | 5     |
| confidence_gap         | attack   | 5     |

**Honesty notes:**

- The corpus is **synthetic.** No production agent traffic. Every
  malicious trace was constructed to match one of the scorer's
  built-in sequence patterns, which measures "do we detect known
  patterns" not "do we detect novel attacks."
- 77 traces is a deliberate first pass. Scaling to thousands of
  production traces is a v2 goal that depends on real deployments.
- Pattern-match level: detection fires on the deterministic sequence
  matcher; the MWU and conformal layers contribute to the decision
  band (allow vs escalate vs deny) but are not load-bearing for
  catching these categories on cold data.

## Methodology

- Evaluator: `bench/scorer_eval.py` (SHA-256:
  `9d92933fe1ad2e2f7cab69a77e1316646ee7d3312804734dc0d56603768bb149`).
- Scorer construction: `AdaptiveScorer()` with default config, which
  pre-seeds the conformal calibrator with 50 synthetic benign
  `(predicted, actual)` pairs so cold and warmed modes produce the
  same decision quality.
- Mode: cold (no extra warmup beyond the synthetic prior).
- One agent per trace, one fresh pipeline per trace.
- Decision under evaluation: the verdict on the trace's last action
  (`target_index = -1` for every corpus entry in v1).
- Decision bands:
  - **Soft caught:** verdict is anything other than `ALLOW`
    (`ESCALATE` + `DENY`). `ESCALATE` routes to human review, which
    is the governance system working as designed, not a false
    positive.
  - **Hard caught:** only `DENY`. The scorer actively refused.

## Headline numbers (Vaara 0.11.0, cold mode)

| metric                                 | value      |
| -------------------------------------- | ---------- |
| Soft TPR (malicious not allowed)       | 100.00%    |
| Hard TPR (malicious outright denied)   | 28.85%     |
| Soft FPR (benign not allowed)          | 20.00%     |
| Hard FPR (benign outright denied)      | 0.00%      |

Every malicious trace in every category was routed away from `ALLOW`.
No benign trace was outright denied. The 20% soft-FPR is entirely
`ESCALATE` on the single-shot benign DEX-swap category (a fresh agent
with no history asking to swap funds is genuinely uncertain, and
routing that to human review is the governance system working as
designed).

Per-category detection: see `bench/vaara-bench-v1-results.json`.

## Reproduction

```bash
git clone https://github.com/vaaraio/vaara
cd vaara
pip install -e '.[ml]'
python3 bench/scorer_eval.py
```

No GPU, no network. Wall-clock under 2 seconds on commodity hardware.
The corpus is deterministic and re-running `bench/build_corpus.py`
produces byte-identical output.

## License

The corpus, generator, evaluator, and results in this directory are
released under **AGPL-3.0-or-later**, inherited from the top-level `LICENSE`
in this repository. Citation suggested but not required:

```text
vaara-bench-v1. Vaara Execution Layer, v0.11.0.
https://github.com/vaaraio/vaara/blob/main/bench/vaara-bench-v1.md
```

## Versioning policy

vaara-bench-**v1** freezes:

- The 77-trace synthetic corpus (`adversarial_corpus.jsonl`).
- The scoring methodology in `scorer_eval.py`.
- The decision-band definitions (soft / hard caught).
- The four headline metrics (soft TPR, hard TPR, soft FPR, hard FPR).

The **Vaara version evaluated** is allowed to change without bumping
the benchmark version. Result files are dated and tagged with the
Vaara version so the methodology stays comparable across Vaara
releases.

A new benchmark version (**v2**) would imply changes to the corpus or
methodology, for example adding production traces, switching to a
held-out novel-attack corpus, or adding new decision bands.

## What vaara-bench-v1 is NOT

- Not a claim of imperviousness to novel adversarial attackers.
  Stronger attackers, longer iteration budgets, or alternate
  strategies may produce different numbers.
- Not a substitute for the production pre-deployment testing
  required by OVERT 1.0 MEA-4 or EU AI Act Article 9 (Risk Management
  System).
- Not a third-party evaluation per OVERT MEA-3. A third party would
  need to run the benchmark independently.
- Not a benchmark of latency, throughput, or operational characteristics.
  See `bench/README.md` for the latency benchmark.

## Cross-references

- `bench/adversarial_corpus.jsonl`: the corpus itself.
- `bench/build_corpus.py`: deterministic corpus generator.
- `bench/scorer_eval.py`: evaluator.
- `bench/vaara-bench-v1-results.json`: machine-readable results.
- `bench/README.md`: latency benchmark and corpus background.
- `COMPLIANCE.md`: Article-level mapping including MEA-2 (S3P).
