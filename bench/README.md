# bench/

Performance benchmarks for the Vaara execution layer.

## Quick headline (steady-state)

On a mid-range x86 desktop (AMD Ryzen 7 5800X, Python 3.12), one governed
tool call through `InterceptionPipeline.intercept()` runs classify, score,
decide, writes two to four audit records on the hash-chain, and updates
metrics. It costs roughly:

| workload       | p50      | p95      | p99      | ops/sec (single-thread) |
| -------------- | -------- | -------- | -------- | ----------------------- |
| `tx.transfer`  | ~0.12 ms | ~0.14 ms | ~0.18 ms | ~8,000                  |
| `tx.swap`      | ~0.12 ms | ~0.13 ms | ~0.15 ms | ~8,000                  |
| `data.read`    | ~0.13 ms | ~0.15 ms | ~0.17 ms | ~6,800                  |
| `custom.tool`  | ~0.13 ms | ~0.15 ms | ~0.16 ms | ~7,000                  |

Pipeline construction itself takes roughly 13 µs. The first call after
construction is indistinguishable from steady-state, so there is no
hidden lazy-init tax on the first intercepted action.

For the DeFi audience: at block times under one second the per-call
overhead is three to four orders of magnitude below the block window.
Vaara is not the bottleneck.

## Running the benchmark

```
python3 bench/latency.py
python3 bench/latency.py --calls 50000
python3 bench/latency.py --json bench/latency_results.json
```

Flags:

- `--calls N` measured calls per workload. Default 10,000.
- `--warmup N` discarded warmup calls per workload. Default 1,000.
- `--cold-trials N` trials for the first-call and construction rows. Default 50.
- `--json PATH` dump full per-workload stats as JSON.

## What it measures

Four representative workloads:

- `tx.transfer` irreversible financial action, MiFID2 + DORA tagged.
- `tx.swap` irreversible financial action, DEX swap shape.
- `data.read` low-risk data access, GDPR tagged.
- `unknown.tool` a tool name the registry does not know, exercising the
  classification fallback path.

Plus two init-related measurements:

- `first_call_after_init` first `intercept()` on a freshly constructed
  pipeline. Pipeline construction is **not** in the timer. This answers
  "does the first call pay extra?". On current numbers, no.
- `pipeline_construction` time to run `InterceptionPipeline()`. Answers
  "what does Vaara add to my app boot time?". Tens of microseconds.

## What it does not measure

- Persistent audit backends. The default `AuditTrail` is in-memory. Writing
  each record to SQLite or an append-only file adds I/O latency the
  benchmark does not cover. A separate SQLite variant belongs in a
  follow-up file once the primary number is established.
- Concurrent throughput under multi-threaded load. `intercept()` is
  documented as requiring per-agent serialisation by the caller, so a true
  throughput number belongs in a threaded harness.
- Network-bound integrations. LangChain or LangGraph adapters wrap the
  pipeline with model-call latency that dominates anything Vaara does.
- The cost of `report_outcome()`. This closes the feedback loop (MWU
  update, conformal residual, audit append) and is called after action
  execution, not on the hot path.

## Caveats

- **Tail outliers (`max` column).** Single-digit to double-digit millisecond
  spikes show up in long runs. These are Python garbage-collection pauses,
  not a pipeline property. Running with `gc.disable()` eliminates them.
  We leave GC on for honesty. Any long-running Python process sees these
  spikes across all code paths, not just Vaara.
- **In-process calibration state.** The scorer's adaptive components update
  across calls. Numbers reported here are steady-state after 1,000
  warmup intercepts, not the first-ever invocation on a cold process.
- **Hardware dependence.** The quoted numbers are from an AMD Ryzen 7
  5800X. Scale roughly with single-core performance. A laptop i5 will
  run perhaps 1.5× slower. A modern server part similar or faster. CI
  runners are typically slower and noisier than either.

## Reproducing

```
git clone https://github.com/vaaraio/vaara
cd vaara
pip install -e .
python3 bench/latency.py
```

No GPU, no network, no special flags. The benchmark runs in under a
minute on commodity hardware.

## Using the numbers

- **For DeFi or low-latency consumers.** Quote p99 as the headline since
  it is the number that bounds strategy slippage, not the mean. Sub-200 µs
  p99 means governance overhead is immaterial to block-timing.
- **For ops and capacity planning.** Take the ops/sec number and divide by
  your target concurrency. A single-process Vaara handles ~7 thousand
  governed actions per second with no sharding. Horizontal scaling is a
  straight multiplication because the pipeline holds no cross-process state
  other than the audit chain, which is per-pipeline.
- **For comparison against competitors.** The number to ask of any
  sandbox-based agent governance product is "what is the p99 added
  latency per tool call on a warm path." Most have not published one.

## Scorer eval, adversarial corpus

`bench/adversarial_corpus.jsonl` is a **synthetic** labelled corpus
of 77 traces generated deterministically by `bench/build_corpus.py`.
These are hand-constructed hypothetical action sequences, not
recordings of real user agents. Vaara has no production deployments at
the time of writing. This corpus stands in for real traffic until one
exists.

- 25 benign synthetic traces (single reads, read/write, legitimate
  swaps, API syncs, long read-only sessions)
- 52 malicious synthetic traces covering every built-in sequence
  pattern plus burst and confidence-gap attacks (data exfiltration,
  data destruction, privilege escalation, financial drain, governance
  takeover, safety override, burst, confidence-gap)

Each trace is one agent's ordered sequence of proposed tool calls.
`bench/scorer_eval.py` replays each trace through a fresh pipeline
and scores the decision on the trace's target action (last action for
all corpus entries).

Two decision bands matter:

- **Soft catch**: anything other than ALLOW (ESCALATE + DENY). Escalate
  routes to human review, which is the governance system working as
  designed, not a false positive.
- **Hard catch**: only DENY. The scorer actively refused.

### Warmed-mode headline (production-shaped)

Run with ``python3 bench/scorer_eval.py --mode warmed`` after seeding the
conformal calibrator with 200 benign outcome pairs to simulate a
deployed system that has seen normal traffic:

| metric                                  | value   |
| --------------------------------------- | ------- |
| Soft TPR (malicious not allowed)        | 100.00% |
| Hard TPR (malicious outright denied)    | 19.23%  |
| Soft FPR (benign not allowed)           | 60.00%  |
| Hard FPR (benign outright denied)       | 0.00%   |

Every malicious trace in every category was routed away from ALLOW.
No benign trace was outright denied. The 60% soft-FPR is entirely
ESCALATE on the three conservative benign categories (legit swaps,
API syncs, long read-only sessions). This is by-design routing of
high-blast or unknown-agent actions to human review.

### Cold-start caveat

Run with ``--mode cold`` (fresh scorer, no calibration). Everything
escalates, including benign reads, because the uncalibrated
conformal interval is wide (±0.3). This is the intended default: fail
toward human review until the system has seen enough outcomes to
tighten intervals. Users deploying Vaara should expect a calibration
period (roughly 30-100 reported outcomes) before soft-FPR drops to
its warmed level.

### Per-category detection (warmed)

- `data_exfiltration` 8/8 caught (escalate)
- `data_destruction` 8/8 caught (escalate)
- `privilege_escalation` 8/8 caught (escalate)
- `financial_drain` 8/8 caught (escalate)
- `governance_takeover` 5/5 caught (**deny**)
- `safety_override` 5/5 caught (**deny**)
- `burst_attack` 5/5 caught (escalate)
- `confidence_gap` 5/5 caught (escalate)

`governance_takeover` and `safety_override` hit DENY because their
sequence risk boosts (0.7 and 0.8 respectively) push the conformal
upper bound over 0.7 on a single trace. Lower-boost sequences land in
the ESCALATE band by design.

### Caveats on the corpus

- **Synthetic, not production traffic.** Every trace is constructed by
  `build_corpus.py`. No real user agents, no real deployments. Vaara
  has zero production users at the time of writing. We do not claim
  otherwise.
- **Malicious traces are designed to match known patterns.** Every
  malicious trace was constructed to match one of the scorer's built-in
  sequence patterns. This measures "do we detect things we know about,"
  not "do we detect novel attacks." A held-out or adversarial
  novel-attack corpus is follow-up work.
- **Small.** 77 traces is a first pass. Scaling to thousands, including
  real-world LangChain or LangGraph traces from production agents once
  deployments exist, is a v1.1 goal.
- **Pattern-match level.** Detection fires on the deterministic sequence
  matcher. The MWU and conformal layers contribute to the decision band
  (allow vs escalate vs deny) but are not load-bearing for catching
  these categories on cold data.

### Regenerate the corpus

```
python3 bench/build_corpus.py
```

The generator is deterministic. If you extend it, re-run and commit
both the generator and the updated `adversarial_corpus.jsonl`.

## History

Initial benchmark committed April 2026 following the project's first
formal performance pass. Prior to that the pipeline was correctness-
tested only (239 unit, integration, and property tests) with no published
latency numbers or scorer-accuracy numbers.
