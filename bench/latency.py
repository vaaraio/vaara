"""Latency micro-benchmark for InterceptionPipeline.intercept().

Measures the per-call overhead an agent pays to run through Vaara's
classify → score → decide → audit path. The output is the number that
answers "can I put this in front of a sub-second agent loop" and
"what does one governed tool call cost" for ops budgeting.

Run::

    python bench/latency.py                     # default: 10k calls, 1k warmup
    python bench/latency.py --calls 50000       # longer run
    python bench/latency.py --json results.json # machine-readable dump

The benchmark holds the scorer, registry, and audit trail at their
library defaults so the number reported is what a user gets out of the
box — no tuning, no pre-warmed calibration. Audit trail is in-memory
(no persistence callback); persistent backends add their own I/O cost
on top and are measured separately in bench/latency_sqlite.py if
needed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Make `import vaara` work when running the file directly from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from vaara.pipeline import InterceptionPipeline  # noqa: E402


@dataclass
class LatencyStats:
    label: str
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    min_ms: float
    max_ms: float
    throughput_per_sec: float


def _percentile(sorted_samples: list[float], p: float) -> float:
    if not sorted_samples:
        return 0.0
    k = (len(sorted_samples) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = k - lo
    return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac


def _summarize(label: str, samples_ms: list[float]) -> LatencyStats:
    s = sorted(samples_ms)
    total_s = sum(samples_ms) / 1000.0
    return LatencyStats(
        label=label,
        n=len(samples_ms),
        mean_ms=round(statistics.fmean(samples_ms), 4),
        p50_ms=round(_percentile(s, 0.50), 4),
        p95_ms=round(_percentile(s, 0.95), 4),
        p99_ms=round(_percentile(s, 0.99), 4),
        p999_ms=round(_percentile(s, 0.999), 4),
        min_ms=round(s[0], 4),
        max_ms=round(s[-1], 4),
        throughput_per_sec=round(len(samples_ms) / total_s, 1) if total_s > 0 else 0.0,
    )


# Representative action mix. Covers:
#   - high-stakes financial (irreversible, regulated)
#   - everyday data access (low risk, hot path)
#   - unknown tool (classify fallback path)
# Parameters are realistic in shape but small — the interesting variable
# is the pipeline path, not parameter serialization.
_WORKLOADS = {
    "tx.transfer": {
        "tool_name": "tx.transfer",
        "parameters": {"to": "0xabc123", "amount": 1000, "token": "USDC"},
        "agent_confidence": 0.82,
    },
    "tx.swap": {
        "tool_name": "tx.swap",
        "parameters": {"from": "USDC", "to": "ETH", "amount": 2500},
        "agent_confidence": 0.77,
    },
    "data.read": {
        "tool_name": "data.read",
        "parameters": {"path": "s3://bucket/key"},
        "agent_confidence": 0.95,
    },
    "unknown.tool": {
        "tool_name": "custom.plugin.invoke",
        "parameters": {"payload": "x"},
        "agent_confidence": 0.5,
    },
}


def bench_workload(
    pipeline: InterceptionPipeline,
    workload_name: str,
    n_calls: int,
    n_warmup: int,
) -> LatencyStats:
    spec = _WORKLOADS[workload_name]
    # Warmup — let JIT/caches/first-touch allocations settle.
    for i in range(n_warmup):
        pipeline.intercept(
            agent_id=f"warmup-{i}",
            tool_name=spec["tool_name"],
            parameters=spec["parameters"],
            agent_confidence=spec["agent_confidence"],
        )

    samples_ms: list[float] = []
    # Use perf_counter_ns for sub-microsecond resolution independent of
    # the pipeline's internal time.monotonic() bookkeeping.
    for i in range(n_calls):
        t0 = time.perf_counter_ns()
        pipeline.intercept(
            agent_id=f"agent-{i & 0xFF}",
            tool_name=spec["tool_name"],
            parameters=spec["parameters"],
            agent_confidence=spec["agent_confidence"],
        )
        samples_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

    return _summarize(workload_name, samples_ms)


def bench_first_call(n_trials: int) -> LatencyStats:
    """First intercept on a freshly-constructed pipeline.

    Pipeline construction is NOT included in the timer — this is the
    caller-visible latency of the first `.intercept()` call after the
    library is already imported and the pipeline object exists.
    Compare against steady-state to see whether the first call pays
    any lazy-init tax beyond ongoing calls.
    """
    samples_ms: list[float] = []
    spec = _WORKLOADS["tx.transfer"]
    for _ in range(n_trials):
        pipeline = InterceptionPipeline()
        t0 = time.perf_counter_ns()
        pipeline.intercept(
            agent_id="cold",
            tool_name=spec["tool_name"],
            parameters=spec["parameters"],
            agent_confidence=spec["agent_confidence"],
        )
        samples_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
    return _summarize("first_call_after_init", samples_ms)


def bench_construction(n_trials: int) -> LatencyStats:
    """Time to construct a fresh InterceptionPipeline() (import excluded).

    This is the one-off startup cost a process pays per pipeline
    instance. Most users construct once at process start, so this is
    the number for "how long does Vaara add to my app's boot time."
    """
    samples_ms: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        InterceptionPipeline()
        samples_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)
    return _summarize("pipeline_construction", samples_ms)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calls", type=int, default=10_000,
                    help="Measured calls per workload (default: 10000)")
    ap.add_argument("--warmup", type=int, default=1_000,
                    help="Warmup calls per workload (default: 1000)")
    ap.add_argument("--cold-trials", type=int, default=50,
                    help="Cold-start trials (default: 50)")
    ap.add_argument("--json", type=str, default=None,
                    help="Write full results to this JSON file")
    args = ap.parse_args()

    results: list[LatencyStats] = []

    # Steady-state: one pipeline reused across workloads, matching how
    # a long-running agent would use it. The scorer's MWU state evolves
    # across calls; this is intentional — it's the realistic shape.
    pipeline = InterceptionPipeline()
    for name in _WORKLOADS:
        stats = bench_workload(pipeline, name, args.calls, args.warmup)
        results.append(stats)

    results.append(bench_first_call(args.cold_trials))
    results.append(bench_construction(args.cold_trials))

    # Human-readable table.
    print()
    print(f"{'workload':<24} {'n':>6} {'mean':>8} {'p50':>8} {'p95':>8} "
          f"{'p99':>8} {'p999':>8} {'max':>8} {'ops/s':>10}")
    print("-" * 96)
    for r in results:
        print(f"{r.label:<24} {r.n:>6} {r.mean_ms:>7.3f}m "
              f"{r.p50_ms:>7.3f}m {r.p95_ms:>7.3f}m {r.p99_ms:>7.3f}m "
              f"{r.p999_ms:>7.3f}m {r.max_ms:>7.3f}m {r.throughput_per_sec:>10.1f}")
    print()

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in results], indent=2)
        )
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
