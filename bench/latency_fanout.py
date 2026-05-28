"""Fan-out HTTP-transport latency micro-benchmark for vaara-mcp-proxy.

Measures per-call overhead Vaara adds to a `tools/call` routed through the
v0.40 streamable-HTTP transport across N upstream slots. Answers "what
does fan-out routing cost on top of the single-upstream baseline" and
gives publishable p50/p95/p99 numbers for the v0.40 PR-body scope note.

Honest scope: the upstream subprocess is mocked at the `UpstreamMCPClient`
boundary, so the measurement isolates Vaara's added cost (HTTP parse,
tenant + upstream header resolution, `Pipeline.intercept`, dispatch). It
does NOT include real stdio JSON-RPC roundtrip to a live MCP server,
which depends on the upstream's own runtime and is out of scope for a
governance-overhead number.

Run::

    python bench/latency_fanout.py                       # default: N=[1,2,4,8], 2000 calls each
    python bench/latency_fanout.py --calls 5000
    python bench/latency_fanout.py --upstreams 1,2,4,8,16
    python bench/latency_fanout.py --json bench/v040_fanout.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from vaara.integrations import mcp_proxy  # noqa: E402
from vaara.integrations.mcp_proxy import VaaraMCPProxy  # noqa: E402
from vaara.pipeline import InterceptionPipeline  # noqa: E402


@dataclass
class FanoutStats:
    n_upstreams: int
    n_calls: int
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


def _summarize(n_upstreams: int, samples_ms: list[float]) -> FanoutStats:
    s = sorted(samples_ms)
    total_s = sum(samples_ms) / 1000.0
    return FanoutStats(
        n_upstreams=n_upstreams,
        n_calls=len(samples_ms),
        mean_ms=round(statistics.fmean(samples_ms), 4),
        p50_ms=round(_percentile(s, 0.50), 4),
        p95_ms=round(_percentile(s, 0.95), 4),
        p99_ms=round(_percentile(s, 0.99), 4),
        p999_ms=round(_percentile(s, 0.999), 4),
        min_ms=round(s[0], 4),
        max_ms=round(s[-1], 4),
        throughput_per_sec=round(len(samples_ms) / total_s, 1) if total_s > 0 else 0.0,
    )


def _make_app(n_upstreams: int, pipeline: InterceptionPipeline):
    """Build a FastAPI app with N mocked upstreams. Returns (app, slot_names)."""
    # The mock returns a fixed tools/list response so the proxy's own
    # response-handling path executes end-to-end without a real subprocess.
    def make_client(*_a, **_kw):
        c = MagicMock()
        c.request.return_value = {
            "jsonrpc": "2.0", "id": 1, "result": {"tools": []},
        }
        return c

    original_cls = mcp_proxy.UpstreamMCPClient
    mcp_proxy.UpstreamMCPClient = MagicMock(side_effect=make_client)
    try:
        if n_upstreams == 1:
            proxy = VaaraMCPProxy(upstream_command=["mock"], pipeline=pipeline)
            slot_names = ["default"]
        else:
            upstreams = {f"u{i}": [f"mock-{i}"] for i in range(n_upstreams)}
            proxy = VaaraMCPProxy(upstreams=upstreams, pipeline=pipeline)
            slot_names = sorted(upstreams.keys())

        # Replicate run_http()'s FastAPI build without uvicorn.run().
        import unittest.mock as um
        captured: dict = {}
        with um.patch("uvicorn.run") as run_mock:
            def fake_run(app, **_kw):
                captured["app"] = app
            run_mock.side_effect = fake_run
            proxy.run_http(host="127.0.0.1", port=0)
        return captured["app"], slot_names
    finally:
        mcp_proxy.UpstreamMCPClient = original_cls


def bench_fanout(n_upstreams: int, n_calls: int, n_warmup: int) -> FanoutStats:
    pipeline = InterceptionPipeline()
    app, slot_names = _make_app(n_upstreams, pipeline)
    client = TestClient(app)
    # Single-upstream keeps v0.39 silent-default contract — no header.
    # Multi-upstream requires explicit X-Vaara-Upstream per v0.40 4XX rule.
    rng = random.Random(0xC0FFEE)

    def call_once(i: int) -> float:
        headers = {"X-Vaara-Tenant": f"t{i & 0x3}"}
        if n_upstreams > 1:
            headers["X-Vaara-Upstream"] = rng.choice(slot_names)
        body = {
            "jsonrpc": "2.0",
            "id": i,
            "method": "tools/call",
            "params": {
                "name": "data.read",
                "arguments": {"path": "s3://bucket/key"},
            },
        }
        t0 = time.perf_counter_ns()
        client.post("/mcp", json=body, headers=headers)
        return (time.perf_counter_ns() - t0) / 1_000_000.0

    for i in range(n_warmup):
        call_once(i)

    samples_ms = [call_once(i) for i in range(n_calls)]
    return _summarize(n_upstreams, samples_ms)


def _print_markdown_table(rows: list[FanoutStats]) -> None:
    print()
    print("| N upstreams | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | p99.9 (ms) | throughput/s |")
    print("|-------------|-----------|----------|----------|----------|------------|--------------|")
    for r in rows:
        print(
            f"| {r.n_upstreams:>11} | {r.mean_ms:>9.3f} | {r.p50_ms:>8.3f} | "
            f"{r.p95_ms:>8.3f} | {r.p99_ms:>8.3f} | {r.p999_ms:>10.3f} | "
            f"{r.throughput_per_sec:>12,.0f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=2000,
                        help="calls per N-upstream configuration (default: 2000)")
    parser.add_argument("--warmup", type=int, default=200,
                        help="warmup calls per configuration (default: 200)")
    parser.add_argument("--upstreams", type=str, default="1,2,4,8",
                        help="comma-separated N values (default: 1,2,4,8)")
    parser.add_argument("--json", type=str, default=None,
                        help="dump machine-readable results to PATH")
    args = parser.parse_args()

    n_values = [int(x) for x in args.upstreams.split(",") if x.strip()]
    rows: list[FanoutStats] = []
    for n in n_values:
        print(f"[bench] N={n} upstreams, {args.calls} calls (+{args.warmup} warmup)...", flush=True)
        rows.append(bench_fanout(n, args.calls, args.warmup))

    _print_markdown_table(rows)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": [asdict(r) for r in rows]}, indent=2))
        print(f"\n[bench] wrote {out}")


if __name__ == "__main__":
    main()
