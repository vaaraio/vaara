# Vaara v0.46 concurrency and governance-overhead bench

This release makes the multi-tenant runtime-governance claim defensible
rather than aspirational. Two correctness properties and one latency
number back it.

## 1. Concurrent requests do not serialise (HTTP transport)

The streamable-HTTP transport runs the blocking `_handle_request` on a
worker thread (`asyncio.to_thread`) with per-request ContextVars copied
across the hop. `tests/test_http_concurrency.py` asserts two in-flight
POSTs to different upstream slots overlap in wall-clock (well under
`2 * upstream_sleep`) and each still routes to its own slot. Before this,
real concurrency was 1: one slow upstream stalled every other POST, SSE
drain, and `/health`.

## 2. Concurrent multi-tenant lifecycles keep their scope (audit trail)

The action -> tenant map (`AuditTrail._tenant_for_action`) is now guarded
by a dedicated lock. `tests/test_v040_tenant.py` adds two proofs under
real thread contention:

- `test_concurrent_multi_tenant_lifecycles_preserve_scope_and_chain`:
  16 threads, each its own tenant, run 40 full lifecycles
  (request, decision, execution) released together on a barrier. After
  join, the hash chain verifies intact and every action's three records
  carry the tenant captured at request time. No cross-tenant bleed.
- `test_concurrent_tenant_map_eviction_is_threadsafe`: 8 threads push 300
  actions each (past the soft cap) so eviction runs concurrently with
  reads. The unguarded dict previously could raise "dictionary changed
  size during iteration"; with the lock it completes cleanly.

## 3. Governance overhead per `tools/call`

Per-call overhead Vaara adds routing a `tools/call` through the HTTP
transport, fan-out across N upstream slots, with the upstream subprocess
mocked at the `UpstreamMCPClient` boundary so the number isolates Vaara's
own cost (HTTP parse, tenant + upstream header resolution,
`Pipeline.intercept`, dispatch). It excludes the real stdio JSON-RPC
roundtrip to a live MCP server, which depends on the upstream's runtime.

Harness: `bench/latency_fanout.py --calls 3000 --upstreams 1,2,4,8`.
Machine: commodity Linux x86-64 dev box. Raw: `bench/v046_fanout.json`.

| N upstreams | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | p99.9 (ms) | throughput/s |
|-------------|-----------|----------|----------|----------|------------|--------------|
|           1 |     1.571 |    1.559 |    1.699 |    1.773 |      1.912 |          636 |
|           2 |     1.574 |    1.562 |    1.699 |    1.790 |      1.999 |          635 |
|           4 |     1.584 |    1.565 |    1.714 |    1.823 |      2.014 |          631 |
|           8 |     1.580 |    1.562 |    1.704 |    1.799 |      2.001 |          633 |

Overhead is sub-2ms p50 and flat across fan-out: routing one upstream or
eight costs the same per call, and the tenant-map lock adds no measurable
regression. Single-process governance is not the bottleneck for an MCP
fleet at this scale.
