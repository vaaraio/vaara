# AGENTS.md

Instructions for coding agents working in or integrating this repository. The human-oriented overview is [README.md](README.md); the machine-oriented index of docs and published numbers is [llms.txt](llms.txt).

## What this is

Vaara is a runtime evidence and enforcement layer for AI agents. It intercepts agent tool calls, scores each one, decides allow / block / escalate against a declarative policy, and writes the call, the decision, and the outcome into a signed, hash-chained audit trail that an outside party can verify offline. Python 3.10+, zero runtime dependencies for the core. License: AGPL-3.0, commercial licenses available.

## Integrating Vaara into an application

The one-line path:

```python
import vaara

@vaara.govern
def transfer_funds(to: str, amount: float) -> str:
    ...
```

A blocked call raises `vaara.Blocked`. The explicit engine is `vaara.pipeline.InterceptionPipeline` (`intercept(...)` then `report_outcome(...)`). Policies are YAML/JSON loaded with `vaara.policy.from_yaml()`; starter policies for common MCP servers are in `examples/policies/mcp-starters/`.

CLI entry points: `vaara` (policy validate/test, trail, review, verify-bundle), `vaara-audit`, `vaara-mcp-proxy` (governs an existing MCP server), `vaara-mcp-server`.

Runnable references, in rough order of usefulness: `examples/quickstart.py`, `examples/prove-it-yourself/` (produce, verify, and tamper-check an evidence bundle), `examples/governance_demo.py`, `examples/langchain_agent.py`, the MCP proxy demos (`examples/github-mcp-proxy-demo/`, `examples/goose-mcp-proxy-demo/`, `examples/sap-mcp-proxy-demo/`).

Optional extras gate heavier features: `vaara[ml]` (classifier), `vaara[export]` (signed bundle export), others listed in `pyproject.toml`.

## Working on this repository

Setup:

```bash
pip install --require-hashes -r requirements-dev.txt
pip install -e . --no-deps
```

Checks that must pass (same as CI):

```bash
ruff check .
pytest -q
mypy src/vaara/policy/
```

The mypy strict set covers `vaara.policy.*` only, per `[[tool.mypy.overrides]]` in `pyproject.toml`; do not widen it casually. Tests run on Python 3.10 through 3.13. Some tests are environment-gated and skip locally (ML gate bundles, live timestamp authority, reportlab); skips are expected, failures are not.

Layout:

- `src/vaara/` core package: `policy/` (schema, validation), `audit/` (trail, receipts, anchoring), `pipeline.py` and `govern.py` (interception), `scorer/`, `integrations/` (MCP, LangChain, CrewAI, OpenAI Agents SDK, cloud guardrails), `compliance/` (AI Act article mapping engine), `server/` (HTTP API).
- `tests/` mirrors the package; conformance vectors live in `tests/vectors/`.
- `docs/` reference documentation; `SPEC.md` is the canonical receipt format.
- `bench/` frozen benchmark methodology and corpus.

## Conventions and constraints

- The public surface is versioned and frozen: the signed envelope (`vaara.receipt/v1`), capability tokens, and conformance vectors must not change shape. New behavior goes behind new fields or new profiles, never by mutating existing vector semantics.
- Zero runtime dependencies in the core is a hard property. New imports in the default path need to come from the standard library; anything heavier belongs behind an optional extra.
- Follow existing style; ruff and the CI mypy set are the arbiters. Do not add TODO markers, bare excepts, or `shell=True`.
- Every claim in docs must be reproducible from the repo (a runnable example, a test, or a published benchmark). Do not add unverifiable numbers.
- Verify evidence-related changes end to end: `python examples/prove-it-yourself/prove_it.py` must exit 0, and `python tests/vectors/external_evidence_v0/_check_independent.py` must keep passing.

## Key documents

- [SPEC.md](SPEC.md): the `vaara.receipt/v1` receipt format.
- [docs/verifying-evidence.md](docs/verifying-evidence.md): every verifier and its trust model.
- [docs/COMPLIANCE.md](docs/COMPLIANCE.md): EU AI Act and DORA article-level evidence mapping.
- [docs/architecture.md](docs/architecture.md): scoring, conformal coverage, time anchoring.
- [docs/logs-vs-evidence.md](docs/logs-vs-evidence.md): the trust model behind the whole design.
- [CHANGELOG.md](CHANGELOG.md): version history.
