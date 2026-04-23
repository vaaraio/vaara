# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-04-23

### Added
- `AdversarialClassifier` — opt-in XGBoost scorer for adversarial tool-call detection. Install with `pip install vaara[ml]`.
- `src/vaara/data/adversarial_classifier_v1.joblib` — 295 KB pre-trained bundle shipped with the wheel.
- `scripts/classifier_vs_heuristic.py` — reproducible comparison harness (by-seed train/test split, no leakage).
- `tests/adversarial/generated/` — 1945 LLM-generated adversarial variants across 8 attack categories (Qwen2.5-3B on MI300X, batched).
- `tests/adversarial/benign_generated/` — 635 benign variants (487 broad + 148 targeted at http_post/send_email/shell_exec legit uses).
- `[project.optional-dependencies.ml]` — `xgboost`, `scikit-learn`, `joblib`, `numpy`. Default install stays zero-dep.

### Benchmarks
On the by-seed held-out test (50 adversarial seeds held out entirely, plus 60 benign entries), classifier at threshold 0.8:
- Attack recall: 61% (vs 44% heuristic)
- Benign false-positive rate: 13% (vs 25% heuristic)
- Balanced accuracy: 74% (vs 50% heuristic)
- Latency: 140 µs mean, 210 µs p99 (commodity CPU, no GPU at inference)

Live dogfood on 174 Qwen-generated prompts with an end-to-end LangChain-style agent loop: recall 70%, FPR 51% — real-world distribution is tougher than the clean held-out test. Ship with `decision="escalate"` in production, not `decision="deny"`.

## [0.4.4] - 2026-04-22

### Fixed
- `vaara_wrap_tool` signature preservation via `functools.wraps` so LangChain `BaseTool.run()` injects `config` and `run_manager` kwargs under langchain-core >= 0.3.

### Added
- `examples/langchain_agent.py` runnable end-to-end with minimal action taxonomy and pre-calibrated pipeline.

## [0.4.3] - 2026-04-21

### Added
- Sigstore-signed release workflow with PyPI trusted publishing and PEP 740 attestations
- CodeQL SAST workflow
- OpenSSF Best Practices (Passing) badge
- CONTRIBUTING.md

### Changed
- Scorecard workflow pinned to peeled commit SHAs, with `workflow_dispatch` trigger added
- SECURITY.md restructured for OpenSSF Scorecard Security-Policy check

## [0.4.2] - 2026-04-20

### Added
- `Pipeline` alias for `InterceptionPipeline` at the top-level import.

## [0.4.1] - 2026-04-20

### Added
- Signed audit-trail export with verification tooling
- CLI surface for common operations
- Boundary sanitization on ingress/egress paths

### Changed
- README quick-start uses a generic filesystem example (was domain-specific)
- `InterceptionPipeline` is now importable directly from the top-level package

### Removed
- Domain-specific taxonomy modules (moved to plugin scope)

## [0.3.0] - 2026-04-18

### Added
- Framework integrations: LangChain, CrewAI, OpenAI Agents
- MCP server surface
- SQLite audit persistence

## [0.1.0] - 2026-04-10

- Initial release: interception pipeline, adaptive scoring, hash-chained audit trail
