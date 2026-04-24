# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2026-04-24

### Fixed
- `AdversarialClassifier` bundle in v0.5.0 and v0.5.1 scored unreliably because of two pipeline bugs. Both now fixed, bundle rebuilt with `version: 1.1`.
  - **Feature-name drift between trainer and runtime.** `scripts/train_adversarial_classifier.py` and `src/vaara/adversarial_classifier.py` each defined their own `_SHELL_PATTERNS`. Only 2 of 5 overlapped. The model learned columns whose regexes did not match the features the runtime built at inference time, silently inverting scores on `shell_exec` traffic.
  - **Silent zero-variant training.** `scripts/classifier_vs_heuristic.py` pointed `VARIANTS_DIR` at `tests/adversarial/generated_by_category/`, which never existed. The script loaded zero attack variants and trained only on 250 seeds + 1,145 benigns. Class balance on `http_post`, `send_email`, `shell_exec` was skewed heavily benign. With the right path, training now loads all 1,945 attack variants and produces 2,740 train / 600 test entries.

### Added
- `_STATIC_FEATURES` constant plus load-time schema-drift check in `src/vaara/adversarial_classifier.py`. A bundle whose `feature_names` tail diverges from the runtime static feature list now raises `ValueError` at construction time, pinpointing the first differing index. This class of bug is no longer shippable without failing loud.

### Changed
- Default threshold: `0.3` (v0.5.1) → `0.5`. Balanced accuracy peaks at 0.5 on the rebuilt bundle. The v0.5.1 claim of "52% recall, 3.3% FPR at threshold 0.3" was itself a recordkeeping error: the bundle saved `0.8`, not `0.3`, and those numbers were measured at 0.8.
- `scripts/train_adversarial_classifier.py` `load_corpus` now uses `rglob` to recurse into `tests/adversarial/generated/` and `benign_generated/` automatically.
- The `data_exfil` and `destructive_actions` regressions disclosed in v0.5.1 were artifacts of the broken bundle. The rebuilt classifier in v0.5.2 beats the heuristic in both: `destructive_actions` +40.2, `data_exfil` +24.7.
- Bundle format `version` bumped 1.0 → 1.1.

### Benchmarks (by-seed held-out, threshold 0.5)
- Attack recall: **85.2%**
- Benign false-positive rate: **23.3%**
- Balanced accuracy: **80.9%**
- Classifier beats heuristic by **+42.0 balanced-accuracy points** overall.
- Wins over heuristic on 8 of 9 attack categories: `ssrf_via_tools` +76.5, `tool_misuse` +73.1, `privilege_escalation` +58.1, `destructive_actions` +40.2, `credential_exfil` +39.0, `benign_control` +38.3, `prompt_injection` +26.6, `data_exfil` +24.7.

### Threshold sweep

| threshold | attack recall | benign FPR | balanced accuracy |
|-----------|---------------|------------|-------------------|
| 0.3 | 92.0% | 38.3% | 76.9% |
| **0.5** | **85.2%** | **23.3%** | **80.9%** |
| 0.7 | 61.5% | 8.3% | 76.6% |
| 0.8 | 47.2% | 3.3% | 71.9% |
| 0.9 | 32.6% | 0.0% | 66.3% |

Use `ESCALATE`, not `DENY`, at threshold 0.5 on mixed-intent tool traffic. 23% FPR is a reviewer queue, not a blast door.

### Known regressions
- `jailbreak`: classifier scores 0% on 11 held-out examples, down from heuristic 90.9%. Stack the heuristic rather than using the classifier alone on this category. v0.5.3 will widen jailbreak variant coverage.
- Legit `read_file` on low-signal paths (e.g. `README.md`) still trips the gate at threshold 0.5. Benign `read_file` is under-represented in training. v0.5.3.

### Recovery note
v0.5.1 remains on PyPI but ships a broken classifier. Upgrade to 0.5.2 to get the fixed bundle. The heuristic-only path (`pip install vaara`, no `[ml]` extras) was unaffected in both releases.

## [0.5.1] - 2026-04-23

### Changed
- `AdversarialClassifier` retrained on an expanded benign corpus to reduce false-positive rate in live agent traffic.
- Recommended operating threshold changed from `0.8` to `0.3` — the added benigns shifted the score distribution, and 0.3 is now the optimal balanced-accuracy point.

### Added
- `tests/adversarial/benign_generated/BT-new-http_post.jsonl` (170 variants)
- `tests/adversarial/benign_generated/BT-new-send_email.jsonl` (170 variants)
- `tests/adversarial/benign_generated/BT-new-shell_exec.jsonl` (170 variants)
- Total benign corpus size: 1,145 (up from 635 in v0.5.0)

### Benchmarks (by-seed held-out, threshold 0.3)
- Attack recall: 52% (vs 61% in v0.5.0 at threshold 0.8)
- Benign false-positive rate: **3.3%** (vs 13% in v0.5.0)
- Balanced accuracy: 74% (same)
- Latency: 140 µs mean, 210 µs p99 (unchanged)

### Known regressions (disclosed)
The new benigns shifted the decision surface toward allow. Per-category accuracy regressed in three attack categories:
- `data_exfil`: 0% (was 28.6% heuristic baseline — classifier now worse than heuristic here)
- `destructive_actions`: 25% (was 87.5% heuristic)
- `jailbreak`: 0% (was 100% heuristic)

The heuristic scorer retains strong coverage in these categories. Stack both rather than using the classifier alone. v0.5.2 will add targeted attack variants to rebalance the training corpus and recover recall without losing the FPR improvement.

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
