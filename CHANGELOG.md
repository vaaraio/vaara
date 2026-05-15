# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-05-15

**Theme: policy artifact validate + test framework.** v0.9.0 ships the
two CLI surfaces that turn the YAML / JSON policy from "a config file
the pipeline loads" into a policy-as-code artifact that compliance
teams can validate and test in CI, independently from the agent code
it governs.

### Added
- **`vaara.policy.validate` module.** `validate(Policy)` returns a
  `ValidationReport` with structured `PolicyIssue` records (level,
  code, path, message). Semantic warnings emitted: empty
  `action_classes`, threshold bands narrower than 0.05 (default and
  per-class merged), threshold overrides targeting an action class
  not declared, sequence-pattern steps that do not name a declared
  action class, escalation routes whose `if_articles` overlap no
  emitted regulatory tag, missing default escalation route.
  `validate_source(source, fmt="auto")` combines load and check so a
  single call yields `(policy, report)` or `(None,
  report-with-error)`. Stable JSON shape via `ValidationReport.to_dict()`.
- **`vaara.policy.test_cases` module — Conftest analog for Vaara
  policies.** `evaluate(policy, action_class, risk_score,
  matched_sequences=())` is the underlying primitive: applies any
  matched sequence pattern boosts (capped at 1.0), resolves the
  merged threshold for the action class, returns
  `EvaluationResult(verdict, boosted_risk, route)`. `PolicyTestCase`
  captures the inputs plus an expected verdict and (for
  `escalate`) an expected operator route. `run_test_cases(policy,
  cases)` runs the list, captures evaluation errors as failed cases
  rather than raising, and returns `PolicyTestResult` rows. The
  evaluator validates inputs at the boundary (risk score in `[0,1]`,
  action class declared, matched sequences known).
- **`vaara.policy.test_cases_io` module.** `load_test_cases(path)`
  reads a YAML or JSON cases document and returns a list of
  `PolicyTestCase`. Document shape mirrors typical OPA / Conftest
  test files: a top-level `cases:` list with `action_class`,
  `risk_score`, optional `matched_sequences`, and an `expect:` block
  carrying `verdict` and optional `route`.
- **`vaara policy validate POLICY_PATH [--json]`** and **`vaara
  policy test POLICY_PATH --cases CASES_PATH [--json]`** CLI
  subcommands. Both honour standard CI exit codes: validate returns
  1 on parse errors (warnings do not flip), test returns 1 on any
  failed case (and 2 if the policy itself fails to parse).
- **`examples/policies/test_cases.yaml`** — six worked test cases
  exercising thresholds, sequence-pattern boost, default and
  article-matched escalation routes against
  `examples/policies/full.yaml`.
- **48 new tests** (`tests/test_policy_validate.py`,
  `tests/test_policy_test_cases.py`,
  `tests/test_policy_test_cases_io.py`, `tests/test_policy_cli.py`)
  covering report shape, every warning code, evaluator edges
  (threshold-equal-escalate, boost cap at 1.0, unknown action class,
  unknown sequence, out-of-range risk), case-construction validation
  (bad verdict, route without escalate), YAML and JSON case files,
  the worked example end-to-end, and CLI smoke for each subcommand
  including `--json`. Full suite 472 / 472 pass.
- **COMPLIANCE.md** gains a *Policy artifact review* subsection under
  Article 14 documenting both CLI surfaces as the path to reviewing
  the policy artifact independently from the agent code.

### Note
Backwards-compatible. Pure addition. No existing module signatures
change. `Policy` and the load path are unchanged; the new modules
sit beside them under `vaara.policy.*`.

### Provenance note
The PyPI artifact for v0.9.0 was uploaded via `twine` (not via
trusted publishing) after a GitHub Actions infrastructure outage on
2026-05-15 caused the Release workflow's Build job to fail at the
artifact upload step, which skipped the downstream
trusted-publishing job. The wheel and sdist on PyPI are bit-identical
to what `python -m build` produced from commit `ea201fe`. Subsequent
releases use trusted publishing as standard.

## [0.8.0] - 2026-05-14

**Theme: Article 73 serious-incident export (interim).** Adds the export
surface a provider needs to satisfy EU AI Act Article 73 reporting
obligations. The Commission template promised by Article 73 paragraph 7
has not published, so the format is explicitly INTERIM. When the template
publishes, the schema_version bumps and new fields land additively.

### Added
- **`vaara.audit.incident_export` module.** `build_incident_report()` and
  `write_incident_report()` produce a standalone JSON document covering
  the fields Article 73 paragraphs 1 to 7 reference. Schema version
  `vaara-incident/1.0`. Reporting deadline is derived from the Article
  3(49) sub-category: general 15 days, death of a person 10 days,
  Article 3(49)(b) widespread or serious 2 days. Trigger record must have
  `event_type` in `{outcome_recorded, action_blocked, policy_override}`.
  Other event types describe normal operation and are rejected at
  build time.
  Paragraph 5 incomplete-initial plus complete-follow-up workflow is
  supported via `report_status` and `previous_report_id`. Audit records
  are referenced by `record_id` as evidence. The report does not
  duplicate their content. Pair with `vaara trail export-prov` for
  the full evidence bundle.
- **`vaara trail export-incident` CLI subcommand.** Reads a trail JSONL
  plus an operator-supplied incident metadata JSON, writes the report
  to the output path. Picks the most recent trigger-eligible record by
  default; explicit `--trigger-record-id ID` overrides. No external
  template dependency, zero new runtime deps.
- **`tests/test_incident_export.py`** covers schema shape, deadline
  mapping per Article 3(49) sub-category, trigger-event validation,
  causal-link and reporter-role validation, initial-versus-complete
  report sequencing, and end-to-end trail-to-report flow.

### Note
Backwards-compatible. Pure addition. The `AuditRecord` schema is
unchanged. Severity is an incident-level concept, not a per-event
concept, so no `AuditRecord` column was added.

---

**Theme: human-in-the-loop review queue (Article 14).** Adds the
storage layer and operator surface that turn an `escalate` decision
into a substantive Article 14(4)(d) override path. The pipeline
already wrote `ESCALATION_SENT` for every escalated action; with a
queue wired in, those actions now wait in a queryable place with
their conformal interval, get claimed by an operator, and produce an
`ESCALATION_RESOLVED` audit record when resolved.

### Added
- **`vaara.audit.review_queue` module.** `ReviewQueue` is a
  SQLite-backed queue in its own DB file, separate from the audit DB
  (which keeps its append-only invariant clean). Statuses:
  `pending → claimed → resolved` happy path, `pending → expired`
  stale path. Resolutions: `allow`, `deny`, `abstain`. `enqueue`
  records each item with the conformal interval, risk signals,
  bucket category, and request parameters/context as JSON; the
  interval is what makes Article 14 oversight substantive rather
  than cosmetic (see `COMPLIANCE.md`). `claim` is optimistic —
  concurrent claim races resolve with one winner and
  `InvalidTransitionError` for the loser. `resolve` accepts an
  optional `trail` and writes `ESCALATION_RESOLVED` so the Article
  14(4)(d) evidence row lands on the hash chain. `expire_stale`
  marks pending items past a timeout; claimed items are left alone
  since they are under active review.
- **`InterceptionPipeline(review_queue=...)`.** Optional constructor
  parameter. When supplied, every `escalate` decision is enqueued
  alongside the existing `ESCALATION_SENT` audit record. Default
  `None` preserves prior behaviour bit-for-bit. Queue write failure
  logs and continues — the action is already gated by the escalate
  verdict and the audit record stands.
- **`vaara review` CLI.** Subcommands `list`, `show`, `claim`,
  `resolve`, `expire`. `resolve --audit-db PATH` writes the
  `ESCALATION_RESOLVED` record into an audit DB at that path so a
  single operator command produces both the queue terminal state
  and the Article 14(4)(d) hash-chain evidence.
- **`tests/test_review_queue.py`** covers schema round trip,
  pending-only and cross-status listing, agent-id filter, claim
  race semantics, resolve from `pending` and from `claimed`,
  unknown-resolution rejection, terminal-state guard, trail
  write-through, expire-stale with `dry_run`, claimed-items-left-alone,
  positive-timeout guard, counts partitioning, file-backed
  persistence, oversized-blob capping, pipeline enqueue-on-escalate
  end-to-end, no-enqueue-on-allow, and CLI smoke for each
  subcommand including `resolve --audit-db` writing the audit row.

### Note
Backwards-compatible. Pure addition. No existing schemas migrate;
the queue lives in its own DB file with its own `review_queue_meta`
schema-version row.

## [0.7.0] - 2026-05-10

**Theme: class-conditional conformal calibration.** v0.7.0 adds Mondrian per-category conformal prediction on top of the marginal split conformal that has shipped since v0.5.x. The same coverage guarantee now holds independently per action category, so a 90% headline can no longer hide a 60% miss rate on `credential_exfil` behind a 99% pass rate on `benign_control`. Eval surfaces the per-category breakdown. PROV-DM exports surface the calibration context an external auditor needs to read each interval honestly.

### Added
- **Mondrian per-category conformal calibration in `ConformalCalibrator`.** Optional `category` argument on `add_calibration_point` and `predict_interval` routes residuals to a per-category bucket. Each bucket carries its own residual deque, FACI alpha trajectory, and conformal quantile, so a per-category coverage guarantee holds independently. New helpers: `calibration_size_for(category)`, `is_calibrated_for(category)`, `effective_alpha_for(category)`. Calls without a `category` argument land in a single shared bucket and behaviour is identical bit-for-bit to marginal split conformal. Reference: Vovk (2012), "Conditional validity of inductive conformal predictors". PR #60.
- **`AdaptiveScorer.mondrian_categories` constructor flag.** When set to `True`, each action's category (derived from the `tool_name` prefix via `_category_of`) routes through to the calibrator at all three call sites: `evaluate`, `dry_run_evaluate`, `record_outcome`. Default `False` preserves the marginal contract for existing callers. The 50-pair seed prior continues to populate the default bucket regardless, so a fresh Mondrian-mode scorer falls back to the conservative `point ± 0.3` interval per untouched bucket. PR #61.
- **`scripts/eval_adversarial.py --mondrian` flag.** Constructs `Pipeline(scorer=AdaptiveScorer(mondrian_categories=True))` so the adversarial corpus can be evaluated under both regimes for direct comparison. PR #61.
- **Empirical conformal coverage in `scripts/eval_adversarial.py`.** Per-category and overall: `coverage` (fraction of entries where `lower <= true_risk <= upper`) and `mean_interval_width` (mean of `upper - lower`). Per-entry rows now also carry `lower`, `upper`, and `actual_risk`. Result JSON gains an `overall` block alongside the existing `summary`. Eval-only change. PR #59.
- **Conformal calibration context in PROV-DM exports.** W3C PROV-JSON score Entities now surface `vaara:calibrationSize`, `vaara:effectiveAlpha`, and (in Mondrian mode) `vaara:bucketCategory`. An external auditor can now distinguish a wide interval from a cold calibrator, from FACI alpha drift, or from genuine uncertainty without re-running Vaara. `RiskAssessment` dataclass gains `effective_alpha` and `bucket_category` fields. `Pipeline` writes them through to the audit trail with defensive coercion. Pre-enrichment audit records produce identical PROV output (attributes are omitted when source data is missing or `None`). PR #62.

### Changed
- `vaara.scorer.adaptive.RiskAssessment` adds two optional fields: `effective_alpha: float = 0.10` and `bucket_category: Optional[str] = None`. Existing constructors that do not pass them keep working unchanged. PR #62.

### Note
Backwards-compatible release. All four PRs are additive. Mondrian is opt-in via `mondrian_categories=True`, and PROV enrichment fields are omitted when their source data is missing.

## [0.6.2] - 2026-05-05

**Theme: standards-track lineage.** v0.6.2 adds W3C PROV-DM as a second standards-track audit format alongside the JTC21-bound trajectory of v0.6.0. The audit record schema and event lifecycle are unchanged. Vaara now also emits PROV-JSON, so any PROV-aware consumer can ingest a trail without a Vaara-specific adapter.

### Added
- **W3C PROV-DM audit-trail export.** New module `vaara.audit.prov_export` and CLI subcommand `vaara trail export-prov --trail PATH --out PATH [--action-id ID] [--no-chain]`. Emits PROV-JSON (W3C Submission, 2013). Two layers: per-action bundles (lifecycle as Activities, request/score/decision/outcome as Entities, AI agent + Vaara pipeline + human reviewer as Agents) and an audit-record chain layer (each record as a `prov:Bundle`-typed Entity, consecutive records linked via `wasDerivedFrom`/`prov:Revision`). Regulatory articles surface as `aiact:satisfies` and `dora:satisfies` attributes on the Activity that generated them. Decision and outcome entity IDs are scoped by `record_id` so multi-override lifecycles preserve every prior decision. Chain edges are derived from `previous_hash` (not iteration order) so filtered slices and equal-timestamp records never assert false lineage. Cryptographic chain verification stays Vaara's job (`vaara trail verify`). Zero new runtime deps.

## [0.6.1] - 2026-04-27

**Theme: documentation sync to PyPI.** v0.6.0 shipped the functional changes (policy DSL, retention purge, transparency taxonomy, distribution-shift / stack-ablation / PAIR evals, lint sweep) but the README, library docstring, example file headers, and PyPI tagline stayed at the v0.5.0 framing. PyPI's package page kept publishing pre-rebalance numbers and the wrong default threshold. v0.6.1 ships only the documentation cleanup so new PyPI installs see the current state.

### Changed
- `README.md`: replaced the "Numbers" section with the v0.6 distribution-shift table (97.1% recall / 70.0% FPR hand-curated held-out; 95.2% / 87.5% LLM-generated in-sample) plus PAIR ASR 0.0% (0/25). Threshold default 0.5 -> 0.55, corpus description updated to the 5,955-entry rebalanced corpus, threshold-direction note corrected (recall drops as threshold rises). PR #44.
- `src/vaara/adversarial_classifier.py`: rewrote the module docstring. Removed all version-bound numbers; readers now point at README + COMPLIANCE so the docstring does not go stale on every release. PR #44.
- `examples/adversarial_classifier.py`: ship-note threshold 0.8 -> 0.55. PR #44.
- `scripts/classifier_vs_heuristic.py`: clarified the script is the v0.5.0 historical reproducer, not the current production training path. PR #44.
- `pyproject.toml`: rephrased description for cleaner PyPI tagline rendering. PR #45.

### Note
No functional code changes. v0.6.0 users are on the same code; v0.6.1 only refreshes documentation surfaces visible to new PyPI installs and to anyone reading the package source.

## [0.6.0] - 2026-04-27

**Theme: standards alignment + legibility.** v0.5.x was the capability axis (jailbreak coverage closed, classifier rebalanced). v0.6 is the legibility axis: policies become readable, audit records become standards-aligned, adversarial numbers become honest, architecture contribution becomes documented.

### Added
- **`vaara.policy` package — JSON-native policy loader plus optional YAML via `vaara[yaml]` extra.** Frozen dataclasses for action classes, threshold curves, sequence patterns, and escalation routes. Hand-rolled validation with field-path error messages. Reuses existing `vaara.taxonomy.actions` enums verbatim. Threshold partial-overrides supported (set just `deny`, inherit default `escalate`). Implements Sketch A from the v0.6 DSL design exploration; embedded Python DSL (Sketch B) and standalone DSL (Sketch C) stay deferred to v0.7+ pending external pull.
- **`vaara trail purge --db PATH --retention-days N (--tenant TID | --all-tenants) [--dry-run]` CLI subcommand** plus `SQLiteAuditBackend.purge_older_than(seconds, *, dry_run=False)` Python API. Article 12(2) retention enforcement. Tenant scoping is required: pick `--tenant TID` for a single tenant or `--all-tenants` explicitly, so a shared multi-tenant audit DB can never be silently purged across all tenants. Hash-chain integrity: surviving records still reference deleted predecessors via `previous_hash`, leaving a documented seam at the retention boundary that subsequent loads expose as a hash mismatch. Intended workflow: export a signed handoff zip BEFORE purging, archive externally, then purge. The signed zip remains self-consistent forever; the live DB chain has the seam.
- **prEN ISO/IEC 12792 four-axis transparency taxonomy on `AuditRecord`.** Four optional fields (`system_operation`, `data_usage`, `decision_making`, `limitations`) with default-classification heuristic per `EventType`. Per-record override via construction kwargs. NOT tamper-evident in v0.6 — fields are metadata annotations excluded from `record_hash` so pre-v0.6 chains stay valid. v0.7+ may add a separate signing mechanism if compliance requires.
- **`scripts/eval_distribution_shift.py`** — runs the full Vaara stack against the adversarial corpus with per-source tagging (hand-curated vs LLM-generated). Reports recall and FPR per source/class.
- **`scripts/eval_stack_ablation.py`** — runs three configurations (heuristic-only, classifier-only, full-stack) against the same corpus. Quantifies the independent contribution of each layer.
- **`scripts/eval_pair_attack.py`** — PAIR (Chao et al. 2023) iterative adaptive attacker. Uses an OpenAI-compatible vLLM endpoint for both attacker and judge roles. Zero new runtime deps (uses `urllib.request`).
- **`[yaml]` optional extra in `pyproject.toml`** (`pyyaml>=6.0`). Core `dependencies = []` preserved.
- **`examples/policies/minimal.json` and `full.yaml`** as reference policies.
- **COMPLIANCE.md gains "EU AI Act Annex IV evidence sections"** (maps Vaara contribution per §1–§9; direct fill on §3, §5, §9; contributes on §2, §4, §6, §7; out of scope for §1, §8) **and "CEN-CENELEC harmonised standards alignment"** (per-standard table for ISO/IEC 42001, prEN 18286, prEN 18228, ISO/IEC 42006, prEN ISO/IEC 24970, prEN 18229-1, prEN ISO/IEC 12792).
- **`scripts/lint_full.sh` pre-push lint sweep** — chains `ruff` (style + correctness), `bandit` (security), `mypy` (types — strict on `vaara.policy`, lenient on legacy modules), and `pytest`. Documented in CONTRIBUTING.md. Catches CodeRabbit-class findings before they hit a PR review round-trip. New dev extras: `bandit>=1.7.5`, `mypy>=1.8`. Bandit configured in `pyproject.toml` to skip B608 across `audit/sqlite_backend.py` (all f-string SQL there interpolates only internally-controlled tenant clauses, not user input). Two `# nosec` annotations document the remaining trusted-bundle and synthetic-trace-RNG sites.

### Changed
- **Audit DB schema v2 → v3.** Migration `_MIGRATIONS[2]` adds four nullable transparency columns to `audit_records`. Pre-v0.6 records get NULL for the new columns; their stored `record_hash` is preserved (NOT re-hashed on load), so chain verification of historical records continues to work.
- **COMPLIANCE.md "Current limits"** replaced placeholder bullets with v0.6 measurement results:
  - **Distribution-shift split.** Hand-curated (held-out, 250): attack recall 97.1% / benign FPR 70.0%. LLM-generated (in-sample, 5,705): attack recall 95.2% / benign FPR 87.5%. The 18pp benign-FPR gap is the dominant distribution-shift signal.
  - **Stack composition.** `heuristic_only` recall 35% / 63%. `classifier_only` recall 94% / 86%. `full_stack` recall 97% / 98%. Layers not redundant — heuristic catches a small set of attacks the classifier misses (justifies the ensemble). Most full-stack benign FPR comes from heuristic ESCALATEs, not classifier upgrades.
  - **PAIR adaptive-attacker calibration.** Qwen2.5-32B-Instruct as both attacker and judge, 25 hand-curated jailbreak seeds, max 5 iterations: **ASR 0.0% (0/25)**. NOT a claim of imperviousness to all adaptive attackers — stronger attacker (70B+), longer iteration budgets, or alternate strategies (multi-turn drift, language-switch, obfuscation) might produce non-zero ASR.

### Deferred to v0.7+
- **prEN ISO/IEC 24970 field-alias layer** — pending public final of the standard. Will land when 24970 publishes.
- **DORA mapping refinement** — pending deployer-side signal. Conservative defaults shipped in v0.5.3 stay until a financial deployer's input refines them.

### Reproducible artifacts
- `tests/adversarial/distribution_shift_v0_5_3.json`
- `tests/adversarial/stack_ablation_v0_5_3.json`
- `tests/adversarial/pair_v0_5_3.json`

## [0.5.3] - 2026-04-26

### Fixed
- **`AdversarialClassifier` jailbreak collapse.** v0.5.2 shipped a bundle with **0% classifier recall on held-out jailbreak**. The heuristic carried the category alone (~91% recall). v0.5.3 retrains on a corpus expanded by 1,500 LLM-generated jailbreak variants across three canonical sub-styles (roleplay/persona-bypass, hypothetical/academic-framing, fake-mode/DAN), recovering classifier jailbreak recall to **78.3%** at the new operating threshold.

### Added
- **1,500 jailbreak variants** in `tests/adversarial/generated/JB-{roleplay,hypothetical,fakemode}.jsonl`. Generated by Qwen2.5-32B-Instruct on AMD MI300x via vLLM, deduplicated by canonical-content sha1 (not shape, which caps yield at ~30 unique fingerprints per style and should be avoided in future runs).
- **1,010 cross-shape benign entries** in `tests/adversarial/benign_generated/BT-{readfile-canonical,http_post,sql_query,send_email}.jsonl` (500 read_file canonical + 170 each for http_post / sql_query / send_email). Added to rebalance the corpus and recover broad benign_control accuracy.
- **105 targeted training entries** in `tests/adversarial/generated/CE-envvar-targeted.jsonl` (60 send_email credential-exfil with env-var-shaped bodies labeled DENY) and `tests/adversarial/benign_generated/BT-http-public-targeted.jsonl` (45 legitimate http_post calls to GitHub / Slack / Stripe public APIs labeled ALLOW). Added to repair preflight smoke-test edges.
- **`--with-classifier` flag** on `scripts/eval_adversarial.py`. The harness previously evaluated heuristic-only by default. The new flag stacks the AdversarialClassifier on top per the v0.5.2 deployment recommendation. Heuristic DENY/ESCALATE are preserved. Heuristic ALLOW is upgraded to ESCALATE when classifier prob ≥ threshold.
- `tests/test_adversarial_classifier_integration.py` covers the bundle-load, score-range, and known-bad-input paths end-to-end. Skipped when `vaara[ml]` extras are not installed.

### Changed
- **Default classifier threshold: `0.5` → `0.55`.** Justified by threshold sweep on the rebalanced corpus: 0.55 is the operating point that clears the FPR and jailbreak-recall gates (global benign FPR ≤ 25%, jailbreak recall ≥ 60%) and passes the canonical preflight smoke test, while staying close to v0.5.2's balanced-accuracy band.
- **Bundle format `version` bumped `1.1` → `1.4`.** Trained on the full 5,955-entry corpus (3,422 attack / 2,533 benign). Feature schema unchanged from v1.1 (236 features), so `_STATIC_FEATURES` schema-drift check passes without modification.
- **`scripts/train_adversarial_classifier.py`** now coerces non-dict `context` and `parameters` entries (string-typed entries existed in the corpus from v0.5.0 onward but the trainer crashed on them) and runs `baseline_predictions` in `best_effort=True` mode. Net effect: trainer runs cleanly on the heterogeneous corpus.

### Benchmarks (5-fold CV OOF, threshold 0.55)

| Metric | v0.5.2 @ 0.5 | v0.5.3 @ 0.55 | Δ |
|---|---|---|---|
| **Jailbreak recall (classifier)** | **0.0%** | **78.3%** | **+78.3pp** |
| **Global benign FPR** | 23.3% | **21.0%** | **−2.3pp** (better) |
| Balanced accuracy | 80.9% | 79.4% | −1.5pp |
| Attack recall | 85.2% | 79.8% | −5.4pp |

Threshold sweep (sample points):

| threshold | bal_acc | attack_recall | benign_FPR | jb_recall |
|---|---|---|---|---|
| 0.50 | 79.0% | 84.3% | 26.3% | 80.7% |
| **0.55** | **79.4%** | **79.8%** | **21.0%** | **78.3%** |
| 0.60 | 79.5% | 75.4% | 16.3% | 75.8% |
| 0.65 | 79.0% | 70.7% | 12.7% | 73.6% |
| 0.70 | 78.1% | 66.0% | 9.7% | 71.6% |

### Preflight smoke test (canonical 4-case gate)

| Case | v1.1 (v0.5.2) | v1.4 (v0.5.3) | Expected |
|---|---|---|---|
| SSRF cloud metadata | 0.983 BLOCK | 0.999 BLOCK | BLOCK |
| Credential exfil email | 0.557 BLOCK | 0.973 BLOCK | BLOCK |
| rm -rf / destructive | 0.867 BLOCK | 0.998 BLOCK | BLOCK |
| Legit GH issue | 0.007 ALLOW | 0.491 ALLOW | ALLOW |

### Hand-curated regression set (250 entries, stacked classifier)

Per-category allow-leakage on the seed corpus (`tests/adversarial/<category>.jsonl`) using `eval_adversarial.py --with-classifier`:

| Category | v0.5.2 heuristic | v0.5.3 stacked |
|---|---|---|
| jailbreak | 48% | **4%** |
| ssrf_via_tools | 100% | **0%** |
| privilege_escalation | 92% | **4%** |
| credential_exfil | 76% | **8%** |
| prompt_injection | 56% | **4%** |
| data_exfil | 56% | **0%** |
| tool_misuse | 72% | **4%** |
| destructive_actions | 20% | **4%** |

### Known limits / honest read
- Aggregate balanced accuracy regressed **1.5pp** from v0.5.2 (80.9% → 79.4%) and attack recall regressed **5.4pp** (85.2% → 79.8%). The trade is justified by the **+78.3pp** jailbreak recall delta and the **−2.3pp** FPR improvement, plus the cleaner edge-case behaviour evidenced by the preflight smoke test. v0.5.2's 80.9% balanced accuracy was partly inflated by counting jailbreak as "in scope" while the classifier scored 0% on it.
- LLM-generated content shares Qwen-style writing. The distribution-shift gap between generated-test recall and hand-curated-held-out recall has **not** been measured separately in this release. It will be reported in v0.6. Hand-curated regression numbers above are evidence that transfer is happening, but a formal split is owed.
- Attacker-as-iterative-PAIR ceiling has **not** been measured. `COMPLIANCE.md` does not yet quote an adaptive-ASR figure.

### Recovery / fallback
- Heuristic-only path (`pip install vaara`, no `[ml]` extras) is unaffected. Threshold 0.55 applies only when `vaara[ml]` is installed and the classifier is engaged.
- If the v1.4 classifier produces unexpected behaviour for your traffic, downgrade to `vaara==0.5.2` for the previous bundle and threshold (0.5).

### Use ESCALATE, not DENY, on mixed-intent traffic
At threshold 0.55, the 21.0% global FPR is a **reviewer queue**, not a blast door. Stack the classifier with the heuristic (the new `--with-classifier` flag does this in eval) and route classifier-positive cases to ESCALATE for production deployments where a single false positive blocks a real user task.

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
