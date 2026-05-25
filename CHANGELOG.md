# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.31.0] - 2026-05-25

**Theme: industrial-grade adversarial benchmark.** v0.31 retires the
cross-validated headline numbers and replaces them with a 70/15/15
stratified split, threshold picked on VAL only, and Wilson 95%
intervals on every figure cited in the release. The classifier
bundle is retrained on TRAIN entries only. A single command
(`make repro-v031-bench`) reproduces every number from a fresh clone
against the four chain-of-custody hashes printed by every script.

### Added
- `bench/vaara-bench-v0.31.md`: methodology spec parallel to
  `vaara-bench-v1.md`. Chain of custody, split methodology,
  threshold pick by Youden's J and balanced accuracy on VAL,
  headline numbers with Wilson 95% intervals, named limits,
  reproduction recipe.
- `scripts/build_train_val_test_split.py`: deterministic 70/15/15
  split stratified by `(category, source)`, per-stratum seeded RNG,
  key format `<rel_path>#L<line>` because raw `id` has 1,855
  cross-file collisions.
- `scripts/eval_pipeline_attribution.py`: per-entry attribution
  audit running every corpus entry through a fresh `Pipeline`
  (cold-start methodology) plus the classifier side-by-side.
- `scripts/three_way_variants.py`: rules-only / classifier-only /
  both-OR comparison on any fold with per-category and per-source
  breakdowns.
- `scripts/threshold_sweep_val.py`: classifier threshold sweep on
  VAL only.
- `scripts/wilson_intervals.py`: Wilson 95% intervals on every
  headline metric. PAIR ASR upper bound at 0/25 = 13.3%.
- `Makefile` with `repro-v031-bench` target: eight-step end-to-end
  reproduction.
- 2,000-entry v0.31 corpus extension: `JB-roleplay-v031.jsonl`,
  `JB-hypothetical-v031.jsonl`, `JB-fakemode-v031.jsonl`,
  `BT-canonical-v031.jsonl`. Qwen2.5-72B-Instruct on MI300x with
  documented parameters.

### Changed
- Runtime classifier switched from `adversarial_classifier_v1.joblib`
  (v0.5.3-trained, cross-validation point estimates) to
  `adversarial_classifier_v2.joblib` (v0.31-trained on TRAIN-only).
- `detect_injection()` default threshold raised from 0.55 to 0.90.
  The 0.55 was the v1 bundle's escalation band against a
  cross-validated corpus. 0.90 is the VAL-pick threshold against
  the v2 bundle, chosen by Youden's J and balanced accuracy.
  Callers passing an explicit threshold are unaffected.
- `README.md` headline numbers swap to v0.31 honest values: recall
  53.9% [50.3, 57.5] at FPR 4.6% [3.1, 7.0] on held-out TEST
  n=1,196. The earlier 97.1% number was a contaminated read because
  the threshold tuning saw the test fold.

### Notes
- 771 tests pass under the runtime switch + threshold change.
- The chain of custody is the v0.31 contribution. Reviewers cannot
  tear down the methodology, only the numbers, and the numbers are
  intervals not point estimates.

## [0.30.0] - 2026-05-24

**Theme: matrix pre-work. Vaara's coverage of OWASP Top 10 for
Agentic Applications 2026 and OVERT 1.0 Part 3 controls, written as
two standalone documents so an enterprise reader can see the honest
mapping without having to read the rest of the repository.** Carlos
Hernandez (Bosch AI Officer, awesome-eu-ai-act curator) offered a
post-2026-06-18 side-by-side mapping of Vaara, Watcher, MS Agent
Governance Toolkit, and OWASP controls on the GenAI Gurus platform.
The Vaara column for that matrix needs to be drafted under the
assumption of expert-review by the maintainers of the other
columns. This release ships the two reference documents that anchor
the Vaara entries for the matrix.

### Added
- `OWASP_AGENTIC.md`: Vaara mapping to OWASP Top 10 for Agentic
  Applications 2026 (ASI01 through ASI10). Per-risk coverage with
  ✅ / ◐ / ◯ status badge, the OWASP mitigation list quoted under
  CC BY-SA 4.0, the Vaara primitive that satisfies each mitigation,
  and an explicit "deployer-owned" note per risk. Includes a
  cross-mapping summary table and the source citation (genai.owasp.org).
- `OVERT_CONTROLS.md`: standalone OVERT 1.0 Part 3 (Agentic AI
  Controls) mapping, extracted from `COMPLIANCE.md`. Same
  ✅ / ◐ / ◯ status convention. Covers TOOL-*, MCP-*, MULTI-*,
  CAP-*, DISC-*, HITL-*, DRIFT-* control families plus the S3P
  measurement primitive in Section 9, MEA-2.
- `OWASP_AGENTIC.md` and `OVERT_CONTROLS.md` cross-linked from the
  README "Where things live" table.

### Changed
- `COMPLIANCE.md` "OVERT 1.0 Part 3 (Agentic AI Controls) mapping"
  section now points readers to the new `OVERT_CONTROLS.md` and
  `OWASP_AGENTIC.md` documents. The full inline mapping content is
  retained in `COMPLIANCE.md` for backward compatibility with
  existing anchors and references.

### Unchanged
- Hash chain format, OVERT envelope schema, MCP proxy semantics,
  CLI surface, HTTP API, release workflow. This release ships
  documentation surfaces only. The compliance engine, scorer,
  audit, OVERT, and policy paths are byte-identical to v0.29.0.

### Notes for deployers
- The OWASP Top 10 for Agentic Applications 2026 is published by
  the OWASP GenAI Security Project, Agentic Security Initiative,
  December 2025, under Creative Commons CC BY-SA 4.0. The mapping
  in `OWASP_AGENTIC.md` is referenced under that license with
  attribution.
- The OVERT 1.0 Part 3 mapping references controls defined by Glacis
  Technologies (overt.is). The status assessments in
  `OVERT_CONTROLS.md` are Vaara's own reading of which controls the
  shipped code satisfies, not an OVERT-issued certification.

## [0.29.0] - 2026-05-24

**Theme: chronology anchor for Vaara's load-bearing concepts and a
neutral list of adjacent published work.** As runtime-evidence,
hash-chained audit, and conformal-interval framings appear in
recent pre-prints (Protocol-Driven Development, Subjective Logic
runtime confidence updates, formal-methods runtime monitors), a
reader comparing Vaara against newer proposals deserves a single
file that anchors when each Vaara concept first shipped in a tagged
public release and lists adjacent peer-reviewed and pre-print work
without competitive framing. This release ships that file and adds a
short related-work section to the conformal-prediction explainer.

### Added
- `PRIOR_ART.md`: per-concept chronology table mapping each
  load-bearing Vaara concept to its first shipped version and a path
  into the codebase or docs. Includes a related-work section listing
  adjacent pre-prints (Protocol-Driven Development, Formal Methods
  Meet LLMs, Subjective Logic runtime confidence updates,
  mechanistic interpretability for EASA learning-assurance,
  backchaining LoC mitigations from national security benchmarks)
  and classical foundations (conformal prediction, Linear Temporal
  Logic runtime verification). Inclusion is neutral attribution, not
  ranking.
- `PRIOR_ART.md` cross-link in the README "Where things live" table.

### Changed
- `docs/conformal-prediction.md`: appended a "Related
  runtime-confidence work" section citing the Subjective Logic
  safety-arguments pre-print (arXiv:2605.22530v1) as a complementary
  research line, with a cross-link to `PRIOR_ART.md` for the broader
  chronology and related-work list. The plain-language explanation
  and the Article 15(1) mapping are unchanged.

### Unchanged
- Hash chain format, OVERT envelope schema, MCP proxy perimeter
  semantics, CLI surface, HTTP API, release workflow. This release
  ships documentation surfaces only. The compliance engine,
  scorer, audit, OVERT, and policy paths are byte-identical to
  v0.28.0.

### Notes for deployers
- The chronology table in `PRIOR_ART.md` is reconstructed from
  `CHANGELOG.md` and the git tags. Both can be verified
  independently against PyPI release history and the GitHub tag
  list. Where a deployer or auditor wants a paper trail for "since
  when has Vaara done X," `PRIOR_ART.md` is the file to cite.

## [0.28.0] - 2026-05-22

**Theme: making the evidence-sufficiency rules and the conformal
interval readable by the people who use them.** The compliance engine
has been producing `verdict_inputs` (the threshold-vs-observed snapshot
that drove each article's verdict) and `verdict_reasons` (human-readable
rationale lines) since v0.26.0, and conformal intervals on every risk
score since the first public release. What was missing was a public,
linkable reference for the threshold values per article and a
plain-language explainer of the conformal interval for compliance
reviewers and legal counsel who are not statisticians. This release
ships both.

### Added
- `VERDICTS.md`: per-article evidence sufficiency reference. Documents
  the EU AI Act (Articles 9, 11-15, 17, 61) and DORA (Articles 10, 12,
  13) defaults the engine ships with: minimum record count, staleness
  window, strong-strength count and freshness bounds, future-timestamp
  downgrade, chain-integrity pin, and the status/strength decision
  tree. Includes worked JSON examples of `verdict_inputs` for a
  sufficient article and the same article after a chain break.
  Cross-linked from `COMPLIANCE.md` and from the README "Where things
  live" table.
- `docs/conformal-prediction.md`: plain-language companion to
  `docs/formal_specification.md`. Explains why a point risk score is
  not enough, what the interval gives a reader, and how the
  distribution-free coverage guarantee maps to Article 15(1)
  ("appropriate level of accuracy"). Cross-linked from the README
  "Where things live" table.

### Changed
- `COMPLIANCE.md` "EU AI Act Article Mapping" intro now points readers
  to `VERDICTS.md` for the threshold rules behind status assignment.

### Notes for deployers
- The threshold defaults in `VERDICTS.md` are the values shipped in
  `EU_AI_ACT_REQUIREMENTS` and `DORA_REQUIREMENTS` in
  `vaara/compliance/engine.py`. A deployer can override any of them
  by registering a custom `RegulatoryRequirement` with the
  `ComplianceEngine`. The document is the public starting point, not
  a constraint.

## [0.27.0] - 2026-05-22

**Theme: continuous fuzzing on the parsers that ingest attacker-controlled
input.** The OVERT envelope decoder, the audit-record `from_dict`
deserialiser, and the policy YAML/JSON loader all sit at the boundary
between Vaara and untrusted bytes. A crash, hang, or unhandled exception
in any of them is a denial-of-service vector at minimum, and a
deserialisation hazard at worst. This release wires ClusterFuzzLite into
CI so those three parsers get continuously fuzzed on every PR and nightly
in batch, and ships the first finding the local smoke test caught
(`from_yaml` leaking `OSError(ENAMETOOLONG)` past the `PolicyError`
contract).

### Added
- `fuzz/fuzz_overt_envelope.py`: atheris target that decodes attacker
  CBOR bytes, validates the closed 9-field schema, reconstructs a
  `BaseEnvelope`, and signature-checks against a dummy pubkey. Mirrors
  the attack surface of `vaara overt verify`.
- `fuzz/fuzz_audit_from_dict.py`: atheris target for `AuditRecord.from_dict`
  + `compute_hash()` + the `narrative` property. Models the JSONL-replay
  path where trail records get reloaded from disk.
- `fuzz/fuzz_policy_loader.py`: atheris target for `from_json` and
  `from_yaml`. Exercises both text paths with attacker-controlled strings.
- `.clusterfuzzlite/Dockerfile` and `.clusterfuzzlite/build.sh`: builder
  image based on `gcr.io/oss-fuzz-base/base-builder-python`, installs
  Vaara with `[attestation,yaml]` extras, compiles each `fuzz/fuzz_*.py`
  target with `compile_python_fuzzer`.
- `.github/workflows/cflite_pr.yml`: PR-triggered fuzzing for 300s under
  both address and undefined sanitizers, code-change mode.
- `.github/workflows/cflite_batch.yml`: nightly cron, 3600s batch fuzzing
  under both sanitizers.
- `.github/workflows/cflite_cifuzz.yml`: build-sanity on push to main, so
  a broken Dockerfile or build.sh surfaces immediately rather than at
  next PR.
- `tests/test_policy.py`:
  `test_from_yaml_oversize_single_line_treated_as_content_not_path`
  pins the regression behaviour of the loader fix below.
- SECURITY.md: brief note on the continuous-fuzzing posture so reporters
  know the parsers are under active fuzz coverage.

### Fixed
- `vaara.policy.loader.from_yaml`: a single-line YAML string longer than
  the OS path limit (~255 bytes on most filesystems) previously caused
  `Path(source).is_file()` to raise `OSError(ENAMETOOLONG)` directly,
  bypassing the loader's `PolicyError` contract. Any caller that loads
  YAML from attacker-controlled config could be DoS'd by an oversized
  single-line payload. The is_file probe is now wrapped: any stat
  failure is interpreted as "not a path" and the input falls through to
  YAML parsing, where it surfaces as a normal `PolicyError`. Found by
  the local smoke run of `fuzz_policy_loader.py` before any atheris
  fuzzing ran.

### Unchanged
- Hash chain format, OVERT envelope schema, MCP proxy perimeter semantics,
  CLI surface, HTTP API, release.yml SLSA provenance. The fuzz targets,
  Dockerfile, and CFLite workflows are additive supply-chain
  infrastructure; nothing in the runtime kernel moves.

## [0.26.0] - 2026-05-21

**Theme: per-article verdict drill-down inside the compliance report.** A
reviewer reading a v0.25 evidence report could see that Article 9(1) was
`evidence_sufficient` with `strength: strong`, but had to re-run the engine
or open the raw audit DB to learn which records the verdict sat on or which
threshold pushed the strength up. v0.26.0 attaches that reasoning to every
article so a regulator can trace status to threshold delta to concrete event
in one read. The change extends existing report machinery and stays
backwards compatible at the wire level. Existing fields keep their shape.
Two new fields appear next to them.

### Added
- `ArticleEvidence.verdict_inputs`: dict of the threshold-vs-observed
  inputs the engine compared against to produce status and strength.
  Surfaces `min_evidence_count`, `staleness_hours`, `evidence_event_types`,
  `evidence_count_observed`, `freshest_evidence_age_hours`,
  `oldest_evidence_age_hours`, `future_timestamp_count`, `chain_intact`,
  a `strength_thresholds` sub-dict (the 2x / staleness/4 bounds used for
  `STRONG`), and a `verdict_reasons` list of human-readable rationale lines
  explaining why this article landed at its current status and strength.
  JSON-strict: no inf/NaN at the dict boundary.
- `ArticleEvidence.contributing_events`: list of the most recent (up to 5)
  qualifying audit records the verdict sits on. Each entry carries
  `record_id`, `action_id`, `event_type`, `timestamp_iso`, `age_hours`,
  `agent_id`, `tool_name`, the record's `narrative`, and a curated
  `drill_down` dict containing only the data fields that directly fed
  risk/decision/outcome reasoning (point estimate, conformal interval,
  decision, reason, outcome severity, escalation target, resolution,
  reviewer, override reason). Free-form `data` keys are not inlined so a
  caller-supplied secret cannot leak into a regulator-facing summary.
- Renderers updated end to end. `render_markdown` adds a `Verdict inputs`
  table and `Contributing events` table per article. `render_narrative`
  appends a `Why:` rationale line and an `Event:` line per article.
  `render_pdf` adds the same two tables to each per-article section.
  `compliance.dashboard.render_html` adds the same drill-down to each card
  with HTML-escaped values. Still no JavaScript and no external assets.
- Broken-chain handling extends to drill-down: when `verify_chain()`
  fails, every article's `verdict_inputs.chain_intact` flips to `False`
  and the rationale list prepends a chain-integrity warning, matching
  the existing status / strength downgrade.
- Tests: 11 new tests across `tests/test_compliance.py`,
  `tests/test_compliance_render.py`, and `tests/test_compliance_dashboard.py`
  covering verdict_inputs shape, contributing-events ordering, drill-down
  key filtering (free-form data must not leak), broken-chain marking,
  empty-trail INSUFFICIENT pinning, JSON strict-mode round-trip, and
  renderer output across markdown / narrative / dashboard / PDF.

### Changed
- `.github/workflows/release.yml`: the `build` job now generates SLSA
  build provenance via `actions/attest-build-provenance@v4.1.0` after the
  wheel and sdist are built. The attestation binds workflow + commit SHA
  + builder identity to the artefact digests and ships alongside the
  Release for downstream `slsa-verifier` / `gh attestation verify` use.
  Required new permissions on the build job: `id-token: write`,
  `attestations: write`.

### Unchanged
- Hash chain format, OVERT envelope schema, MCP proxy perimeter semantics,
  CLI surface (`vaara compliance report` still takes the same flags),
  HTTP API. The `to_dict()` boundary adds two keys but does not rename or
  remove existing ones, and v0.25 consumers ignore the additions cleanly.

## [0.25.0] - 2026-05-21

**Theme: streaming notifications inside the audit boundary.** Long-running
upstream MCP tools emit `notifications/progress` and `notifications/message`
between the request and the final response. Until v0.25.0 those flowed
through the proxy untouched: forwarded to the client, but invisible to the
audit chain and the OVERT receipt directory. A regulator reconstructing the
session from receipts only could see the request and the result, never the
events the upstream emitted while working. v0.25.0 brings each notification
inside the same audit + attestation perimeter that already governs
`tools/call`, `resources/read`, and `prompts/get`. The notification still
forwards to the client unchanged. Observation never blocks streaming.

### Added
- Streaming-notification interception in `vaara.integrations.mcp_proxy`.
  When a `tools/call` arrives with `params._meta.progressToken` set, the
  proxy records the token in an in-flight map alongside the originating
  `action_id`, `agent_id`, and tool name. The map is cleared in a
  `finally` block when the call returns or raises.
- Every `notifications/progress` arriving from the upstream is recorded
  as a perimeter audit pair (`tool_name="mcp.notification.progress"`,
  `decision="observed"`) carrying the parent `action_id` and tool name
  pulled from the in-flight map. When OVERT is configured, the
  notification additionally emits a dedicated Base Envelope with action
  class `mcp.notification.progress` and `parent_action_id` /
  `parent_tool` / `progress_token` in `non_content_metadata`.
- Every `notifications/message` arriving from the upstream is similarly
  recorded as `mcp.notification.message` and, when OVERT is configured,
  emits a Base Envelope carrying the `level` and `logger` fields.
- Orphan progress notifications (token with no matching in-flight call)
  still audit and emit, with an empty `parent_action_id`, so the
  regulator can spot dangling events rather than have them disappear.
- Tests: 10 new tests across `tests/test_integrations_mcp_proxy.py` and
  `tests/test_integrations_mcp_proxy_overt.py`. Coverage includes
  correlation to in-flight `tools/call`, map cleanup on success and on
  raise, audit-failure-still-forwards (observation must not block
  streaming), orphan-progress, and OVERT envelope contents for both
  progress and message surfaces.

### Changed
- `.github/workflows/release.yml`: npm publish step reverts to OIDC-only
  (no `NODE_AUTH_TOKEN`, no `NPM_TOKEN`) now that the `@vaara/client`
  package has a Trusted Publisher configured. Provenance flag preserved.

### Unchanged
- Pipeline, audit chain hash format, perimeter allow/deny semantics, MCP
  wire format, OVERT envelope closed-schema fields. Notifications still
  forward to the client byte-identically. A v0.24.0 deployment behaves
  the same when it does not see streaming traffic.

## [0.24.0] - 2026-05-20

**Theme: MCP proxy emits OVERT 1.0 Base Envelopes per interaction.** With
the v0.21-v0.23 surface coverage in place across tools, resources, and
prompts, the next step is making each interaction independently verifiable
by a regulator or auditor. The MCP proxy can now sign every governed
interaction as an OVERT 1.0 Protocol Profile 1.0 Base Envelope and write
it to disk in canonical CBOR, ready for `vaara overt verify` or any other
conformant OVERT verifier. The OVERT substrate (emitter, verifier, schema,
CLI) shipped in earlier releases. This release wires the emitter into the
MCP proxy as the first concrete production-shape OSS OVERT 1.0 integration.

### Added
- `vaara-mcp-proxy` learns three new flags: `--overt-signing-key PATH`
  (Ed25519 PEM private key), `--overt-operator-key PATH` (raw bytes for
  the HMAC request_commitment, minimum 16 bytes; also accepts
  `VAARA_OVERT_OPERATOR_KEY_HEX` from the environment), and
  `--overt-receipts-dir DIR` (where canonical-CBOR envelopes land). All
  three are off unless set. When set, every allowed and every
  perimeter-blocked `tools/call`, `resources/read`, and `prompts/get`
  produces one Base Envelope on disk.
- `vaara.integrations._mcp_overt`: internal `OVERTReceiptEmitter` with
  per-process monotonic counter under a lock, per-process
  `arbiter_instance_identifier`, `encoder_binary_identity` derived from
  the Vaara version plus SHA-256 of the canonical perimeter config
  (allow/deny lists across all three primitives), and a pinned
  `pubkey.bin` written into the receipts directory on emitter start.
- Receipt layout: `DIR/{nanosecond_timestamp}-{counter:010d}.cbor` plus
  `DIR/pubkey.bin` (32 raw Ed25519 public-key bytes).
- `non_content_metadata` carries structural fields only: action class
  (`mcp.tool.call` / `mcp.resource.read` / `mcp.prompt.get`),
  tool/resource/prompt identifier, decision, reason, agent_id, action_id.
  Request content never leaves the operator environment. Only its
  HMAC-SHA256 commitment crosses the trust boundary, per OVERT Annex B.4.
- Tests: `tests/test_integrations_mcp_proxy_overt.py`. Coverage includes
  emitter-disabled-by-default, one envelope per call site (allowed and
  filtered), monotonic counter strictly increases, envelopes verify
  under the pinned public key via `verify_base_envelope`, and the CLI
  rejects misconfigurations.

### Unchanged
- Pipeline, audit chain, perimeter allow/deny semantics, MCP wire format.
  When `--overt-signing-key` is absent, proxy behavior is byte-identical
  to v0.23.1.
- OVERT envelope schema (closed 9-field Annex B.6) and `vaara overt
  verify` semantics. The verifier reads canonical-CBOR files one at a
  time, as it did since v0.17.0.

## [0.23.1] - 2026-05-20

**Theme: CI fix.** The v0.23.0 release workflow flipped `npm publish` to
OIDC-only trusted publishing, but the npmjs.com side of the
`@vaara/client` package was not yet configured for a trusted publisher.
The npm leg of the v0.23.0 release therefore failed with a 404 on PUT,
even though PyPI shipped cleanly and the GitHub Release was signed and
attached. This patch restores the token + provenance model that v0.22.0
used.

### Fixed
- `.github/workflows/release.yml`: re-add `NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}`
  env on the npm publish step. The `--provenance` flag is preserved, so
  the npm package still ships with a Sigstore-attested provenance
  statement from the GitHub Actions OIDC token. Auth flows via the
  token, provenance flows via OIDC. OIDC-only publishing can return
  later once trusted publishing is configured on the npm package
  settings page.

### Unchanged
- Library code, public API, audit chain format, OVERT envelope schema,
  MCP proxy behavior. v0.23.1 is a pure CI fix.

## [0.23.0] - 2026-05-20

**Theme: MCP proxy closes the surface-coverage gap.** Resources and
prompts join tools as governed MCP primitives. The same operator
allow/deny perimeter applies symmetrically to `resources/list` /
`resources/read` and `prompts/list` / `prompts/get`, and every allowed
read or prompt retrieval writes a request+decision audit pair to the
hash chain. Vaara now covers all three MCP primitives, not one.

### Added
- `src/vaara/integrations/mcp_proxy.py`: `VaaraMCPProxy.__init__`
  accepts `resource_allowlist`/`resource_denylist` and
  `prompt_allowlist`/`prompt_denylist`. New `_handle_list` helper is
  shared by `tools/list`, `resources/list`, and `prompts/list`. New
  `_handle_resources_read` and `_handle_prompts_get` enforce the
  perimeter and write a request+decision audit pair on allow.
  Perimeter blocks for resources/prompts surface as JSON-RPC errors
  (code -32000) with a `data` payload mirroring the tool-call block
  shape (`vaara_blocked: true`, `decision: "FILTERED"`, `reason`).
- CLI: `--allow-resource URI`, `--deny-resource URI`,
  `--allow-prompt NAME`, `--deny-prompt NAME`, all repeatable.
  Backward compatible: no flags = passthrough.
- `tests/test_integrations_mcp_proxy.py`: twelve new tests covering
  resources/list filtering (denylist, allowlist, no-policy), prompts/
  list filtering, blocked resources/read returns JSON-RPC error,
  blocked prompts/get returns JSON-RPC error, allowed resources/read
  writes the audit pair without invoking the risk scorer, and
  allowed prompts/get writes the audit pair.

### Design
Resource reads and prompt gets are read-oriented MCP surfaces. They
need audit coverage so regulators can reconstruct what an agent
accessed, but they do not run through the risk scorer (the scorer is
for actions). The operator perimeter is the gate. The audit chain
captures the evidence. Tool calls keep the full pipeline (classify,
score, decide, audit).

### Fixed
- `src/vaara/__init__.py`: `__version__` was stale at `0.21.0` after
  v0.22.0 shipped. Bumped to `0.23.0` together with `pyproject.toml`
  and `clients/ts/package.json`.

## [0.22.0] - 2026-05-20

**Theme: MCP proxy operator-side tool filtering.** Adds two repeatable
CLI flags to the MCP proxy that let an operator shape the upstream
tool surface visible to the AI client: `--allow-tool NAME` (if any are
given, only those tools pass through) and `--deny-tool NAME` (those
tools are filtered, wins on overlap with allowlist). Filtered tools
are dropped from `tools/list` responses before the client sees them,
and any `tools/call` to a filtered tool is rejected at the proxy
perimeter with an MCP `isError: true` payload (`decision: "FILTERED"`,
`reason: "Tool filtered by operator policy"`) without forwarding to
the upstream or invoking the risk pipeline.

### Added
- `src/vaara/integrations/mcp_proxy.py`: `VaaraMCPProxy.__init__`
  accepts `allowlist: Optional[set[str]]` and `denylist:
  Optional[set[str]]`. New `_tool_filtered(name)` helper applied to
  both `tools/list` (filters the `result.tools` array) and
  `tools/call` (returns a `FILTERED` block payload before the
  pipeline runs).
- CLI: `--allow-tool NAME` and `--deny-tool NAME`, both repeatable.
  Backward compatible: no flags = passthrough.
- `tests/test_integrations_mcp_proxy.py`: eight new tests covering
  denylist drops, allowlist restricts, denylist-wins-on-overlap,
  no-policy passthrough, filtered tools/call returns block, and
  allowlisted tools/call still routes through the pipeline.

### Verified
- End-to-end smoke against `github/github-mcp-server` (42 real tools):
  `--deny-tool` drops named entries from `tools/list` and rejects
  matching `tools/call`; `--allow-tool` restricts `tools/list` to the
  allowlist and routes allowlisted calls through the pipeline as
  before; no-flag run is identical to v0.21.0 behaviour.

### Use case
Hide write/delete tools (e.g. `delete_file`, `merge_pull_request`)
when the upstream MCP server exposes more capability than the
deployment policy allows. The LLM client never learns the tool
exists, which is materially different from instructing the model
not to call it.

## [0.21.0] - 2026-05-19

**Theme: MCP-aware proxy.** Adds
`vaara.integrations.mcp_proxy.VaaraMCPProxy`, a transparent MCP proxy
that sits between an MCP client (Claude Code, Cursor, any MCP-capable
host) and an upstream MCP server (SAP ADT MCP, SAP Graph API MCP, SAP
Cloud ALM MCP, any community-built MCP server). Every `tools/call`
request from the client routes through Vaara's interception pipeline
before reaching the upstream. Allowed calls forward transparently and
report the upstream outcome back to the scorer. Blocked calls return
an MCP `isError: true` response with the block reason. Other MCP
methods (initialize, tools/list, resources, notifications) forward
unchanged.

### Added
- `src/vaara/integrations/mcp_proxy.py`: `VaaraMCPProxy` and CLI entry
  point. Invoke as `python -m vaara.integrations.mcp_proxy --upstream
  <cmd> [--upstream-arg ...]`.
- `src/vaara/integrations/_mcp_upstream.py`: `UpstreamMCPClient`,
  subprocess lifecycle plus JSON-RPC id demultiplexing on a background
  reader thread. Internal module, not part of the public surface.
- `tests/test_integrations_mcp_proxy.py`: six smoke tests covering
  blocked tool calls, allowed forward, severity mapping, the
  `_vaara_agent_id` strip, non-tools/call passthrough, and invalid
  request handling.

### Strategic frame
The community SAP MCP servers shipped at SAP Sapphire 2026 plus the
Anthropic-SAP partnership announcement put SAP ABAP / Graph / Cloud
ALM behind Claude Code in enterprise developer workflows. None of the
parties (SAP, Anthropic, the community MCP server authors) ships the
runtime governance layer the EU AI Act high-risk obligations require
for those tool calls. The proxy is that layer in OSS today.

## [0.20.0] - 2026-05-18

**Theme: OSS guardrail adapters.** Adds four adapters that take findings
from NVIDIA NeMo Guardrails, Guardrails AI, LLM Guard, and Rebuff and
route them through Vaara's hash-chained audit trail and OVERT envelope
with EU AI Act article tags. Same pattern as v0.19.0's cloud adapters.
The OSS guardrail runs as an upstream signal in the deployer's
environment. Vaara records what the guardrail flagged, normalises it
to a shared vocabulary, and tags each finding against Art. 5, 10, 13,
15, and 53.

The mapping itself is the value. Each adapter is a thin SDK wrapper
around a published category-to-article table extended in
`_content_safety_articles.py`. A deployer can read the table, dispute
a row, and override mappings without touching adapter code.

### Added
- 41 new mapping rows in `_content_safety_articles.py` across the four
  OSS providers (7 NeMo, 10 Guardrails AI, 20 LLM Guard, 4 Rebuff),
  each tagged with EU AI Act articles and OWASP LLM Top 10 codes.
  Seven new Vaara categories: `secrets_leak`, `schema_violation`,
  `output_validation`, `bias`, `language`, `sentiment`,
  `resource_limit`.
- `src/vaara/integrations/nemo_guardrails.py`:
  `NemoGuardrailsAdapter` plus `parse_generation_response`. Maps input
  rails, dialog rails, output rails, and retrieval rails from the
  `GenerationResponse.log.activated_rails` payload.
- `src/vaara/integrations/guardrails_ai.py`: `GuardrailsAIAdapter` plus
  `parse_validation_outcome`. Maps `ValidationOutcome` summary shape
  to findings with PascalCase normalisation across hub validators.
- `src/vaara/integrations/llm_guard.py`: `LLMGuardAdapter` plus
  `parse_scan_result`. Wraps `scan_prompt` and `scan_output`
  callables. Scanner-callable injection for test isolation.
- `src/vaara/integrations/rebuff.py`: `RebuffAdapter` plus
  `parse_detect_response` and `parse_canary_leak`. Records all three
  injection-detection layers (heuristic, model, vector) and the
  canary-word leak on responses.
- Four test files exercising parsers and adapters with no SDK install
  required (29 new tests, 712 total).

### Pyproject
- New optional extras: `nemo-guardrails`, `guardrails-ai`, `llm-guard`,
  `rebuff`. Base install stays free of OSS guardrail dependencies.
- HuggingFace Space added as `Demo` URL in `[project.urls]`.

### README
- HuggingFace Space badge added to the badge row.

### Strategic frame
The OSS guardrails are inputs to Vaara, not Vaara's replacement.
Vaara stays the schema findings flow into: hash-chained audit trail,
OVERT envelope, EU AI Act article-level evidence. Add NeMo, Guardrails
AI, LLM Guard, or Rebuff to your stack and their findings land in the
same Vaara audit record as Bedrock, Azure, and GCP findings did in
v0.19.0.

## [0.19.1] - 2026-05-18

**Patch: audit DB upgrade safety.** Opening an existing audit DB at
any schema version older than the current one crashed on first
MCP-server boot with `no such column: tenant_id`. The init path ran
`SCHEMA_SQL` before migrations, and `SCHEMA_SQL` contains indexes on
columns that later migrations add.

Init now runs migrations from the stored version (or from v0 for
pre-versioned DBs that have no `audit_meta` row yet) BEFORE running
`SCHEMA_SQL` idempotently. Fresh DBs continue to use the existing
single-pass `SCHEMA_SQL` path.

Tests added for the v0 (pre-versioning), v1, and current-version
open paths.

## [0.19.0] - 2026-05-17

**Theme: Big Cloud guardrail adapters.** Adds three adapters that take
findings from AWS Bedrock Guardrails, Azure AI Content Safety, and GCP
Model Armor and route them through Vaara's hash-chained audit trail
and OVERT envelope with EU AI Act article tags. The cloud filter runs
as an upstream signal in the deployer's environment. Vaara records
what the filter flagged, normalises it to a shared vocabulary, and
tags each finding against Art. 5, 10, 13, 15, 53, and the
CSAM-specific obligation from the May 2026 Digital Omnibus political
agreement.

The mapping itself is the value. Each adapter is a thin SDK wrapper
around a published category-to-article table in
`_content_safety_articles.py`. A deployer can read the table, dispute
a row, and override mappings without touching adapter code.

### Added
- `src/vaara/integrations/_content_safety_articles.py`: canonical
  mapping rows for 27 provider categories across the three vendors
  (10 Bedrock, 9 Azure, 8 GCP), each tagged with EU AI Act articles
  and OWASP LLM Top 10 codes.
- `src/vaara/integrations/_content_safety_base.py`: shared
  `ContentSafetyFinding` and `FindingCategory` dataclasses,
  `ContentSafetyScorer` protocol, verdict and severity aggregators.
  Adapter output ships with `to_audit_context()` for
  `pipeline.intercept(context=...)` and `to_overt_metadata()` for OVERT
  envelope `non_content_metadata` (decimal-string severities only per
  Protocol Profile 1.0).
- `src/vaara/integrations/bedrock_guardrails.py`:
  `BedrockGuardrailsAdapter` plus `parse_apply_guardrail_response`.
  Maps Bedrock's five policy buckets (topicPolicy, contentPolicy,
  wordPolicy, sensitiveInformationPolicy, contextualGroundingPolicy).
  Normalises `ANONYMIZED` to the shared `REDACTED` action.
- `src/vaara/integrations/azure_content_safety.py`:
  `AzureContentSafetyAdapter` plus `parse_responses`. Wraps
  `analyze_text`, Prompt Shields, Protected Material, and Groundedness
  Detection behind a single `scan_prompt` / `scan_response` pair.
  Block threshold defaults to severity 4.
- `src/vaara/integrations/gcp_model_armor.py`: `GcpModelArmorAdapter`
  plus `parse_sanitize_response`. Wraps `sanitize_user_prompt` and
  `sanitize_model_response`. CSAM, prompt injection, malicious URIs,
  and SDP findings always block. Responsible-AI filters block at
  `MEDIUM_AND_ABOVE` by default.
- Three test files exercising the parsers and adapters with canned
  fixtures (44 new tests, 680 total). No hard dependency on boto3,
  azure-ai-contentsafety, or google-cloud-modelarmor.

### Pyproject
- New optional extras: `bedrock`, `azure-content-safety`,
  `gcp-model-armor`. Base install stays free of cloud SDK
  dependencies.

### Strategic frame
The cloud filters are inputs to Vaara, not Vaara's replacement.
Vaara stays the schema findings flow into: hash-chained audit trail
with explicit article tags, OVERT envelope metadata, EU AI Act and
OWASP vocabulary on every finding. Not "Vaara works with AWS / Azure
/ GCP." The other direction: AWS Bedrock Guardrails, Azure AI Content
Safety, and GCP Model Armor work inside Vaara's compliance kernel.

## [0.18.1] - 2026-05-17

**Release-bookkeeping patch.** The v0.18.0 release shipped Python to PyPI
successfully but the npm publish step failed because
`clients/ts/package.json` was not bumped from 0.17.0. The TS client is
unchanged in behaviour; this patch restores PyPI/npm version lockstep
established in v0.15.0. No Python code changes versus 0.18.0.

### Changed
- `clients/ts/package.json`: 0.17.0 → 0.18.1 (lockstep with PyPI).
- `pyproject.toml`, `src/vaara/__init__.py`: 0.18.0 → 0.18.1.

## [0.18.0] - 2026-05-17

**Theme: hardware TEE attestation hook (experimental).** Adds an optional
hardware-rooted attestation layer alongside the Ed25519 (or ML-DSA-65)
signature already on the OVERT 1.0 Base Envelope. Initial backend is AMD
SEV-SNP, the natural fit for the confidential-VM deployment model used
in agent runtimes. Intel TDX and SGX backends are tracked for later
releases. The OVERT envelope schema is unchanged (closed per spec); the
TEE report is a sibling artefact bound to a specific envelope by placing
`SHA-512(canonical_cbor(envelope))` into the report's 64-byte
`REPORT_DATA` field.

### Added
- `src/vaara/attestation/tee.py` module with:
  - `parse_sev_snp_report`: binary parser for the 1184-byte SEV-SNP
    attestation report (AMD SEV Secure Nested Paging Firmware ABI
    Specification rev. 1.55, Table 22).
  - `bind_overt_envelope_to_report_data`: computes the 64-byte
    `REPORT_DATA` value that binds a TEE report to a specific OVERT
    envelope. Hash covers all 9 envelope fields including the inner
    Ed25519 signature.
  - `verify_sev_snp_report_signature`: validates the ECDSA P-384
    over the report body against a caller-supplied VCEK PEM.
  - `verify_envelope_binding`: confirms `REPORT_DATA` matches
    SHA-512 of the supplied envelope.
  - `MockSEVSNPAttester`: deterministic in-memory attester for tests
    and CI, building byte-compatible report blobs.
  - `SEVSNPHostAttester`: skeleton that wraps `/dev/sev-guest` and
    raises a clear error when not on an SEV-SNP host.
- `vaara tee parse REPORT` CLI: dump key fields of a SEV-SNP report as
  JSON.
- `vaara tee verify REPORT --vcek VCEK.pem [--overt ENVELOPE.cbor]` CLI:
  signature check plus optional binding check against an OVERT envelope.
- 16 tests in `tests/test_attestation_tee.py` covering parse round-trip,
  size rejection, mock-attester emission, signature verification,
  tamper detection, wrong-VCEK rejection, envelope binding, wrong-curve
  rejection, non-EC-key rejection, unsupported-algo rejection, and the
  non-SEV-SNP host error path.

### Not in this release
- AMD KDS-based cert-chain validation (VCEK → ASK → ARK). Validating
  against AMD's Key Distribution Service requires a network fetch
  against `https://kdsintf.amd.com/` and is tracked for v0.19+.
- Live `/dev/sev-guest` ioctl emission. The `SNP_GET_REPORT` ioctl path
  is documented in `linux/sev-guest.h` and is tracked for v0.19+ once a
  tested SEV-SNP guest is available for integration testing.
- Intel TDX, Intel SGX backends. Same module shape will accommodate them
  via additional attester classes.

### Fixed
- `src/vaara/__init__.py` `__version__` had drifted to `0.15.0` while
  `pyproject.toml` had moved through 0.16.0 and 0.17.0. Both now read
  `0.18.0`.

### Notes
- The OVERT 1.0 envelope schema is closed and unchanged. TEE attestation
  is a strictly additive sibling artefact; an OVERT verifier without TEE
  awareness still validates Vaara envelopes exactly as before.
- 636 Python tests pass (was 620 + 16 new).

## [0.17.0] - 2026-05-17

**Theme: Vaara as the OVERT 1.0 reference verifier.** Vaara has emitted
OVERT 1.0 Protocol Profile 1.0 Base Envelopes since v0.10.0 and Phase 3
IAP attestations since v0.13.0. This release adds the CLI surface so any
operator, auditor, or Independent Attestation Provider can verify a Base
Envelope produced by any conformant emitter, not only Vaara. The verifier
is implementation-agnostic and reads the canonical CBOR wire format
defined in Annex B.6 verbatim.

### Added
- **`vaara overt verify RECEIPT.cbor --pubkey-file PUB.bin` CLI.**
  Validates an OVERT 1.0 Base Envelope from a canonical CBOR file against
  a supplied raw 32-byte Ed25519 public key. The schema is closed per
  the OVERT 1.0 spec, so envelopes carrying unknown fields are rejected.
  Exit 0 on valid, 1 on invalid envelope or signature mismatch, 2 on
  argument or I/O errors. Requires the existing `vaara[attestation]`
  extra (already present for emission). Accepts `--pubkey-hex` as an
  inline alternative to `--pubkey-file`.
- 10 new tests in `tests/test_overt_verify_cli.py` covering happy path,
  wrong pubkey, hex form, malformed hex, wrong-length pubkey, missing
  receipt, malformed CBOR, missing required fields, closed-schema
  rejection of extra fields, and tampered-signature rejection.

### Notes
- Anyone implementing OVERT 1.0 (overt.is) can route their conformance
  check through `vaara overt verify` without taking a runtime dependency
  on Vaara's emitter side. The CLI reads only the Annex B.6 wire format.
- 620 Python tests pass (was 610 + 10 new). TypeScript client bumped to
  0.17.0 for PyPI/npm lockstep, no code change.

## [0.16.0] - 2026-05-17

**Theme: auditor-shippable PDF.** The article-evidence report has rendered
as Markdown, JSON, and narrative since v0.13.0; auditors and Notified
Bodies consume PDFs. ``vaara compliance report --format pdf --out FILE``
now ships a styled, single-file PDF suitable for attaching to a conformity
submission or compliance binder. No new wire-format surface. The underlying
``ConformityReport`` is the same artefact in a different render.

### Added
- **`render_pdf(report, path)` in `vaara.compliance.render`.** Single-file
  PDF: cover with system metadata + integrity, per-domain article tables,
  per-article detail sections with status, strength, evidence age, gaps,
  recommendations, and sample record IDs. Layout mirrors the Markdown
  rendering. Returns bytes written.
- **`vaara compliance report --format pdf --out FILE` CLI option.**
  ``--out`` is required for ``pdf`` (binary output, no stdout path).
  Missing reportlab raises a clear ``ImportError`` pointing at
  ``pip install 'vaara[pdf]'``.
- **`vaara[pdf]` optional extra.** ``reportlab>=4.0``. Pure-Python, no
  native deps. Keeps the base install lean for deployers who only need
  Markdown/JSON.
- 4 new tests in ``test_compliance_render.py``: PDF magic bytes + EOF
  marker, HTML metachar escaping in ``system_name`` (so a hostile
  deployer name cannot smuggle ``<b>`` into a regulator-facing PDF),
  broken-chain rendering, and the missing-reportlab error path.

### Notes
- A3 commit-prove receipt pair is **already shipped** as of v0.13.0
  (``vaara.audit.receipts``, ``vaara trail receipt`` CLI); this release
  closes the remaining A-tier item from the post-v0.15.0 competitive
  differentiation move list.
- 610 tests pass (was 606 + 4 new).

## [0.15.0] - 2026-05-17

**Theme: Vaara reaches JavaScript.** First-party TypeScript HTTP client
for the v1 API so JS/TS agents (LangChain.js, Vercel AI SDK, MCP, any
Node service) can call Vaara without spawning a Python sidecar.

### Added
- **`@vaara/client` npm package (clients/ts).** Typed wrapper over
  every v1 endpoint: ``score`` / ``reportOutcome`` /
  ``appendAuditEvent`` / ``getActionChain`` / ``verifyAuditChain`` /
  ``reloadPolicy`` / ``detectInjection`` / ``detectPII`` /
  ``serverInfo`` / ``health``. ESM-only, Node 18+ (global ``fetch``),
  TypeScript declarations shipped. Server-side ``{error: {code,
  message}}`` bodies map to ``VaaraError`` (carries ``status`` +
  ``code`` for pattern matching against the spec); network failures
  map to ``VaaraTransportError``. Optional ``fetch`` injection makes
  the client trivially mockable in tests.
- **Release workflow gains a publish-npm job.** Guarded on the
  repository variable ``NPM_PUBLISH_ENABLED`` so the first publish
  stays manual. Once enabled and an ``NPM_TOKEN`` secret is set, every
  tag push publishes ``@vaara/client`` to npm with provenance.
- 6 new TypeScript tests covering URL construction, JSON body
  serialisation, the 4xx → ``VaaraError`` path with server-supplied
  code, the network-failure → ``VaaraTransportError`` path, the
  detector response shape, and constructor input validation.

### Notes
- Python package is unchanged in v0.15.0 beyond the version bump; this
  release ships the JS/TS surface alongside the existing PyPI artefact.

## [0.14.0] - 2026-05-17

**Theme: audit-retention horizon + composer position.** Two additions.
A pluggable signer abstraction lets operators sign the regulator-handoff
export envelope with ML-DSA-65 (FIPS 204) for retention horizons that
cross the credible quantum threshold. A composite-scorer module reuses
Vaara's own v0.10.0 /v1/score wire contract on the inbound side so
external scorers (NeMo Guardrails, another Vaara instance, any peer)
slot in alongside Vaara's adaptive scorer with a single object.

### Added
- **Pluggable signer abstraction (`vaara.audit.signer`).** New
  ``Signer`` / ``Verifier`` protocols with ``Ed25519Signer`` and
  ``MLDSA65Signer`` implementations. ``export_signed`` accepts a
  ``signer=`` keyword (mutually exclusive with the legacy
  ``signer_key`` Ed25519 PEM path). The export manifest carries
  ``signature_algorithm`` so verifiers dispatch automatically.
  Ed25519 exports still write ``signer_pubkey.pem`` for backward-
  compatible verification by older clients; ML-DSA-65 exports write
  ``signer_pubkey.bin`` (raw bytes, 1,952 bytes). ``verify_signed``
  reads the algorithm field and routes to the right verifier.
  ML-DSA-65 requires the new ``vaara[pq]`` extra
  (``pip install 'vaara[pq]'``), which pulls pure-Python
  ``dilithium-py``.
- **External-scorer composition (`vaara.scorer.composition`,
  `vaara.scorer.composite`).** ``ExternalScorer(url)`` calls a remote
  ``/v1/score`` endpoint (any service that implements the Vaara
  v0.10.0 wire contract) and returns the result in Vaara's internal
  scorer dict shape. Transport errors, malformed bodies, and non-JSON
  responses fail closed to ``deny`` with a structured reason in the
  ``raw_result``. ``CompositeScorer(scorers, combine="max"|"mean"|
  callable)`` runs Vaara's ``AdaptiveScorer`` alongside one or more
  external scorers and merges the results. The composite preserves
  the strongest decision (``deny`` > ``escalate`` > ``allow``) so one
  member firing high blocks the action. The composite's
  ``evaluate(context) -> dict`` shape matches what
  ``InterceptionPipeline`` already expects, so it drops into existing
  pipelines as a direct replacement.
- 20 new tests (10 signer + Ed25519 + ML-DSA-65 round-trip and zip
  tamper-detection; 10 composition + external-scorer error handling
  + composite combine modes).

## [0.13.0] - 2026-05-17

**Theme: operator surface + OVERT Phase 3 path.** Four additions that
close the most legible competitive gaps without diluting the kernel
position. Hot policy reload meets the Galileo Agent Control selling
point on its own ground. The OVERT 1.0 Phase 3 Independent Attestation
Provider (IAP) reference closes the AAL-3 → AAL-4 promotion path that
v0.11.0's Provisional Receipt opens, so Vaara owns the full path
without forcing dependence on an external IAP vendor. Named injection
and PII detectors expose existing scoring surface under buyer-visible
labels. A static HTML article-coverage dashboard adds the auditor-
facing visual artefact that the peer set has converged on.

### Added
- **Hot policy reload.** New `vaara.policy.controller.PolicyController`
  owns the live `Policy` and runs registered listeners under a write
  lock on `reload()`. `AdaptiveScorer.apply_policy(policy)` rebinds
  thresholds and sequence patterns atomically under the scorer's own
  RLock; an `evaluate()` call in flight on another thread either sees
  the old `(allow, deny)` pair or the new one, never a torn half.
  Conformal calibration, MWU expert state, and agent profiles are
  preserved across reloads. Malformed reloads are rejected with the
  previous policy left live. `POST /v1/policy/reload` accepts a
  server-side path or an inline body; `vaara serve --policy PATH`
  enables the endpoint; `vaara policy reload POLICY_PATH` triggers
  reload over HTTP from the operator's shell.
- **OVERT 1.0 AAL-4 Phase 3 IAP reference.** New
  `vaara.attestation.iap` ships a `Phase3Attestation` dataclass that
  wraps a Vaara `BaseEnvelope` with a notary Ed25519 signature (over a
  domain-separated prefix + canonical-CBOR of the inner envelope
  including its signature) and a transparency-log inclusion proof.
  Structural independence between the Arbiter key and the notary key
  is enforced at both emit and verify. New
  `vaara.attestation.transparency_log.InProcessTransparencyLog`
  implements an RFC 6962-style binary Merkle tree with domain-
  separated leaf and internal hashes; `append()` /
  `inclusion_proof()` / `root_hash` match the shape a sigstore Rekor
  adapter would expose, so a production deployment can swap in Rekor
  at the same call sites without changing the IAP contract.
- **Named injection + PII detector aliases.** `vaara.detect.detect_injection`
  routes free text through the same AdversarialClassifier behind
  vaara-bench-v1's published numbers (heuristic fallback when the ml
  extra is absent; the `backend` field reports which path served the
  call). `vaara.detect.detect_pii` is a zero-dependency regex extractor
  over six categories — email, phone, US SSN, IPv4, credit_card
  (Luhn-checked), IBAN (mod-97 checksum). `POST /v1/detect/injection`
  and `POST /v1/detect/pii` mirror the CLI. `vaara detect injection`
  and `vaara detect pii` read text from `--text`, `--file`, or
  `--stdin` and exit non-zero when the detector fires.
- **Static HTML article-coverage dashboard.**
  `vaara.compliance.dashboard.render_html` produces a single
  self-contained HTML page with embedded CSS, no JavaScript, no
  external assets, no network calls. Same content as the Markdown
  renderer (system metadata, audit-trail integrity, summary, critical
  gaps, per-domain article tables, detailed per-article sections) with
  status badges as colored pills and a print-friendly stylesheet.
  `vaara compliance dashboard --db PATH --out PATH` writes the page;
  a trailing slash or existing directory drops `index.html` inside.
- **OpenAPI spec coverage.** `docs/openapi.yaml` adds
  `/v1/detect/injection`, `/v1/detect/pii`, and `/v1/policy/reload`
  with full request and response schemas. The spec remains the
  authoritative integration surface.
- 53 new tests (14 PolicyController + reload HTTP; 11 IAP +
  transparency-log; 19 detect (injection + PII + HTTP); 9 HTML
  dashboard). Total 586 passing, 12 skipped.

## [0.12.0] - 2026-05-16

**Theme: agentic OVERT reference, published benchmark, product liability hook.**
v0.12.0 ships three additions that widen Vaara's category position.
First, an OVERT 1.0 MEA-2 Statistical Safety Signal Protocol (S3P)
emitter with exact Clopper-Pearson confidence intervals and a proposed
Protocol Profile extension that reports aggregate statistics over
Vaara's per-action conformal prediction intervals. Second, an explicit
control-by-control mapping of Vaara to OVERT 1.0 Part 3 (Agentic AI
Controls) in COMPLIANCE.md, covering Tool-Call Governance (Section 11),
MCP Server Trust Governance (Section 11.5), Multi-Agent System Controls
(Section 12), Capability-Based Access Control (Section 13), Agent
Disclosure (Section 14), Human-in-the-Loop Attestation (Section 15),
and Behavioural Drift Governance (Section 16). Third, vaara-bench-v1: a
versioned, reproducible adversarial-detection benchmark with frozen
corpus, methodology, and headline numbers under the v0.11.0 scorer
(soft TPR 100%, hard FPR 0% across 77 synthetic traces). Plus a forward
section on EU Product Liability Directive 2024/2853 (Article 9
rebuttable-presumption hook), transposition deadline 9 December 2026.

### Added
- **OVERT 1.0 MEA-2 S3P emitter (`vaara[attestation]` extra).** New
  module `vaara.attestation.s3p` provides exact Clopper-Pearson
  binomial confidence intervals via a pure-Python regularized
  incomplete beta function (no scipy dependency), plus a closed-schema
  S3P attestation per MEA-2.6 (Ed25519-signed, canonical CBOR
  encoding, decimal-string rates per Protocol Profile 1.0 numeric
  rules). Public API: `emit_s3p_attestation`, `verify_s3p_attestation`,
  `make_epoch_nonce_commitment`, `clopper_pearson_ci`,
  `regularized_incomplete_beta`, `S3PAttestation`,
  `ConformalExtension`.
- **`ConformalExtension`** Protocol Profile extension (proposed).
  Optional field in the signed S3P metadata that reports aggregate
  statistics over Vaara's per-action conformal prediction intervals
  alongside the standard binomial CI. Carries the same non-parametric
  coverage guarantee with no distributional assumption.
  Standard OVERT verifiers ignore the extension. Vaara-aware
  verifiers cross-check it against the per-action receipts.
- **vaara-bench-v1.** Versioned adversarial-detection benchmark with
  frozen corpus (`bench/adversarial_corpus.jsonl`, SHA-256
  `7a3219776e1c93a5127ab3b63832d73ba75f32fa044cabdbaa4e5d7088b33ff2`),
  frozen methodology (`bench/scorer_eval.py`), and frozen headline
  numbers under v0.11.0 (soft TPR 100%, soft FPR 20%, hard TPR
  28.85%, hard FPR 0%). Spec doc at `bench/vaara-bench-v1.md`.
  Machine-readable results at `bench/vaara-bench-v1-results.json`.
  Apache-2.0 licensed.
- 18 new tests in `tests/test_attestation_s3p.py` covering
  Clopper-Pearson against textbook values, the k=0 and k=n
  endpoints, round-trip emit-and-verify, status-band selection
  (compliant / threshold_exceeded / insufficient_sample), conformal
  extension attachment, tampered-field rejection, wrong-key rejection,
  and invalid-count rejection.

### Documentation
- **COMPLIANCE.md** gains two sections. **OVERT 1.0 Part 3 (Agentic AI
  Controls) mapping** walks Vaara's coverage of every control in
  Sections 11 through 16 with a ✅ / ◐ / ◯ marker for satisfied,
  partial, or gap-to-deployer / future-work. Covers TOOL-1 through
  TOOL-5, MCP-1/2/3, MULTI-1/2, CAP-1/2, DISC-1, HITL-1/2/3/4,
  SESS-1..5, STATE-1/2, IDENT-1, and DRIFT-1, plus a final S3P
  (Section 9, MEA-2) subsection. **EU Product Liability Directive
  2024/2853** is added as a new forward-looking section covering
  Articles 7, 8, and 9 (rebuttable presumption of defectiveness)
  and how Vaara's evidence stream supports the Article 9 rebuttal.
  Transposition deadline 9 December 2026 (Article 22).
- **README.md** updated to reference vaara-bench-v1 in the Numbers
  section and to mention the v0.12.0 S3P emitter and Part 3 mapping
  in the OVERT 1.0 attestation section. Sample code uses
  `arbiter_version="vaara/0.12.0"`.
- **bench/README.md** points to the vaara-bench-v1 spec doc as the
  canonical citation surface for external references to the
  benchmark.

## [0.11.0] - 2026-05-16

**Theme: first OSS Python reference implementation of OVERT 1.0.** v0.11.0
ships an emitter for the OVERT 1.0 Protocol Profile 1.0 Base Envelope
(Glacis Technologies, overt.is, March 2026) at AAL-3 Phase 2 (Provisional
Receipt). Vaara operates as the Arbiter in OVERT terms. Phase 3 notary
attestation and AAL-4 promotion are left to external Independent
Attestation Providers, per OVERT's structural-independence principle and
Vaara's always-OSS-kernel rule. Plus eIDAS NOTICE and OVERT position
statement in COMPLIANCE.md.

### Added
- **OVERT 1.0 Protocol Profile 1.0 Base Envelope emitter (`vaara[attestation]` extra).**
  First OSS Python reference implementation of the OVERT 1.0 AAL-3
  Phase 2 (Provisional Receipt) emission path. New module
  `vaara.attestation.overt` produces 9-field closed-schema Base
  Envelopes per Annex B.6, canonically CBOR-encoded per RFC 8949
  Section 4.2, Ed25519-signed per RFC 8032, with HMAC-SHA256 keyed
  request commitments per Annex B.4 and SHA-256 encoder-identity
  derivation. IEEE-754 floats are rejected at the canonical-encoding
  boundary per Protocol Profile 1.0 numeric rules. Vaara operates as
  the **Arbiter** in OVERT terms. External Independent Attestation
  Providers promote AAL-3 emission to AAL-4 by attaching Phase 3
  notary signatures and transparency-log inclusion proofs. Public API:
  `BaseEnvelope`, `emit_base_envelope`, `verify_base_envelope`,
  `make_request_commitment`, `encoder_binary_identity`, `canonical_cbor`.
- 13 new attestation tests (`tests/test_attestation_overt.py`).
  Round-trip verification, IEEE-754-float rejection at every nesting
  level, key-identifier binding, monotonic-counter binding, tampered-field
  rejection.

### Documentation
- **COMPLIANCE.md** gains two sections under the deployer-vs-Vaara
  ownership boundary: an explicit eIDAS NOTICE (Vaara hash chains and
  commit-prove receipts are technical evidence, not Qualified Electronic
  Attestations of Attributes under Regulation (EU) 910/2014 Article
  3(45), not a qualified trust service under Article 3(16)) and a
  position statement relative to OVERT 1.0 (Glacis Technologies). Vaara
  is structurally independent of the agent it governs and maps to
  OVERT AAL-3 operator-controlled attestation. Reaching AAL-4 requires
  pairing Vaara with an external Independent Attestation Provider. The
  design admits an external IAP layer without internal change.

## [0.10.0] - 2026-05-16

**Theme: Vaara as the kernel others build around.** v0.10.0 ships the
network-callable surface, the auditor-facing evidence artefact, and the
offline-verifiable receipt pair. Each of the three pieces is additive
and backward-compatible. Together they reposition Vaara from a Python
library to a runtime kernel that control planes, audit consumers, and
orchestration frameworks reference. The HTTP contract at
`docs/openapi.yaml` is versioned `/v1/` independently of the project
version, following the OPA pattern.

### Added
- **HTTP API reference server (`vaara[server]` extra).** Exposes the
  conformal scorer and hash-chained audit trail over HTTP per the
  contract in `docs/openapi.yaml`. Endpoints: `POST /v1/score`,
  `POST /v1/score/outcome`, `POST /v1/audit/events`,
  `GET /v1/audit/actions/{action_id}/chain`, `POST /v1/audit/verify`,
  `GET /v1/server`, `GET /v1/health`. The spec is authoritative. The
  reference server in `src/vaara/server/` is a FastAPI implementation
  suitable for local development and modest production loads.
- **`vaara serve`** CLI subcommand.
- **OpenAPI 3.1 contract at `docs/openapi.yaml`.** Stable v1 surface,
  intended as the integration point for control planes, orchestration
  frameworks, and audit consumers. Vaara defines the interface. The
  vendors call it.
- 11 new HTTP server tests (`tests/test_server.py`).
- **Auditor-facing evidence report rendering.** New module
  `vaara.compliance.render` with `render_markdown`, `render_json`, and
  `render_narrative` for the `ConformityReport` produced by
  `ComplianceEngine.assess`. Markdown output has per-domain article
  tables, per-article detail sections, evidence status badges,
  audit-chain integrity flagging, and a deployer-owns-the-decision
  disclaimer suitable for shipping to a regulator or attaching to an
  internal conformity submission.
- **`vaara compliance report --db PATH --format md|json|narrative
  [--out FILE]`** CLI subcommand. Loads an audit SQLite DB, runs
  `ComplianceEngine.assess`, renders to chosen format.
- 5 new compliance-render tests (`tests/test_compliance_render.py`).
- **Article 12 commit-prove receipt pair.** New module
  `vaara.audit.receipts` derives an offline-verifiable receipt from the
  existing audit chain: a `commit_hash` covering the gate-time decision
  (action_id, decision, risk_score, thresholds, decided_at) and an
  `outcome_hash` covering the post-execution outcome and embedding the
  commit_hash. Open-standards SHA-256 over canonical JSON, no external
  cryptography library required. Verification needs only `hashlib`,
  enabling per-action handoff to auditors without sharing the full chain
  or key material.
- **`vaara trail receipt --db PATH --action-id ID [--out FILE]`** CLI
  subcommand. Extracts and verifies the receipt pair, prints JSON.
- 11 new receipt tests (`tests/test_receipts.py`).

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
- **`vaara.policy.test_cases` module - Conftest analog for Vaara
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
- **`examples/policies/test_cases.yaml`** - six worked test cases
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
change. `Policy` and the load path are unchanged. The new modules
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
  default. Explicit `--trigger-record-id ID` overrides. No external
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
already wrote `ESCALATION_SENT` for every escalated action. With a
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
  bucket category, and request parameters/context as JSON. The
  interval is what makes Article 14 oversight substantive rather
  than cosmetic (see `COMPLIANCE.md`). `claim` is optimistic -
  concurrent claim races resolve with one winner and
  `InvalidTransitionError` for the loser. `resolve` accepts an
  optional `trail` and writes `ESCALATION_RESOLVED` so the Article
  14(4)(d) evidence row lands on the hash chain. `expire_stale`
  marks pending items past a timeout. Claimed items are left alone
  since they are under active review.
- **`InterceptionPipeline(review_queue=...)`.** Optional constructor
  parameter. When supplied, every `escalate` decision is enqueued
  alongside the existing `ESCALATION_SENT` audit record. Default
  `None` preserves prior behaviour bit-for-bit. Queue write failure
  logs and continues - the action is already gated by the escalate
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
Backwards-compatible. Pure addition. No existing schemas migrate.
The queue lives in its own DB file with its own `review_queue_meta`
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
- `README.md`: replaced the "Numbers" section with the v0.6 distribution-shift table (97.1% recall / 70.0% FPR hand-curated held-out. 95.2% / 87.5% LLM-generated in-sample) plus PAIR ASR 0.0% (0/25). Threshold default 0.5 -> 0.55, corpus description updated to the 5,955-entry rebalanced corpus, threshold-direction note corrected (recall drops as threshold rises). PR #44.
- `src/vaara/adversarial_classifier.py`: rewrote the module docstring. Removed all version-bound numbers. Readers now point at README + COMPLIANCE so the docstring does not go stale on every release. PR #44.
- `examples/adversarial_classifier.py`: ship-note threshold 0.8 -> 0.55. PR #44.
- `scripts/classifier_vs_heuristic.py`: clarified the script is the v0.5.0 historical reproducer, not the current production training path. PR #44.
- `pyproject.toml`: rephrased description for cleaner PyPI tagline rendering. PR #45.

### Note
No functional code changes. v0.6.0 users are on the same code. V0.6.1 only refreshes documentation surfaces visible to new PyPI installs and to anyone reading the package source.

## [0.6.0] - 2026-04-27

**Theme: standards alignment + legibility.** v0.5.x was the capability axis (jailbreak coverage closed, classifier rebalanced). v0.6 is the legibility axis: policies become readable, audit records become standards-aligned, adversarial numbers become honest, architecture contribution becomes documented.

### Added
- **`vaara.policy` package - JSON-native policy loader plus optional YAML via `vaara[yaml]` extra.** Frozen dataclasses for action classes, threshold curves, sequence patterns, and escalation routes. Hand-rolled validation with field-path error messages. Reuses existing `vaara.taxonomy.actions` enums verbatim. Threshold partial-overrides supported (set just `deny`, inherit default `escalate`). Implements Sketch A from the v0.6 DSL design exploration. Embedded Python DSL (Sketch B) and standalone DSL (Sketch C) stay deferred to v0.7+ pending external pull.
- **`vaara trail purge --db PATH --retention-days N (--tenant TID | --all-tenants) [--dry-run]` CLI subcommand** plus `SQLiteAuditBackend.purge_older_than(seconds, *, dry_run=False)` Python API. Article 12(2) retention enforcement. Tenant scoping is required: pick `--tenant TID` for a single tenant or `--all-tenants` explicitly, so a shared multi-tenant audit DB can never be silently purged across all tenants. Hash-chain integrity: surviving records still reference deleted predecessors via `previous_hash`, leaving a documented seam at the retention boundary that subsequent loads expose as a hash mismatch. Intended workflow: export a signed handoff zip BEFORE purging, archive externally, then purge. The signed zip remains self-consistent forever. The live DB chain has the seam.
- **prEN ISO/IEC 12792 four-axis transparency taxonomy on `AuditRecord`.** Four optional fields (`system_operation`, `data_usage`, `decision_making`, `limitations`) with default-classification heuristic per `EventType`. Per-record override via construction kwargs. NOT tamper-evident in v0.6 - fields are metadata annotations excluded from `record_hash` so pre-v0.6 chains stay valid. v0.7+ may add a separate signing mechanism if compliance requires.
- **`scripts/eval_distribution_shift.py`** - runs the full Vaara stack against the adversarial corpus with per-source tagging (hand-curated vs LLM-generated). Reports recall and FPR per source/class.
- **`scripts/eval_stack_ablation.py`** - runs three configurations (heuristic-only, classifier-only, full-stack) against the same corpus. Quantifies the independent contribution of each layer.
- **`scripts/eval_pair_attack.py`** - PAIR (Chao et al. 2023) iterative adaptive attacker. Uses an OpenAI-compatible vLLM endpoint for both attacker and judge roles. Zero new runtime deps (uses `urllib.request`).
- **`[yaml]` optional extra in `pyproject.toml`** (`pyyaml>=6.0`). Core `dependencies = []` preserved.
- **`examples/policies/minimal.json` and `full.yaml`** as reference policies.
- **COMPLIANCE.md gains "EU AI Act Annex IV evidence sections"** (maps Vaara contribution per §1–§9. Direct fill on §3, §5, §9. Contributes on §2, §4, §6, §7. Out of scope for §1, §8) **and "CEN-CENELEC harmonised standards alignment"** (per-standard table for ISO/IEC 42001, prEN 18286, prEN 18228, ISO/IEC 42006, prEN ISO/IEC 24970, prEN 18229-1, prEN ISO/IEC 12792).
- **`scripts/lint_full.sh` pre-push lint sweep** - chains `ruff` (style + correctness), `bandit` (security), `mypy` (types - strict on `vaara.policy`, lenient on legacy modules), and `pytest`. Documented in CONTRIBUTING.md. Catches CodeRabbit-class findings before they hit a PR review round-trip. New dev extras: `bandit>=1.7.5`, `mypy>=1.8`. Bandit configured in `pyproject.toml` to skip B608 across `audit/sqlite_backend.py` (all f-string SQL there interpolates only internally-controlled tenant clauses, not user input). Two `# nosec` annotations document the remaining trusted-bundle and synthetic-trace-RNG sites.

### Changed
- **Audit DB schema v2 → v3.** Migration `_MIGRATIONS[2]` adds four nullable transparency columns to `audit_records`. Pre-v0.6 records get NULL for the new columns. Their stored `record_hash` is preserved (NOT re-hashed on load), so chain verification of historical records continues to work.
- **COMPLIANCE.md "Current limits"** replaced placeholder bullets with v0.6 measurement results:
  - **Distribution-shift split.** Hand-curated (held-out, 250): attack recall 97.1% / benign FPR 70.0%. LLM-generated (in-sample, 5,705): attack recall 95.2% / benign FPR 87.5%. The 18pp benign-FPR gap is the dominant distribution-shift signal.
  - **Stack composition.** `heuristic_only` recall 35% / 63%. `classifier_only` recall 94% / 86%. `full_stack` recall 97% / 98%. Layers not redundant - heuristic catches a small set of attacks the classifier misses (justifies the ensemble). Most full-stack benign FPR comes from heuristic ESCALATEs, not classifier upgrades.
  - **PAIR adaptive-attacker calibration.** Qwen2.5-32B-Instruct as both attacker and judge, 25 hand-curated jailbreak seeds, max 5 iterations: **ASR 0.0% (0/25)**. NOT a claim of imperviousness to all adaptive attackers - stronger attacker (70B+), longer iteration budgets, or alternate strategies (multi-turn drift, language-switch, obfuscation) might produce non-zero ASR.

### Deferred to v0.7+
- **prEN ISO/IEC 24970 field-alias layer** - pending public final of the standard. Will land when 24970 publishes.
- **DORA mapping refinement** - pending deployer-side signal. Conservative defaults shipped in v0.5.3 stay until a financial deployer's input refines them.

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
- Recommended operating threshold changed from `0.8` to `0.3` - the added benigns shifted the score distribution, and 0.3 is now the optimal balanced-accuracy point.

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
- `data_exfil`: 0% (was 28.6% heuristic baseline - classifier now worse than heuristic here)
- `destructive_actions`: 25% (was 87.5% heuristic)
- `jailbreak`: 0% (was 100% heuristic)

The heuristic scorer retains strong coverage in these categories. Stack both rather than using the classifier alone. v0.5.2 will add targeted attack variants to rebalance the training corpus and recover recall without losing the FPR improvement.

## [0.5.0] - 2026-04-23

### Added
- `AdversarialClassifier` - opt-in XGBoost scorer for adversarial tool-call detection. Install with `pip install vaara[ml]`.
- `src/vaara/data/adversarial_classifier_v1.joblib` - 295 KB pre-trained bundle shipped with the wheel.
- `scripts/classifier_vs_heuristic.py` - reproducible comparison harness (by-seed train/test split, no leakage).
- `tests/adversarial/generated/` - 1945 LLM-generated adversarial variants across 8 attack categories (Qwen2.5-3B on MI300X, batched).
- `tests/adversarial/benign_generated/` - 635 benign variants (487 broad + 148 targeted at http_post/send_email/shell_exec legit uses).
- `[project.optional-dependencies.ml]` - `xgboost`, `scikit-learn`, `joblib`, `numpy`. Default install stays zero-dep.

### Benchmarks
On the by-seed held-out test (50 adversarial seeds held out entirely, plus 60 benign entries), classifier at threshold 0.8:
- Attack recall: 61% (vs 44% heuristic)
- Benign false-positive rate: 13% (vs 25% heuristic)
- Balanced accuracy: 74% (vs 50% heuristic)
- Latency: 140 µs mean, 210 µs p99 (commodity CPU, no GPU at inference)

Live dogfood on 174 Qwen-generated prompts with an end-to-end LangChain-style agent loop: recall 70%, FPR 51% - real-world distribution is tougher than the clean held-out test. Ship with `decision="escalate"` in production, not `decision="deny"`.

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
