# Prior art and related work

This document anchors when each load-bearing Vaara concept first shipped
in a tagged public release, and lists adjacent published work in the
same lane. It exists so that a reader comparing Vaara against newer
academic or industry proposals can check the published timeline rather
than relying on marketing claims.

The chronology is reconstructed from `CHANGELOG.md` and the git history
of this repository. Dates are calendar dates of the tagged PyPI release.
Version numbers and dates can be verified against
[https://pypi.org/project/vaara/#history](https://pypi.org/project/vaara/#history)
and the `vX.Y.Z` tags on
[https://github.com/vaaraio/vaara/tags](https://github.com/vaaraio/vaara/tags).

## When each Vaara concept first shipped

| Concept | First shipped | Where to read it |
|---|---|---|
| Interception pipeline for agent tool calls | v0.1.0, 2026-04-10 | `src/vaara/pipeline.py`, `CHANGELOG.md` v0.1.0 |
| Adaptive risk scoring with conformal interval on every score | v0.1.0, 2026-04-10 | `docs/formal_specification.md`, `docs/conformal-prediction.md` |
| Hash-chained audit trail | v0.1.0, 2026-04-10 | `src/vaara/audit/`, `COMPLIANCE.md` |
| Framework integrations (LangChain, CrewAI, OpenAI Agents) and MCP server surface | v0.3.0, 2026-04-18 | `src/vaara/integrations/` |
| Signed audit-trail export and verification CLI | v0.4.1, 2026-04-20 | `src/vaara/audit/`, `docs/vaara-audit-cli.md` |
| Sigstore-signed release workflow with PyPI trusted publishing and PEP 740 attestations | v0.4.3, 2026-04-21 | `.github/workflows/release.yml`, `docs/signing-keys.md` |
| Opt-in XGBoost adversarial classifier with by-seed held-out benchmarks | v0.5.0, 2026-04-23 | `src/vaara/adversarial_classifier.py`, `bench/` |
| Callable kernel HTTP surface (Vaara as the schema, not the plug-in) | v0.10.0, 2026-05-16 | `docs/openapi.yaml`, `src/vaara/integrations/http.py` |
| Article 12 commit-prove receipt pair | v0.10.0, 2026-05-16 | `src/vaara/audit/`, `CHANGELOG.md` v0.10.0 |
| `vaara-bench-v1` reproducible benchmark harness | v0.12.0, 2026-05-16 | `bench/` |
| Hot policy reload without pipeline restart | v0.13.0, 2026-05-17 | `src/vaara/policy/`, `CHANGELOG.md` v0.13.0 |
| Static HTML article-coverage dashboard | v0.13.0, 2026-05-17 | `src/vaara/compliance/dashboard.py` |
| Pluggable signer with optional ML-DSA-65 (FIPS 204) post-quantum scheme | v0.14.0, 2026-05-17 | `src/vaara/audit/signer.py` |
| External-scorer composition over the same HTTP interface | v0.14.0, 2026-05-17 | `src/vaara/policy/composition.py` |
| TypeScript client (`@vaaraio/client`) for the HTTP surface | v0.15.0, 2026-05-17 | `clients/ts/` |
| PDF auditor evidence export (per-article rollup) | v0.16.0, 2026-05-17 | `src/vaara/compliance/render.py` |
| OVERT 1.0 reference verifier CLI (`vaara overt verify`) | v0.17.0, 2026-05-17 | `src/vaara/overt/`, `docs/openapi.yaml` |
| Streaming-notification interception inside the audit and OVERT perimeter | v0.25.0, 2026-05-21 | `src/vaara/integrations/mcp_proxy.py` |
| Per-article verdict drill-down: `verdict_inputs`, `verdict_reasons`, `contributing_events` | v0.26.0, 2026-05-21 | `src/vaara/compliance/engine.py`, `VERDICTS.md` |
| SLSA build provenance attestation on every release | v0.26.0, 2026-05-21 | `.github/workflows/release.yml` |
| Continuous fuzzing of the OVERT decoder, audit `from_dict`, and policy loader via ClusterFuzzLite | v0.27.0, 2026-05-22 | `fuzz/`, `.clusterfuzzlite/`, `.github/workflows/cflite_*.yml` |
| `VERDICTS.md` per-article evidence sufficiency reference | v0.28.0, 2026-05-22 | `VERDICTS.md` |
| `docs/conformal-prediction.md` plain-language explainer | v0.28.0, 2026-05-22 | `docs/conformal-prediction.md` |
| This document (`PRIOR_ART.md`) | v0.29.0, 2026-05-24 | `PRIOR_ART.md` |
| Cross-model held-out methodology with public 4,176-entry eval fold | v0.36.0, 2026-05-25 | `bench/vaara-bench-v0.36.md`, `tests/adversarial/v036_holdout.json` |
| Destination-aware features (`dst__*`) and v7 production classifier | v0.36.0, 2026-05-25 | `src/vaara/adversarial_classifier.py`, `scripts/train_adversarial_classifier.py` |

The `CHANGELOG.md` entry for each version carries the substantive
description and, where relevant, the failure mode that motivated the
change.

## Related published work

The following peer-review and pre-print papers describe approaches in
the same lane as Vaara (runtime evidence, hash-chained or signed audit
trails, conformal calibration, behavioural-constraint monitoring,
safety cases with runtime updates). They are listed here as related
reading, not as competitors. Where the publication post-dates Vaara's
shipped feature for the same idea, that is a chronological fact rather
than a judgment of the work.

### Runtime evidence and behavioural monitoring

- **Protocol-Driven Development: Governing Generated Software Through
  Invariants and Continuous Evidence.** arXiv:2605.12981v2, published
  2026-05-15. Introduces an "Evidence Chain" of compliance for
  generated implementations and a "Dynamic Evidence Ledger" for
  deployed systems, with signed runtime observations appended by
  verifiers. Conceptually adjacent to Vaara's hash-chained audit trail
  with article-explicit evidence (shipped v0.1.0, 2026-04-10) and to
  the per-article `verdict_inputs` and `contributing_events`
  drill-down (shipped v0.26.0, 2026-05-21).
- **Formal Methods Meet LLMs: Auditing, Monitoring, and Intervention
  for Compliance of Advanced AI Systems.** arXiv:2605.16198v1,
  published 2026-05-15. Proposes runtime monitors using Linear
  Temporal Logic for product-specific behavioural constraints, with
  intervening monitors that act at runtime to preempt predicted
  violations. Adjacent to Vaara's policy-driven runtime decisions and
  external-scorer composition (shipped v0.14.0, 2026-05-17).

### Safety cases with runtime confidence updates

- **A Subjective Logic-based method for runtime confidence updates in
  safety arguments.** arXiv:2605.22530v1, published 2026-05-21.
  Describes a method for continuously updating static safety cases
  using runtime Safety Performance Indicators, propagating confidence
  through a Subjective Logic assurance case. Adjacent to Vaara's
  evidence-sufficiency framework shipped in `VERDICTS.md` (v0.28.0,
  2026-05-22) and to the conformal interval that ships with every
  Vaara risk score (v0.1.0, 2026-04-10).

### Calibration and external validation

- **Calibration, Uncertainty Communication, and Deployment Readiness
  in CKD Risk Prediction: A Framework Evaluation Study.** arXiv:2605.21566v1,
  published 2026-05-20. Trains five classifiers on the UCI CKD dataset
  (400 patients) and evaluates each across calibration quality, conformal
  prediction coverage, and an eight-criterion deployment readiness
  framework. Reports internal AUROC 1.00 collapsing to 0.48-0.58 on the
  MIMIC-IV external cohort, with split-conformal coverage falling from
  0.80-0.98 internal to 0.21-0.25 against a 90% target. Domain
  incomparable to Vaara, but the methodological lesson (internal test is
  a ceiling, the external gap is visible only against a held-out
  generator) motivates Vaara's v0.36 cross-model held-out corpus
  (`bench/vaara-bench-v0.36.md`).

### Selective inference on conformal prediction sets

- **Selecting Informative Conformal Prediction Sets with an Optimized
  FCR-Controlled Approach.** arXiv:2605.22004v1, published 2026-05-21.
  Formalises selective inference on conformal prediction sets with
  finite-sample false coverage rate guarantees. Methodology pointer for
  Vaara's planned FPR-bounded three-stage combiner (rules-veto in the
  uncertain band), scheduled for v0.37+. Not yet implemented in Vaara.

### Aviation learning-assurance

- **Mechanistic Interpretability for Learning Assurance of a
  Vision-Based Landing System.** arXiv:2605.20607v1, published
  2026-05-20. Applies mechanistic interpretability to an EASA
  learning-assurance scenario, including out-of-model-scope runtime
  monitoring against the operational design domain. Vaara does not
  currently target aviation directly, but EASA learning-assurance is
  in the same harmonisation surface as the AI Act Article 6(1) /
  Annex I product-safety route.

### National security threat-modelling

- **Backchaining Loss of Control Mitigations from Mission-Specific
  Benchmarks in National Security.** arXiv:2605.21095v1, published
  2026-05-20. Methodology for national security deployers to
  back-chain affordance and permission constraints from
  use-case-specific benchmarks. Adjacent in motivation to Vaara's
  policy-driven decisions over agent tool calls, in a deployment
  context Vaara does not currently target.

### Classical foundations

- **Conformal prediction.** Vovk, Gammerman, Shafer. *Algorithmic
  Learning in a Random World* (Springer). Vaara implements
  split-conformal prediction with a distribution-free coverage
  guarantee, as documented in `docs/formal_specification.md` and
  explained in plain language in `docs/conformal-prediction.md`.
- **Linear Temporal Logic and runtime verification.** Pnueli (1977),
  Bauer, Leucker, Schallhart (2011). Background for the runtime-monitor
  literature cited above.

## What this document is not

This document is not a competitive matrix. It deliberately omits
vendor comparisons, feature checklists against named peers, and any
"first to ship" claims framed as authority rather than chronology.
Inclusion of a paper here means the work is in the same lane and
worth reading, not that it is positioned as inferior or superior to
Vaara.

For vendor positioning relative to commercial peers, see the
discussion in `COMPLIANCE.md` and the framework integrations under
`src/vaara/integrations/`.

## How to keep this current

When a tagged release adds a load-bearing concept (a new audit
primitive, a new evidence shape, a new public surface, or a new
formal property), add a row to the chronology table above with the
version, date, and a path into the codebase or docs. When a relevant
paper or peer specification appears in the wider literature, add it
to the related-work section with a neutral one-paragraph summary and
the publication date. The goal is a paper trail that a reader can
verify against PyPI, the git tags, and the cited URLs.
