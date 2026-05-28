# vaara-bench-v0.34

Methodology delta against [vaara-bench-v0.33](vaara-bench-v0.33.md).
Three changes, in order of decreasing scope:

1. **Adversarial corpus extended from 7,955 to 10,055 entries.** 700 new
   targeted entries each across the three weakest v0.32 TEST categories
   (`tool_misuse`, `privilege_escalation`, `data_exfil`), generated via
   Qwen2.5-72B-Instruct on AMD MI300X. Schema-validated, deduplicated
   within and across files. `tests/adversarial/MANIFEST.sha256`
   regenerated to anchor the new files.
2. **Fresh stratified split**: `tests/adversarial/v034_split.json`,
   70/15/15 by (category, source), n=7,033 TRAIN / 1,511 VAL / 1,511 TEST.
3. **An experimental v5 classifier bundle trained on the v034 TRAIN
   fold did NOT clear the v0.34 ship gate** for the production swap.
   See "What did not ship" below. The production loader continues to
   use `adversarial_classifier_v3.joblib` from v0.33.

## What v0.34 ships

- The extended corpus (publicly committed, anchored in MANIFEST).
- `tests/adversarial/v034_split.json` as the canonical evaluation fold.
- `src/vaara/data/adversarial_classifier_v5.joblib` as an A/B artefact
  not loaded by default, documented here for reproducibility.
- A README refactor toward an industry-standard single-pointer bench
  reference, version-agnostic `make bench` target, dynamic version in
  the OVERT example.
- This bench doc and the methodology lesson it records.

## What did not ship, and why

A v5 classifier bundle was trained on the v034 TRAIN fold under the
same hparams as v0.32 (`n_estimators=400, max_depth=6,
learning_rate=0.10`, MiniLM embeddings, revision pinned). Evaluated
against the v3 (v0.33) production bundle on the same v034_split TEST,
at matched FPR ~5%, the per-category picture is mixed:

| category | v3 @ T=0.80 (FPR 5.3%) | v5 @ T=0.95 (FPR 4.8%) | Δ |
|---|---|---|---|
| privilege_escalation | 74.1% | **87.4%** | +13.3pp |
| data_exfil | 90.8% | 91.5% | +0.7pp |
| tool_misuse | 77.5% | 76.8% | -0.7pp |
| credential_exfil | 68.9% | 66.7% | -2.2pp |
| jailbreak | 96.2% | 95.2% | -1.0pp |
| ssrf_via_tools | 87.2% | 79.5% | **-7.7pp** |
| destructive_actions | 79.5% | 69.2% | **-10.3pp** |
| prompt_injection | 90.6% | 78.1% | **-12.5pp** |

One targeted category lifted significantly (privilege_escalation
+13.3pp). The two other targeted categories were flat. **Three
untargeted categories regressed by 7-13 pp.** Ship gate failed.

**Root cause.** Adding 2,100 adversarial-only entries shifted the
TRAIN positive_rate from 0.620 to 0.701. The classifier was given
more high-confidence patterns to fit but no matched benign coverage,
so it traded discrimination on the untouched categories to fit the
new ones. This is a class-balance artefact, not a model architecture
problem. The corpus extension is sound. The retraining recipe was not.

**What v0.35 will do differently.** Generate matched benign coverage
for the three weakest categories before retraining. The corpus
extension that landed in v0.34 keeps its value as a permanent
benchmark anchor.

## Chain of custody

| anchor | path / value | what it pins |
|---|---|---|
| corpus manifest | `tests/adversarial/MANIFEST.sha256` (293 lines) | SHA-256 of every JSONL including the v034 additions |
| split manifest | `tests/adversarial/v034_split.json` | entry-key to fold, 10,055 entries, stratified 70/15/15 |
| production bundle | `src/vaara/data/adversarial_classifier_v3.joblib` | unchanged from v0.33, MiniLM revision `c9745ed1` pinned in metadata |
| A/B bundle (not loaded by default) | `src/vaara/data/adversarial_classifier_v5.joblib` | trained on v034_split TRAIN, documented above |
| droplet session logs | `bench/v034_droplet_logs/` | Qwen-72B vLLM session + per-category generator logs |

## Production headline (unchanged)

The production classifier is still v3 from v0.33. Numbers carry
forward unchanged:

| metric | value |
|---|---|
| TEST recall at calibrated threshold | 84.3% [81.5, 86.7] |
| TEST FPR at calibrated threshold | 4.6% [3.1, 7.0] |
| Calibrated threshold | 0.9226 |
| Evaluated against | `v031_split.json` TEST, n=1,196 |
| MiniLM embedding revision | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |

Wilson 95% intervals. Verification command:

```
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v3.joblib \
    --target-fpr 0.05 \
    --json-out bench/v033_eval_final.json
```

## Cross-eval on the new TEST (v034_split)

For reproducibility, v3 and v5 are both evaluated on the new harder
v034_split TEST (n=1,511) and recorded:

| bundle | calibrated T (VAL FPR ≤ 5%) | TEST recall | TEST FPR |
|---|---|---|---|
| v3 (production, v0.33) | 0.9140 | 79.8% [77.3, 82.1] | 2.9% [1.7, 4.8] |
| v5 (A/B, not shipped) | 0.9873 | 77.7% [75.1, 80.2] | 3.1% [1.8, 5.1] |

The v034_split TEST contains 2,100 entries the v3 model has never
seen at training time, which is why v3 also drops from its
v031_split TEST headline of 84.3%. This is held-out methodology
working as intended: the harder TEST exposes real generalisation,
not just memorisation.

Files: `bench/v034_eval_v3_cross.json`,
`bench/v034_eval_v5.json`,
`bench/v034_per_category_v3_at_T080.json`,
`bench/v034_per_category_v5_at_T095.json`.

## Reproduction recipe

```
# Verify corpus integrity (includes v034 additions)
cd tests/adversarial && sha256sum -c MANIFEST.sha256

# Re-evaluate the production v3 bundle on v031_split TEST
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v3.joblib \
    --target-fpr 0.05 \
    --json-out bench/v033_eval_final.json

# Cross-eval on v034_split TEST
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v3.joblib \
    --split-manifest tests/adversarial/v034_split.json \
    --target-fpr 0.05 \
    --json-out bench/v034_eval_v3_cross.json
```

The Makefile target `make bench` runs the full v0.34 chain
(integrity to split to evaluate v3 on both splits to cross-eval v5).
