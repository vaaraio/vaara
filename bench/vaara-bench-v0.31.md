# vaara-bench-v0.31

Methodology spec for the v0.31 adversarial benchmark. Tightens v1 in
three ways. Adds a 70/15/15 stratified split, retrains the classifier
on TRAIN only, reports held-out TEST numbers with Wilson 95% intervals
instead of cross-validated point estimates.

vaara-bench-v0.31 is the source of the recall and FPR numbers cited in
v0.31 release notes. A reviewer matching all three SHAs (corpus
manifest, split manifest, training commit) gets bit-identical artefacts.

## Chain of custody

Four hashes lock the benchmark. Every cited number is reproducible from
these anchors plus the scripts in `scripts/`.

| anchor | path | what it pins |
|---|---|---|
| corpus manifest | `tests/adversarial/MANIFEST.sha256` | SHA-256 of every JSONL |
| split manifest | `tests/adversarial/v031_split.json` | entry-key to fold |
| classifier bundle | `src/vaara/data/adversarial_classifier_v2.joblib` | trained model + provenance |
| training commit | recorded in the bundle | git SHA at training time |

The bundle records the SHAs of the corpus and split manifests it was
trained against. The split manifest records the SHA of the corpus
manifest it was generated against. Mismatched SHAs mean different
artefacts, not different numbers on the same benchmark.

## Corpus

7,955 entries, nine categories, two sources.

| source | count | how it was built |
|---|---|---|
| hand_curated | 250 | top-level `tests/adversarial/*.jsonl`, manually written |
| llm_generated | 7,705 | `generated/` + `benign_generated/`, Qwen2.5-72B on MI300x |

Per-category counts and v0.31 generator parameters live in
`tests/adversarial/README.md`.

## Split

`scripts/build_train_val_test_split.py` produces a 70/15/15 split
stratified by (category, source). Composite stratification keeps
per-source recall computable on held-out folds.

- Ratio: 70 / 15 / 15
- Strata: 19 non-empty (category, source) buckets
- Smallest stratum (25-entry hand_curated attack category) lands 17 / 4 / 4
- Random state: 42, per-stratum seed `f"{42}::{category}::{source}"`
- Key format: `<relative_path>#L<line_index>` because raw `id` is not
  unique across files (1,855 cross-file collisions in v0.31)

Two consecutive runs produce identical assignments under stable-hash
comparison.

| fold | n |
|---|---|
| train | 5,563 |
| val | 1,196 |
| test | 1,196 |

## Classifier bundle

`scripts/save_classifier_bundle.py --split-manifest tests/adversarial/v031_split.json`
trains XGBoost on TRAIN only. The bundle records `version`,
`default_threshold`, `trained_at`, `training_commit`, `n_entries`,
`positive_rate`, `training_corpus_manifest_sha256`,
`split_manifest_sha256`, `training_fold`.

The runtime bundle (`adversarial_classifier_v1.joblib`) is the v0.5.3
classifier that ships with the package. `_v2.joblib` is gitignored
until the v0.31 release commit flips the runtime pointer in
`src/vaara/adversarial_classifier.py` from `_v1` to `_v2`.

## Threshold selection

`scripts/threshold_sweep_val.py` sweeps thresholds on VAL only. No
TEST contact.

| rule | threshold | recall | FPR |
|---|---|---|---|
| max Youden's J | 0.90 | 0.525 | 0.049 |
| max balanced accuracy | 0.90 | 0.525 | 0.049 |
| max F1 | 0.16 | 1.000 | 0.852 |
| max recall at FPR ≤ 5% | 0.90 | 0.525 | 0.049 |
| max recall at FPR ≤ 10% | 0.89 | 0.533 | 0.065 |

v0.31 ships threshold **0.90**, picked by both Youden's J and balanced
accuracy on VAL.

## Headline on TEST (n=1,196)

Threshold 0.90 chosen on VAL, applied to held-out TEST. Wilson 95%
intervals on every number.

| variant | recall | FPR |
|---|---|---|
| rules-only | 0.422 [0.387, 0.458] | 0.361 [0.318, 0.406] |
| classifier-only | **0.539 [0.503, 0.575]** | **0.046 [0.031, 0.070]** |
| both (OR) | 0.777 [0.746, 0.805] | 0.403 [0.358, 0.449] |

The v0.31 headline path is **classifier-only at threshold 0.90: recall
53.9% [50.3%, 57.5%] at FPR 4.6% [3.1%, 7.0%] on TEST n=1,196.**

OR-combining lifts recall to 77.7% but inflates FPR to 40.3% because
rules-only alone has 36.1% FPR. Ensemble combiners that bound FPR
require a separate calibration step planned for v0.32.

## Per-category recall (classifier-only, TEST, threshold 0.90)

| category | n | recall | FPR |
|---|---|---|---|
| jailbreak | 491 | 0.654 | 0.182 |
| ssrf_via_tools | 41 | 0.590 | 0.000 |
| prompt_injection | 40 | 0.562 | 0.250 |
| data_exfil | 41 | 0.300 | 0.000 |
| destructive_actions | 41 | 0.282 | 0.000 |
| credential_exfil | 50 | 0.244 | 0.200 |
| tool_misuse | 40 | 0.206 | 0.000 |
| privilege_escalation | 40 | 0.172 | 0.182 |
| benign_control | 336 | --- | 0.042 |

Jailbreak is the strongest path. Five attack categories sit below 30%
recall. Closing those is v0.32 feature-engineering work.

## Multi-attacker PAIR

Three attackers, 25 jailbreak seeds each, max_iters 5. All three hit
identical seed indices (JB-001 through JB-025), so ASR numbers are
directly comparable.

| attacker | seeds | successes | ASR | 95% Wilson upper |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 25 | 0 | 0.000 | 0.133 |
| Qwen2.5-72B-Instruct | 25 | 0 | 0.000 | 0.133 |
| Llama-3.3-70B-Instruct | 25 | 0 | 0.000 | 0.133 |

The honest ceiling at 0/25 successes is "no more than 13.3% ASR with
95% confidence", not "0% ASR".

## Named limits

Surface these before a reviewer does.

1. **Eval corpus is single-action.** A fresh Pipeline per entry means
   `agent_history`, `sequence_pattern`, `action_frequency`, and
   `confidence_gap` all start at zero. Only `taxonomy_base` tops a
   decision in the attribution audit. Four of five expert signals are
   structurally dormant on this benchmark. Multi-action agent sessions
   are a separate evaluation track.

2. **PAIR sample size.** n=25 per attacker is small. Wilson upper
   bound at 0 successes is 13.3%, not 0%.

3. **Five categories under 30% recall.** credential_exfil, data_exfil,
   destructive_actions, privilege_escalation, tool_misuse. The
   classifier is jailbreak-strong and weaker elsewhere.

4. **Both-variant FPR.** OR-combining lifts recall to 77.7% but FPR to
   40.3%. Not the headline number.

5. **Small hand_curated TEST denominators.** Per-source breakdowns for
   hand_curated carry wide Wilson intervals. Cite per-category
   instead of per-source on small denominators.

## Reproduction

```
cd tests/adversarial && sha256sum -c MANIFEST.sha256
.venv/bin/python scripts/build_train_val_test_split.py
.venv/bin/python scripts/save_classifier_bundle.py \
    --version v0.31 --threshold 0.90 \
    --split-manifest tests/adversarial/v031_split.json \
    --bundle-out src/vaara/data/adversarial_classifier_v2.joblib
.venv/bin/python scripts/eval_pipeline_attribution.py --fold all
.venv/bin/python scripts/three_way_variants.py --fold val --classifier-threshold 0.90
.venv/bin/python scripts/three_way_variants.py --fold test --classifier-threshold 0.90 \
    --out tests/adversarial/v031/test_final_eval_v031.json
.venv/bin/python scripts/threshold_sweep_val.py --fold val
.venv/bin/python scripts/wilson_intervals.py
```

The four anchor SHAs printed by these scripts must match the values
cited in release notes for the numbers to be the same numbers.
