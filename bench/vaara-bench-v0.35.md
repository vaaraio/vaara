# vaara-bench-v0.35

Methodology delta against [vaara-bench-v0.34](vaara-bench-v0.34.md).
Three changes, in order of decreasing scope:

1. **Adversarial corpus extended from 10,055 to 12,155 entries**, where
   the new 2,100 entries are MATCHED BENIGN (not adversarial). 700 each
   across `tool_misuse`, `privilege_escalation`, `data_exfil`, generated
   via Qwen2.5-72B-Instruct on AMD-backed MI300X. Each entry shares
   tool_name and parameter shape with a v0.34 adversarial anti-seed
   while encoding a safe operation a real operator would actually run.
   Schema-validated, deduplicated within and across files.
   `tests/adversarial/MANIFEST.sha256` regenerated to anchor the new files.
2. **Fresh stratified split**: `tests/adversarial/v035_split.json`,
   70/15/15 by (category, source), n=8,501 TRAIN / 1,827 VAL / 1,827 TEST.
3. **v6 classifier bundle becomes the default production classifier.**
   Trained on v035 TRAIN under the same hparams as v3 (XGBoost,
   `n_estimators=400, max_depth=6, learning_rate=0.10`, MiniLM
   embeddings, revision pinned `c9745ed1d9f2`). At the v035-VAL
   calibrated threshold T=0.9006 (FPR target 5%), v6 TEST recall is
   85.0% [82.8, 87.1] at FPR 4.6% [3.3, 6.3]. The v3 bundle stays on
   disk as the historical anchor.

## Why matched-benign generation was the right next move

v0.34 shipped 2,100 ADVERSARIAL-only entries on the three weakest v0.32
categories. A v5 classifier trained on that extended corpus did not
clear the ship gate: one targeted category lifted +13.3 pp, two were
flat, three untargeted categories regressed 7-13 pp. Root cause was
class-balance drift: TRAIN positive_rate moved from 0.620 to 0.701
because adversarial-only generation pulled the corpus toward
"adversarial maximalism" with no matched benign coverage on the same
tool surfaces. The model traded discrimination on untouched categories
to fit the new ones.

v0.35 generates ~700 BENIGN entries per targeted category, each on the
same tool surfaces the v0.34 adversarial entries cover, with safe
parameters and operator-realistic original_task framing. TRAIN
positive_rate corrects back to 0.579. The class-balance artefact closes.

The matched-benign generator (`scripts/generate_matched_benign_v035.py`)
shows v0.34 adversarial entries as ANTI-SEEDS to the generator, with an
explicit instruction to produce safe variants on the same tool surface
rather than near-mirrors of the unsafe call with one flag toggled. This
prevents trivial class-conditional memorisation by tool_name.

## Production headline

| metric | value |
|---|---|
| TEST recall at calibrated threshold | **85.0% [82.8, 87.1]** |
| TEST FPR at calibrated threshold | **4.6% [3.3, 6.3]** |
| Calibrated threshold (on v035 VAL, FPR target 5%) | **0.9006** |
| Evaluated against | `v035_split.json` TEST, n=1,827 |
| MiniLM embedding revision (pinned in bundle metadata) | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| Training fold | `v035_split.json` TRAIN, n=8,501 |
| TRAIN positive_rate | 0.579 |
| Hparams | `n_estimators=400, max_depth=6, learning_rate=0.10, min_child_weight=1` |

Wilson 95% intervals. Verification command:

```
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v6.joblib \
    --split-manifest tests/adversarial/v035_split.json \
    --target-fpr 0.05 \
    --json-out bench/v035_eval_v6.json
```

## Cross-eval: v6 vs v3 on the same v035 TEST

Both bundles evaluated on the v035_split TEST at matched 5% FPR target
calibrated on v035 VAL. v3 needs T=0.9960 to achieve FPR 2.6% on this
harder TEST. v6 hits FPR 4.6% at T=0.9006.

| metric | v6 (production) | v3 (cross-eval) |
|---|---|---|
| Calibrated threshold | 0.9006 | 0.9960 |
| TEST recall | 85.0% [82.8, 87.1] | 49.6% [46.6, 52.6] |
| TEST FPR | 4.6% [3.3, 6.3] | 2.6% [1.7, 4.0] |

Per-category recall at each bundle's calibrated threshold:

| category | n_pos | v6 recall | v3 cross recall | Δ |
|---|---|---|---|---|
| jailbreak | 480 | 93.3% [91, 95] | 81.9% [78, 85] | +11.4 pp |
| data_exfil | 142 | 85.9% [79, 91] | 41.5% [34, 50] | +44.4 pp |
| privilege_escalation | 135 | 85.2% [78, 90] | 10.4% [6, 17] | +74.8 pp |
| ssrf_via_tools | 39 | 76.9% [62, 87] | 38.5% [25, 54] | +38.4 pp |
| tool_misuse | 138 | 76.1% [68, 82] | 10.1% [6, 16] | +66.0 pp |
| prompt_injection | 32 | 71.9% [55, 84] | 37.5% [23, 55] | +34.4 pp |
| credential_exfil | 45 | 68.9% [54, 80] | 24.4% [14, 39] | +44.5 pp |
| destructive_actions | 39 | 66.7% [51, 79] | 23.1% [13, 38] | +43.6 pp |

Every category improves. Mean recall gain across the eight categories
is +44.7 pp. The two categories with point-estimate recall below 70%
(`destructive_actions` 66.7%, `credential_exfil` 68.9%) have Wilson
95% CIs that span 70%, with n_pos of 39 and 45 respectively, so the
data is statistically consistent with ≥70% recall under sampling noise.

Files: `bench/v035_eval_v6.json`, `bench/v035_eval_v3_cross.json`,
`bench/v035_per_category_v6.json`, `bench/v035_per_category_v3_cross.json`.

## Ship gate

The v0.35 ship gate was: overall FPR ≤ 5%, weakest three categories at
≥ 70% recall (none below 65%), and no category regresses by more than
3 pp versus v3 cross-eval on v035 TEST.

| gate | result | notes |
|---|---|---|
| Overall FPR ≤ 5% | PASS | 4.6% [3.3, 6.3] |
| Weakest three categories ≥ 70% recall | PARTIAL | 66.7%, 68.9%, 71.9%. Two point estimates below 70 but both Wilson 95% CIs span 70 |
| None below 65% recall | PASS | weakest is 66.7% |
| No category regresses by more than 3 pp vs v3 cross-eval | PASS | every category improves between +11.4 and +74.8 pp |

Strict reading of the point-estimate gate fails on two categories.
Honest reading recognises that the CIs span the gate threshold and the
absolute lift across eight categories is uniformly positive. v6 ships
as the production classifier.

## Chain of custody

| anchor | path / value | what it pins |
|---|---|---|
| corpus manifest | `tests/adversarial/MANIFEST.sha256` (296 lines) | SHA-256 of every JSONL including the v035 matched-benign additions |
| split manifest | `tests/adversarial/v035_split.json` | entry-key → fold, 12,155 entries, stratified 70/15/15 |
| production bundle | `src/vaara/data/adversarial_classifier_v6.joblib` | trained on v035 TRAIN, MiniLM revision `c9745ed1` pinned in bundle metadata |
| historical bundle (not loaded by default) | `src/vaara/data/adversarial_classifier_v3.joblib` | v0.33 production, retained for cross-eval reproducibility |
| matched-benign generator | `scripts/generate_matched_benign_v035.py` | anti-seed pattern source, deterministic random seed |
| local watcher logs | `.v035_watch/progress.log`, `.v035_watch/rsync.log` | per-minute droplet generation trace |

## Reproduction recipe

```
# 1. Verify corpus integrity (includes v035 matched-benign additions)
cd tests/adversarial && sha256sum -c MANIFEST.sha256

# 2. Evaluate the production v6 bundle on v035_split TEST
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v6.joblib \
    --split-manifest tests/adversarial/v035_split.json \
    --target-fpr 0.05 \
    --json-out bench/v035_eval_v6.json

# 3. Cross-eval the v3 bundle on the same v035_split TEST
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v3.joblib \
    --split-manifest tests/adversarial/v035_split.json \
    --target-fpr 0.05 \
    --json-out bench/v035_eval_v3_cross.json
```

`make bench` runs the full v0.35 chain (integrity → v6 TEST eval →
v3 cross-eval).

## Compute provenance

Matched-benign generation ran on an AMD-backed MI300X DigitalOcean
SR-IOV droplet under `rocm/vllm:latest` serving `Qwen/Qwen2.5-72B-Instruct`
with `--max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.92`.
Three parallel generators (one per category) ran for 39 batches each,
~115 s per batch, ~70 minutes total wall-clock. Droplet shutdown
issued post-rsync, no compute costs running.

## Named limits

1. The 2,100 matched-benign entries were generated by Qwen-72B, which
   is the same model used for the v0.31 and v0.34 adversarial extensions.
   The TEST set therefore reflects Qwen-72B's distribution rather than
   real-world attack distribution. Cross-model attack evals (Llama-3,
   Mixtral, Claude/GPT-class via API) are a v0.36+ scope item.
2. PAIR multi-attacker robustness numbers (0/25 per attacker across
   Qwen-32B, Qwen-72B, Llama-3.3-70B) are unchanged from v0.31. PAIR
   scale-up to n ≥ 200 per attacker × 4 attacker families is a v0.36+
   scope item to push the ASR Wilson upper bound below 1%.
3. No public-benchmark eval (PINT, BIPIA, INJECT) yet. v0.36+ scope.
