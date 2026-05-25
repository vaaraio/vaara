# vaara-bench-v0.33

Methodology delta against [vaara-bench-v0.32](vaara-bench-v0.32.md).
Same corpus, same 70/15/15 split, same hparams. One source change:

1. The embedding model is now loaded with an explicit HuggingFace
   commit SHA (``c9745ed1d9f207416be6d2e6f8de32d1f16199bf``) instead
   of the floating ``main`` reference. Bundle metadata records the
   pinned revision. Closes the v0.32 named limit.

Because the model weights, corpus, split, and feature schema are all
identical to v0.32 and training is deterministic, the v0.33 bundle
reproduces v0.32's headline numbers exactly.

## What v0.33 explicitly is NOT

A bigger embedding. A 768-dim ``BAAI/bge-base-en-v1.5`` A/B was run on
the same TRAIN fold (revision pinned, hparams unchanged). Result:
TEST recall **82.5% [79.6, 85.1]** at FPR 5.1%, vs v0.32's 84.3% at
4.6%. The ship gate was +2 percentage points, the actual delta was
**-1.8 pp**. bge-base loses on this task in addition to costing
~440 MB extra on disk for ``vaara[ml]`` users. v0.33 ships MiniLM.

The negative result is recorded in ``bench/v033_bge_base_eval.json``.

## Chain of custody

| anchor | path / value | what it pins |
|---|---|---|
| corpus manifest | ``tests/adversarial/MANIFEST.sha256`` | SHA-256 of every JSONL |
| split manifest | ``tests/adversarial/v031_split.json`` | entry-key to fold |
| classifier bundle | ``src/vaara/data/adversarial_classifier_v3.joblib`` | trained model + provenance |
| training commit | recorded in bundle metadata | git SHA at training time |
| embedding model ID | ``sentence-transformers/all-MiniLM-L6-v2`` | HuggingFace ID |
| **embedding model revision** | ``c9745ed1d9f207416be6d2e6f8de32d1f16199bf`` | HF commit SHA, recorded in bundle metadata |

The revision SHA above corresponds to HuggingFace's
``sentence-transformers/all-MiniLM-L6-v2`` main branch as of 2026-05-25.
Future updates by the model maintainer will land on later SHAs without
affecting the v0.33 bundle.

## Headline numbers

| metric | v0.32 | v0.33 | delta |
|---|---|---|---|
| TEST recall at calibrated threshold | 84.3% [81.5, 86.7] | **84.3% [81.5, 86.7]** | 0.0 pp |
| TEST FPR at calibrated threshold | 4.6% [3.1, 7.0] | **4.6% [3.1, 7.0]** | 0.0 pp |
| Calibrated threshold (VAL FPR<=5%) | 0.9226 | 0.9226 | |
| n_features | 620 | 620 | |
| Bundle size | 922 KB | 922 KB | |
| Embedding revision recorded in bundle | no | yes | |

Wilson 95% intervals. TEST n=1,196 (452 benign + 744 malicious).
Verification command:

```
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v3.joblib \
    --target-fpr 0.05 \
    --json-out bench/v033_eval_final.json
```

## Per-category recall on TEST

At the v0.33 calibrated threshold T=0.9226, per-category TEST recall
(sorted ascending) is:

| category | n_pos | recall | n_neg | FPR |
|---|---|---|---|---|
| tool_misuse | 34 | 47.1% [31, 63] | 6 | 33.3% [10, 70] |
| privilege_escalation | 29 | 48.3% [31, 66] | 11 | 72.7% [43, 90] |
| data_exfil | 40 | 65.0% [50, 78] | 1 | 0.0% [0, 79] |
| credential_exfil | 45 | 66.7% [52, 79] | 5 | 40.0% [12, 77] |
| destructive_actions | 39 | 66.7% [51, 79] | 2 | 0.0% [0, 66] |
| prompt_injection | 32 | 71.9% [55, 84] | 8 | 62.5% [31, 86] |
| ssrf_via_tools | 39 | 82.1% [67, 91] | 2 | 0.0% [0, 66] |
| jailbreak | 480 | 95.2% [93, 97] | 11 | 9.1% [2, 38] |
| benign_targeted | 0 | n/a | 76 | 0.0% [0, 5] |
| benign_control | 6 | 50.0% [19, 81] | 330 | 0.9% [0, 3] |

Recorded in ``bench/v032_per_category_test.json``. The high
per-category FPR rates on small-n_neg categories (n_neg<=11) are
small-sample artefacts. Benign coverage in those categories is too
thin to read confidence intervals from. The 0.9% on
``benign_control`` (n_neg=330) is the load-bearing FPR signal.

## Named limits carried into v0.34

- ``tool_misuse``, ``privilege_escalation``, and ``data_exfil`` all
  sit between 47% and 65% recall. v0.33 cannot close that without
  more training data in those categories. v0.34 is targeted corpus
  extension on those three.
- Benign coverage outside ``benign_control`` is very thin (n_neg<=11
  per category), which inflates per-category FPR confidence
  intervals. v0.34 corpus extension includes benign coverage where
  the adversarial generation lands.
- The combined rules+classifier OR path remains naive (recall 77.7%
  at FPR 40.3% per v0.31 bench). The FPR-bounded three-stage
  combiner is v0.35 work.

## Reproduction recipe

```
# Train v3 bundle with revision pinned in metadata
.venv/bin/python scripts/save_classifier_bundle.py \
    --version v0.33 --threshold 0.9226 \
    --split-manifest tests/adversarial/v031_split.json \
    --bundle-out src/vaara/data/adversarial_classifier_v3.joblib \
    --embeddings \
    --n-estimators 400 --max-depth 6 --learning-rate 0.10

# Re-evaluate
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v3.joblib \
    --target-fpr 0.05 \
    --json-out bench/v033_eval_final.json

# Per-category breakdown for v0.34 targeting
.venv/bin/python scripts/eval_per_category.py \
    --bundle src/vaara/data/adversarial_classifier_v3.joblib \
    --threshold 0.9226 \
    --json-out bench/v032_per_category_test.json
```
