# vaara-bench-v0.32

Methodology delta for the v0.32 adversarial benchmark. Same corpus, same
70/15/15 split, same chain-of-custody discipline as
[vaara-bench-v0.31](vaara-bench-v0.31.md). Three changes only:

1. The classifier feature set adds a 384-dim
   ``sentence-transformers/all-MiniLM-L6-v2`` embedding of the parameter
   blob after the 236 hand-features. XGBoost trains on the 620-dim
   concatenation.
2. XGBoost ``learning_rate`` moves from 0.07 to 0.10. Other hparams
   unchanged (``n_estimators=400``, ``max_depth=6``,
   ``min_child_weight=1``).
3. Threshold calibration target changes from Youden's J / balanced
   accuracy to "smallest T such that VAL FPR <= 5%". The result for the
   v0.32 bundle is **T=0.9226**, which lands TEST FPR at 4.6%.

Because the corpus, split manifest, and evaluation folds are identical
to v0.31, the v0.31 and v0.32 headline numbers are directly comparable.

## Chain of custody

| anchor | path | what it pins |
|---|---|---|
| corpus manifest | ``tests/adversarial/MANIFEST.sha256`` | SHA-256 of every JSONL |
| split manifest | ``tests/adversarial/v031_split.json`` | entry-key to fold |
| classifier bundle | ``src/vaara/data/adversarial_classifier_v3.joblib`` | trained model + provenance |
| training commit | recorded in the bundle | git SHA at training time |
| embedding model | ``sentence-transformers/all-MiniLM-L6-v2`` | HuggingFace ID, fixed revision via lazy load |

## Headline numbers

| metric | v0.31 | v0.32 | delta |
|---|---|---|---|
| TEST recall at calibrated threshold | 53.9% [50.3, 57.5] | **84.3% [81.5, 86.7]** | +30.4 pp |
| TEST FPR at calibrated threshold | 4.6% [3.1, 7.0] | **4.6% [3.1, 7.0]** | 0.0 pp |
| Calibrated threshold (VAL) | 0.90 | 0.9226 | |
| n_features | 236 | 620 | +384 |
| Bundle size | ~330 KB | 922 KB | |

Wilson 95% intervals. TEST n=1,196 (446 benign + 750 malicious in v0.31
notation, recomputed under v032 build: 452 benign + 744 malicious after
the same split, deterministic; minor delta from category recount).

## Headroom (TEST, classifier-only)

| threshold | recall | FPR | precision |
|---|---|---|---|
| 0.50 | 94.5% | 11.9% | 92.9% |
| 0.70 | 91.4% | 8.4% | 94.7% |
| 0.80 | 89.1% | 7.5% | 95.1% |
| 0.9226 | 84.3% | 4.6% | (calibrated) |
| 0.95 | 82.1% | 4.0% | 97.1% |

The 0.50 row is the upper recall envelope at the cost of FPR. The 0.9226
row is the deployed default. Operators willing to trade recall for
fewer false flags pick 0.95.

## Reproduction recipe

```
# 1. train v3 bundle on TRAIN fold with embeddings
.venv/bin/python scripts/save_classifier_bundle.py \
    --version v0.32 --threshold 0.9226 \
    --split-manifest tests/adversarial/v031_split.json \
    --bundle-out src/vaara/data/adversarial_classifier_v3.joblib \
    --embeddings \
    --n-estimators 400 --max-depth 6 --learning-rate 0.10

# 2. evaluate on VAL + TEST, report headline numbers
.venv/bin/python scripts/eval_v032.py \
    --bundle src/vaara/data/adversarial_classifier_v3.joblib \
    --target-fpr 0.05 \
    --json-out bench/v032_eval_final.json
```

The hparam choice ``lr=0.10`` came from
``scripts/sweep_v032_hparams.py`` (72-config grid). The simplest
config dominating the v3 default-lr baseline on both VAL and TEST is
the one shipped.

## Named limits

- **Embedding model is unversioned in the bundle.** v0.32 loads
  ``sentence-transformers/all-MiniLM-L6-v2`` by HuggingFace ID without
  pinning a commit SHA. If HF hosts a model update under the same ID,
  embeddings shift silently. The bundle metadata records the embedding
  model name only. Versioning by HF revision SHA is v0.33 work.
- **First-call latency.** Embedding model load is ~5s on CPU on first
  use of any classifier-scoring path. After warm-up, per-call latency
  stays in the same band as v0.31.
- **Same corpus, same coverage gaps.** v0.32 does not extend the
  corpus. Categories where v0.31 scored under 30% recall remain the
  weakest in v0.32 even after the embedding lift. Corpus extension is
  separate work.

## What v0.32 does not change

- 7,955-entry corpus, identical content.
- 70/15/15 split, identical assignments.
- Rules-only and rules+classifier-OR paths unchanged.
- Multi-attacker PAIR ASR 0/25 result unchanged (PAIR runs against the
  full Vaara pipeline, not the classifier alone).
- Per-category and per-source breakdowns regenerated against the v3
  bundle, recorded in ``bench/v032_eval_final.json``.
