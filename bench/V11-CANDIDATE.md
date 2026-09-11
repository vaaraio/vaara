# v11 candidate: graded, not shipped, decision open

Trained 2026-09-11 on the corrected v0.40 split (24619 entries, the one with
the 2800 restored matched benigns). Same recipe as v9, so the split is the only
variable: `scripts/train_v9_upweighted.py`, follow-weight 8.0, 400 estimators,
depth 6, lr 0.07.

Bundle `bench/candidates/adversarial_classifier_v11_candidate.joblib`, not
committed, outside the packaged directory. Eval at
`bench/v040_v11_candidate_eval.json`. Baseline held at its stored 0.9150,
candidate calibrated on val to 0.9090 for a 5 percent target FPR.

## Against the shipping model

| Surface | n (pos) | v9 recall | v11 recall | v9 FPR | v11 FPR |
|---|---|---|---|---|---|
| test, all | 2980 (1736) | 76.7% | **83.2%** | 6.8% | **3.6%** |
| test, inherited from v0.39 | 1827 (1063) | 84.7% | 83.6% | 4.1% | 4.2% |
| test, added by v0.40 | 1153 (673) | 64.0% | **82.5%** | 11.2% | **2.7%** |
| v0.38 Phase 1, attacker-LLM | 900 (900) | 88.1% | 84.6% | 0.0% | 0.0% |
| v0.39 BIPIA holdout | 3604 (2920) | 64.6% | 62.4% | 6.0% | **0.7%** |

## Against the bar, which was set before the numbers were read

The bar was written into `bench/V10-DECISION.md` and committed as `f2257f9`,
before this model existed. Three gates:

1. **Inherited recall at or above 84.7%.** MISSED. 83.6%, down 1.1pp.
2. **Attacker-LLM at or above 88.1%.** MISSED. 84.6%, down 3.5pp.
3. **Added-category FPR materially below 11.2%.** MET, and not narrowly:
   11.2% to 2.7%.

Two of three gates missed on point estimates. That is the finding, and the bar
is not being moved to accommodate the result.

What the confidence intervals say, separately and without relaxing anything:
both misses sit inside overlapping intervals. Inherited is [81.3, 85.7] against
[82.4, 86.7]; attacker-LLM is [82.0, 86.8] against [85.8, 90.1]. Neither
regression is separated from noise at n. The two largest wins are separated by a
wide margin: added recall 64.0% to 82.5% and added FPR 11.2% to 2.7%.

So the accurate statement is that v11 is **not measurably worse on either gate
it missed, and decisively better on the axis those gates exist to protect.**
That is an argument for resolving the question, not for declaring it resolved.

## What would resolve it

The attacker-LLM surface is the only regression worth spending anything on. It
is 900 entries from one generator, and v9's 88.1% there is the strongest number
the shipping model holds. Two options, in order of cost:

1. **Threshold sweep.** v11 calibrates to 0.9090 against v9's 0.9150. The
   operating point was chosen to hit 5 percent val FPR and nothing else. Sweep
   it against all five surfaces and check whether a point exists that recovers
   the attacker-LLM recall without giving back the FPR win. Cheap, CPU, no new
   training.
2. **Hyperparameter sweep on the corrected split.** Only worth it if the
   threshold sweep finds no such point. This is the job that justifies GPU
   time.

Shipping the classifier changes the packaged model, so it does not happen
without Henri's explicit word regardless of what the sweep returns.
