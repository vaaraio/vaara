# v10 candidate: not shipped

Decided 2026-09-11. The shipping classifier stays at v9
(`src/vaara/data/adversarial_classifier_v9.joblib`, sha256 `2566da22bf52`).

The candidate is `adversarial_classifier_v10_candidate.joblib`,
sha256 `c5112828d14d`, trained on the v0.40 split
(`tests/adversarial/v040_split.json`). It is not in the packaged directory and
is not committed. Numbers below come from
`scripts/eval_v039_v9.py` with `--split tests/adversarial/v040_split.json`,
output at `bench/v040_v10_candidate_eval.json`.

## What was measured

Baseline held at its stored threshold 0.9150. Candidate recalibrated on the
v0.40 val fold to the same 5 percent target FPR, giving 0.9460. Every figure
below is at those two thresholds.

| Surface | n (pos) | v9 recall | v10 recall | Change |
|---|---|---|---|---|
| test, all | 2500 (1736) | 76.7% | 83.9% | +7.2pp |
| test, entries inherited from v0.39 | 1827 (1063) | 84.7% | 81.3% | -3.4pp |
| test, entries added by v0.40 | 673 (673) | 64.0% | 88.0% | +24.0pp |
| v0.38 Phase 1, attacker-LLM | 900 (900) | 88.1% | 82.9% | -5.2pp |
| v0.39 BIPIA holdout | 3164 (2920) | 64.6% | 63.0% | -1.6pp |

False-positive rate on the full test fold moves 4.1% to 3.7%.

## Why not

The headline +7.2pp is one population, not a general gain. Split the same test
fold by whether an entry existed before the v0.40 refill and the aggregate
comes apart: the candidate gains 24 points on the four refilled categories and
loses 3.4 points on everything that was already there. It also loses 5.2 points
on the attacker-LLM surface and 1.6 on the BIPIA holdout. Three of the four
populations that can be compared regress. The one that improves is the one the
candidate was trained alongside.

A second limit bounds how far the gain can be read at all. All 3883 v0.40
additions are attack-labelled. The corpus has 4800 benign entries, every one of
them inherited from v0.39 and carrying `category: benign_control`, matched to
no category. So the four refilled categories reach the test fold with zero
negatives, and their false-positive rate is unmeasurable at any threshold. The
+24pp is a recall figure with no cost term next to it.

Shipping a model that trades measured recall on three populations for
unmeasured-cost recall on a fourth is the wrong direction for a component whose
false positives block legitimate work.

## What would change the answer

1. Generate matched benigns for the four categories so v0.41 can put a
   false-positive rate next to the +24pp. This is the prerequisite, not a
   nice-to-have.
2. Retrain so the refilled categories are learned without giving up inherited
   recall. The candidate's loss pattern is consistent with the new attack-only
   mass pulling the decision surface; class-balanced sampling or a heavier
   weight on the inherited distribution is where to start.
3. Regrade against v9 on all five rows above. A candidate that holds inherited
   recall flat and keeps the attacker-LLM surface at or above 88.1% ships.

## Fixed at the same time

`scripts/eval_v039_v9.py` now splits the test fold by origin whenever the
manifest declares `inherits_from`, and warns when a population has no negatives.
The first candidate graded on an extended split hid a 3.4pp regression behind a
7.2pp headline, and nothing in the output revealed it.

`scripts/build_v040_split.py` claimed to add the four categories "with their
matched benigns". It never did. Its benign collector looked for `BT-v035-*`
file naming that has never been on disk, matched nothing, and said nothing. The
claim is corrected in the docstring and in the manifest metadata, and the
collector now warns when it comes back empty. Assignments are unchanged and
byte-identical, so the split itself is the same split; only its metadata and
sha256 moved, and the eval was rerun so its recorded provenance matches.
