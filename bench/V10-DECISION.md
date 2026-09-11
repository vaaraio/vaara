# v10 candidate: not shipped

Decided 2026-09-11. The shipping classifier stays at v9
(`src/vaara/data/adversarial_classifier_v9.joblib`, sha256 `2566da22bf52`).

The candidate is `adversarial_classifier_v10_candidate.joblib`,
sha256 `c5112828d14d`. It is not in the packaged directory and is not
committed. Numbers come from `scripts/eval_v039_v9.py` with
`--split tests/adversarial/v040_split.json`, output at
`bench/v040_v10_candidate_eval.json`.

Read the corpus defect below first. It is the reason the first grading of this
candidate produced numbers that pointed the other way.

## The corpus defect

`scripts/build_v040_split.py` derived a file's category prefix from
`fp.name.split("-")`. Attack files are `SR-v037-llama33.jsonl`, prefix first,
so field 0 never contained the extension and the read was correct. Matched
benign files are `BT-v035-SR.jsonl`, prefix LAST, so field 2 was `"SR.jsonl"`
and matched no known prefix. Every file it touched was skipped in silence.

The effect: **2800 matched benign entries, 700 for each of the four v0.40
categories, sat on disk correctly labelled `benign_control` / `expected: ALLOW`
and were assigned to no fold in either v0.39 or v0.40.** The three categories
whose benigns did land (TM, PE, DE) reached v0.39 by a different path, which is
why the gap was invisible.

Fixed by reading `fp.stem` instead of `fp.name`. The split now carries 24619
entries against 21819 before, leakage check clean.

## What was measured, with the benigns present

Baseline held at its stored threshold 0.9150. Candidate recalibrated on the
v0.40 val fold to a 5 percent target FPR, which the restored negatives move
from 0.9460 to 0.9800.

| Surface | n (pos) | v9 recall | v10 recall | v9 FPR | v10 FPR |
|---|---|---|---|---|---|
| test, all | 2980 (1736) | 76.7% | 70.1% | 6.8% | 4.5% |
| test, inherited from v0.39 | 1827 (1063) | 84.7% | 67.7% | 4.1% | 1.8% |
| test, added by v0.40 | 1153 (673) | 64.0% | 73.8% | **11.2%** | 8.8% |
| v0.38 Phase 1, attacker-LLM | 900 (900) | 88.1% | 67.9% | 0.0% | 0.0% |
| v0.39 BIPIA holdout | 3604 (2920) | 64.6% | 46.7% | 6.0% | 4.8% |

## Why not

At the threshold the restored negatives force, the candidate loses on every
population that can be compared: 17.0pp of inherited test recall, 20.2pp on the
attacker-LLM surface, 17.9pp on the BIPIA holdout. It keeps a 9.8pp recall
advantage on the four refilled categories and pays for it everywhere else. This
is a clearer verdict than the first grading gave, and it points the same way.

The first grading, run before the benigns were restored, reported the candidate
ahead by 7.2pp on aggregate test recall. That number came from a val fold with
480 negatives missing, which let the candidate calibrate to a threshold it
could not hold once they were present.

## The finding that outranks the decision

**v9 false-positives on 11.2% of benign calls in the four categories.** More
than one in nine legitimate operations across credential handling, SSRF-adjacent
tool use, destructive actions and prompt injection would be flagged. That is the
shipping model, and the figure had never been computed because the negatives
that make it computable were orphaned.

No published claim is affected. README line 203 scopes its figure to held-out
TEST n=1,827, which is the inherited block, and that block still reads 84.7%
recall at 4.1% FPR exactly. The 11.2% is a surface no shipped claim covers.

## Next

1. Retrain on the corrected split. The four categories now carry 2800 matched
   negatives that no model has ever trained against, which is the direct
   remedy for both the 64.0% recall and the 11.2% FPR.
2. Regrade on all five rows above. Bar for shipping: inherited recall at or
   above 84.7%, attacker-LLM at or above 88.1%, and the added-category FPR
   materially below 11.2%.
3. Retire the v10 candidate. It was trained on a split missing 2800 negatives
   and there is no reason to carry it forward.

## Also fixed

`scripts/eval_v039_v9.py` now splits the test fold by origin whenever the
manifest declares `inherits_from`, and warns when a population has no negatives.
The first grading of this candidate hid a regression behind an aggregate, and
that warning is what surfaced the orphaned benigns.
