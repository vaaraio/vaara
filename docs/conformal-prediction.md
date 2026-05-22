# Conformal prediction in plain language

This document is the non-technical companion to Vaara's formal
specification of adaptive risk scoring. The formal version lives at
[`docs/formal_specification.md`](formal_specification.md). If you are a
compliance reviewer, legal counsel, or procurement officer reading a
Vaara report and wondering what the conformal interval next to each
risk score actually means, this is the document for you.

## What is a risk score, and why is one number not enough?

When Vaara intercepts an action by an LLM agent (a tool call to read a
file, send an email, transfer funds, and so on), it produces a risk
score between 0 and 1. A score of 0 means "no harm expected" and a
score of 1 means "catastrophic harm expected." The score is the
system's best guess at the action's true risk.

A single number is not enough on its own. The same score of 0.6 can
come from two very different places: a confident estimate based on
many similar past actions, or an uncertain guess based on weak
signals. A compliance reviewer who only sees the number cannot tell
which one they are looking at.

This is the gap conformal prediction fills.

## What conformal prediction gives you

Instead of one number, conformal prediction gives you an **interval**.
A typical Vaara output looks like this:

> Risk score: 0.6, interval [0.42, 0.78].

The interval is read as "the true risk is somewhere between 0.42 and
0.78, and we have a known probability of being right about that range."

Two properties make the interval useful:

1. **Distribution-free.** The interval is valid without assuming the
   data follows any particular shape (normal, uniform, anything). This
   matters because real-world action sequences from LLM agents do not
   follow a textbook distribution.
2. **Coverage guarantee.** You set an error budget (say, 10%), and
   the interval comes with the guarantee that the true risk lands
   inside the interval at least 90% of the time, on average, over many
   actions. The guarantee holds without assuming the model is
   well-calibrated to begin with.

A narrow interval like [0.58, 0.62] tells a reviewer the system is
confident. A wide interval like [0.2, 0.95] tells them it is not. The
risk score itself can be identical in both cases. The interval is what
makes the difference visible.

## Why this matters under the EU AI Act

**Article 15(1)** requires that "high-risk AI systems shall be
designed and developed in such a way that they achieve an appropriate
level of accuracy, robustness, and cybersecurity, and that they
perform consistently in those respects throughout their lifecycle." The phrase "appropriate
level" is meaningless without a way to measure it. A point score
cannot be measured against an accuracy target because it has no
notion of how often it is wrong. A conformal interval is measurable: a
deployer can publish the guarantee (e.g. "the interval covers the true
risk at least 90% of the time") and an auditor can check it against
observed outcomes.

A widening interval is itself a real-time signal that the model is
moving into a region where its predictions cannot be trusted. A point
estimate does not surface that drift until ground-truth labels arrive
and the error rate is reconstructed after the fact.

In short: a conformal interval converts "trust me, the score is 0.6"
into "the score is 0.6, here is the range we expect to be in, here is
the error rate, and you can hold us to it."

## What this is not

Conformal prediction is not a confidence score in the
common-marketing sense (e.g. "the model is 87% confident"). Those
numbers usually have no formal meaning and degrade silently as the
underlying distribution shifts. The conformal interval has a
mathematical guarantee that holds without distributional assumptions.

Conformal prediction is also not a substitute for ground-truth outcome
labels. The system still needs to observe what actually happened after
each action to update its calibration. Vaara records that signal via
`OUTCOME_RECORDED` events. See [`COMPLIANCE.md`](../COMPLIANCE.md) for
the article mapping and [`VERDICTS.md`](../VERDICTS.md) for the
staleness windows the engine applies to those records.

## Where to read more

- Formal definition, coverage proof, and the exact scorer Vaara
  ships: [`docs/formal_specification.md`](formal_specification.md).
- How outcome records feed the calibration loop, and the staleness
  rules that decide when calibration is fresh enough to trust:
  [`VERDICTS.md`](../VERDICTS.md), Article 15(1) row.
- The original conformal prediction literature (Vovk, Gammerman,
  Shafer) is a good academic anchor if you want the textbook proof.
  Vaara implements split-conformal prediction, the most operationally
  practical variant.
