# Logs vs evidence: proving what an AI agent did

A log is what your system says happened. Evidence is what a party who does not trust you can confirm happened. For AI agents acting on their own, the second is the one that settles a dispute, and most tooling only produces the first.

This page explains the distinction, what the EU AI Act does and does not require, and how to produce evidence rather than logs. If you want to see it rather than read about it, [examples/prove-it-yourself/](../examples/prove-it-yourself/) is a working end to end run.

## The distinction

A log records activity for the operator's benefit: what to debug, what to monitor, what to bill. Its trust model assumes the reader trusts the writer. That assumption holds inside your own team and breaks the moment the reader is a regulator, an auditor, or a customer after an incident, because you could have edited the log before showing it.

Evidence is built for a reader who does not extend that trust. It carries its own proof of integrity and origin, so the reader checks the record instead of trusting the recorder. Concretely, an evidence record is:

- **Signed**, so its origin is verifiable against a public key.
- **Hash-chained**, so each entry commits to the previous one and any edit, insertion, or deletion breaks the chain.
- **Independently verifiable**, so the reader reproduces every verdict from the bytes alone, ideally with software that is not yours.
- **Fail-closed on authenticity**, so a missing or altered entry is a provable gap, not a silent one.

## Why observability tooling does not cover this

Datadog, Splunk, the OpenTelemetry stack, and application logs answer "what does the system report." They are excellent at that. They are not built to answer "can an outside party confirm this was not changed," and for AI agents they usually do not capture the load-bearing facts at all: the per-agent decision, the policy it was checked against, the reasoning, and the integrity of the sequence of actions. A screenshot of a Datadog dashboard is a claim. A signed, hash-chained record an auditor re-derives themselves is evidence.

GRC platforms like Vanta and Drata sit at the other end. They collect static control evidence for SOC 2 or ISO 27001, snapshots that a configuration was in place. They do not produce a runtime record of what an autonomous agent actually did on a given day. That runtime record is the gap.

## What the EU AI Act actually requires

Be precise here, because over-claiming the law is its own credibility problem.

Article 12 requires high-risk AI systems to automatically record events (logs) over their lifetime and to retain them for an appropriate period. It mandates that logging happens and that records are kept. It does **not** mandate cryptographic tamper-evidence, hash chains, or independent verifiability. A plain retained log can satisfy the letter of Article 12.

So why produce evidence rather than logs? Because the obligation you are really preparing for is not "did you keep a log" but "prove what the system did" when it is challenged, under Article 12 record-keeping, Article 14 human oversight, an incident investigation, a liability claim, or a procurement review. A plain log answers the first and fails the second. Tamper-evident, independently verifiable evidence answers both. The hash chain is not a compliance checkbox. It is what makes the record hold up in front of someone who has every reason to doubt it.

(Note on timing: the exact enforcement dates for high-risk obligations are in legislative flux and have shifted during 2026. Do not build a plan around a single hardcoded date; check the current text of the regulation.)

## How to produce evidence

The mechanics are not exotic. At the point an agent acts, record the action, the decision, and the outcome into a signed, append-only, hash-chained store, and make the export verifiable with a public key and, ideally, a checker that does not depend on your own code.

With Vaara that is the default behavior rather than an add-on:

```python
import vaara

@vaara.govern
def transfer_funds(to: str, amount: float) -> str:
    ...
```

Every call is decided against your policy, and the decision, the call, and the outcome land in a signed record anyone can verify offline. The [prove-it-yourself example](../examples/prove-it-yourself/) shows the full loop, produce a record, verify it, then watch a single forged byte get caught. The trust model for each verifier is in [verifying-evidence.md](verifying-evidence.md).

## Questions

**Is a hash-chained audit trail required by the EU AI Act?** No. Article 12 requires automatic logging and retention, not tamper-evidence. Tamper-evidence is what makes the log hold up when challenged, which is a stronger and more useful property than the minimum the law names.

**Can I add this without replacing my logging?** Yes. Evidence and logs answer different questions and coexist. Keep your observability stack; add a verifiable record at the decision point.

**Who verifies the evidence?** Whoever needs to: the regulator, the auditor, the customer, or you. Verification needs the record and a public key, not access to your systems, and Vaara ships a standalone checker so an independent party can reproduce the result without running your software.
