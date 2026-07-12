# Vaara vs observability vs GRC platforms: who produces evidence of what an AI agent did?

They answer three different questions. Observability platforms (Datadog, Splunk, the OpenTelemetry stack) answer "what does my system report is happening", for the operator. GRC platforms (Vanta, Drata and similar) answer "were the required controls in place", for a certification. Vaara answers "can a party who does not trust us confirm what our AI agent actually did", for a regulator, auditor, customer, or court. Most teams running agents in production eventually need all three, because none of the three substitutes for the others.

This page lays out the differences so you can decide what you are missing. It is written by the Vaara project, so read it as a vendor's map of the terrain, checked against what each category says about itself.

## The comparison

| | Observability | GRC / compliance automation | Vaara |
|---|---|---|---|
| Question answered | What is the system doing right now, and why is it slow or broken | Are the controls required by a framework in place | What did the agent do, provably, for a reader who does not trust the producer |
| Built for | Operators and developers | Compliance teams pursuing SOC 2, ISO 27001 and similar | Whoever must prove or verify agent behavior: deployer, auditor, regulator, counterparty |
| Unit of record | Logs, metrics, traces | Control attestations and configuration snapshots | Per-action decision records: proposed call, policy decision, outcome |
| Trust model | Reader trusts the producer and the platform | Reader trusts the platform's collection | Reader verifies signature and hash chain from the bytes; trusts no one's word |
| Tamper handling | Access control; edits are not self-evident | Access control; snapshots are point-in-time | Any post-hoc edit or deletion fails verification with a named reason |
| Enforcement | None; it observes | None at runtime; it attests | Allow, block, or escalate each tool call against policy before it runs |
| Verifiable offline by an outsider | No | No | Yes, from the bundle and a public key, with a standalone checker |

## What each is genuinely good at

**Observability** is how you run software. Traces, dashboards, alerting, and retention at volume are mature and battle-tested, and agent frameworks increasingly emit OpenTelemetry spans you can collect today. Keep it. Nothing on this page argues against observability. The gap is the trust model: it reports what the system says to people who already trust the system, and an exported dashboard does not survive the question "how do I know this was not edited?"

**GRC platforms** compress the certification grind. They gather control evidence across your stack and keep an auditor supplied with attestations that configurations were in place. That is real, recurring work done well. The gap is runtime: a SOC 2 attestation says your organization has controls. It does not contain a record of what a specific agent did on a specific day, which is the artifact an incident, dispute, or AI Act record-keeping question actually asks for.

**Vaara** sits at the decision point. It scores and decides each agent tool call against your policy, then writes the call, the decision, and the outcome into a signed, hash-chained trail that an outside party verifies offline without running any of our code. It also enforces, which the other two by design do not: a destructive call can be blocked or escalated before it runs, and the block lands in the same trail. It runs in your environment with no SaaS and no telemetry, and maps its records to EU AI Act articles ([COMPLIANCE.md](COMPLIANCE.md)). What it is not: a metrics platform, a dashboard, or a certification workflow.

## Using them together

The stacking is straightforward. Observability keeps carrying operational telemetry. The GRC platform keeps carrying certification evidence. Vaara adds the per-action evidence layer at the agent boundary, and its exports become inputs to the other two: an evidence bundle is a strong artifact to attach in a GRC workflow, and decision events can flow into your log pipeline like any other structured source.

If you want to test the difference in trust model concretely, [examples/prove-it-yourself/](../examples/prove-it-yourself/) produces a signed trail, verifies it offline, and then shows a one-byte forgery failing the check. No observability or GRC product attempts that property, because it was never their question.

## Questions

**We already ship agent traces to Datadog. Is that not an audit trail?** It is an operational record, and a good one, but its integrity rests on access control and on trusting whoever operates the account. It does not verify offline in an outsider's hands, and it usually lacks the policy decision per action. See [logs-vs-evidence.md](logs-vs-evidence.md) for the full argument.

**Our GRC tool has an AI category now. Does that cover this?** Check what the artifact is. If it attests that policies and configurations exist, it is control evidence, which is necessary and different. The runtime question, what did the agent do and can you prove it, needs a record produced at the decision point.

**Does the EU AI Act force a choice between these?** No. Article 12 requires automatic event logging and Articles 19 and 26(6) require retention; how you meet that is up to you. The precise reading, including what the law does not require, is in [eu-ai-act-article-12.md](eu-ai-act-article-12.md).

**Is Vaara a replacement for any of them?** No. It replaces nothing you run today. It adds the layer none of them produce: enforceable policy at the agent boundary and evidence that survives a hostile reader.
