# The AI Act deadline that didn't move

Most of what you have read about "the August 2026 AI Act deadline" is now
wrong. The Omnibus VII simplification, approved by the Council on
[29 June 2026](https://www.consilium.europa.eu/en/press/press-releases/2026/06/29/artificial-intelligence-council-gives-final-green-light-to-simplify-and-streamline-rules/),
moved the high-risk obligations to December 2027 and August 2028. Half the
compliance industry exhaled and moved on.

They exhaled past the part that stayed. [Article 50 transparency](https://artificialintelligenceact.eu/article/50/)
applies from 2 August 2026, and unlike the high-risk chapter it is not gated
on any risk classification. If your system talks to people, it must tell them
they are talking to AI. If it generates content, the output must carry
machine-readable marking. Deepfakes and AI-written public text must be
disclosed. And the notice has to arrive at the first interaction, not
somewhere in your terms of service. This reaches ordinary chatbots,
assistants, and agents. Yours, probably.

Here is the part that gets uncomfortable later. Article 50 duties are
behavioral: inform, mark, disclose, at a defined moment. The enforcement
question is retrospective: show that it happened. A screenshot of your
disclosure banner is a claim about today. It says nothing about the ten
thousand sessions before the auditor asked, and your application logs will
not settle it either, because you could have edited them. A log persuades
people who already trust you. The person asking, by definition, does not.
The full argument is in [logs vs evidence](logs-vs-evidence.md).

What answers the question is a record made at the moment of the interaction,
carrying which notice was shown, when, in which session, next to what the AI
then did, in a form an outside party can verify without access to your
systems. That is evidence, and it is a different artifact from a log.

This is what Vaara does. It is open source, runs entirely in your own
environment, and records disclosures into the same
[signed, hash-chained trail](tamper-evident-audit-trail.md) as the agent's
actions:

```python
from vaara.audit.article50 import record_disclosure

record_disclosure(
    trail,
    paragraph="50(1)",
    statement="You are chatting with an AI assistant operated by Demo Oy.",
    session_id=session_id,
    channel="chat_ui",
)
```

When someone asks, one command exports a signed package with per-paragraph
coverage and the timing check built in, and the recipient
[verifies it offline](verifying-evidence.md) with a standalone checker that
imports none of our code:

```bash
vaara trail export-article50 --db audit.db --out article50.zip --key signing.pem
```

The report states what it proves and what it does not. Where evidence is
missing it says so, because a compliance report that cannot say
"insufficient" is worthless to everyone, including you.

On the high-risk postponement: two years of runway is time to build the
record-keeping habit cheaply, not a reason to ignore it. Evidence is the one
thing you cannot buy retroactively. The organisations that treat December
2027 the way most treated this August, as a surprise, will discover that.

Article 50 evidence works today: `pip install vaara`, one function call at
your disclosure point. The detailed legal breakdown is in
[EU AI Act on 2 August 2026](eu-ai-act-august-2026.md), and the concept
timeline with shipped versions is anchored in [PRIOR_ART.md](PRIOR_ART.md).
