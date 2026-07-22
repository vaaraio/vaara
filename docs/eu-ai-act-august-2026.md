# EU AI Act on 2 August 2026: what actually applies, and what moved

Much of what is written about "the August 2026 AI Act deadline" is now wrong. The Omnibus VII simplification regulation, given final approval by the Council on 29 June 2026, moved the high-risk obligations to 2 December 2027 (stand-alone systems) and 2 August 2028 (systems embedded in products). What it did **not** move is Article 50 transparency and the Commission's enforcement of the general-purpose AI rules. Those arrive on schedule. Source: [Council press release, 29 June 2026](https://www.consilium.europa.eu/en/press/press-releases/2026/06/29/artificial-intelligence-council-gives-final-green-light-to-simplify-and-streamline-rules/).

This page states what applies from 2 August 2026, who it applies to, and what evidence a deployer of AI agents should be able to produce. As with [Article 12](eu-ai-act-article-12.md), precision matters in both directions: under-doing it is a compliance gap, over-claiming it is a credibility problem.

## What applies from 2 August 2026

**Article 50 transparency** ([full text](https://artificialintelligenceact.eu/article/50/)). Unlike the high-risk chapter, Article 50 is not gated on Annex III classification. It reaches ordinary chatbots, assistants, and agents:

- **50(1), providers:** an AI system that interacts directly with people must inform them they are interacting with AI, unless that is obvious from context.
- **50(2), providers:** systems generating synthetic audio, image, video, or text must mark outputs as artificially generated in a machine-readable format. For generative systems already on the market, Omnibus VII gives a grace period for this marking until 2 December 2026; new systems get none.
- **50(3), deployers:** people exposed to emotion recognition or biometric categorisation must be informed.
- **50(4), deployers:** deepfakes must be disclosed, and AI-generated text published to inform the public must be disclosed unless it went through human editorial review.
- **50(5):** the information above must arrive at the latest at the first interaction or exposure, in a clear and distinguishable manner, meeting accessibility requirements.

**The Commission's guidance on Article 50** (C(2026) 5054 final, 20 July 2026) confirms the 2 August 2026 date for 50(1) with no grace period, and spells out what agent disclosure means in paragraph 31: an AI agent must disclose both its artificial nature and the person on whose behalf it acts, covering the delegation of authority and accountability for its actions, to the persons instructing it at key steps (authorisation, reporting, validation), at every new interaction, and when it relies on output from other AI systems rather than a person. Paragraph 40 adds that disclosure is due whenever the system is asked about its nature. The guidance is non-binding but is the Commission's own reading of the obligation.

**General-purpose AI enforcement.** The obligations for general-purpose AI model providers have applied since 2 August 2025; what starts on 2 August 2026 is the Commission's power to enforce them, including fines. This concerns model providers, not agent deployers, but it keeps supervisory attention on exactly the question Article 50 asks deployers: can you show what your AI did and what the humans around it were told?

**Already in force since February 2025:** the Article 5 prohibitions and the AI literacy duty. Omnibus VII also adds a new prohibition on generating non-consensual intimate content and child sexual abuse material, from December 2026.

## What moved

- High-risk obligations (Articles 8 to 27, including [Article 12 record-keeping](eu-ai-act-article-12.md) and Article 14 human oversight): 2 December 2027 for stand-alone high-risk systems, 2 August 2028 when embedded in products.
- National regulatory sandboxes: deadline postponed to 2027 (reporting differs between August and December; the Official Journal text settles it).

The new dates bind once the regulation is published in the Official Journal, expected before 2 August 2026; until publication the original timeline remains the legal baseline.

Two years of runway on high-risk is time to build the record-keeping habit cheaply, not a reason to ignore it. The organisations that treat December 2027 the way most treated August 2026, as a surprise, will be buying evidence retroactively, which is precisely the thing evidence cannot be.

## The evidence question Article 50 raises

Article 50 obligations are behavioral: inform, mark, disclose, at a defined time, in a defined manner. When a market-surveillance authority or a counterparty asks, the question becomes retrospective: **show that the disclosure happened.** A screenshot of a banner is a claim about today, not about the ten thousand sessions before it. The record that answers the question has the same properties Article 12 evidence needs:

- recorded automatically at the moment of the interaction, not reconstructed;
- carrying the fact of disclosure (which notice, shown when, in which interaction) next to the actions the AI then took;
- tamper-evident and independently verifiable, because the reader is by definition someone questioning you.

This is the same evidence problem Vaara already solves for tool calls: the [signed, hash-chained trail](tamper-evident-audit-trail.md) records events at the decision point, and an outside party [verifies the export offline](verifying-evidence.md). A disclosure event is one more event type in the same chain. The [logs vs evidence distinction](logs-vs-evidence.md) applies unchanged: a disclosure line in an application log persuades people who already trust you; a verifiable record persuades the ones who do not.

## Recording and exporting Article 50 evidence

Vaara treats a disclosure as one more event in the same signed,
hash-chained trail as the agent's actions. At the moment the notice is
shown:

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

For an AI agent, the guidance's paragraph 31 asks for more than "this is
AI": on whose behalf the agent acts, and disclosure at the key steps of
the delegation. The agent profile records exactly those fields, and can
thread the disclosure into the delegation chain the trail already keeps:

```python
from vaara.audit.article50 import record_agent_disclosure

record_agent_disclosure(
    trail,
    statement="I am an AI agent acting for Demo Oy.",
    on_behalf_of="Demo Oy",
    step="first_interaction",   # or authorisation, reporting, validation,
                                # new_interaction, ai_output_received, on_inquiry
    session_id=session_id,
    channel="chat_ui",
    authority_ref="mandate-2026-041",
    attestation=attestation_claims,   # optional: pins an eIDAS-style
                                      # attestation by SHA-256 (footnote 21)
)
```

No Python required — the same record from the command line:

```bash
vaara trail record-disclosure --db audit.db \
  --statement "I am an AI agent acting for Demo Oy." \
  --on-behalf-of "Demo Oy" --step first_interaction
```

When someone asks, one command produces the package:

```bash
vaara trail export-article50 --db audit.db --out article50.zip --key signing.pem
```

The zip is the standard signed trail with `article50_report.md` folded
in: disclosure counts per paragraph, the events themselves, and 50(1)
session coverage including whether each disclosure preceded the
session's first action (the 50(5) timing question). Agent-profile
disclosures get their own section: counts per key step, how many named
the principal, how many carried an authority reference, and per-session
step coverage. The zip also carries `README_FOR_AUTHORITY.md`: a cover
note written for the market-surveillance officer receiving it — what is
inside, how to verify it offline without trusting the operator, and what
it does and does not establish. The report states
plainly what it proves (the record was made then and has not been
altered) and what it does not (that pixels rendered, or that wording met
accessibility requirements). Verify offline with
`vaara trail verify --zip article50.zip --pubkey <key>`.

## Questions

**Does Article 50 apply to my internal-only agent?** 50(1) turns on interaction with natural persons; an internal tool where every user knows it is AI is typically covered by the "obvious from the context" carve-out. Document that reasoning; the assessment itself is worth recording.

**We only deploy, we don't build. Which parts are ours?** 50(3) and 50(4) bind deployers directly. 50(1) and 50(2) bind providers, but if you substantially modify or white-label a system, provider duties can become yours. The provider/deployer line is the first thing to establish, not the last.

**Is watermarking enough?** Watermarking addresses 50(2) marking. It does nothing for 50(1), 50(4), or 50(5), which are about informing people, and nothing about proving any of it happened. Marking is one obligation of five.

**High-risk moved to 2027, so can this wait?** Article 50 cannot; it applies from 2 August 2026. And the high-risk delay is dated relief: an agent trail started now is two years of accumulated, cheap evidence by December 2027.
