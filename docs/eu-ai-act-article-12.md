# EU AI Act Article 12: what record-keeping actually requires, and what tooling has to produce

Article 12 of the EU AI Act requires that high-risk AI systems "technically allow for the automatic recording of events (logs) over the lifetime of the system" (Article 12(1), [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)). Providers must keep those logs for at least six months (Article 19(1)), and deployers who control the logs must do the same (Article 26(6)). That is the whole legal core: automatic logging must be possible, logging must happen, and the records must be retained.

This page states precisely what the text requires, what it does not, and what that means when you evaluate tooling. Getting this wrong in either direction is costly: under-doing it is a compliance gap, and over-claiming it is a credibility problem in front of exactly the audience the records are for.

## What the text requires

- **Automatic recording.** The system must technically allow events to be recorded automatically over its lifetime (Article 12(1)). Manual note-taking or after-the-fact reconstruction does not meet "automatic".
- **Relevance to risk and oversight.** Logging must enable identifying situations that may present a risk or a substantial modification, facilitate post-market monitoring, and support the deployer's monitoring duty under Article 26(5) (Article 12(2)).
- **Minimum content for remote biometric systems.** For the Annex III point 1(a) systems, Article 12(3) names specifics: period of each use, the reference database checked, the input data matched, and the identity of the persons who verified the results under Article 14(5).
- **Retention.** Logs are kept "for a period appropriate to the intended purpose", of at least six months, by providers (Article 19(1)) and by deployers to the extent the logs are under their control (Article 26(6)). Union or national law, in particular data-protection law, can change that period.

Timing caveat: the enforcement dates for high-risk obligations have been in legislative flux during 2026. Do not anchor a compliance plan to a hardcoded date from a blog post, this one included. Check the current consolidated text.

## What the text does not require

Article 12 does not mandate hash chains, digital signatures, tamper-evidence, or independent verifiability. A plain application log, retained for six months, can satisfy the letter of the record-keeping articles. Any vendor telling you the AI Act requires cryptographic proof is over-claiming, and you should treat the rest of their claims accordingly.

## Why tamper-evidence is worth having anyway

The records exist to be used, and they get used in the least friendly settings available: an incident investigation, a market-surveillance inquiry, a liability claim, a procurement review. In each of those, the party reading the log is deciding whether to believe the party who produced it. A plain log cannot help them, because the producer could have edited it. A signed, hash-chained record that the reader verifies from the bytes alone can.

So the practical standard is not "did you keep a log" but "will your record survive a challenge". Meeting Article 12 with evidence rather than plain logs costs little at runtime and converts a retention obligation into an asset. The full argument, including the trust model, is in [logs-vs-evidence.md](logs-vs-evidence.md).

## What to require from tooling

Evaluating an Article 12 tooling claim comes down to five questions:

1. Does it record automatically at the decision point, covering the proposed action, the decision, and the outcome, for allowed and blocked actions alike?
2. Are the records retained and exportable as a self-contained artifact?
3. Can a party who does not trust you verify the export offline, from the bytes and a public key, without running your software?
4. Is a missing or edited record a detectable failure rather than a silent one?
5. Do the records map to the articles an auditor will ask about, so evidence assembly is a query rather than a reconstruction project?

Vaara's answers are documented rather than asserted: the article-by-article mapping from runtime events to AI Act obligations (Articles 9, 11 to 15, 61, 73) is in [COMPLIANCE.md](COMPLIANCE.md), the verifier trust models are in [verifying-evidence.md](verifying-evidence.md), and [examples/prove-it-yourself/](../examples/prove-it-yourself/) is a runnable demonstration that produces a record, verifies it, and catches a forged byte.

## Questions

**Does Article 12 apply to AI agents?** Article 12 applies to high-risk AI systems as classified by Article 6 and Annex III. An agent is in scope when the system it belongs to is. Agentic systems make the logging question harder in practice, because the events that matter are tool calls and decisions, which application logs rarely capture in a usable form.

**Who is responsible, the provider or the deployer?** Both, differently. The provider must build the logging capability in (Article 12) and retain logs under its control (Article 19). The deployer must keep the logs it controls, at least six months (Article 26(6)), and monitor operation (Article 26(5)).

**Is six months enough?** Six months is the floor, not a recommendation. The text says "a period appropriate to the intended purpose", and other law can extend it. For systems where disputes surface slowly, appropriate is likely longer.

**Does a hash-chained trail satisfy Article 12 by itself?** The chain satisfies the integrity part of nothing by itself; what satisfies Article 12 is automatic recording of the relevant events plus retention. Vaara does the recording and retention and adds the integrity properties on top, which is the part that makes the record usable under challenge.
