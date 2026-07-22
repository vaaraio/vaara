# The Code of Practice and the evidence you still need

The Code of Practice on Transparency of AI-Generated Content, published 10 June 2026, is the Union-wide framework for demonstrating compliance with the marking and labelling obligations of Article 50(2) and 50(4). The Commission's Article 50 guidance (C(2026) 5054 final, section 8.1) sets out what signing does and does not change, and it draws a line worth reading carefully before 2 August 2026.

Vaara is not a watermarking or content-marking tool and does not compete with the Code's technical measures. What follows maps where the tamper-evident record fits around it.

## What the Code covers, and what it does not

The Code has two sections: provider commitments on marking and detection of AI-generated content (50(2)), and deployer commitments on labelling deepfakes and public-interest AI text (50(4)). It does not cover 50(1), the obligation on interactive AI systems and AI agents to disclose their artificial nature. There is no code of practice for agent disclosure; that obligation applies in full from 2 August 2026 with no grace period, and every provider of an interactive system or agent is on the "demonstrate it yourself" path for it.

## The two positions the guidance creates

**Signatories** get focused supervision: authorities assess whether they adhered to the Code and implemented its measures (para 147). Adherence is a commitment about what you implement. When an authority checks that adherence, the question becomes retrospective: show that the measures ran, in production, across the period in question. A signature form is not that record.

**Non-signatories** must demonstrate compliance "through other adequate means", are expected to run a gap analysis against the Code's measures, and should expect "a larger number of requests for information and requests for access to assess the effectiveness, interoperability, robustness and reliability" of what they implemented (para 148). The guidance says plainly that authorities will need more detailed information from them.

Both positions end at the same desk: an official asking what your system actually did. The difference is how much paper they will ask for.

## What a Vaara record supplies in each obligation

| Obligation | The Code / guidance asks for | What the signed trail records |
| --- | --- | --- |
| 50(1) agent disclosure (outside the Code) | Disclosure of artificial nature and the principal, at key steps and every new interaction (guidance para 31) | The agent-profile disclosure receipt: statement, principal, step, authority reference, threaded into the delegation chain |
| 50(2) marking | Machine-readable marking that is effective, interoperable, robust, reliable | Not the marking itself. A disclosure event per marked output, with the notice or mark pinned by hash, proving the marking step ran when and where claimed |
| 50(4) labelling | Labels on deepfakes and public-interest AI text | A disclosure event per labelled item, with subject and channel, in the same chain as the actions that produced the content |
| 50(5) timing and manner | Information at the latest at the first interaction, clear and distinguishable | The timing check in the export: whether each session's disclosure preceded its first action, computed from the signed records |

The export (`vaara trail export-article50`) folds this into one package: the signed trail, a human-readable report per obligation and per key step, and a cover note written for the market surveillance officer receiving it, with offline verification instructions that do not require trusting the operator.

## The honest boundary

A Vaara record proves that the operator's system made and kept the disclosure record at the stated moment inside a tamper-evident chain. It does not prove that a watermark survives downstream processing, that pixels rendered on a screen, or that wording met accessibility requirements. The Code's technical measures and the record of having run them are different layers; you need both, and the report says so in as many words.

## If you are deciding right now

Signing the Code answers the 50(2) and 50(4) question for generative systems. It answers nothing for 50(1), and it does not by itself produce the operational record an adherence check or an access request asks for. Whichever position you take, start the record before 2 August: evidence accumulates forward only, and the first request for access will ask about the period you are in now.
