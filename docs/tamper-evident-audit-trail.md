# A tamper-evident audit trail for AI agents

A tamper-evident audit trail is an append-only record in which every entry cryptographically commits to the one before it, the whole sequence is signed, and any edit, insertion, or deletion after the fact makes verification fail. It does not prevent tampering. It makes tampering detectable by anyone holding the record and the public key, which for audit purposes is the property that matters: the reader no longer has to trust the party who kept the record.

For AI agents the trail records tool calls: what the agent proposed, what the policy decided, and what actually happened. This page explains the mechanics, and, just as important, what a tamper-evident trail does not guarantee.

## The mechanics

**Append-only entries.** Each agent action produces records at the decision point: action requested, risk scored, decision made, outcome recorded. Nothing is written retroactively.

**Hash chain.** Every entry carries the SHA-256 hash of the previous entry. Verification recomputes each hash from the entry bytes and compares. Change one byte anywhere and every subsequent link fails, so an edit cannot be local or quiet. Deleting a record breaks the chain the same way, which turns "the log is complete" from an assertion into a checkable claim.

**Signed export.** The trail exports as a bundle whose manifest lists the content digests, signed with the producer's key. The signature binds origin; the digests bind content. Verification is offline: bundle plus public key, no network, no access to the producing system.

**Fail-closed verification.** The verifier reports ok only when the signature is actually established and the chain re-derives end to end. A missing key, an unverifiable signature, or a digest mismatch is a named failure, never a shrug. In [examples/prove-it-yourself/](../examples/prove-it-yourself/) you can watch a single edited byte get caught and named:

```
verify_signed(evidence.zip)           ->  ok=True
verify_signed(evidence_tampered.zip)  ->  ok=False
  caught: trail.jsonl SHA-256 does not match manifest.trail_sha256
```

## What it does not guarantee

Honesty about limits is part of the trust model, so here are the limits.

**It does not prevent lying at write time.** A hash chain proves the record was not altered after writing. It cannot prove the writer recorded the truth in the first place. The mitigation is architectural: the record is produced by the interception layer at the moment of decision, not by the agent or the application after the fact, which shrinks the window in which a false record can be constructed.

**It does not by itself prove when records were written.** A producer could regenerate an entire trail and re-sign it. Anchoring mitigates this: Vaara supports RFC 3161 timestamping and, where hardware offers it, binding the trail identity to a TPM 2.0 or confidential-VM root, so backdating requires forging an external witness too.

**It does not replace access control.** Tamper-evidence tells you the record was altered; it does not stop the alteration. Keep the trail on storage with appropriate permissions. Detection plus retention is the audit property; prevention is an operations property.

Claims beyond these limits should make you suspicious of any audit product, ours included.

## Cost and adoption

The trail is written by the same interception that enforces policy, so there is no second pipeline to operate. The rule-based scoring hot path adds 140 microseconds mean per call on a commodity CPU, reproducible with `make bench`; the benchmark method and corpus are frozen in [bench/vaara-bench-v1.md](../bench/vaara-bench-v1.md). Adoption is one decorator on a governed function, an MCP proxy in front of an existing server, or framework adapters for LangChain, CrewAI, and the OpenAI Agents SDK. Everything runs in your environment; there is no SaaS and no telemetry.

```python
import vaara

@vaara.govern
def transfer_funds(to: str, amount: float) -> str:
    ...
```

## Questions

**How is this different from writing logs to WORM storage?** WORM storage protects whatever bytes land on it, on that system, under that configuration. The reader still has to trust your storage setup. A hash-chained, signed trail carries its integrity with it: the bundle proves itself anywhere, including in the reader's hands.

**Can I verify a trail without installing the producer's software?** With Vaara, yes. A standalone checker with no Vaara imports re-derives every verdict from the bundle and the public key, and public conformance vectors let you check the checker. Trust models per verifier are in [verifying-evidence.md](verifying-evidence.md).

**Is a tamper-evident trail required by the EU AI Act?** No. Article 12 requires automatic logging and Articles 19 and 26(6) require retention of at least six months; none of it mandates cryptographic integrity. The precise reading is in [eu-ai-act-article-12.md](eu-ai-act-article-12.md). Tamper-evidence is what makes the retained record persuasive under challenge.

**What gets recorded for a blocked action?** The proposed call, the risk score, the decision, and the reason, in the same chain as allowed actions. Blocks are evidence that the control operated, which is usually what an auditor asks about first.
