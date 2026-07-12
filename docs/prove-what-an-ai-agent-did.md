# How to prove what an AI agent did

To prove what an AI agent did, you need a record of every action that was written at the moment the action happened, is cryptographically signed, chains each entry to the previous one by hash, and can be verified by the other party using only the record and a public key. A record with those four properties settles the question even when the person asking does not trust you. A record without them is a log, and a log only persuades people who already trust the machine that wrote it.

This page is the method. A working end to end run of it is [examples/prove-it-yourself/](../examples/prove-it-yourself/), and the argument for why logs are not enough is [logs-vs-evidence.md](logs-vs-evidence.md).

## Who asks, and what they will accept

The question "prove what your agent did" arrives from a few directions, and none of them accept your word for it:

- A regulator, under record-keeping obligations such as EU AI Act Article 12 ([Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)).
- An auditor assembling conformity or certification evidence.
- A customer or counterparty after an incident, deciding whether to escalate.
- A court or insurer in a liability dispute, where the standard is not "plausible" but "provable".

What they share is a trust model: the record must hold up on its own, because the producer is the party under scrutiny. Screenshots, dashboards, and exported log files all fail this test for the same reason. You could have edited them.

## The four properties a provable record needs

**Written at the decision point.** The record must be produced where the action is decided, not reconstructed afterwards from application logs. A record assembled after the incident is a narrative. For agents this means intercepting the tool call itself: the proposed action, the policy decision, and the outcome, captured before and after the call runs.

**Signed.** A signature binds the record to a key, so origin is checkable against the public half. Without it, anyone with file access can rewrite history and nothing detects it.

**Hash-chained.** Each entry carries a cryptographic hash of the previous entry. Editing, inserting, or deleting any record breaks every hash after it, so the whole tail fails verification. This is what turns "trust me, it's complete" into "check it yourself". A missing entry becomes a provable gap rather than a silent one.

**Independently verifiable.** The other party must be able to re-derive every verdict from the bytes and a public key, offline, ideally with a checker that is not your software. If verification requires your code or your infrastructure, the trust problem has just moved, not gone away.

## The verification step, concretely

Whoever receives the record runs three checks: the signature verifies against the published public key, each entry's hash chain re-derives from the entry bytes, and the manifest digests match the files in hand. Any failed check names what broke. With Vaara that is one command against an exported bundle:

```bash
vaara verify-bundle evidence-bundle.json
```

Vaara also ships a standalone checker that imports no Vaara code, so an auditor reproduces every verdict without running the producer's software. The trust model for each verifier, including what each one does and does not prove, is documented in [verifying-evidence.md](verifying-evidence.md).

## Doing this from one decorator

```python
import vaara

@vaara.govern
def transfer_funds(to: str, amount: float) -> str:
    ...
```

Every call to a governed function is decided against policy before the body runs, and the call, the decision, and the outcome land in the signed, hash-chained trail. The same interception is available for MCP servers and for LangChain, CrewAI, and OpenAI Agents SDK via adapters. It runs in your environment, with no SaaS and no telemetry, and the rule-based hot path adds 140 microseconds mean per call on a commodity CPU, reproducible with `make bench`.

## Questions

**Can I prove what an agent did from my existing logs?** Only to someone who trusts you. Ordinary logs are editable by the party that holds them, and they usually do not capture the per-action decision or the integrity of the sequence. See [logs-vs-evidence.md](logs-vs-evidence.md).

**Does proving what an agent did require special hardware?** No. Signature and hash-chain verification work from the recorded bytes alone. A TPM 2.0 or confidential-VM root, where present, adds a hardware-bound identity on top, but nothing depends on it.

**What if the agent's action was blocked, not executed?** Record it anyway. The blocked proposals are often the record that matters most after an incident, because they show the control operating. Vaara writes allow and block decisions into the same chain.

**Is this required by law?** The EU AI Act requires automatic event logging and retention for high-risk systems (Article 12, with retention of at least six months under Articles 19 and 26(6)). It does not mandate cryptographic tamper-evidence. Tamper-evidence is what makes the retained record hold up when someone challenges it. The precise reading is in [eu-ai-act-article-12.md](eu-ai-act-article-12.md).
