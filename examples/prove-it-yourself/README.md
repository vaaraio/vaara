# Prove what your AI agent actually did

The short version: run one Python file and you get a signed, hash-chained record of what an AI agent did, which you then verify yourself, offline, without trusting the machine that produced it. Change a single byte of the record and the check fails. That is the difference between a log and evidence.

```bash
pip install 'vaara[export]'
python prove_it.py
```

## What it shows

An agent proposes four tool calls. Vaara scores and decides each one before it runs:

```
   run  ALLOW    risk=0.113  read a project file
   run  ALLOW    risk=0.111  search the support knowledge base
 BLOCK  ESCALATE risk=0.236  run a destructive shell command
 BLOCK  DENY     risk=0.173  fetch the cloud instance-metadata endpoint (classic SSRF)
```

Every decision, the two that ran and the two that were blocked, is written into one hash-chained record. The run then does the part that matters:

```
 verify_signed(evidence.zip)           ->  ok=True
 verify_signed(evidence_tampered.zip)  ->  ok=False
   caught: trail.jsonl SHA-256 does not match manifest.trail_sha256
```

The first bundle verifies from its own bytes and a public key. The second is the same bundle with one recorded action edited after the fact, and the check catches it. No network, no access to the machine, no taking anyone's word for it.

## Why this is not just logging

A log answers "what does the system say happened." Evidence answers "what can a party who does not trust the producer confirm happened." Datadog and Splunk record activity, but you have to trust that the records were not edited, and they do not capture the per-agent decision, its reasoning, or the integrity of the sequence. Vaara's record is content-addressed and fails closed on authenticity: a dropped or altered entry is a provable gap, not a silent one.

This matters the moment someone who does not trust you asks you to prove what your agent did: a regulator under EU AI Act Article 12 record-keeping, an auditor, or a customer after an incident. Your own logs will not settle it, because you could have edited them. This record settles it.

## Questions people ask

**How do I prove what an AI agent actually did?** Record each action and its decision into a signed, hash-chained trail at the point the action happens, then hand that trail to whoever needs to check it. They verify the signature and re-derive the chain from the bytes. This example is a working end to end instance of that.

**Can the person checking it trust the result without trusting me?** Yes, that is the design goal. Verification needs only the bundle and a public key. Vaara also ships a standalone checker that imports no Vaara code, so an independent party reproduces every verdict. See [docs/verifying-evidence.md](../../docs/verifying-evidence.md).

**What happens if a record is tampered with?** The verification fails and names the reason, as the tampered run above shows. The hash chain links each record to the previous one, so an edit anywhere breaks the chain.

**Does this need special hardware?** No. It recomputes verification from the recorded bytes. When a TPM 2.0 or confidential-VM root is present, Vaara can bind the record to it as an optional upgrade, but nothing here requires it.

## Where next

- The one-line adoption path, `@vaara.govern`, is in [examples/quickstart.py](../quickstart.py).
- Putting Vaara in front of a real MCP server (GitHub, filesystem, Slack, Postgres, Google Workspace) is in [examples/policies/mcp-starters/](../policies/mcp-starters/README.md).
- The trust model for every verifier is in [docs/verifying-evidence.md](../../docs/verifying-evidence.md).
- The full logs-versus-evidence argument is in [docs/logs-vs-evidence.md](../../docs/logs-vs-evidence.md).
