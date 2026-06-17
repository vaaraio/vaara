# Architecture

How Vaara processes a tool call, how it scores, and how the audit trail is anchored in time. The formal guarantees (MWU regret bound, conformal coverage, security properties) are in [formal_specification.md](formal_specification.md); the benchmark numbers are in the project [README](../README.md#how-it-scores) and under [bench/](../bench/).

## How it works

Every tool call an agent makes passes through Vaara before it runs:

1. **Intercept.** Vaara catches the call (`fs.write_file`, `tx.transfer`, an MCP `tools/call`, and so on) through your framework's own hook, or transparently as an MCP proxy in front of an upstream server.
2. **Score and decide.** Each call gets a risk score and an allow / block / escalate decision against your policy.
3. **Record.** The call, the score, the decision, and the real-world outcome are written to a hash-chained audit trail. An outside auditor can verify the chain is intact without trusting your stack or your word.

The scoring blends five expert signals and keeps adapting as outcomes come back, and each risk score carries a confidence interval with a coverage guarantee that holds regardless of the input distribution. Those are the properties an auditor can check independently; the math is in [formal_specification.md](formal_specification.md) and a plain-language version for compliance reviewers and legal counsel is in [conformal-prediction.md](conformal-prediction.md).

## External time anchor

The hash chain proves order and integrity but not *when* it existed: every timestamp comes from your own clock, so a compromised signing key could in principle be used to forge a backdated chain. Vaara can anchor the current chain head to an external RFC 3161 Time-Stamp Authority, the standard behind eIDAS qualified electronic timestamps. The authority signs the chain head and the time, so the chain's existence is provable against a clock you do not control. Verification is offline.

```bash
pip install 'vaara[timeanchor]'
```

```python
from vaara.audit.timeanchor import RFC3161TimeAnchorClient

# Periodically, or after a batch of high-risk actions:
trail.anchor_head(RFC3161TimeAnchorClient("https://freetsa.org/tsr"))
```

The anchor also folds into the one-command regulator package: `vaara trail export-article12 --anchor-tsa https://freetsa.org/tsr` writes the timestamp beside the signed trail as Article 19 existence-in-time evidence, and `vaara trail verify-anchor --zip <package>.zip` checks it offline.

The same command folds cross-org handoffs and confidential-VM enforcement evidence into the package as verified sidecars (`--handoffs ./handoffs --enforcements ./enforced`); an attachment that does not verify fails the export, so the package never ships evidence it cannot back. It is a more complete pack, not a certificate. See [verifying-evidence.md](verifying-evidence.md).
