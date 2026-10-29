# Architecture

How Vaara processes a tool call, how it scores, and how the audit trail is anchored in time. The formal guarantees (MWU regret bound, conformal coverage, security properties) are in [formal_specification.md](formal_specification.md); the benchmark numbers are under [bench/](../bench/).

## How it works

Every tool call an agent makes passes through Vaara before it runs:

1. **Intercept.** Vaara catches the call (`fs.write_file`, `tx.transfer`, an MCP `tools/call`, and so on) through your framework's own hook, or transparently as an MCP proxy in front of an upstream server.
2. **Score and decide.** Each call gets a risk score and an allow / block / escalate decision against your policy.
3. **Record.** The call, the score, the decision, and the real-world outcome are written to a hash-chained audit trail. An outside auditor can verify the chain is intact without trusting your stack or your word.

The scoring blends five expert signals and keeps adapting as outcomes come back, and each risk score carries a confidence interval with a coverage guarantee that holds regardless of the input distribution. Those are the properties an auditor can check independently; the math is in [formal_specification.md](formal_specification.md) and a plain-language version for compliance reviewers and legal counsel is in [conformal-prediction.md](conformal-prediction.md).

## One interception layer per call

The interception points in step 1 are alternatives. Pick the one that matches your client.

| Client | Use | Why |
|---|---|---|
| Has a hook Vaara can register in (Claude Code, and any framework with a pre-call hook) | The hook | It sees every call the client makes, MCP calls included, and it does not own any other process |
| Has no hook (a bare MCP client, a custom runtime) | The MCP proxy | Sitting in the stdio pipe is the only interception point available |

Registering both puts two decision points on the same action. Each MCP call is then scored and decided twice and written to the trail twice, under two different names: the client's namespaced name from the hook (`mcp__server__tool`) and the bare wire name from the proxy (`tool`). The chain stays intact and every verifier still passes, so nothing reports an error. The trail simply counts one action as two, which shifts any figure derived from event counts.

The proxy detects this at startup and names the layer it found:

```
vaara-mcp-proxy: a second Vaara governance layer is already in front of this proxy.
  found: ~/.claude/settings.json: vaara hook pre-tool-use
```

Control it with `--stacked warn` (default), `--stacked fail` to refuse to start, or `--stacked ignore`. Detection reports and never changes what is recorded, because the proxy owns the upstream server's process and anything that can abort there takes the upstream down with it.

Process lifetime differs between the two. Under stdio, the proxy starts the upstream MCP server as a child process, since owning the pipe is what makes the traffic readable. The upstream's lifetime is therefore bound to the proxy's, and a proxy that exits takes the MCP server down with it. A hook has no such relationship. On clients that offer one, the hook is the smaller commitment.

## Delegated grants

An action can run under a capability grant (`intercept(capabilities=...)`) and name the action that delegated to it (`parent_action_id`). The pipeline then checks that the child's grant allows no more than the parent's, and denies a broadening grant whatever its risk score.

The grant is written into the action's `action_requested` record. The pipeline keeps recent grants in memory, and when the parent is not there (evicted, or recorded by another process such as an earlier hook call) it reads the parent's grant from the trail, within the same tenant. When the parent cannot be found:

| Parent | Decision |
|---|---|
| Not on the trail in this tenant (an invented id, another tenant's action) | Denied, `violation_type` `privilege_attenuation`, `unavailable_input` `parent_grant` |
| On the trail, grant unreadable (cut at the size cap, or does not parse) | Denied the same way |
| On the trail, ran under no grant | Not denied on this ground; there is no bound to exceed |

`InterceptionPipeline(unknown_parent="allow")` lets the first case through to the scorer, for a deployment whose parents run behind another gate. The setting is in the decision state the record digests.

Every decision on an action that carries a grant and names a parent says what the check found, as `attenuation` on the decision record: `status` `held`, `violated` or `unverified`, with `reason` when it did not hold, and `source` (`memory` or `trail`) when the parent grant was found. A check that could not run is on the record, not silent.

The pipeline check depends on the trail it can read. The credential gateway does not: it verifies each call against the signed grant the call presents, so a call outside its grant is refused there whatever the pipeline decided.

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

The anchor also folds into the one-command regulator package: `vaara trail export-article12 --anchor-tsa https://freetsa.org/tsr` writes the timestamp beside the signed trail as Article 19 existence-in-time evidence, and `vaara trail verify-anchor --zip <package>.zip` checks it offline; add `--tsa-cert` or `--tsa-issuer` to hold it to the authority you trust, otherwise it reports the time as unpinned.

The same command folds cross-org handoffs and confidential-VM enforcement evidence into the package as verified sidecars (`--handoffs ./handoffs --enforcements ./enforced`); an attachment that does not verify fails the export, so the package never ships evidence it cannot back. It is a more complete pack, not a certificate. See [verifying-evidence.md](verifying-evidence.md).
