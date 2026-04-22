# Action governance is not model governance

Most AI governance tools score models. They ask: is the model biased?
Is it drifting? Is it producing hallucinations? They run at evaluation
time, at deployment time, at audit time, but not at the moment a
model-derived agent picks up a tool and does something.

That moment is where the risk actually lives.

`read_data` is benign. `export_data` is benign. `delete_data` is benign.
`read_data` → `export_data` → `delete_data` is a data exfiltration
pattern, and every existing model-governance dashboard will tell you
the model is fine.

## The category gap

Model governance is a crowded market. Arize, Weights & Biases, Fiddler,
Datadog AI, NeMo Guardrails, Lakera, and dozens of other vendors score the model
itself. They answer: is this model safe to run?

Action governance answers a different question: is this action safe to
execute? And the answer depends on who's acting, what came before, what
the blast radius is, whether it's reversible, and what regulatory
domain applies.

That's the gap Vaara fills.

## Why the EU AI Act forces the issue

Article 14 requires human oversight of high-risk AI systems. Article 12
requires event logging. Article 26 puts the responsibility on the
deployer, not just the model vendor. Together they mean: if you're
running an agent in production in the EU, you need a runtime gate.
not a deployment-time scorecard.

Most high-risk provisions apply from 2 August 2026. Enforcement is
not theoretical. Fines reach up to €35M or 7% of global turnover for
the most serious tier, and up to €15M or 3% for high-risk compliance
failures under Articles 9–15.

## What the Vaara primitive looks like

Every agent action hits one function:

```python
result = pipeline.intercept(
    agent_id="agent-007",
    tool_name="tx.transfer",
    parameters={"to": "0xabc", "amount": 1000},
)

if result.allowed:
    execute(...)
    pipeline.report_outcome(result.action_id, outcome_severity=0.0)
else:
    # result.decision is "deny" or "escalate"
    # result.risk_score is a point estimate
    # result.risk_interval is a conformal lower / upper bound
    handle_intervention(result)
```

That is the entire API surface for the runtime side. The policy backend
underneath learns online from outcomes, adapts weights across five
expert signals, and writes a hash-chained audit trail of every
classification, score, and decision.

It ships as an imported library, not as a service or dashboard or
SaaS subscription.

## Why library, not SaaS

Runtime gates can't sit behind a network call. If the only thing
stopping a bad action is a round trip to someone else's service, the
action either happens before the gate answers, or the whole agent
stalls under latency.

Vaara runs in-process. The same Python process that executes the tool
calls the gate first. Zero runtime dependencies: no Redis, no
vector DB, no remote scorer. The audit trail is local and portable,
export it, sign it, send it to your regulator, cross-check it against
your chain-of-custody.

## What just shipped

Vaara 0.3.2 is live on PyPI:

```
pip install vaara
```

The three framework integrations (LangChain/LangGraph, CrewAI, and
OpenAI Agents SDK) are included. The action taxonomy ships with EU AI
Act, GDPR, DORA, NIS2, MiFID II, HIPAA, SOC 2, and product-liability
regulatory-domain tags. The
audit trail is hash-chained by default and has an optional SQLite
persistence backend.

If you're running agents in production, especially in finance,
healthcare, infrastructure, or anywhere a regulator will eventually
ask "what stopped this?", it's five minutes to find out whether the
abstraction fits.

## Links

- PyPI: https://pypi.org/project/vaara/
- LangGraph integration walkthrough: `docs/blog/01_langgraph_article_14_runtime_gate.md`
- EU AI Act text: https://artificialintelligenceact.eu/
