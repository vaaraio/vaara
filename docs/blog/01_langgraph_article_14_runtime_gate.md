# An EU AI Act Article 14 runtime gate for LangGraph in five minutes

Article 14 of the EU AI Act says high-risk AI systems must support
"effective oversight by natural persons." For an agent framework that
means: before an action ships, something outside the model must be able
to see it and score it, then stop it or route
it to a human.

LangGraph makes the observation part easy. It does not give you the
scoring or the stop. That is the piece you import.

This post wires Vaara's runtime gate into a LangGraph agent in one file.
Zero runtime dependencies. No retraining. No threshold tuning.

## What a runtime gate actually does

An agent produces a proposed action. Before the action executes, a gate:

1. Classifies it (category, reversibility, blast radius, regulatory domain).
2. Scores it using a confidence interval rather than a point estimate.
3. Returns one of three decisions: `allow`, `escalate`, or `deny`.
4. Writes the classification, score, and decision to a hash-chained
   audit trail so a regulator can reconstruct the decision path later.

That last part is the Article-14 byproduct: the oversight record exists
whether a human reviews it or not, and its integrity is verifiable.

## Install

```
pip install vaara
```

Python 3.10+. Zero runtime dependencies.

## Wire it into LangGraph

LangChain's `BaseCallbackHandler` fires an event on every tool call.
Vaara plugs straight into it:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from vaara.pipeline import InterceptionPipeline
from vaara.integrations.langchain import (
    VaaraCallbackHandler,
    ToolExecutionBlocked,
    ToolExecutionEscalated,
)


@tool
def send_funds(to: str, amount: float) -> str:
    """Send funds to an address."""
    return f"sent {amount} to {to}"


pipeline = InterceptionPipeline()
handler = VaaraCallbackHandler(
    pipeline,
    agent_id="treasury-bot",
    block_on_escalate=False,  # escalated calls still execute; event is logged
)

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[send_funds])

try:
    result = agent.invoke(
        {"messages": [("user", "send 100 USD to 0xabc")]},
        config={"callbacks": [handler]},
    )
except ToolExecutionBlocked as e:
    print(f"Denied: {e.tool_name} - {e.reason} (risk {e.risk_score:.2f})")
except ToolExecutionEscalated as e:
    print(f"Escalated for review: {e.tool_name} - {e.reason}")
```

On every tool call the handler:

- records an ACTION_REQUESTED event in the audit trail,
- scores risk with conformal prediction (returns a lower and upper
  bound, not just a point estimate),
- records the RISK_SCORED event,
- decides `allow`, `escalate`, or `deny`,
- raises `ToolExecutionBlocked` if the decision is deny.

When the tool finishes, the handler reports the outcome back to the
scorer so it adapts online without a retraining loop or manual labels.

## The audit trail

```python
from vaara.audit.trail import EventType

# Every allow/escalate decision
decisions = pipeline.trail.get_records_by_type(EventType.DECISION_MADE)
# Everything that was actually blocked
blocks = pipeline.trail.get_records_by_type(EventType.ACTION_BLOCKED)

for r in decisions + blocks:
    print(r.narrative)

# Verify integrity: hash chain must be intact
assert pipeline.trail.chain_intact  # True if the chain is unbroken
```

Every event carries the SHA-256 hash of the previous record. Tamper
with any entry and `verify_chain()` returns false. This is the piece a
regulator or internal auditor actually looks at under Article 14.

## Why conformal prediction, not a single score

A point estimate of "risk 0.73" is not an oversight signal. It is a
claim. A conformal interval of `[0.41, 0.92]` tells a human reviewer:
"the model isn't sure, escalate." Article 14 wants the reviewer to be
able to intervene meaningfully, which is hard to do on top of a single
number. The interval does the heavy lifting.

The scorer combines five expert signals via multiplicative weight
update. Weights adapt online from outcome feedback. There is no
retraining loop and no threshold file to manage.

## Scope and extensions

- `VaaraCallbackHandler` is the LangChain-family integration. The
  same `InterceptionPipeline` can back CrewAI
  (`vaara.integrations.crewai.VaaraCrewGovernance`) and OpenAI Agents
  SDK (`vaara.integrations.openai_agents.VaaraToolGuardrail`).
- `block_on_escalate=True` converts escalations into hard blocks, which suits
  in narrow, regulated flows (financial transfers, infrastructure writes).
- The action taxonomy ships with EU AI Act, GDPR, DORA, NIS2, MiFID II,
  HIPAA, SOC 2, and product-liability regulatory-domain tags on each
  action type. Compliance-engine reports fall out of it.

## Where to go next

- PyPI: https://pypi.org/project/vaara/
- EU AI Act Article 14 text: https://artificialintelligenceact.eu/article/14/
- LangChain callback docs: https://python.langchain.com/docs/modules/callbacks/

If you run an agent in production in the EU, especially in a high-risk
category, the oversight question is not optional. The engineering
question is just whether you write it yourself or import it.
