# Comparison with adjacent tools

This doc compares Vaara against the open-source tools most often
named in the same breath: **NVIDIA NeMo Guardrails**, **Guardrails AI**,
**OpenAI Guardrails** (for Agents SDK), **LangChain callback handlers**,
and the **OWASP LLM Top 10** threat taxonomy.

No benchmark numbers are cited for the other tools here. Each one
solves a different problem than Vaara, so a head-to-head TPR/FPR on
Vaara's adversarial corpus would score them near zero. Not because
they are bad tools, but because they are not scoring tool calls at
runtime. That number would be unfair and not informative.

What follows is a capability matrix plus one-paragraph summaries of
what each tool actually does. If you are an AI Act Article 14
deployer comparing options, read the matrix. If you want the longer
prose, read the sections below it.

## Capability matrix

| Concern                                          | Vaara | NeMo Guardrails | Guardrails AI | OpenAI Guardrails | LangChain callbacks | OWASP LLM Top 10 |
| ------------------------------------------------ | :---: | :-------------: | :-----------: | :---------------: | :-----------------: | :--------------: |
| Validates tool-call **arguments** at runtime     |   ✓   |        ✗        |       ✗       |         ✗         |    observes only    |   not software   |
| Probabilistic / conformal risk scoring per call  |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |
| Detects temporal **sequence** patterns           |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |
| Hash-chained, regulator-exportable audit trail   |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |
| EU AI Act Art. 12 / 14 / 26 evidence mapping     |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |
| Validates LLM *output text* (PII, toxicity, etc) |   ✗   |        ✓        |       ✓       |         ✓         |          ✗          |   advisory only  |
| Validates LLM *input prompt* (jailbreak etc)     |   ✗   |        ✓        |       ✓       |         ✓         |          ✗          |   advisory only  |
| Structured-output validation (schema / regex)    | partial|        ✓        |       ✓       |         ✓         |          ✗          |        ✗         |
| Self-hostable Python library (no SaaS required)  |   ✓   |        ✓        |       ✓       |         ✓         |          ✓          |     document     |
| Apache-2.0                                       |   ✓   |     Apache-2.0  |     Apache-2.0|        MIT        |        MIT          |      CC-BY       |

Reading the matrix: Vaara and the output-validation tools are
complementary, not competitive. A real deployment uses output
validation **and** tool-call governance. Vaara does not validate LLM
text output, so use Guardrails AI or NeMo for that. NeMo and Guardrails
AI do not validate tool-call arguments at runtime, so use Vaara for that.

## One paragraph each

**NVIDIA NeMo Guardrails.** Programmable rails for LLM input and
output, plus dialog flows defined in Colang. The project's docs are
explicit that input and output rails do not see tool-call arguments:
when the LLM emits a tool call, the arguments are in the `tool_calls`
field and are not passed through the rails. Output rails can validate
the final user-facing response, but not the arguments to a
`tx.transfer`. Good at content moderation, topical rails, and jailbreak
defence at the prompt level. Not a runtime gate on what an agent
actually executes.

**Guardrails AI.** Output validators built as composable pipelines of
type checks, regex matches, PII redaction, and LLM-based judgements.
Validators are installed from the Guardrails Hub. Excellent at
structured-output enforcement. If you need "this response must be
valid JSON with these fields and no PII in the name field," Guardrails
AI is the right choice. Not a runtime risk scorer on tool invocations.

**OpenAI Guardrails (for the OpenAI Agents SDK).** MIT-licensed
input/output validation with a tripwire that halts agent execution on
violation. Tool guardrails run before and after each function-tool
invocation. The checks are deterministic (schema, regex, policy) and
the framework is tightly coupled to the OpenAI Agents SDK runtime.
Useful if you're all-in on that stack, less useful outside it. No
probabilistic scoring, no conformal intervals, no regulator-facing
audit artefact.

**LangChain callback handlers.** Hooks that fire on LLM start/end,
chain start/end, and tool start/end. Useful for observability and
logging. They do not decide. The agent still executes whatever the
LLM returned. You can build a policy layer on top of the callbacks,
but then you are re-building a runtime gate and you might as well
use Vaara, which is that layer.

**OWASP LLM Top 10.** A threat taxonomy (prompt injection, insecure
output handling, training data poisoning, model denial-of-service, etc.).
Extremely useful as a security-review checklist and as a shared
vocabulary. Not software, so there is nothing to install. Vaara's
signals and sequence patterns are informed by this taxonomy, but the
taxonomy itself does not do runtime enforcement.

## Where Vaara fits

Vaara is the gate between an AI agent's *decision* to take an action
and the actual *execution* of that action. It classifies the action,
scores the risk, decides allow/escalate/deny, and writes a
hash-chained audit record suitable for Article 14 oversight evidence.
It is framework-agnostic: LangChain, LangGraph, any MCP-compatible
runtime, or a custom loop.

The three things Vaara does that the tools above do not:

1. Look at the **arguments** of the tool call, not just the LLM text.
2. Score the tool call probabilistically against an **adaptive**
   model (MWU + conformal prediction + taxonomy base rate).
3. Produce **regulator-ready** evidence: cryptographic audit chain,
   signal breakdown per decision, conformity report.

The three things Vaara does not do that the tools above handle well:

1. LLM output validation (PII, toxicity, schema).
2. LLM input guardrails (jailbreak detection, topical rails).
3. Constrained decoding and structured output generation.

If you are building an agent that writes to user-visible text **and**
executes tools, you want both Vaara and one of the output-validation
tools wired in. They run in different places in the stack.

## Numbers we publish

See `bench/README.md` for the numbers on Vaara itself: per-intercept
latency (sub-200 μs p99), throughput (~7-8k ops/sec single-threaded),
and scorer detection on a 77-trace **synthetic** labelled corpus
(100% soft TPR, 0% hard FPR). Vaara has no production deployments at
the time of writing, so these are benchmark numbers, not traffic
numbers. We have not found published per-intercept latency numbers
for any of the adjacent tools. If you are evaluating a runtime
governance layer for a low-latency use case, that is a question worth
asking of anything you shortlist.
