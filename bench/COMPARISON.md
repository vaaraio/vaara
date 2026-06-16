# Comparison with adjacent tools

This doc compares Vaara against the open-source tools most often
named in the same breath. Two clusters: LLM-text rails and output
validators on one side (**NVIDIA NeMo Guardrails**, **Guardrails AI**,
**OpenAI Guardrails** for Agents SDK, **LangChain callback handlers**,
and the **OWASP LLM Top 10** threat taxonomy), and agent governance
plus attestation tools on the other (**Glacis Python SDK** and
**Microsoft Agent Governance Toolkit**).

No benchmark numbers are cited for the other tools here. Each one
solves a different problem than Vaara, so a head-to-head TPR/FPR on
Vaara's adversarial corpus would score them near zero. Not because
they are bad tools, but because they are not scoring tool calls at
runtime. That number would be unfair and not informative.

What follows is a capability matrix plus one-paragraph summaries of
what each tool actually does. If you are an AI Act Article 14
deployer comparing options, read the matrix. If you want the longer
prose, read the sections below it.

## Capability matrix (as of 2026-05-16)

| Concern                                          | Vaara | NeMo Guardrails | Guardrails AI | OpenAI Guardrails | LangChain callbacks | OWASP LLM Top 10 | Glacis Python SDK | MS Agent Governance Toolkit |
| ------------------------------------------------ | :---: | :-------------: | :-----------: | :---------------: | :-----------------: | :--------------: | :---------------: | :-------------------------: |
| Validates tool-call **arguments** at runtime     |   ✓   |        ✗        |       ✗       |         ✗         |    observes only    |   not software   |         ✗         |              ✓              |
| Probabilistic / conformal risk scoring per call  |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              |
| Detects temporal **sequence** patterns           |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              |
| Hash-chained, regulator-exportable audit trail   |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |  partial (Merkle) |      partial (logging)      |
| EU AI Act Art. 12 / 14 / 26 evidence mapping     |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              |
| OVERT 1.0 Base Envelope emission (RFC 8949 CBOR) |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              |
| RFC 6962 Merkle inclusion proof integration      |  ext. IAP  |     ✗      |       ✗       |         ✗         |          ✗          |        ✗         |    ✓ (hosted)     |              ✗              |
| Validates LLM *output text* (PII, toxicity, etc) |   ✗   |        ✓        |       ✓       |         ✓         |          ✗          |   advisory only  |         ✗         |              ✗              |
| Validates LLM *input prompt* (jailbreak etc)     |   ✗   |        ✓        |       ✓       |         ✓         |          ✗          |   advisory only  |         ✗         |              ✗              |
| Structured-output validation (schema / regex)    | partial|        ✓        |       ✓       |         ✓         |          ✗          |        ✗         |         ✗         |          partial            |
| Zero-trust agent identity primitives             |   ✗   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✓              |
| Capability-based access control                  | policy schema |  ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✓              |
| Execution sandboxing                             |   ✗   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✓              |
| Multi-language SDKs                              | Python only |     N/A    |   Python      |  Python (Agents)  |   Python / JS       |      N/A         |    Python only    |              ✓              |
| Self-hostable Python library (no SaaS required)  |   ✓   |        ✓        |       ✓       |         ✓         |          ✓          |     document     |         ✓         |              ✓              |
| License                                          | AGPL-3.0 |   Apache-2.0 |   Apache-2.0 |        MIT        |        MIT          |      CC-BY       |    Apache-2.0     |             MIT             |

Reading the matrix: Vaara and the other tools are complementary, not
competitive. Different cells of the matrix. Different parts of the
stack. A real production agent deployment uses several of these at
once. Vaara owns the runtime risk-scoring + Article 14 evidence +
OVERT 1.0 attestation slice. NeMo and Guardrails AI cover the LLM
text-rail slice. Microsoft AGT covers the agent identity, capability,
and sandboxing slice. Glacis SDK is a client to Glacis's hosted
attestation service. Vaara does not validate LLM text output, so use
Guardrails AI or NeMo for that. Vaara does not provide zero-trust
agent identity, so use Microsoft AGT for that. The text-rail tools do
not validate tool-call arguments at runtime, so use Vaara for that.

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

**Glacis Python SDK.** Apache-2.0 client library for Glacis
Technologies' hosted attestation service, using RFC 8785 canonical
JSON, SHA-256 hashing, Ed25519 signatures, and RFC 6962 Merkle
inclusion proofs delivered in-line by the hosted service. Glacis
Technologies also authored OVERT 1.0, the open standard for
runtime trust in AI systems, published at overt.is in March 2026.
Either tool can be used depending on whether you need a
Glacis-hosted-service client or an OVERT 1.0 Base Envelope emitter
in your runtime.

**Microsoft Agent Governance Toolkit.** MIT-licensed toolkit for
agent identity, capability-based access control, execution sandboxing,
and reliability engineering. The toolkit frames its surface around
the OWASP Agentic Top 10 and zero-trust principles, with multi-language
SDKs for deployers running heterogeneous agent stacks. Where Vaara
provides runtime risk scoring and Article 14 audit evidence, AGT
provides agent identity primitives and the sandboxing layer that
isolates agent execution from the host environment. The two tools
cover different layers of the same governance stack. The
`GenAI-Gurus/awesome-eu-ai-act` curator places Vaara and AGT side
by side in the AI Agent Governance section for exactly this reason:
deployers running production agents typically want both wired in.

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

The things Vaara does not do that the tools above handle well:

1. LLM output validation, PII redaction, toxicity filtering (NeMo,
   Guardrails AI, OpenAI Guardrails).
2. LLM input guardrails, jailbreak detection, topical rails (same).
3. Constrained decoding and structured output generation (same).
4. Zero-trust agent identity primitives and capability-based access
   control as first-class types (Microsoft Agent Governance Toolkit).
5. Execution sandboxing as a built-in primitive (Microsoft AGT).
6. Hosted Merkle-inclusion-proof attestation as a managed service
   (Glacis Python SDK).

If you are building an agent that writes to user-visible text **and**
executes tools, you want Vaara plus one of the output-validation
tools wired in. If you are running agents in production, you want
Vaara plus Microsoft AGT for the identity, capability, and sandboxing
layer Vaara does not cover. They run in different places in the
stack and the matrix above shows where each tool lives.

## Sources

Each row in the matrix is grounded in publicly available project documentation
or source code. Verified as of 2026-05-16.

- **Vaara**: this repository at [github.com/vaaraio/vaara](https://github.com/vaaraio/vaara).
- **NVIDIA NeMo Guardrails**: [github.com/NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails).
- **Guardrails AI**: [github.com/guardrails-ai/guardrails](https://github.com/guardrails-ai/guardrails).
- **OpenAI Guardrails** (for OpenAI Agents SDK): [openai.github.io/openai-agents-python/guardrails/](https://openai.github.io/openai-agents-python/guardrails/).
- **LangChain callback handlers**: [python.langchain.com/docs/concepts/callbacks/](https://python.langchain.com/docs/concepts/callbacks/).
- **OWASP LLM Top 10**: [genai.owasp.org/llm-top-10/](https://genai.owasp.org/llm-top-10/).
- **Glacis Python SDK**: [github.com/Glacis-io/glacis-python](https://github.com/Glacis-io/glacis-python). Capabilities recorded by source-read of the repository on 2026-05-16.
- **Microsoft Agent Governance Toolkit**: [github.com/microsoft/agent-governance-toolkit](https://github.com/microsoft/agent-governance-toolkit).

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
