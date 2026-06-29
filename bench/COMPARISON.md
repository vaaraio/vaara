# Comparison with adjacent tools

This doc compares Vaara against the open-source tools most often
named in the same breath. Two clusters: LLM-text rails and output
validators on one side (**NVIDIA NeMo Guardrails**, **Guardrails AI**,
**OpenAI Guardrails** for Agents SDK, **LangChain callback handlers**,
and the **OWASP LLM Top 10** threat taxonomy), and agent governance
plus attestation tools on the other (**Glacis Python SDK**,
**Microsoft Agent Governance Toolkit**, and **Apollo Watcher**).

No benchmark numbers are cited for the other tools here. Each one
solves a different problem than Vaara, so a head-to-head TPR/FPR on
Vaara's adversarial corpus would score them near zero. Not because
they are bad tools, but because they are not scoring tool calls at
runtime. That number would be unfair and not informative.

What follows is a capability matrix plus one-paragraph summaries of
what each tool actually does. If you are an AI Act Article 14
deployer comparing options, read the matrix. If you want the longer
prose, read the sections below it.

## Capability matrix (as of 2026-06-29, Vaara v1.20.0)

| Concern                                          | Vaara | NeMo Guardrails | Guardrails AI | OpenAI Guardrails | LangChain callbacks | OWASP LLM Top 10 | Glacis Python SDK | MS Agent Governance Toolkit | Apollo Watcher |
| ------------------------------------------------ | :---: | :-------------: | :-----------: | :---------------: | :-----------------: | :--------------: | :---------------: | :-------------------------: | :------------: |
| Validates tool-call **arguments** at runtime     |   ✓   |        ✗        |       ✗       |         ✗         |    observes only    |   not software   |         ✗         |              ✓              | partial (hook escalation) |
| Probabilistic / conformal risk scoring per call  |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              | ✗ (rubric grading) |
| Detects temporal **sequence** patterns           |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              | partial (session drift) |
| Hash-chained, regulator-exportable audit trail   |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |  partial (Merkle) | partial (Merkle, self-keyed)| ✗ |
| EU AI Act Art. 12 / 14 / 26 evidence mapping     |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |           partial           | ✗ |
| Independently-recomputable conformance corpus    |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              | ✗ |
| Spec in a standards process (IETF / MCP)         |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |     taxonomy     |  OVERT (self-pub) |              ✗              | ✗ |
| OVERT 1.0 Base Envelope emission (RFC 8949 CBOR) |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              | ✗ |
| RFC 6962 Merkle inclusion proof integration      |  ext. IAP  |     ✗      |       ✗       |         ✗         |          ✗          |        ✗         |    ✓ (hosted)     |              ✗              | ✗ |
| Validates LLM *output text* (PII, toxicity, etc) |   ✗   |        ✓        |       ✓       |         ✓         |          ✗          |   advisory only  |         ✗         |              ✗              | ✗ |
| Validates LLM *input prompt* (jailbreak etc)     |   ✗   |        ✓        |       ✓       |         ✓         |          ✗          |   advisory only  |         ✗         |              ✗              | ✗ |
| Structured-output validation (schema / regex)    | partial|        ✓        |       ✓       |         ✓         |          ✗          |        ✗         |         ✗         |          partial            | ✗ |
| Real-time agent-session behavioral oversight     |partial|        ✗        |       ✗       |         ✗         |          ✗          |   advisory only  |         ✗         |              ✗              | ✓ |
| Zero-trust agent identity primitives             |   ✗   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✓              | ✗ |
| Capability-based access control                  |   ✓   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✓              | ✗ |
| Attestation-bound credential gateway             | ✓ (opt-in) |   ✗      |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✗              | ✗ |
| Execution sandboxing                             |   ✗   |        ✗        |       ✗       |         ✗         |          ✗          |        ✗         |         ✗         |              ✓              | ✗ |
| Multi-language SDKs                              | Python + TS |   N/A    |   Python      |  Python (Agents)  |   Python / JS       |      N/A         |    Python only    |              ✓              | Claude Code hooks |
| Self-hostable (no SaaS / vendor cloud required)  |   ✓   |        ✓        |       ✓       |         ✓         |          ✓          |     document     |         ✓         |              ✓              | ✗ (monitors run on Apollo cloud) |
| License                                          | AGPL-3.0 |   Apache-2.0 |   Apache-2.0 |        MIT        |        MIT          |      CC-BY       |    Apache-2.0     |             MIT             | source-available (NOASSERTION) |

Reading the matrix: Vaara and the other tools are complementary, not
competitive. Different cells of the matrix. Different parts of the
stack. A real production agent deployment uses several of these at
once. Vaara owns the runtime risk-scoring + Article 14 evidence +
OVERT 1.0 attestation slice, and the part no one else does: a public
conformance corpus that an independent party recomputes offline with a
checker that imports none of Vaara's code. NeMo and Guardrails AI cover
the LLM text-rail slice. Microsoft AGT covers the agent identity,
capability, and sandboxing slice. Glacis SDK is a client to Glacis's
hosted attestation service. Vaara does not validate LLM text output, so
use Guardrails AI or NeMo for that. Vaara does not provide zero-trust
agent identity or execution sandboxing, so use Microsoft AGT for those.
The text-rail tools do not validate tool-call arguments at runtime, so
use Vaara for that. Apollo Watcher is the closest to Vaara's lane, both
sit at agent runtime, but Watcher grades a session's behavior on Apollo's
cloud while Vaara gates the individual call locally and emits a
recomputable record, so the two cover different jobs at the same layer.

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
cover different layers of the same governance stack: deployers running
production agents typically want both wired in. Recent versions add an audit layer with a Merkle audit chain, signed entries, and an EU AI Act compliance mapping. By the toolkit's own documentation that integrity is partial. One chain verifier is a stub that always returns true, audit entries are signed with symmetric HMAC keys an insider could forge, and the default retention falls short of Article 26(6). The audit is verified with Microsoft's own keys, with no public conformance corpus and no way for an outside party to recompute a record without trusting the vendor.

**Apollo Watcher.** A real-time oversight layer for coding agents from
Apollo Research, a Delaware public-benefit corporation with servers in
Western Europe. It installs hooks into a Claude Code session, captures
every tool call and permission decision, and grades them against
configurable rubrics to flag failure modes such as deception, instruction
drift, scope creep, and dangerous commands. A flagged action can be
escalated to a human through the blocking permission hook, and full
pre-execution blocking ("Active Control") is described as a roadmap goal.
The code is public on GitHub but carries no open-source license (the
GitHub API reports it as NOASSERTION), it is free during alpha, and the
monitoring runs on Apollo's cloud rather than your own machine, so agent
trajectories leave your control unless you opt out. Watcher and Vaara are
complementary: Watcher watches a session's behavior and grades it, Vaara
gates the individual call and writes a signed record you recompute and
have checked offline.

## Where Vaara fits

Vaara is the gate between an AI agent's *decision* to take an action
and the actual *execution* of that action. The entry point is one
decorator, `@vaara.govern`: every call to a governed function is
risk-scored and decided against your policy before the body runs, a
blocked call raises `vaara.Blocked`, and the decision, the call, and
the outcome land in a signed record anyone can verify offline. Vaara
classifies the action, scores the risk, decides allow/escalate/deny,
and writes a hash-chained audit record suitable for Article 14
oversight evidence. It is framework-agnostic: LangChain, LangGraph,
any MCP-compatible runtime, or a custom loop.

The four things Vaara does that the tools above do not:

1. Look at the **arguments** of the tool call, not just the LLM text.
2. Score the tool call probabilistically against an **adaptive**
   model (MWU + conformal prediction + taxonomy base rate).
3. Produce **regulator-ready** evidence: cryptographic audit chain,
   signal breakdown per decision, conformity report.
4. Ship a **public conformance corpus** with a standalone checker, so
   any third party recomputes every verdict offline, with no key, no
   access, and none of Vaara's code.

The things Vaara does not do that the tools above handle well:

1. LLM output validation, PII redaction, toxicity filtering (NeMo,
   Guardrails AI, OpenAI Guardrails).
2. LLM input guardrails, jailbreak detection, topical rails (same).
3. Constrained decoding and structured output generation (same).
4. Zero-trust agent identity primitives (Microsoft Agent Governance
   Toolkit).
5. Execution sandboxing as a built-in primitive (Microsoft AGT).
6. Hosted Merkle-inclusion-proof attestation as a managed service
   (Glacis Python SDK).
7. Real-time behavioral grading of an agent session for deception,
   instruction drift, and scope creep (Apollo Watcher).

Vaara does carry its own access-control layer: typed capability scopes
that bound what a governed call may do, plus an opt-in credential
gateway that refuses a tool call unless it carries a credential bound
to the runtime's attestation. That is access control as evidence, not
as a sandbox; AGT's identity and sandboxing primitives sit at a
different layer and the two compose.

If you are building an agent that writes to user-visible text **and**
executes tools, you want Vaara plus one of the output-validation
tools wired in. If you are running agents in production, you want
Vaara plus Microsoft AGT for the identity and sandboxing layer Vaara
does not cover. They run in different places in the
stack and the matrix above shows where each tool lives.

## Sources

Each row in the matrix is grounded in publicly available project documentation
or source code. Verified as of 2026-06-29.

- **Vaara**: this repository at [github.com/vaaraio/vaara](https://github.com/vaaraio/vaara).
  The `@vaara.govern` entry point (`src/vaara/govern.py`), the TypeScript
  client (`clients/ts/`), the conformance corpus and standalone checkers
  (`conformance/` and `tests/vectors/`), and the Internet-Draft
  `draft-sirkkavaara-vaara-receipt` (Independent Submission stream,
  Informational, under consideration) plus SEP-2828 (Server-Side Signed
  Execution Record) in the MCP process.
- **NVIDIA NeMo Guardrails**: [github.com/NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails).
- **Guardrails AI**: [github.com/guardrails-ai/guardrails](https://github.com/guardrails-ai/guardrails).
- **OpenAI Guardrails** (for OpenAI Agents SDK): [openai.github.io/openai-agents-python/guardrails/](https://openai.github.io/openai-agents-python/guardrails/).
- **LangChain callback handlers**: [python.langchain.com/docs/concepts/callbacks/](https://python.langchain.com/docs/concepts/callbacks/).
- **OWASP LLM Top 10**: [genai.owasp.org/llm-top-10/](https://genai.owasp.org/llm-top-10/).
- **Glacis Python SDK**: [github.com/Glacis-io/glacis-python](https://github.com/Glacis-io/glacis-python). Capabilities recorded by source-read of the repository on 2026-05-16.
- **Microsoft Agent Governance Toolkit**: [github.com/microsoft/agent-governance-toolkit](https://github.com/microsoft/agent-governance-toolkit). Audit-chain and EU AI Act mapping features verified by source-read on 2026-06-29.
- **Apollo Watcher**: [github.com/ApolloResearch/watcher](https://github.com/ApolloResearch/watcher) and the [product page](https://www.apolloresearch.ai/products/introducing-watcher-for-ai-oversight/). License reported by the GitHub API as NOASSERTION (no open-source license) on 2026-06-29; alpha status, cloud-hosted monitoring, and EU-based servers per the product page and docs.

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
