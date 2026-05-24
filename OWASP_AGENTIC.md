# Vaara mapping to OWASP Top 10 for Agentic Applications 2026

This document maps Vaara's runtime controls to the **OWASP Top 10 for
Agentic Applications 2026** (ASI01 through ASI10), published by the
OWASP GenAI Security Project, Agentic Security Initiative, December
2025, under Creative Commons CC BY-SA 4.0.

For each ASI risk, the mapping states whether Vaara satisfies the
mitigation surface today (✅), partially satisfies it (◐), or leaves
it as explicit gap-to-deployer or future work (◯). The mapping does
not establish legal compliance with any regulation. It records
technical correspondence so an enterprise reader can answer the
question "where does Vaara fit in the OWASP Agentic threat model"
without reading every section of `COMPLIANCE.md`.

The companion mapping to OVERT 1.0 Part 3 controls lives in
[`OVERT_CONTROLS.md`](OVERT_CONTROLS.md). The two documents address
overlapping concerns from different framings: OWASP names threat
classes, OVERT 1.0 Part 3 names enforcement controls.

## Source

- **OWASP Top 10 For Agentic Applications 2026** (Version 2026,
  December 2025), OWASP GenAI Security Project, Agentic Security
  Initiative. Project page:
  [genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/).
- Authors named in the Letter from The Agentic Top 10 Leaders: John
  Sotiropoulos (Chair), Keren Katz (Lead), Ron F. Del Rosario
  (ASI Co-lead).
- License CC BY-SA 4.0. Threat titles and the Appendix A cross-
  mapping matrix are referenced under that license with attribution.

## Status legend

- ✅ Vaara satisfies the mitigation in a tagged release today.
- ◐ Vaara partially satisfies it. Remainder is deployer-owned or
  future work.
- ◯ Out of Vaara's current surface, deployer-owned, or future work.

## Cross-mapping summary

| ASI risk | Vaara | Primary Vaara surfaces |
|---|---|---|
| ASI01 Agent Goal Hijack | ◐ | Intercept boundary, classifier, audit chain, OVERT envelope |
| ASI02 Tool Misuse and Exploitation | ✅ | Policy DSL, intercept boundary, audit chain, commit-prove receipts |
| ASI03 Identity and Privilege Abuse | ◐ | Caller identity in audit, taxonomy scopes, review queue |
| ASI04 Agentic Supply Chain Vulnerabilities | ◐ | SLSA provenance, Sigstore signing, ClusterFuzzLite, policy hash binding |
| ASI05 Unexpected Code Execution (RCE) | ◐ | Policy DSL allowlists, intercept-time approval gates |
| ASI06 Memory and Context Poisoning | ◯ | Adaptive scorer drift signal only |
| ASI07 Insecure Inter-Agent Communication | ◐ | OVERT signing per call, audit chain integrity |
| ASI08 Cascading Failures | ◐ | Hash-chained audit, circuit breakers, FACI drift signal |
| ASI09 Human-Agent Trust Exploitation | ◐ | Conformal interval, signed narratives, escalation queue |
| ASI10 Rogue Agents | ◐ | Hash-chained audit, per-agent identity, drift signals |

Vaara is strongest on **tool-call governance, runtime evidence
emission, and audit traceability** (ASI02, ASI05, ASI08, ASI10).
Vaara is weakest on **memory poisoning, transport-layer security,
and external supply chain provenance** (ASI06, parts of ASI04 and
ASI07).

## ASI01 Agent Goal Hijack ◐

Attackers manipulate the agent's objectives or decision pathways
through prompt-based manipulation, deceptive tool outputs, malicious
artefacts, forged agent-to-agent messages, or poisoned external data.

- ✅ Treat natural-language inputs as untrusted at the tool-call
  boundary: `InterceptionPipeline.intercept()` is the enforcement
  point (`src/vaara/pipeline.py`).
- ✅ Least privilege and human approval for high-impact actions:
  policy DSL + SQLite-backed review queue
  (`src/vaara/audit/review_queue.py`).
- ✅ Comprehensive logging and continuous monitoring with a
  behavioural baseline: hash-chained `AuditTrail` plus the adaptive
  scorer in `src/vaara/scorer/adaptive.py`.
- ◐ Intent capsule binding declared goal to each execution cycle in
  a signed envelope: the OVERT Base Envelope (v0.11.0) carries
  `encoder_binary_identity`, action class, and decision, but Vaara
  does not yet emit a separate intent capsule at task start.
- ◯ Sanitise connected data sources, prompt-carrier detection:
  deployer concern. Vaara intercepts tool calls, not RAG inputs.

**Deployer-owned:** input sanitisation upstream of the model, RAG
retrieval filtering, prompt-injection detection between the agent
and untrusted external content.

## ASI02 Tool Misuse and Exploitation ✅

Agents misapply legitimate tools due to prompt injection,
misalignment, or unsafe delegation, leading to exfiltration, tool
output manipulation, workflow hijacking, or denial-of-wallet. This
is Vaara's core wedge.

- ✅ Least Agency and Least Privilege for Tools (per-tool scopes,
  rate caps, egress allowlists): policy DSL (`src/vaara/policy/`).
- ✅ Action-Level Authentication and Approval, dry-run before
  high-impact actions: the Article 12 commit-prove receipt pair
  (shipped v0.10.0).
- ✅ Policy Enforcement Middleware (PEP/PDP that validates intent
  and arguments at runtime): Vaara is the PEP/PDP.
- ✅ Semantic and Identity Validation, fully qualified tool names
  with version pins: policy DSL pins tool identifiers, hash flows
  into the OVERT Base Envelope.
- ✅ Logging, Monitoring, and Drift Detection (immutable logs +
  anomalous-pattern monitoring): hash-chained `AuditTrail` + the
  adaptive scorer's FACI drift signal.
- ◐ Execution Sandboxes and Egress Controls: policy DSL declares
  destinations. Process-level sandboxing is deployer-owned.
- ◐ Adaptive Tool Budgeting (cost / rate / token ceilings): adaptive
  scorer tracks velocity, policy DSL expresses rate caps. Per-tool
  cost budgets not yet first-class.
- ◯ Just-in-Time and Ephemeral Access (short-lived credentials):
  deployer issues tokens.

**Deployer-owned:** process-level sandboxing, network egress
enforcement at host or container, credential issuance, upstream IAM.

## ASI03 Identity and Privilege Abuse ◐

Dynamic trust and delegation are exploited to escalate access by
manipulating delegation chains, role inheritance, control flows, and
agent context.

- ✅ Mandate Per-Action Authorization, central policy engine that
  re-verifies each privileged step: every `intercept()` call.
- ✅ Apply Human-in-the-Loop for Privilege Escalation: review queue
  routes `ESCALATE` verdicts, records `ESCALATION_RESOLVED` events
  with reviewer identity, timestamp, decision.
- ◐ Enforce Task-Scoped, Time-Bound Permissions: policy DSL declares
  per-tool scopes. Vaara does not issue or rotate the tokens.
- ◐ Isolate Agent Identities and Contexts: Vaara records active
  `agent_id` per call. Memory segmentation is a deployer
  architecture choice.
- ◐ Define Intent, bind tokens to signed intent: partial via OVERT
  Base Envelope. External OAuth token binding is deployer-owned.
- ◯ Evaluate Agentic Identity Management Platforms (Entra, Bedrock
  Agents, Agentforce): deployer concern.

**Deployer-owned:** identity provider, token issuance and rotation,
per-session memory segmentation, federated delegation chain
integrity.

## ASI04 Agentic Supply Chain Vulnerabilities ◐

Agents, tools, and related artefacts from third parties may be
malicious, compromised, or tampered with in transit. Splits cleanly
into Vaara's own supply chain (strong) and the external agent / MCP
supply chain the deployer connects through Vaara (mostly
observational).

- ✅ Provenance and SBOMs for Vaara itself: SLSA Build Level 3
  provenance on every release (`.github/workflows/release.yml`,
  v0.26.0), Sigstore-signed PyPI release with PEP 740 attestations
  (v0.4.3), npm provenance on `@vaara/client`.
- ✅ Pinning by content hash and commit ID: PyPI Trusted Publishing
  binds the release to workflow + commit + builder identity.
- ◐ Dependency gatekeeping (allowlist and pin upstream tools):
  policy DSL pins tool identifiers, hash flows into
  `encoder_binary_identity`. External MCP server / agent provenance
  is not yet verified by Vaara.
- ◐ Continuous validation and monitoring (recheck signatures at
  runtime): OVERT envelope re-verifies policy hash + Vaara binary
  identity per call. External tool re-check not yet shipped.
- ◯ AIBOM emission for the agent ecosystem Vaara protects: future
  work. The auditor-facing evidence export (v0.16.0) is the
  candidate surface when CycloneDX-AI / SPDX 3.0 schemas stabilise.
- ◐ Supply chain kill switch (instant disable across deployments):
  hot policy reload (v0.13.0) lets an operator revoke a tool's
  scope without restarting the pipeline.

**Deployer-owned:** verifying upstream MCP servers, third-party
tool providers, external agent registries connected through Vaara.

## ASI05 Unexpected Code Execution (RCE) ◐

Agentic systems generate and execute code at runtime. Attackers
exploit code generation or embedded tool access to escalate into
RCE, local misuse, or sandbox escape.

- ✅ Separate code generation from execution with validation gates
  (Architecture and design): the intercept boundary is exactly this
  gate. Code is generated by the model, execution requires a tool
  call, the tool call is audited and gated.
- ✅ Access control and approvals, human approval for elevated
  runs, allowlist for auto-execution: policy DSL + review queue.
- ✅ Code analysis and monitoring, log all generation and runs: the
  hash-chained audit trail with conformal risk interval, plus the
  adversarial classifier (v0.5.0+) scoring shell-like and
  exfiltration-like patterns.
- ◐ Ban eval in production agents, taint tracking: policy DSL can
  deny `eval`-shaped sinks. Taint tracking across artefacts requires
  deployer instrumentation.
- ◯ Execution environment security (sandboxed containers, syscall
  filters): host and container concern.
- ◯ Prevent direct agent-to-production, vibe coding pre-production
  checks: deployer architecture choice.

**Deployer-owned:** the execution sandbox (container isolation,
syscall filtering, network egress), pre-production gates for vibe
coding artefacts, static analysis on generated code.

## ASI06 Memory and Context Poisoning ◯

Adversaries corrupt or seed an agent's memory or retrievable context
with malicious or misleading data, causing future reasoning,
planning, or tool use to become biased, unsafe, or aid exfiltration.

Vaara is a tool-call governor, not a memory layer. Memory poisoning
is largely out of scope, with one cross-cut:

- ◐ Resilience and verification, snapshots, version control on
  memory stores: the hash-chained audit is itself a rollback-safe
  record of every action the agent has taken, which is adjacent to
  (but not the same as) a memory snapshot. A deployer can correlate
  audit drift against memory contents.
- ◐ FACI drift signal as early warning: the adaptive scorer tracks
  conformal coverage error per agent. Drifting coverage is an
  observable signal that the agent's behaviour has shifted, which
  may indicate memory or context poisoning. Observational, not
  preventative.
- ◯ Baseline data protection, content validation, memory
  segmentation, access and retention, provenance and anomalies: all
  deployer-owned.

**Deployer-owned:** memory storage architecture, embedding
provenance, RAG ingestion sanitization, cross-tenant memory
isolation, source-attribution metadata on retrieved chunks.

## ASI07 Insecure Inter-Agent Communication ◐

Multi-agent systems coordinate via APIs, message buses, and shared
memory. Weak inter-agent controls let attackers intercept,
manipulate, spoof, or block messages, including MITM, replay,
downgrade, and descriptor forgery.

- ✅ Message integrity and semantic protection, digitally sign
  messages, hash payload and context: every Vaara-governed action
  ships as a hash-chained audit record and, when OVERT is
  configured, an Ed25519-signed (or optional ML-DSA-65 signed for
  the post-quantum scheme) Base Envelope.
- ◐ Agent-aware anti-replay (nonces, session identifiers,
  timestamps): the audit chain carries timestamps and per-record
  hash. Dedicated session nonces are deployer concern.
- ◐ Discovery and routing protection (authenticate discovery using
  cryptographic identity): OVERT Base Envelope carries the
  governing Vaara instance identity. Cross-agent discovery is
  deployer-owned.
- ◐ Secure agent channels (mTLS, per-agent credentials): transport
  is deployer-owned. Vaara is the governance layer above the
  transport.
- ◯ Protocol and capability security (disable weak modes, enforce
  version pinning at gateways): transport policy concern.
- ◯ Attested registry and agent verification, signed agent cards:
  future work. Candidate surface is a `vaara overt verify` mode
  consuming external agent cards.

**Deployer-owned:** TLS / mTLS, agent discovery directory,
cross-agent PKI, inter-agent protocol selection and downgrade
prevention.

## ASI08 Cascading Failures ◐

A single fault (hallucination, malicious input, corrupted tool,
poisoned memory) propagates across autonomous agents, compounding
into system-wide harm.

- ✅ Independent policy enforcement, separate planning and
  execution via an external policy engine: Vaara is the external
  policy engine. Planning happens in the model, every execution
  lands at the Vaara boundary.
- ✅ Output validation and human gates, checkpoints, human review
  for high risk: review queue + static HTML dashboard (v0.13.0).
- ✅ Rate limiting and monitoring, detect fast-spreading commands
  and throttle: policy DSL supports rate caps, adaptive scorer's
  velocity signal feeds risk scoring.
- ✅ Behavioral and governance drift detection, track decisions vs
  baselines, flag gradual degradation: FACI coverage-error signal
  in the adaptive scorer is this drift detector.
- ✅ Logging and non-repudiation, tamper-evident logs bound to
  cryptographic agent identities with lineage metadata: exactly the
  hash-chained `AuditTrail`, OVERT Base Envelope, and Article 12
  commit-prove receipt pair.
- ◐ JIT one-time tool access with runtime checks: the intercept is
  the runtime check. JIT credential issuance is deployer-owned.
- ◐ Blast-radius guardrails (quotas, progress caps, circuit
  breakers between planner and executor): circuit-breaker semantics
  are expressible in the policy DSL but not yet a first-class
  receipt artefact.
- ◐ Zero-trust model in application design: applies at the
  intercept boundary. The broader architecture is deployer-owned.

**Deployer-owned:** infrastructure-level rate limiting at the API
gateway, replicated planner / executor topology choices, upstream
alerting on observed drift signals.

## ASI09 Human-Agent Trust Exploitation ◐

Intelligent agents establish strong trust with human users through
natural-language fluency and perceived expertise. Adversaries
exploit this trust to influence user decisions or extract sensitive
information. Over-reliance bypasses oversight.

- ✅ Explicit confirmations, multi-step approval or human-in-the-
  loop before sensitive actions: review queue + `ESCALATE` decision
  class.
- ✅ Immutable logs, tamper-proof records of user queries and agent
  actions for audit and forensics: hash-chained `AuditTrail`.
- ✅ Allow reporting of suspicious interactions, plain-language
  risk summary: the conformal interval is the plain-language summary
  (one number with documented coverage guarantee, explained for non-
  statisticians in `docs/conformal-prediction.md`). Reviewer-facing
  reports ship in PDF and Markdown.
- ✅ Content provenance and policy enforcement, verifiable metadata,
  source identifiers, timestamps, integrity hashes: every audit
  record carries policy id, action context, conformal interval,
  audit hash. OVERT envelope additionally signs the action class.
- ◐ Adaptive Trust Calibration, adjust agent autonomy based on
  contextual risk scoring: the risk score per call is contextual.
  the reviewer's autonomy adjustment is policy-driven by the
  deployer.
- ◐ Human-factors and UI safeguards, visually differentiate
  high-risk recommendations: static HTML dashboard renders status
  badges. The per-call risk surface lives in the deployer's UI.
- ◐ Plan-divergence detection, compare action sequences against
  approved workflow baselines: the adaptive scorer's drift signal
  is a workflow-level divergence detector at the risk-score layer.

**Deployer-owned:** user-facing UI, training of reviewers on
automation bias, separating preview from effect at the UI layer,
choosing which actions cross the trust threshold.

## ASI10 Rogue Agents ◐

Malicious or compromised AI Agents deviate from their intended
function or authorized scope, acting harmfully, deceptively, or
parasitically. The focus is on the loss of behavioural integrity
once drift begins, not the initial intrusion.

- ✅ Governance and Logging, comprehensive, immutable, and signed
  audit logs of all agent actions, tool calls, and inter-agent
  communication: hash-chained `AuditTrail` plus optional Ed25519 or
  ML-DSA-65 signing on each OVERT Base Envelope.
- ✅ Monitoring and Detection, behavioural detection focusing on
  collusion patterns and coordinated false signals: the adversarial
  classifier scores anomalous patterns per call, the adaptive
  scorer tracks per-agent baselines, the audit chain preserves the
  full sequence for retrospective detection.
- ◐ Isolation and Boundaries, Trust Zones with strict inter-zone
  communication rules: policy DSL declares per-agent boundaries.
  Process and network isolation is deployer-owned.
- ◐ Containment and Response, kill-switches and credential
  revocation: hot policy reload (v0.13.0) lets an operator revoke a
  tool from a rogue agent's permitted set without restarting the
  pipeline. Credential revocation is deployer-owned.
- ◐ Identity Attestation and Behavioral Integrity Enforcement,
  per-agent cryptographic identity attestation, signed behavioural
  manifests: OVERT Base Envelope carries the governing Vaara
  instance identity and the policy hash. A per-agent signed
  behavioural manifest is candidate future work
  (DRIFT-1 in the OVERT 1.0 Part 3 mapping).
- ◐ Require periodic behavioural attestation, signed BOM for
  prompts and tools, per-run ephemeral credentials: the S3P
  attestation (v0.12.0) is the closest in-tree primitive, shipping
  statistical safety signals over windowed observations.

**Deployer-owned:** per-agent credential rotation, network trust
zones, the agent identity provider Vaara reads from, the incident
response runbook that consumes Vaara's audit chain.

## What this document is not

This document does not claim Vaara is a one-stop OWASP Agentic Top
10 mitigation. Several risks (ASI06 Memory and Context Poisoning,
parts of ASI04 Agentic Supply Chain Vulnerabilities, parts of ASI07
Insecure Inter-Agent Communication) sit primarily outside Vaara's
runtime tool-call interception surface and are recorded as
deployer-owned or future work.

The Vaara wedge is strongest on tool-call governance, runtime
evidence emission, and audit traceability (ASI02, ASI05, ASI08,
ASI10). The document is published so an enterprise reader can see
the honest coverage without reading the rest of the repository.

## How to keep this current

When the OWASP project publishes a new ASI revision, replace the
per-risk descriptions and the cross-mapping table accordingly. When
Vaara ships a feature that moves a status from ◐ to ✅ or fills a ◯,
update the relevant section and the summary table. When in doubt,
prefer the more conservative status badge.
