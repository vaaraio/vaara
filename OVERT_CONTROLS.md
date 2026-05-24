# Vaara mapping to OVERT 1.0 Part 3 (Agentic AI Controls)

OVERT 1.0 Part 3 (Sections 11-16) defines the agentic-specific
execution controls: tool-call governance, MCP server trust,
multi-agent boundaries, capability mediation, agent disclosure,
human-in-the-loop attestation, and behavioural drift governance.

The mapping below states, control by control, whether Vaara satisfies
the requirement today (✅), partially satisfies it (◐), or leaves it
as explicit gap-to-deployer or future work (◯). This mapping does
not establish legal compliance with any regulation. It records
technical correspondence.

The companion mapping to the OWASP Top 10 for Agentic Applications
2026 lives in [`OWASP_AGENTIC.md`](OWASP_AGENTIC.md). The two
documents address overlapping concerns from different framings:
OVERT 1.0 Part 3 names enforcement controls, OWASP names threat
classes. The longer architectural context (Position relative to open
runtime-attestation standards, S3P, OVERT envelope binding to a
hardware-rooted measurement) lives in
[`COMPLIANCE.md`](COMPLIANCE.md).

## Source

OVERT 1.0 (Glacis Technologies). Project page:
[overt.is](https://overt.is). The "Part 3 (Agentic AI Controls)"
designation covers Sections 11 through 16 plus the S3P measurement
primitive in Section 9, MEA-2.

## Section 11 - Tool-Call Governance

- **TOOL-1.1** (intercept all tool calls before execution) - ✅.
  `InterceptionPipeline.intercept()` is the enforcement boundary. No
  tool call proceeds without a governance decision.
- **TOOL-1.2** (evaluate against capability policy) - ✅. The policy
  DSL declares permitted tools, parameter ranges, destinations, and
  approval gates. `policy.evaluate` returns the verdict carried in
  the per-call receipt.
- **TOOL-1.3** (denial receipt with policy reference and violation
  type) - ✅. Denials emit a `DENY` event on the hash chain with
  policy id and violation reason.
- **TOOL-1.4** (provisional receipt before execution, upgrade to full
  attestation after notary validation) - ✅ structurally at AAL-3,
  with the AAL-3 → AAL-4 path now implementable in-tree. The Article
  12 commit-prove receipt pair (shipped v0.10.0) is the Phase 2
  Provisional Receipt. The v0.11.0 OVERT Base Envelope is the
  attested form. v0.13.0 ships a reference Phase 3 IAP
  (`vaara.attestation.iap.emit_phase3_attestation`) that notary-signs
  the Provisional Receipt and anchors it in a transparency log.
  Reaching AAL-4 still requires the notary keys to live with an
  independent operator.
- **TOOL-2.1** (explicit function allowlist with hash in policy
  attestation) - ✅. Policy hash flows into `encoder_binary_identity`
  in the Base Envelope (v0.11.0).
- **TOOL-2.2** (parameter schema validation before execution) - ✅
  for declared parameter shapes. ◐ for arbitrary deep schemas (the
  policy DSL is intentionally bounded).
- **TOOL-2.3** (rejection receipt with parameter violation detail) -
  ✅.
- **TOOL-3.1** (per-tool rate limits with attested enforcement) - ◐.
  The adaptive scorer applies velocity-aware risk signals. Explicit
  per-tool calls-per-epoch counters are not yet emitted as standalone
  receipts.
- **TOOL-3.2** (per-session / per-user velocity caps) - ◐ via the
  agent profile in `scorer/adaptive.py`.
- **TOOL-3.3** (circuit breakers on error / violation rate) - ◐ in
  policy DSL. Circuit-breaker receipt not yet a first-class artefact.
- **TOOL-3.4** (recursion-depth termination per trace_id) - ◯.
  Not implemented. Agent-loop termination is currently the deployer's
  responsibility.
- **TOOL-4** (human approval gates) - ◐. The SQLite-backed review
  queue (`vaara.audit.review_queue`) routes `ESCALATE` verdicts to
  human reviewers and records `ESCALATION_RESOLVED` events with
  reviewer identity, timestamp, and decision. TOOL-4.4 approval-
  velocity caps are not enforced.
- **TOOL-5** (tamper-evident tool-call log with epoch attestation) -
  ✅ for TOOL-5.1 and TOOL-5.2 (hash-chained `AuditTrail`, Article
  12 commit-prove receipt pair). TOOL-5.3 epoch notary attestation
  is satisfied by the v0.13.0 reference IAP
  (`vaara.attestation.iap`) paired with the in-process transparency
  log. A sigstore Rekor-backed log can substitute at the same call
  sites.

## Section 11.5 - MCP Server Trust Governance

Vaara ships an MCP server (`vaara.integrations.mcp_server`) that
exposes governance tools to MCP clients. It does not currently act
as an MCP *client* governing tools hosted on third-party MCP
servers. The MCP-1/2/3 control set therefore applies to Vaara only
in the **custom (operator-hosted)** mode (MCP-2): the operator runs
the Vaara MCP server in their own environment.

- **MCP-2.1** (server binary identity in co-epoch binding) - ◐ at
  v0.12.0: arbiter binary identity is captured in
  `encoder_binary_identity`. A dedicated MCP-server binary identity
  field is future work.
- **MCP-2.2** (network topology attestation) - ◯. Deployer concern.
  Vaara does not measure its own network position.
- **MCP-2.3** (per-call authorization at the MCP server boundary) -
  ✅. Every MCP tool invocation passes through `intercept()`.
- **MCP-2.4** (configuration change detection within an epoch) - ◯.
  Future work.
- **MCP-1** and **MCP-3** (managed-vendor and external-third-party
  MCP servers) - outside Vaara's current surface. An operator using
  Vaara as the *governor in front of* a third-party MCP server would
  need adapter work. The architecture admits it but no implementation
  ships today.

## Section 12 - Multi-Agent System Controls

- **MULTI-1** (inter-agent trust boundaries) - ◯. Per-agent policy
  evaluation works today (each `intercept()` call carries an
  `agent_id`), but agent-vs-agent trust separation is not enforced
  beyond what the deployment policy declares.
- **MULTI-2** (agent composition / topology attestation) - ◯.
  Deployer-side documentation. No Vaara-emitted topology receipt.

## Section 13 - Capability-Based Access Control

- **CAP-1** (data provenance tracking) - ◐. The taxonomy and policy
  DSL accept provenance tags on actions. Transformation propagation
  (CAP-1.2) is the deployer's responsibility because Vaara intercepts
  tool calls, not arbitrary data transformations inside the agent
  process.
- **CAP-2** (architectural separation of planning from untrusted
  data) - ◯. AAL-2 documentation at most. This is a deployer-side
  architecture choice that Vaara records but does not enforce.

## Section 14 - Agent Disclosure and Transparency

- **DISC-1.1** (capability documentation) - ◐ via the deployer's
  policy file + `vaara compliance report`.
- **DISC-1.2** (AIBOM in CycloneDX-AI or SPDX 3.0) - ◯. Future
  work. The auditor-facing evidence export (v0.10.0) is a candidate
  surface to embed AIBOM references.
- **DISC-1.3** (attestation summary with coverage ratio, S3P
  signals, override frequency) - ◐ from v0.12.0: Vaara emits S3P
  attestations (`vaara.attestation.s3p`) carrying coverage ratio and
  binomial CI. The deployer aggregates these for disclosure.

## Section 15 - Human-in-the-Loop Attestation

- **HITL-1** (consent attestation) - ◯. Deployer-side concern.
  Vaara does not collect end-user consent.
- **HITL-2** (human review attestation) - ◐. Review-queue resolution
  events on the audit chain carry reviewer identity (when supplied
  by the deployer), timestamp, decision, and reference to the
  original `ESCALATE` verdict by `action_id`. AAL-4 identity binding
  is the deployer's responsibility.
- **HITL-3** (human correction and override) - ◐ via `report_outcome`
  and the review-queue resolution event.
- **HITL-4** (policy and configuration approval with separation of
  duties) - ◯ at the receipt level. Policy-change approval is
  currently a git-history artefact, not an attested OVERT event.
- **SESS-1..5** (session-scoped attestation) - ◯.
- **STATE-1, STATE-2** (durable state sealing and prompt artifact
  binding) - ◯.
- **IDENT-1** (federated identity / token provenance chain) - ◐.
  `vaara.auth` accepts authenticated caller identity into the audit
  record. Full delegation-chain attestation per IDENT-1.2 is future
  work.

## Section 16 - Behavioural Drift Governance

- **DRIFT-1** (baseline intent declaration) - ◯. Future work. The
  policy DSL is the candidate surface for machine-readable
  behavioural bounds.
- **DRIFT-2** and downstream drift controls - ◐ in spirit. The
  adaptive scorer tracks coverage error via FACI
  (`scorer/adaptive.py`) and emits drift signals through audit
  events, but these are not yet packaged as DRIFT-* receipts.

## S3P (Section 9, MEA-2) statistical safety signal

S3P sits in Domain 5 (MEASURE), not Part 3, but it is the agentic-
relevant measurement primitive that ties everything above together.

- **MEA-1** (deterministic sampling infrastructure) - ◯. Vaara
  evaluates every intercepted action. Sampling-rate-based measurement
  is opt-in. A deployer who wants S3P sampling provides the PRF tag
  and threshold.
- **MEA-2.1** (epoch nonce commitment) - ✅ via
  `vaara.attestation.s3p.make_epoch_nonce_commitment`.
- **MEA-2.4** (exact binomial CI) - ✅. Pure-Python Clopper-Pearson
  via the regularized incomplete beta function. No scipy dependency.
- **MEA-2.6** (closed-schema S3P attestation, Ed25519-signed,
  canonical CBOR per Protocol Profile 1.0) - ✅ via
  `emit_s3p_attestation`.
- **Vaara conformal extension (proposed Protocol Profile
  extension):** the `ConformalExtension` field reports aggregate
  statistics over Vaara's per-action conformal prediction intervals
  alongside the standard Clopper-Pearson CI. The conformal aggregates
  carry the same non-parametric coverage guarantee with no
  distributional assumption, exactly the property MEA-2.4 requires
  from a method offered as an alternative to (or complement of)
  Clopper-Pearson. The extension rides in a single field in the
  signed metadata. Standard OVERT verifiers ignore it.
