# Vaara mapping to SAFR v1.1

This document maps Vaara to **Safeguards for Agentic Finance at Runtime
(SAFR)**, white paper Version 1.1, 9 October 2026, developed by the financial
industry under the Monetary Authority of Singapore's BuildFin.ai initiative.
Page numbers below refer to that PDF.

SAFR places a governance checkpoint between an agent's proposed action and
its execution. It names four runtime components (Agent Identity, Controls
Repository, Disposition Engine, Audit Log) joined by a Governance Envelope,
and two ways to deploy them: native integration and a gateway. Vaara is a
runtime layer of the same shape, built for agents in any sector.

Each row states whether Vaara covers the SAFR element today (Yes), covers
part of it (Partly), or leaves it to the deployer (Deployer). The mapping records technical
correspondence. It is not an endorsement by MAS or any SAFR contributor, and
it does not establish that a deployment meets any supervisory expectation.
SAFR itself says it is not regulatory guidance (p. 5).

## Source

- Safeguards for Agentic Finance at Runtime, White paper Version 1.1,
  October 2026. Monetary Authority of Singapore,
  [mas.gov.sg](https://www.mas.gov.sg/publications/monographs-or-information-paper/2026/safeguards-for-agentic-finance-at-runtime).
  Version 1.0 was published 3 July 2026.

## Governance Envelope

| SAFR element (page) | Vaara | |
|---|---|---|
| Action, action trace and context metadata packaged before execution (p. 9) | Every governed call reaches the pipeline as agent id, tool name, parameters and context, and is decided before it runs. With `vaara[attestation]` each decision leaves a signed receipt. | Yes |
| The envelope is not self-verifying; check it against records captured independently of the agent (p. 9) | The MCP proxy, LLM API proxy, shell proxy, host hooks and the Linux OS layer record each call at the boundary, outside the agent. The action trace on the trail is that system-generated record, so it does not depend on the agent's own account. | Yes |
| Bound to the verified agent, session and mandate, and protected against alteration (p. 9) | Receipts are signed and hash-chained. `aud` and `taskId` sit inside the signed preimage. A per-call credential grant binds the call to its scope. A principal-signed mandate object, as SAFR describes it, is carried only through the AP2 payment-receipt profile. | Partly |

## Agent Identity

| SAFR element (page) | Vaara | |
|---|---|---|
| Resolve the agent against a registry before any other evaluation; a failed check is rejected and logged (p. 10) | Every record carries the agent id, and the credential gateway mints short-lived grants bound to the attestation digest. `vaara scan` finds the agents on a machine and says which are governed. A registry lookup that rejects an unknown agent before scoring is the deployer's to configure, for example as a deny rule or a proxy allow list. | Partly |
| Open networks with several registries (p. 10) | Producer identity travels as a DID document with key history and revocations in a cross-organisation handoff (`vaara build-handoff`). Choosing the authoritative registry is the deployer's. | Partly |

## Controls Repository

| SAFR element (page) | Vaara | |
|---|---|---|
| A configurable rulebook from policy, regulation, product rules and user mandates (pp. 10-11) | Declarative policy in JSON or YAML: thresholds, per-tool overrides, action classes, sequence patterns and escalation routes. One format on every surface, hot reload that keeps the old policy if the new one fails, one policy per tenant, and `vaara policy validate` and `vaara policy test` in CI. | Yes |
| Mandates: authority is explicit and an agent cannot extend it (p. 11) | Capability scopes on the credential grant bound what a call may do, and delegated-privilege attenuation refuses any step down a delegation chain that broadens authority. | Yes |
| Authorisation: which agents act for which principals, and to what delegation depth (p. 18) | Delegation chains are reconstructed and checked for attenuation. A per-principal authorisation table is the deployer's policy. | Partly |
| Exposure limits: per-action and aggregate value thresholds (p. 18) | The policy thresholds apply to the risk score only. The x402 and AP2 bindings record amounts in the evidence. An amount limit is the deployer's rule. | Deployer |
| Rate limits per time window (p. 18) | The LLM API proxy enforces rate policy on model calls. Tool-call rate limits are not a policy field. | Partly |
| Evidence quality: a minimum confidence for autonomous execution (p. 18) | The agent's stated confidence (`agent_confidence`) feeds the score, and the conformal interval widens when outcome history is thin, which pushes a call toward escalation. A hard minimum is expressed through thresholds. | Partly |

## Disposition Engine

| SAFR element (page) | Vaara | |
|---|---|---|
| Deny, Escalate, Auto-Execute, Observe, with a specific reason recorded (p. 12) | Allow, escalate and deny, each with its reason. A deny carries a refusal block whose basis is `evaluated`, `not_evaluated` or `undetermined`. Observe and shadow modes score and record without blocking. A per-action Observe outcome is recorded as an allow with its signals. | Partly |
| Calibration by reversibility, materiality, customer impact, regulatory sensitivity and novelty (p. 12) | Action types carry reversibility, blast radius and regulatory domains, and the conformal interval reflects how novel a call is for that bucket. Monetary materiality is not a native factor. | Partly |
| Each step is evaluated independently; no authority carries into the next (p. 13) | Every call is scored and decided on its own, and with the credential gateway each allowed call gets its own grant. Sequence patterns raise the risk of a step when the steps before it form a known attack. | Yes |

## Audit Log

| SAFR element (page) | Vaara | |
|---|---|---|
| Tamper-evident, append-only, cannot be modified without detection (pp. 12-13) | Linear SHA-256 hash chain, signed receipts, signed sequence numbers that make a dropped record a provable gap, and optional RFC 3161 or eIDAS timestamps. | Yes |
| Entry content: envelope, mandate, outcome, rules applied, basis, time per stage (p. 13) | The record holds the call, the decision, the score and its interval, the signals, the policy that applied and the refusal basis. Evaluation time is recorded once per decision. | Partly |
| Independent of any party with an interest in how events are characterised (p. 13) | A trail verifies offline from its bytes and a public key. Standalone checkers import no Vaara code, and 52 conformance suites are public. | Yes |

## Human escalation

| SAFR element (page) | Vaara | |
|---|---|---|
| Escalation timeout that defaults to block (p. 19) | An escalated call waits for a signed human answer. A timeout or a corrupt answer stays a refusal, and the trail closes the hold as a deny by policy. | Yes |
| Escalation volume the institution can process (p. 19) | `vaara trail shadow-report` groups what would have been held, by tool, so the volume is measured before enforcement starts. | Yes |
| Reviewer authority to approve, modify or decline (p. 19) | The review queue resolves to allow, deny or abstain, and the resolution is written into the chain. Modifying the action is not a resolution. | Partly |

## Integration patterns

| SAFR element (page) | Vaara | |
|---|---|---|
| Native integration: the agent emits an envelope before each action (p. 17) | `@vaara.govern`, `InterceptionPipeline.intercept`, the HTTP API and the TypeScript client. | Yes |
| Gateway integration: intercept outbound calls with no change to agent code (p. 18) | `vaara-mcp-proxy`, `vaara llm-proxy`, `vaara proxy-shell`, the host hooks for six coding agents, and the Linux OS layer. | Yes |

## Gaps

- Monetary exposure limits, per action and aggregate, are not a policy field.
- Tool-call rate limits are not a policy field.
- Observe is not a distinct outcome label on the record.
- A reviewer cannot modify an escalated action, only allow, deny or abstain.
- Evaluation time is recorded per decision, not per stage.
