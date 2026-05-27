# Vaara mapping to the Qi et al. agentic-AI trustworthiness survey

This page maps Vaara's runtime governance surface against the taxonomy
in Qi et al., [*Towards Trustworthy Agentic AI: A Comprehensive Survey
of Safety, Robustness, Privacy, and System Security*](https://arxiv.org/abs/2605.23989)
(arXiv:2605.23989, 2026-05-17). The survey organises agentic-AI risks
along the agent workflow (Perceive, Plan, Act, Reflect, Learn,
Multi-agent, Long-horizon) under two top-level dimensions: Safety and
Robustness, and Privacy and System Security. It also defines a
consolidated evaluation framework covering outcome vs. process
signals, trajectory- and step-level metrics, judge reliability, and a
release-pipeline shape from offline regression through canary rollout
to production monitoring.

The mapping is deliberately honest. Vaara is a runtime evidence layer
at the agent tool-call boundary. It addresses risks that materialise
at that boundary (Act-stage and trail-side Reflect-stage risks) and
does not address risks that live at perception, planning, learning,
or training-time. Naming the gap is the point.

## Mapping convention

Each row names one risk category from the survey, a Vaara mechanism
that produces evidence against it, and a coverage classification:

- **Direct**: Vaara records or enforces against this risk at the
  tool-call boundary.
- **Partial**: Vaara captures the trail-side evidence (what happened,
  when, by which agent) but the detection or mitigation lives in an
  upstream component that Vaara composes with.
- **Out of scope**: the risk lives at a layer Vaara is not designed
  to address (training data, model internals, planner reasoning).

## Safety and Robustness (Section 3.1)

### Perceive stage

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Data poisoning | Out of scope (training-time). | Out of scope |
| Adversarial perturbations on inputs | Out of scope (model-side). | Out of scope |
| Indirect prompt injection | Adversarial classifier (`adversarial_classifier_v9.joblib`) scores `tools/call` actions for injection-shaped patterns; v9 retrain on BIPIA follows brings FPR to 1.2% [0.4, 3.6] across four backends under BIPIA pressure. | Direct |
| Sensor / observation spoofing | Out of scope (sensor layer). | Out of scope |
| Instruction–data boundary confusion | MCP proxy perimeter separates resource/prompt operator scope from tool-call client scope; per-action audit captures the boundary crossing. | Direct |

### Plan stage

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| OOD generalisation failures | Cross-model held-out evaluation on Mixtral and Claude Sonnet 4.6; conformal prediction intervals on every risk score quantify model uncertainty rather than hiding it. | Partial |
| Specification gaming | Out of scope (planner-side). Trail records the executed actions and outcomes that a downstream specification-gaming detector can consume. | Partial |
| Goal misgeneralisation | Out of scope (planner-side). | Out of scope |
| Miscalibrated uncertainty in world models | Out of scope (planner-side). Vaara's own scorer is conformal-calibrated; planner uncertainty is not. | Out of scope |
| "Happy-path" brittle strategies | Adversarial robustness measured against three PAIR attacker families at n=300 each; v0.38 ships 88.4% recall under PAIR pressure. | Partial |

### Act stage

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Dangerous tool use | Policy gate enforces allow/escalate/deny at the MCP boundary. Adversarial classifier scores `tools/call` for `tool_misuse`, `privilege_escalation`, `data_exfil`. | Direct |
| Cascading failures from upstream errors | Per-action audit captures the full action chain. Conformal interval widens when context diverges from calibration distribution. | Direct |
| Irreversibility in high-impact actions | ESCALATE routes high-score calls to Article 14 human-in-loop queue before execution. Reviewer decision recorded back into the chain. | Direct |
| Partial automation failures | `evidence_insufficient` status returned honestly when an article has no recorded events. | Direct |
| Tool chaining cascades | Multi-agent action attribution via `agent_id`; composable audit chains across a Vaara fleet. | Direct |

### Reflect stage

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Unsafe self-assessment | Out of scope (model-side). | Out of scope |
| Deceptive rationalisation | Out of scope (model-side). | Out of scope |
| Evaluator spoofing | Hash-chained audit trail prevents post-hoc rewriting of the trail an evaluator reads. SEP-2787 v2 attestation binds tool name + server fingerprint + args commitment per call. | Direct |
| Over-confidence | Conformal interval is honest by construction (distribution-free coverage). | Direct |
| Incomplete trace evidence | `evidence_insufficient` status when an Article report has no recorded events; the trail does not rubber-stamp gaps. | Direct |

### Learn stage

All Learn-stage risks (reward hacking, safety regression, capability–
constraint imbalance, catastrophic forgetting, adversarial pattern
importation) are out of scope. Vaara is a runtime substrate, not a
training pipeline.

### Multi-agent

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Collusion to bypass constraints | Per-agent action attribution via `agent_id`; cross-agent trail composition surfaces collusive patterns to downstream analysis. | Partial |
| Misinformation amplification | Out of scope (content-side). | Out of scope |
| Negative externalities via competitive equilibria | Out of scope (system-design-side). | Out of scope |
| Communication channel compromise | MCP proxy enforces perimeter; audit chain captures every channel crossing. | Direct |

### Long-horizon

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Compounding error | Per-action conformal interval widens as context drifts; outcome feedback loop recalibrates. | Partial |
| Delayed side effects | `outcome_recorded` events accept post-execution feedback at arbitrary delay; chain preserves the original decision context. | Direct |
| Value drift, memory accumulation, stale goal contamination | Out of scope (planner / memory-side). | Out of scope |

## Privacy and System Security (Section 3.2)

### Perceive stage

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Direct / indirect prompt injection | Adversarial classifier (see Safety section above). | Direct |
| Multimodal inference attacks | Out of scope (modality-side). | Out of scope |
| Obfuscated input attacks | Classifier corpus includes obfuscation-class adversarial samples; cross-model held-out evaluation tests generalisation. | Partial |
| Retrieval layer poisoning | Out of scope (retrieval-side); trail captures the resulting tool calls. | Partial |
| Zero-click injection | Classifier scores every `tools/call` regardless of provenance. | Direct |

### Plan and Act stages (privacy)

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Regurgitation / reconstruction of private content | Out of scope (content-side). | Out of scope |
| Memory poisoning (delayed triggers) | Out of scope (memory-side); trail records the action chain that delayed triggers eventually produce. | Partial |
| Tool-mediated leakage | OVERT 1.0 envelope keeps request content local; only HMAC-SHA256 commitment crosses the trust boundary. SEP-2787 v2 hash-only-identity projection achieves the same property for MCP. | Direct |
| Credential theft via tool access | MCP proxy enforces allow/deny on credential-bearing tools; per-call attestation binds server fingerprint. | Direct |
| Exfiltration via authorised channels | Classifier specifically targets `data_exfil` patterns; conformal score quantifies exfiltration risk per call. | Direct |
| Side-channel leakage (timing, error codes) | Out of scope (transport-side). | Out of scope |
| SQL injection | Out of scope at the tool-call layer; trail captures the tool call and its risk score. | Partial |

### Reflect stage (privacy)

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Cross-component propagation | Per-action audit chain captures every component crossing. | Direct |
| Trace over-collection | OVERT envelope is commitment-only by default; raw content stays local unless explicitly attached. | Direct |
| Rationale leakage | Out of scope (rationale-side); trail records the action, not the model's chain of thought. | Out of scope |
| Protocol-level failures (tampering, impersonation) | Hash-chained trail, Sigstore signing on export, optional ML-DSA-65 post-quantum signer. SEP-2787 v2 binds signer-secret-version per envelope. | Direct |
| Replay / downgrade attacks | SEP-2787 v2 nonce + TTL is the per-envelope replay guard; the verifier exposes step 5 (argument commitment) for caller composition. | Direct |

### Learn, Multi-agent, Credential-management

| Risk | Vaara mechanism | Coverage |
|---|---|---|
| Privacy risk persistence across updates | Out of scope (training-side). | Out of scope |
| Insider threats | `policy_override` event captures every manual override with overrider identity and reason; the override is itself audited. | Direct |
| Compromised tool / API backdoors | MCP proxy server-fingerprint binding; allow-list enforces known fingerprints only. | Direct |
| Supply chain poisoning | SLSA build provenance on every PyPI release; Sigstore signatures on the published wheels. | Direct |
| Non-parametric update vulnerabilities | Out of scope (memory-side). | Out of scope |
| Shared context disclosure | Multi-tenancy isolation at the Vaara layer is v0.40 scope; current single-tenant deployments isolate by Vaara instance. | Partial |
| Privilege escalation | Policy gate; classifier scores `privilege_escalation` class explicitly. | Direct |
| Collusive exfiltration | Cross-agent trail composition (see Multi-agent above). | Partial |
| Long-lived token exposure, recovery gaps | Out of scope at the runtime layer (credential-management lives upstream); trail captures use of credentialed tools. | Partial |

## Evaluation framework (Section 4)

The survey's consolidated evaluation framework maps to Vaara's own
bench discipline as follows.

| Survey concept | Vaara correspondence |
|---|---|
| Outcome vs. process evaluation (4.2.1) | Both. `outcome_recorded` events carry outcome; the full lifecycle log is the process record. |
| Trajectory- vs. step-level metrics (4.2.2) | Trajectory: per-action conformal interval. Step: per-event hash-chained record. |
| Long-horizon evaluation (4.2.3) | Outcome feedback loop; `outcome_recorded` accepts arbitrary delay. |
| Multi-agent evaluation (4.2.4) | `agent_id` attribution; cross-agent composition. |
| Judge reliability and adversarial robustness (4.2.5) | PAIR robustness at n=300 per attacker family; cross-model held-out evaluation on Mixtral and Claude Sonnet 4.6. |
| Offline regression (replay known failures) | `tests/adversarial/` corpus replay. |
| Sandboxed execution (ToolEmu-style) | MCP proxy enforces perimeter in sandbox deployments. |
| Red teaming | PAIR three-family adversarial sweep. |
| Shadow mode (read-only deployment) | Audit-only mode (no enforcement) ships with the policy gate; both shapes write the same hash-chained trail. |
| Canary rollout | Out of scope (deployment-shape, not runtime). v0.40 deployment scope includes hot-reload extensions that compose with canary patterns. |
| Production monitoring | Per-action conformal scoring is the runtime monitor. |

## Summary

Vaara provides direct evidence against Act-stage and Reflect-stage
trail-integrity risks across both top-level dimensions of the Qi
survey, with partial coverage of Perceive-stage injection and several
Multi-agent and Long-horizon risks. Plan-stage, Learn-stage, and
content-side privacy risks are out of scope by design and named as
such above.

The survey's open challenges (Section 6) include runtime monitoring,
trustworthy personalisation, standardising explainability, and
closing the accountability gap. Vaara's runtime governance substrate
is one primitive that addresses the first; the audit chain plus
per-Article evidence reports addresses the fourth. The other two sit
at adjacent layers.

## Citation

> Qi, J., Li, M., Liu, J., Shu, Y., Yu, D., Ma, S., Cui, W., Zhao,
> Y., Chen, Y., Jiang, R., King, I., Xu, Z. (2026). "Towards
> Trustworthy Agentic AI: A Comprehensive Survey of Safety,
> Robustness, Privacy, and System Security." arXiv:2605.23989.
