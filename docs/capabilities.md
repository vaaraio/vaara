# What Vaara does

This is the index of Vaara's capabilities, grouped by area. Each entry names the
code or command that implements it, so every line can be checked against the
source. Reconciled against v2.6.0. When this page and the code disagree, the
code is right. For when each concept first shipped, see
[PRIOR_ART.md](PRIOR_ART.md).

Vaara governs every agent tool call and model call against the organisation's
policy, and writes the decision and its outcome into a signed, hash-chained
record that an outside party recomputes from the bytes and verifies offline,
with none of the operator's software. The record can be anchored to up to five
independent roots of trust, with zero-knowledge and post-quantum options. 52
conformance suites each ship a checker that imports no Vaara code.

License: AGPL-3.0-or-later.

## 1. Governance and enforcement

| Capability | Where | What it does |
|---|---|---|
| One-line tool-call governance | `vaara.govern`, `src/vaara/pipeline.py` | Classify, score with a conformal risk interval, allow, escalate or deny, write the hash-chained record. Fail-closed: a refused call raises `vaara.Blocked`. |
| MCP proxy | `vaara-mcp-proxy`, `integrations/mcp_proxy.py` | Fronts any MCP server over stdio or HTTP. Enforces by default, `--shadow` observes. Operator allow and deny lists for tools, resources and prompts. Checks each `tools/call` against the tool's own `inputSchema` before it is scored. Mints a scoped credential per allowed call. `--policy` loads a policy file. |
| Model proxy | `vaara proxy`, `integrations/infer_proxy.py` | Governs the model call itself (OpenAI and Ollama compatible) and signs which model, input and output. `--policy` loads a policy file. |
| LLM API proxy | `vaara llm-proxy`, `integrations/llm_proxy.py`, `_llm_proxy_app.py` | In front of a hosted provider. Chat, Messages and Responses calls are checked against model and rate policy and recorded with their prompt. Tool calls in model replies are decided, and with `--policy` scored against the policy. Named secrets (`--seal-file`), known credential formats and context-detected credentials (`--seal-known-secrets`) are sealed before the call leaves and restored in the reply. A call it cannot record is refused. |
| Agent host hooks | `vaara hook pre-tool-use --client ...`, `integrations/claude_code_hooks.py` | Governs Claude Code, Cursor, Codex CLI, Gemini CLI, OpenCode and Copilot CLI through each host's own hook API. The Claude Code hook takes a policy file. |
| Deny rules | `deny_rules.py`, `integrations/claude_code_deny.json` | 46 rules over shell commands, web access, secret reads, file mutation and the agent's own meta-actions (spawning agents, scheduling, remote triggers, harness configuration). They also protect Vaara's own trail and configuration from the agent it governs. Each rule names its operator lift. |
| Shell proxy | `vaara proxy-shell` | Every shell command an agent issues is classified, scored, decided and recorded through the same pipeline. |
| Linux OS layer | `vaara run`, `vaara os-guard`, `vaara os-layer`, `src/vaara/oslayer/` | Governs what an agent does to files whatever the agent. `vaara run` confines it with an AppArmor profile in its own cgroup; `os-guard` decides each open and exec in the folders the operator picks. `vaara os-layer harden on` adds no_new_privs and a seccomp filter to every launch; `vaara os-layer egress HOST...` locks the network with Landlock so the launch reaches only an egress proxy that lets out the hosts listed, each connection a decision on the trail. |
| Cage layer | `vaara cage`, `src/vaara/cage/`, `docs/cage.md` | The containment an agent runs in, as a choosable part. Twelve drivers, each over the cage's own tool unmodified: the Vaara cage, NVIDIA OpenShell, the Codex sandbox, Anthropic sandbox-runtime, nono, gVisor, Kata, kubernetes-sigs agent-sandbox, Firecracker, microsandbox, E2B self-hosted, Apple's container. Start, stop, status, enforcement state, events; `vaara cage drivers` says which are ready on the machine. Every decision record and its signed receipt carry a `cage` block: which cage the deciding process ran in, the digest of its effective configuration, and whether the kernel confirmed the confinement at decision time. A run outside any cage says so. |
| Whole-machine install | `vaara init`, `vaara ungovern`, `integrations/init_governance.py` | Writes the host hooks, routes MCP client configs through the proxy and installs a managed launchd or systemd service. Reversible. |
| Agent discovery | `vaara scan`, `vaara scan --watch`, `integrations/_mcp_beacon.py` | Finds AI agents on the machine (processes talking to a model API or a local model server, MCP clients) and says whether Vaara governs each. `--watch` records each new agent it sees. The Vaara MCP server and `vaara-mcp-proxy` do the same for whatever connects to them on stdio: the client is read from the process table above them, alongside its own `clientInfo`, and recorded on the same agents trail, so an ungoverned client is announced when it connects. `VAARA_BEACON=0` turns this off. |
| Credential gateway | `credential/gateway.py`, `credential/_grant_*` | Short-lived scoped grants per call, capability-scope enforcement, and delegated-privilege attenuation: authority cannot grow down a delegation chain. |
| Egress guard | `integrations/_egress_guard.py` | Blocks DNS rebinding, cloud-metadata and private-network egress before any socket opens; pins the resolved IP; strips `Authorization` and `Cookie` on cross-origin redirects. |
| Enforcement modes | `pipeline.py`, `vaara-mcp-proxy --shadow`, hook `mode` | Observe, shadow (score and record, block nothing) and enforce. The recorded numbers are the same in every mode, so a deployment can be measured before it blocks anything. |

## 2. The record

| Capability | Where | What it does |
|---|---|---|
| Canonical receipt | `attestation/receipt.py`, `SPEC.md` | `vaara.receipt/v1`, JCS (RFC 8785) canonical, ES256, RS256 or HS256. |
| A receipt per decision | `pipeline.py` with `vaara[attestation]` | Every allow, escalate and deny leaves a signed decision receipt beside the trail. |
| Hash-chained audit trail | `audit/trail.py`, `audit/sqlite_backend.py` | Append-only, linear SHA-256 chain, tamper-evident by recomputing from the bytes. |
| SEP-2828 execution and decision records | `attestation/decision.py`, `attestation/tool_call_attestation.py` | Decision basis carried natively: `rationale`, `binding`, `decisionProof`. |
| Evidence ingestion | `vaara ingest`, `attestation/_ingest_emit.py` | Foreign evidence in, one signed record out. |
| Declarative source profiles | `attestation/profiles/*.json`, `attestation/_declarative.py` | SLSA, C2PA, in-toto agent decisions, ACP checkout, AP2 payment receipts, x402 settlement and CAEP security events, each bound by a JSON profile with no code. |
| Delegation attribution | `audit/delegation.py`, `credential/_grant_attenuation.py` | Reconstructs delegation chains and refuses any step that broadens authority. |
| Gap-evident completeness | `vaara verify-contiguity` | Signed sequence numbers and a running count make a dropped record a provable gap. Run sealing and a sealed class gate. |
| Cross-organisation handoff | `vaara build-handoff`, `vaara verify-handoff`, `attestation/_handoff*.py` | A package of record, DID document, key history, revocations and anchor, with the producer identity pinned. EU AI Act Article 26(6). |
| Access records | `audit/access.py` (`ACCESS_RECORDED`) | A read is an event: who read, on whose behalf, a digest of the returned set, and whether an outcome was recorded. |
| Audience and task binding | `attestation/_receipt_audience.py`, `attestation/_receipt_task.py` | `aud` and `taskId` inside the signed preimage. Verdicts `bound`, `conflict` or `unsupported`; a missing field never passes. |
| Decision-before-effect ordering | `attestation/_decision_verifier.py` (`effect_ordering`) | Reports `ordered`, `effect_precedes_decision` or `not_comparable`. |
| Trail repair with declared gaps | `vaara trail repair`, `audit/sqlite_backend.py` (`TrailRepair`) | Keeps the damaged file and appends a `repair_gap` event naming every lost sequence number. |
| Trail registry | `~/.vaara/sources.json` | Every trail the engine writes is listed, so none goes unnoticed. |
| Confidential-VM enforcement binding | `vaara verify-enforcement`, `attestation/_enforcement*.py` | Binds a signed record to an AMD SEV-SNP VM (`REPORT_DATA = sha512(jcs(record))`), with an optional expected measurement. |

## 3. Witness roots

The same record can be anchored to any or all of these.

| Root | Where | What it gives |
|---|---|---|
| RFC 3161 self-hosted timestamp | `audit/receipt_anchor.py` (`SelfHostedTSA`) | Immediate and offline, no third party. |
| eIDAS qualified timestamp | `audit/receipt_anchor.py` (`QualifiedTSA`), `scripts/qualified_anchor_dss_demo.py` | The demo submits the timestamp to the European Commission's DSS validator, which returns PASSED QTSA. Legal presumption under eIDAS Article 41; certificate pinned to an EU trusted list. |
| Transparency log and SCITT | `attestation/transparency_log.py`, `attestation/cose_receipt.py`, `audit/scitt_anchor.py`, `audit/scitt_service.py` | RFC 9162 Merkle inclusion proofs and SCITT-compatible COSE receipts. Registers a SCITT Signed Statement with an IETF SCITT Transparency Service over SCRAPI; a weekly CI job round-trips a receipt through Microsoft's scitt-ccf-ledger and verifies it offline. |
| TPM 2.0 | `attestation/_tpm*.py` | Hardware root binding. |
| Confidential-VM TEE | `attestation/tee.py`, `vaara tee` | AMD SEV-SNP attestation (experimental), with the report's signing key verified up to AMD's root for Milan, Genoa and Turin. |

## 4. Privacy and post-quantum cryptography

| Capability | Where | What it does |
|---|---|---|
| Zero-knowledge decision proof | `attestation/zk/`, `attestation/_decision_proof_verify.py` | Proves the verdict follows from a committed score and thresholds without revealing them. No trusted setup. Bound to its record so it cannot be moved to another. |
| Post-quantum signing | `audit/signer.py` (ML-DSA-65, FIPS 204), `attestation/_receipt_pq.py` | Hybrid signatures (`pq_hybrid_v0`), installed with `vaara[pq]`. |
| Crypto-posture record | `attestation/_receipt_cbom.py` | CycloneDX 1.6 CBOM, with the NIST post-quantum level inside the signed bytes. |
| Verifiable Credentials and DID | `attestation/_receipt_vc.py`, `attestation/_receipt_identity*.py` | W3C Verifiable Credentials and DID identity binding with live resolution. |

## 5. Inference attestation

`attestation/inference.py` and `attestation/_inference_*` bind a model call:
which model, given what input, returned what output.

- Integrity tier: binds model, input and output. Replay tier: byte reproducibility, gated by a determinism check that samples several outputs.
- Model-diversity cross-check (`attestation/_inference_crosscheck.py`): a second local model of a different identity judges the output for semantic equivalence.
- Session manifest (`attestation/_inference_session.py`): folds ordered receipts into one record that can be bound to a TPM evidence chain; `verify_inference_chain` checks TPM evidence, session and receipts as one verdict.
- `vaara receipt verify-inference` checks the attestation and receipt pairs the model proxy writes, one file or a whole directory.

## 6. OVERT

`attestation/overt.py`, `attestation/iap.py`, `attestation/s3p.py`,
`attestation/tee.py`, `docs/OVERT_CONTROLS.md`. Vaara implements OVERT 1.0:
base envelopes, AAL-3 and AAL-4 assurance levels, an Independent Attestation
Provider, and S3P, a safety-violation-rate measurement an auditor can reproduce.

## 7. Regulatory coverage

- EU AI Act Article 12, record-keeping: `audit/article12_export.py`, `compliance/`
- EU AI Act Article 14, human oversight: `audit/review_queue.py` (section 14)
- EU AI Act Article 50, transparency disclosures: `audit/article50.py`
- EU AI Act Article 73, serious-incident report (interim): `audit/incident_export.py`, `vaara trail export-incident`
- DORA Articles 9, 10 and 13: event-to-article mappings in `audit/trail.py`, `compliance/engine.py`
- SOC 2 Trust Services Criteria CC6.1 to CC6.3, CC7.2, CC7.3 and CC8.1: event-to-criterion mappings in `audit/trail.py`, `compliance/engine.py`, `docs/COMPLIANCE.md`
- GDPR Chapter V, data locality and transfers: `attestation/data_locality.py`, `data_locality_v0`
- GDPR Article 17, erasure: read-time redaction in `audit/sqlite_backend.py`
- eIDAS 2.0 qualified electronic ledger profile: `docs/eidas-qel-profile.md`
- prEN ISO/IEC 12792 transparency taxonomy: per-record tagging in `audit/trail.py`
- W3C PROV-DM and PROV-JSON export: `audit/prov_export.py`, `vaara trail export-prov`
- IETF RATS EAR (AR4SI) trustworthiness claims: `vaara export-attestation-result`
- Guardrail findings tagged to articles, with deployer overrides: `integrations/_content_safety_articles.py`
- Threat and control mappings: OWASP Top 10 for Agentic Applications 2026 (`docs/OWASP_AGENTIC.md`), OWASP AISVS C9.2.3 and C9.2.4, MITRE ATLAS (`tests/vectors/atlas_threat_v0/`), MIT AI Risk Repository (`docs/mit_ai_risk_repository_mapping.md`), article-level EU AI Act, DORA and SOC 2 mapping (`docs/COMPLIANCE.md`)

## 8. Payment and commerce rails

Accountability bindings for x402 (`x402_settlement_v0`), Google AP2 (`ap2_v0`),
Visa TAP (`tap_v0`) and the Agentic Commerce Protocol (`acp_checkout_v0`),
through one schema-agnostic binding, `external_execution_evidence` (`SPEC.md`
section 5.6).

## 9. Integrations

- Agent hosts: Claude Code, Cursor, Codex CLI, Gemini CLI, OpenCode, Copilot CLI (section 1).
- Frameworks: LangChain (`integrations/langchain.py`), CrewAI (`integrations/crewai.py`), OpenAI Agents SDK (`integrations/openai_agents.py`), MCP as server and proxy.
- Guardrail adapters: AWS Bedrock Guardrails, Azure AI Content Safety, GCP Model Armor, NVIDIA NeMo Guardrails, Guardrails AI, LLM Guard, Rebuff (`integrations/`).
- Detection: prompt injection and PII (`detect/`).
- HTTP scoring service: `vaara serve`, `src/vaara/server/`, `docs/openapi.yaml`.
- Packages and deployment: Python (`pip install vaara`), TypeScript client (`@vaara/client` on npm), Helm chart (`deploy/helm/vaara`, policy as YAML), GitHub Action (`action.yml`, Vaara Policy Check).

## 10. Conformance

52 conformance suites under `tests/vectors/` and `conformance/`, each with a
`_check_independent.py` that imports no Vaara code and recomputes its verdicts
from the bytes of its case files. `scripts/conformance_runner.py` runs them all:
50 pass and 2 skip without optional extras (`pq_hybrid_v0` needs `vaara[pq]`,
`qualified_time_v0` needs `vaara[timeanchor]`). `vaara conformance statement`
checks an installed build against the published corpus. Results and outside
reproductions: [vaara.io/conformance.html](https://vaara.io/conformance.html).

## 11. Risk scoring

| Capability | Where | What it does |
|---|---|---|
| Online conformal scorer | `scorer/adaptive.py` (`AdaptiveScorer`) | Expert signals weighted by multiplicative weight updates, under a split-conformal interval with adaptive alpha (FACI) and optional per-category (Mondrian) coverage. The decision compares the interval's upper bound with the thresholds. Per-tenant thresholds; learns from reported outcomes. |
| Dangerous sequences | `scorer/adaptive.py` (`SequencePattern`) | 8 built-in ordered patterns (data exfiltration, read then outbound, data destruction, privilege escalation, financial drain, governance takeover, safety override, rapid rebalance) raise the risk on a match. A policy sequence with `escalate: true` holds the completing call for a human. |
| Trained per-step gates | `scorer/action_gate.py`, `scorer/trained_gate.py`, `scorer/mc_dropout_gate.py`, `scorer/stacked_gate.py` | Gradient-boosted bootstrap ensemble, MC-dropout uncertainty and a logistic stack, deciding by conformal set membership. |
| Composite and remote scoring | `scorer/composite.py`, `scorer/composition.py` | Runs Vaara alongside other scorers, or posts the context to a remote `/v1/score`; a transport error is a deny. |
| Adversarial classifier | `adversarial_classifier.py` | Opt-in (`vaara[ml]`) v11 bundle: 254 hand features plus 384-dimensional MiniLM embeddings. The bundle's SHA-256 is verified before it loads. Recall 85.6% at a 5.1% false-positive rate on held-out test data (README, "How it scores"). |
| Cloud-metadata floor | `scorer/_param_signals.py` | A parameter that targets the instance-metadata address, in any encoding, scores 0.95. |
| Detection primitives | `detect/injection.py`, `detect/pii.py` | Prompt injection (classifier with a pattern fallback) and PII (email, phone, SSN, IPv4, card numbers with Luhn, IBAN with mod 97) with offsets for redaction. |
| Cold-start calibration | `sandbox/trace_gen.py` | Synthetic benign, careless and adversarial traces calibrate the scorer and its interval before any real agent connects. |

## 12. Policy and multi-tenancy

| Capability | Where | What it does |
|---|---|---|
| Declarative policy | `policy/schema.py`, `policy/loader.py` | JSON, or YAML with `vaara[yaml]`. Default thresholds, per-tool overrides, sequence patterns and escalation routes. Unknown keys are refused. |
| Policy on every surface | `vaara serve --policy`, `vaara proxy --policy`, `vaara llm-proxy --policy`, `vaara-mcp-proxy --policy`, Claude Code hook `"policy"` | One policy format across the scoring service, the proxies and the hook. An invalid policy stops the surface at startup. |
| Operating modes | `policy/modes.py`, `vaara mode list`, `show`, `emit` | Eco, balanced, performance and strict presets, emitted as a valid policy. |
| Hot reload | `policy/controller.py` | Validates before it swaps; the old policy stays live if the new one fails. |
| Multi-tenant policy | `policy/registry.py` | One controller per tenant, with a default, loaded from a directory. |
| Tenant isolation in the chain | `audit/trail.py`, `audit/sqlite_backend.py` | Chain version 2 binds `tenant_id` into the record hash; reads are tenant-scoped. |
| Policy CI | `vaara policy validate`, `vaara policy test`, `policy/validate.py`, `policy/test_cases.py` | A semantic validator (narrow threshold band, override for an undeclared tool, unreachable route) and a test framework that asserts expected verdicts. |

## 13. Keys and long-term retention

| Capability | Where | What it does |
|---|---|---|
| Key generation | `vaara keygen` | Ed25519 trail-signing keys or EC P-256 attestation keys, written 0600, fingerprint printed. |
| Key lifecycle events | `audit/trail.py` (`record_key_lifecycle`) | Rotation, revocation and addition recorded as chained, anchorable events, so a revocation can be shown to predate a compromise. |
| Threshold export | `vaara trail export-threshold`, `audit/export.py` | k-of-n custodian signatures on a regulator handoff; no single key can forge it. |
| Long-term verification | `vaara verify-retained` | Verifies a record signed under a rotated-out key through the archived DID document, key history, revocations and an optional time anchor. |
| Retention rotation | `vaara trail rotate`, `vaara trail purge`, `audit/rotate.py` | Exports, re-verifies the export from its own bytes, then purges past retention (EU AI Act Articles 19(1) and 26(6)). `--dry-run` shows the plan. |

## 14. Human oversight (EU AI Act Article 14)

| Capability | Where | What it does |
|---|---|---|
| Review queue | `vaara review list`, `claim`, `resolve`, `audit/review_queue.py` | Pending, claimed and resolved (allow, deny or abstain); the resolution is written into the chain as `ESCALATION_RESOLVED`. |
| Signed approvals | `approvals.py`, host hooks | An escalated call waits for a human answer that is signed, which the governed agent cannot write. The request carries the call's full arguments and their SHA-256. A timeout or a corrupt answer stays a refusal. |
| Shadow report | `vaara trail shadow-report`, `audit/shadow_report.py` | Groups what would have been denied or held, by tool, over a period. |

## Try it

```
pip install vaara
scripts/conformance_runner.py                 # 52 suites, checkers import no Vaara
vaara conformance check PATH                  # keyless SEP-2828 conformance
vaara verify-record FILE --trusted-issuer-cert CA.pem   # qualified or self-asserted time
vaara receipt render RECEIPT.json             # self-contained offline evidence page
scripts/demo_multiagent_attribution.py        # three-agent delegation with attenuation
scripts/verify_vaara_trail.py EXPORT.zip      # standalone trail verification
vaara verify-retained RECORD                  # verify under a rotated-out key
vaara verify-handoff PKG                      # cross-organisation handoff
vaara scan                                    # find the agents on this machine
vaara mode list                               # policy presets
```
