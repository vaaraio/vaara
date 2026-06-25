# Prior art and related work

This document anchors when each load-bearing Vaara concept first shipped
in a tagged public release, and lists adjacent published work in the
same lane. It exists so that a reader comparing Vaara against newer
academic or industry proposals can check the published timeline rather
than relying on marketing claims.

The chronology is reconstructed from `CHANGELOG.md` and the git history
of this repository. Dates are calendar dates of the tagged PyPI release.
Version numbers and dates can be verified against
[https://pypi.org/project/vaara/#history](https://pypi.org/project/vaara/#history)
and the `vX.Y.Z` tags on
[https://github.com/vaaraio/vaara/tags](https://github.com/vaaraio/vaara/tags).

From v1.0.0 (2026-06-17) the project is licensed AGPL-3.0-or-later; releases
through v0.71.0 were published under Apache 2.0. Tag dates and version numbers
verify against the PyPI history and the `vX.Y.Z` git tags regardless of license era.

## When each Vaara concept first shipped

| Concept | First shipped | Where to read it |
|---|---|---|
| Interception pipeline for agent tool calls | v0.1.0, 2026-04-10 | `src/vaara/pipeline.py`, `CHANGELOG.md` v0.1.0 |
| Adaptive risk scoring with conformal interval on every score | v0.1.0, 2026-04-10 | `docs/formal_specification.md`, `docs/conformal-prediction.md` |
| Hash-chained audit trail | v0.1.0, 2026-04-10 | `src/vaara/audit/`, `COMPLIANCE.md` |
| Framework integrations (LangChain, CrewAI, OpenAI Agents) and MCP server surface | v0.3.0, 2026-04-18 | `src/vaara/integrations/` |
| Signed audit-trail export and verification CLI | v0.4.1, 2026-04-20 | `src/vaara/audit/`, `docs/vaara-audit-cli.md` |
| Sigstore-signed release workflow with PyPI trusted publishing and PEP 740 attestations | v0.4.3, 2026-04-21 | `.github/workflows/release.yml`, `docs/signing-keys.md` |
| Opt-in XGBoost adversarial classifier with by-seed held-out benchmarks | v0.5.0, 2026-04-23 | `src/vaara/adversarial_classifier.py`, `bench/` |
| Callable kernel HTTP surface (Vaara as the schema, not the plug-in) | v0.10.0, 2026-05-16 | `docs/openapi.yaml`, `src/vaara/integrations/http.py` |
| Article 12 commit-prove receipt pair | v0.10.0, 2026-05-16 | `src/vaara/audit/`, `CHANGELOG.md` v0.10.0 |
| `vaara-bench-v1` reproducible benchmark harness | v0.12.0, 2026-05-16 | `bench/` |
| Hot policy reload without pipeline restart | v0.13.0, 2026-05-17 | `src/vaara/policy/`, `CHANGELOG.md` v0.13.0 |
| Static HTML article-coverage dashboard | v0.13.0, 2026-05-17 | `src/vaara/compliance/dashboard.py` |
| Pluggable signer with optional ML-DSA-65 (FIPS 204) post-quantum scheme | v0.14.0, 2026-05-17 | `src/vaara/audit/signer.py` |
| External-scorer composition over the same HTTP interface | v0.14.0, 2026-05-17 | `src/vaara/policy/composition.py` |
| TypeScript client (`@vaaraio/client`) for the HTTP surface | v0.15.0, 2026-05-17 | `clients/ts/` |
| PDF auditor evidence export (per-article rollup) | v0.16.0, 2026-05-17 | `src/vaara/compliance/render.py` |
| OVERT 1.0 reference verifier CLI (`vaara overt verify`) | v0.17.0, 2026-05-17 | `src/vaara/overt/`, `docs/openapi.yaml` |
| Streaming-notification interception inside the audit and OVERT perimeter | v0.25.0, 2026-05-21 | `src/vaara/integrations/mcp_proxy.py` |
| Per-article verdict drill-down: `verdict_inputs`, `verdict_reasons`, `contributing_events` | v0.26.0, 2026-05-21 | `src/vaara/compliance/engine.py`, `VERDICTS.md` |
| SLSA build provenance attestation on every release | v0.26.0, 2026-05-21 | `.github/workflows/release.yml` |
| Continuous fuzzing of the OVERT decoder, audit `from_dict`, and policy loader via ClusterFuzzLite | v0.27.0, 2026-05-22 | `fuzz/`, `.clusterfuzzlite/`, `.github/workflows/cflite_*.yml` |
| `VERDICTS.md` per-article evidence sufficiency reference | v0.28.0, 2026-05-22 | `VERDICTS.md` |
| `docs/conformal-prediction.md` plain-language explainer | v0.28.0, 2026-05-22 | `docs/conformal-prediction.md` |
| This document (`PRIOR_ART.md`) | v0.29.0, 2026-05-24 | `PRIOR_ART.md` |
| Cross-model held-out methodology with public 4,176-entry eval fold | v0.36.0, 2026-05-25 | `bench/vaara-bench-v0.36.md`, `tests/adversarial/v036_holdout.json` |
| Destination-aware features (`dst__*`) and v7 production classifier | v0.36.0, 2026-05-25 | `src/vaara/adversarial_classifier.py`, `scripts/train_adversarial_classifier.py` |
| MCP proxy HTTP transport with multi-upstream fan-out | v0.40.0, 2026-05-28 | `src/vaara/integrations/mcp_proxy.py`, `CHANGELOG.md` v0.40.0 |
| Signed execution-receipt envelope paired with request attestation (three blocks: `backLink`, `receiptAsserted`, `outcomeDerived`) | v0.42.0, 2026-05-29 | `src/vaara/attestation/receipt.py`, `docs/execution-receipts.md` |
| Standalone attestation-envelope verifier (`vaara attest verify`) | v0.44.0, 2026-05-30 | `src/vaara/attestation/`, `docs/sep2787-conformance.md` |
| Remote MCP upstream connector (`--upstream-url`) over Streamable HTTP | v0.45.0, 2026-05-30 | `src/vaara/integrations/mcp_proxy.py` |
| External time anchoring of the audit hash chain (RFC 3161 / eIDAS existence-in-time) | v0.48.0, 2026-05-31 | `src/vaara/audit/timeanchor.py`, `README.md` |
| `vaara.attestation.decision`: decision records with supersession resolution | v0.49.0, 2026-05-31 | `src/vaara/attestation/decision.py` |
| Threshold-signed export with chain-anchored key-lifecycle markers | v0.50.0, 2026-06-01 | `src/vaara/audit/export.py`, `CHANGELOG.md` v0.50.0 |
| Receipt identity binding to a DID, with live-resolution audit (`verify_receipt_identity`) | v0.52.0, 2026-06-02 | `src/vaara/attestation/` |
| RFC 9162 transparency-log consistency-proof verification | v0.54.0, 2026-06-05 | `src/vaara/attestation/` |
| Revocation registry and status lens | v0.55.0, 2026-06-05 | `src/vaara/attestation/` |
| One-file evidence bundle verified through six lenses (identity, signature, back-link, inclusion, consistency, revocation) | v0.56.0, 2026-06-05 | `src/vaara/attestation/`, `docs/verifying-evidence.md` |
| Time-anchored Article 12 regulator pack with offline anchor verification | v0.59.0, 2026-06-07 | `src/vaara/audit/`, `docs/verifying-evidence.md` |
| Keyless conformance verifier for any SEP-2828 record, including records Vaara never produced (`verify-record`) | v0.60.0, 2026-06-07 | `src/vaara/attestation/receipt.py` |
| Cross-format normalization of adjacent MCP records (SEP-2643/2787/2817) onto SEP-2828 (`vaara normalize`) | v0.62.0, 2026-06-08 | `src/vaara/attestation/`, `CHANGELOG.md` v0.62.0 |
| Producer conformance self-statement against a versioned published corpus (`conformance-statement`) | v0.63.0, 2026-06-09 | `conformance/`, `src/vaara/attestation/` |
| Retention-window verification of a record under a rotated-out key (`verify-retained`) | v0.64.0, 2026-06-09 | `src/vaara/attestation/`, `docs/verifying-evidence.md` |
| Cross-organisation evidence handoff package, offline-verifiable years later under a rotated key (`build-handoff` / `verify-handoff`, Article 26(6)) | v0.65.0, 2026-06-09 | `src/vaara/attestation/` |
| Attested enforcement: a record bound to an AMD SEV-SNP confidential-VM attestation report (`verify-enforcement`) | v0.66.0, 2026-06-09 | `src/vaara/attestation/`, `docs/verifying-evidence.md` |
| Governed-tool-call agent skill (Article 14 oversight plus Article 12 record in front of a high-risk tool call) | v0.68.0, 2026-06-09 | `examples/skills/vaara-governed-tool-call/` |
| Post-quantum hybrid-signed execution receipts with the signature suite committed in the signed preimage (a stripped PQC signature is a detectable downgrade) | v0.69.0, 2026-06-11 | `src/vaara/attestation/`, `docs/design/pq-hybrid-signing-spec.md` |
| TPM 2.0 plus IMA binding of a signed SEP-2828 record, offline-verifiable on commodity hardware (`verify-tpm-binding`) | v0.70.0, 2026-06-12 | `src/vaara/attestation/`, `scripts/tpm/` |
| Continuous TPM 2.0 plus IMA attestation chain bound to the per-action record (`verify-tpm-chain`) | v0.70.0, 2026-06-12 | `src/vaara/attestation/`, `tests/test_tpm_chain.py` |
| RATS EAR neutral verify: a Vaara attestation verdict re-expressed as an IETF RATS EAR (draft-ietf-rats-ear) carrying an AR4SI trustworthiness vector (draft-ietf-rats-ar4si), root-agnostic across TPM binding, TPM chain, and SEV-SNP (`export-attestation-result`) | v0.71.0, 2026-06-16 | `src/vaara/attestation/`, `docs/design/attestation-result-spec.md` |
| Sovereign inference harness: a local model emits a signed, hardware-rooted inference receipt that a second local model independently cross-checks, with session, chain, cross-check, and determinism verifiers (first public release, AGPL-3.0-or-later) | v1.0.0, 2026-06-17 | `src/vaara/`, `CHANGELOG.md` v1.0.0 |
| evidenceRef worked example recomputes end to end from real tool-surface bytes (surface bytes to surface hash to drift record to content-addressed `evidenceRef`), so a second implementation can reproduce the whole chain | v1.0.2, 2026-06-17 | `tests/vectors/evidence_ref_v0/`, `CHANGELOG.md` v1.0.2 |
| Receipt-bound credential broker (the authority layer): the proxy mints a signed, short-lived credential bound to an attestation digest and scoped to one tool plus an args commitment plus tenant, and a gateway refuses any call lacking a valid, unexpired, non-revoked, attestation-bound grant (off by default) | v1.1.0, 2026-06-18 | `src/vaara/credential/`, `conformance/sep2828/credential_grant_v0/`, `docs/design/credential-broker-spec.md` |
| Canonical `vaara.receipt/v1` parent spec, with the x402 settlement binding and the eIDAS qualified-timestamp profile as downstream profiles that pin to it rather than competing formats | v1.2.0, 2026-06-19 | `SPEC.md`, `CHANGELOG.md` v1.2.0 |
| Self-hosted RFC 3161 timestamp anchors minted offline in pure Python, with `anchoredDigest` equal to the sha256 of the JCS-canonical signed payload, verified through the existing offline anchor verifier | v1.2.0, 2026-06-19 | `src/vaara/audit/receipt_anchor.py` |
| Typed capability scopes on credential grants (`le`/`ge`/`in`/`eq`) with closed coverage, bounding what a tool call may do rather than only pinning an exact argument set | v1.2.0, 2026-06-19 | `src/vaara/credential/_grant_capability.py` |
| Proof-carrying enforcement at the MCP proxy: an allowed `tools/call` mints a signed, independently recomputable authorization receipt next to the grant, carrying a signed coverage boundary (chokepoint identity, capability-surface fingerprint, scope literal) so an absent refusal reads against a declared scope rather than silence (off by default) | v1.3.0, 2026-06-20 | `src/vaara/integrations/_mcp_attest.py`, `src/vaara/credential/`, `tests/vectors/authorization_v0/` |
| Gap-evident completeness: a signed per-boundary sequence and running count on each authorization receipt make a dropped receipt a provable gap from the held receipts alone, with no issuer access and no external witness (`verify-contiguity`), with public conformance vectors (off by default) | v1.4.0, 2026-06-21 | `src/vaara/credential/_contiguity.py`, `tests/vectors/contiguity_v0/`, `SPEC.md` |
| AP2 checkout binding profile: a post-checkout authorization receipt names the AP2 Payment Evidence Frame it followed by content address (`decisionDerived.evidenceRef.ref` equal to `ap2:checkout/<frame_id>`), with the AP2 task as the `coverage` boundary and gap-evident completeness over the receipts inside it, with public conformance vectors | v1.5.0, 2026-06-21 | `SPEC.md` section 5.4, `tests/vectors/ap2_v0/` |
| Visa TAP request binding profile: a decision receipt names the Trusted Agent Protocol request by content address (`decisionDerived.evidenceRef.digest` equal to `sha256(JCS(request))`) across the action lifecycle, the in-progress and terminal receipts carrying distinct `actionRef` join keys, with the verdict recomputable offline from the held bytes and no live verifier endpoint to trust, with public conformance vectors | v1.6.0, 2026-06-22 | `SPEC.md` section 5.5, `tests/vectors/tap_v0/` |
| Worst-case bound on a gap: the sealing record carries the boundary's highest action class (`maxClass`), so a missing receipt's worst case is read from the held set and the seal alone (surfaced as `worst_case_class`), the issuer committing the max at seal time and the verifier reading it back rather than computing an order over labels; the sealed max buys the honest-issuer trimmed-tail case while under-sealing falls to log reconciliation (off by default) | v1.7.0, 2026-06-22 | `src/vaara/credential/_contiguity.py`, `src/vaara/integrations/crewai.py`, `SPEC.md` section 5.3 |
| Generic external-execution-evidence binding profile: the schema-agnostic binding the named profiles are instances of, one mechanism rather than one per plane; a verifier carrying an `external_execution_evidence` slot (`linked_call_id` / `evidence_hash` / `evidence_type`, the shape used by agentrust trace-spec #34 and cMCP #301) resolves it against a `vaara.receipt/v1` authorization receipt as the recomputable producer (the slot's `evidence_hash` equal to the receipt's `evidenceRef.digest`), with gap-evident completeness over the trace and the verdict recomputable offline through a checker that imports no Vaara, with public conformance vectors | v1.8.0, 2026-06-22 | `SPEC.md` section 5.6, `tests/vectors/external_evidence_v0/` |
| Enforcement-time consumption of a sealed worst-case class: the v1.7.0 seal's `maxClass` gates a chain recipient's own next unattended action, permitting iff the sealed class is a member of a held policy set and failing closed when no class is sealed; a membership test, not an ordering over class labels (the spec computes no order), so a permitted class permits even over a gap because the seal bounds the gap at that class, and the recipient consumes the committed bound rather than re-deriving the chain or querying a log (`enforce_on_sealed_class`, `vaara enforce-by-class`, off by default) | v1.9.0, 2026-06-22 | `src/vaara/credential/_contiguity.py`, `tests/vectors/class_gate_v0/`, `SPEC.md` section 5.3 |
| Sealed `maxClass` bound to the signature at every consume path: the sealed class rides under signature only through `decisionDerived.evidenceRef.digest` equal to `sha256:` + JCS(`evidence`), so a class relabeled after signing leaves the record signature intact but no longer binds and contributes no class, the gate failing closed; the consume paths recompute the binding rather than trusting the record signature alone (`evidence_binding_ok`, `all_evidence_bound`), with a `deny_relabeled` conformance vector proving the relabeling attack raised on cosai-oasis/ws4 #99 is rejected | v1.9.1, 2026-06-23 | `src/vaara/credential/`, `tests/vectors/class_gate_v0/` |
| Universal evidence sink (`vaara.ingest/v0`): a signed envelope that seals any record `normalize` understands (SEP-2643 / 2787 / 2817 or an unrecognized record) content-addressed, with the honest gap report, the established proof fields, and the non-proof context all inside the digested object so editing any of them breaks the signature; a sibling of `vaara.receipt/v1` that asserts nothing the source did not establish (no fabricated verdict, no back-link) and cannot reuse the receipt or authorization envelopes, with a per-stream `seq` and `runningCount` making a dropped record a provable gap; the published conformance corpus is generated by a loop over the normalize input registry and reproduced by a checker that imports no Vaara (`vaara ingest`) | v1.10.0, 2026-06-23 | `src/vaara/attestation/_ingest_emit.py`, `tests/vectors/ingest_v0/`, `SPEC.md` section 6 |
| Declarative source profiles: binding a new source format to the ingest sink is a JSON spec rather than a Python adapter, with the spec interpreter compiled once into the same `SourceProfile` pipeline; `slsa-provenance` (SLSA v1 in-toto provenance statement) and `c2pa-manifest` (C2PA manifest) ship as worked profiles; an independent checker exercises each profile from the spec JSON alone, importing no Vaara, and a drift guard fails the test suite if emit logic changes without regenerating the fixtures | v1.11.0, 2026-06-23 | `src/vaara/attestation/profiles/`, `docs/source-profile-contract.md`, `tests/vectors/ingest_v0/` |
| `agent-decision` source profile and DSSE/Ed25519 conformance vector: the in-toto `agent-decision/v0.1` predicate (proposed in in-toto/attestation#554) is mapped onto the SEP-2828 model by a declarative profile; `args_hash` is an argument commitment rather than a back-link to an attested request; the signature lives in a DSSE envelope as `PAE(DSSEv1, type, payload)` over the JCS-canonical statement bytes, and `paeSha256` binds the PAE hash inside the SEP-2828 receipt; the independent checker (`_check_independent.py`) recomputes the carrier end-to-end from the source statement with no Vaara import | v1.12.0, 2026-06-23 | `SPEC.md`, `tests/vectors/agent_decision_v0/` |
| `acp-checkout` source profile and JCS conformance vector: an ACP checkout session is mapped onto the SEP-2828 model by a declarative profile; ACP objects carry no signature, so the recomputable anchor is a JCS (RFC 8785) content digest over the statement rather than a signature envelope; the profile carries the checkout record but no authorization decision or outcome, and `signs: false` in the profile spec declares it as a no-signature plane; the independent checker recomputes every `jcsSha256` with no Vaara import | v1.13.0, 2026-06-23 | `SPEC.md`, `tests/vectors/acp_checkout_v0/` |
| Independent second-generator re-mint of both carriers: `_remint.py` in each of the `agent_decision_v0` and `acp_checkout_v0` vector sets recomputes its carrier byte-exact from the source statement with no Vaara import — the DSSE/Ed25519 re-mint derives the JCS payload bytes, the DSSE pre-authentication encoding, the deterministic Ed25519 signature (RFC 8032 §5.1.6), and `paeSha256` from scratch; the JCS re-mint recomputes `jcsSha256` over the canonical statement bytes; both re-mints fail closed under a tamper pass (a mutated carrier byte changes the recomputed digest and the assertion fires) | v1.14.0, 2026-06-23 | `tests/vectors/agent_decision_v0/_remint.py`, `tests/vectors/acp_checkout_v0/_remint.py` |
| Reproducible conformance vector set for the CrewAI `GovernanceDecision`/`GovernanceOutcome` contract (crewAI#6030): four stream cases (complete, dropped mid-gap caught by running count, tail drop caught only by the terminal seal, and the irreducible tail-unsealed residual) plus four negative fail-closed cases; six derivation preimages (`params_hash`, `intent_digest`, `intent_ref`, `target_state_digest`, `decision_context_hash`, `receipt_ref`) each `sha256:hex(SHA256(JCS(member_set)))`; `intent_ref` is timestamp-free so the same authorized intent recomputes to the same identity on a retry while `receipt_ref` carries `seq` and `timestamp_ms` making a replayed outcome detectable; records are ES256 (P-256 + SHA-256) over `JCS(record)` with a fixed-scalar test key for stable bytes; independent checker (`_check_independent.py`) reproduces every verdict and all six hashes with no Vaara import | PR #306, 2026-06-25 | `tests/vectors/governance_decision_v0/` |
| SEP-2828 fallback binding path conformance vectors: seven vectors proving the portable-projection property — the projection strips observer-local `_meta` fields (progress tokens, trace IDs, injected correlation headers) and hashes only `{arguments, authBinding?, toolName}` JCS-canonical; `observer_stable_a` and `observer_stable_b` carry different `_meta` sidecars but produce identical `attestationDigest` values, confirming that two honest observers on the same call agree on the content-binding hash regardless of transport metadata; three negative vectors (`neg_different_tool`, `neg_different_args`, `neg_different_auth_binding`) confirm that a change to any projection field changes the digest; `basic_no_auth_binding` is the explicitly defined unauthenticated fallback profile where `authBinding` is absent and the digest covers `{arguments, toolName}` only; independent checker (`_check_independent.py`) reproduces every `attestationDigest` with no Vaara import | PR #309, 2026-06-25 | `conformance/sep2828/fallback_projection_v0/` |
| Track 1 gateway enforcement: `AttestPairEmitter.emit_grant()` mints a short-lived `BrokeredCredential` bound to an attestation digest, scoped to one tool name, an RFC 8785 JCS args commitment, and a tenant ID; `CredentialGateway` verifies the credential before forwarding any `tools/call` to the upstream MCP server, failing closed with MCP error `-32603` when the credential is absent, expired, or scope-mismatched; five independently verifiable `credential_binding_v0` conformance vectors (`pos_valid_grant`, `neg_args_changed`, `neg_expired`, `neg_wrong_tenant`, `neg_no_credential`) cover every gate path; independent checker (`_check_independent.py`) reproduces all five verdicts with no Vaara import, using only `hmac`, `hashlib`, `json`, and `rfc8785` | v1.16.0, 2026-06-25 | `src/vaara/credential/`, `src/vaara/integrations/_mcp_attest.py`, `tests/vectors/credential_binding_v0/` |
| MITRE ATLAS threat-detection conformance vectors: five vectors grounding `vaara.receipt/v1` against named AI agent attack patterns — `pos_clean_execution` (control), `neg_injected_args` (Prompt Injection: args commitment diverges from authorization), `neg_tool_substitution` (Unauthorized Access: actionType changed at runtime), `neg_replay` (Replay: receipt re-presented 120 s after issuance, outside the 60 s freshness window), `neg_scope_escalation` (Privilege Escalation: runtime scope exceeds authorized boundary); each receipt is HMAC-SHA256 over RFC 8785 JCS of all receipt fields; independent checker (`_check_independent.py`) reproduces all five verdicts with no Vaara import | v1.17.0, 2026-06-25 | `tests/vectors/atlas_threat_v0/` |

The `CHANGELOG.md` entry for each version carries the substantive
description and, where relevant, the failure mode that motivated the
change.

## Related published work

The following peer-review and pre-print papers describe approaches in
the same lane as Vaara (runtime evidence, hash-chained or signed audit
trails, conformal calibration, behavioural-constraint monitoring,
safety cases with runtime updates). They are listed here as related
reading, not as competitors. Where the publication post-dates Vaara's
shipped feature for the same idea, that is a chronological fact rather
than a judgment of the work.

### Runtime evidence and behavioural monitoring

- **Protocol-Driven Development: Governing Generated Software Through
  Invariants and Continuous Evidence.** arXiv:2605.12981v2, published
  2026-05-15. Introduces an "Evidence Chain" of compliance for
  generated implementations and a "Dynamic Evidence Ledger" for
  deployed systems, with signed runtime observations appended by
  verifiers. Conceptually adjacent to Vaara's hash-chained audit trail
  with article-explicit evidence (shipped v0.1.0, 2026-04-10) and to
  the per-article `verdict_inputs` and `contributing_events`
  drill-down (shipped v0.26.0, 2026-05-21).
- **Formal Methods Meet LLMs: Auditing, Monitoring, and Intervention
  for Compliance of Advanced AI Systems.** arXiv:2605.16198v1,
  published 2026-05-15. Proposes runtime monitors using Linear
  Temporal Logic for product-specific behavioural constraints, with
  intervening monitors that act at runtime to preempt predicted
  violations. Adjacent to Vaara's policy-driven runtime decisions and
  external-scorer composition (shipped v0.14.0, 2026-05-17).
- **Notarized Agents: Receiver-Attested Confidential Receipts for AI
  Agent Actions.** arXiv:2606.04193v1, published 2026-06-02. Inverts
  the trust boundary so the service receiving an agent's call signs a
  receipt of what it observed, encrypts it to the owner, and publishes
  it to a witness-cosigned Merkle log (the Sello protocol). Same problem
  statement as Vaara (the entity producing the log is the entity being
  logged), with a different placement: the signature is on the receiving
  service rather than at a governance proxy in front of the agent, and
  the root of trust is a transparency log rather than a hardware
  attestation. Situates a wider field (Signet, AgentROA, Agent Passport
  System, draft-farley-acta, SCITT). Vaara's signed execution-receipt
  envelope shipped at v0.42.0 (2026-05-29); its transparency-log
  consistency proofs at v0.54.0 (2026-06-05).
- **A Five-Plane Reference Architecture for Runtime Governance of
  Production AI Agents.** arXiv:2606.12320v1, published 2026-06-10. A
  reasoning plane that adjudicates intent over four enforcement planes
  (network, identity, endpoint, data), with stop-anywhere mediation,
  composite principals whose authority attenuates through delegation
  chains, and audit treated as a structured evidence substrate. Direct
  conceptual neighbour to Vaara's interception pipeline (v0.1.0,
  2026-04-10) and per-article `verdict_inputs` / `contributing_events`
  (v0.26.0, 2026-05-21); it specifies a policy-engine core and
  architecture, where Vaara ships a proxy whose audit substrate is
  cryptographically signed and hardware-bound.

### Safety cases with runtime confidence updates

- **A Subjective Logic-based method for runtime confidence updates in
  safety arguments.** arXiv:2605.22530v1, published 2026-05-21.
  Describes a method for continuously updating static safety cases
  using runtime Safety Performance Indicators, propagating confidence
  through a Subjective Logic assurance case. Adjacent to Vaara's
  evidence-sufficiency framework shipped in `VERDICTS.md` (v0.28.0,
  2026-05-22) and to the conformal interval that ships with every
  Vaara risk score (v0.1.0, 2026-04-10).

### Calibration and external validation

- **Calibration, Uncertainty Communication, and Deployment Readiness
  in CKD Risk Prediction: A Framework Evaluation Study.** arXiv:2605.21566v1,
  published 2026-05-20. Trains five classifiers on the UCI CKD dataset
  (400 patients) and evaluates each across calibration quality, conformal
  prediction coverage, and an eight-criterion deployment readiness
  framework. Reports internal AUROC 1.00 collapsing to 0.48-0.58 on the
  MIMIC-IV external cohort, with split-conformal coverage falling from
  0.80-0.98 internal to 0.21-0.25 against a 90% target. Domain
  incomparable to Vaara, but the methodological lesson (internal test is
  a ceiling, the external gap is visible only against a held-out
  generator) motivates Vaara's v0.36 cross-model held-out corpus
  (`bench/vaara-bench-v0.36.md`).

### Selective inference on conformal prediction sets

- **Selecting Informative Conformal Prediction Sets with an Optimized
  FCR-Controlled Approach.** arXiv:2605.22004v1, published 2026-05-21.
  Formalises selective inference on conformal prediction sets with
  finite-sample false coverage rate guarantees. Methodology pointer for
  Vaara's planned FPR-bounded three-stage combiner (rules-veto in the
  uncertain band), scheduled for v0.37+. Not yet implemented in Vaara.

### Aviation learning-assurance

- **Mechanistic Interpretability for Learning Assurance of a
  Vision-Based Landing System.** arXiv:2605.20607v1, published
  2026-05-20. Applies mechanistic interpretability to an EASA
  learning-assurance scenario, including out-of-model-scope runtime
  monitoring against the operational design domain. Vaara does not
  currently target aviation directly, but EASA learning-assurance is
  in the same harmonisation surface as the AI Act Article 6(1) /
  Annex I product-safety route.

### National security threat-modelling

- **Backchaining Loss of Control Mitigations from Mission-Specific
  Benchmarks in National Security.** arXiv:2605.21095v1, published
  2026-05-20. Methodology for national security deployers to
  back-chain affordance and permission constraints from
  use-case-specific benchmarks. Adjacent in motivation to Vaara's
  policy-driven decisions over agent tool calls, in a deployment
  context Vaara does not currently target.

### Agent-action signing and attestation protocols

- **Attested Tool-Server Admission: A Security Extension to the Model
  Context Protocol.** arXiv:2605.24248v2, published 2026-06-01. Three
  additive mechanisms for MCP: an offline-signed clearance assertion a
  tool server publishes at a well-known URI and a host verifies against
  a pinned trust root; a deny-by-default per-server tool allowlist; and
  a flavor-gated enforcement mode writing every decision to a
  tamper-evident audit log, stated in RFC-2119 form as an MCP addendum.
  Adjacent to Vaara's MCP proxy and signed records: admission-time trust
  of a tool server, where Vaara attests the per-call action at runtime.
- **Signed agent-action receipt and identity proposals in the framework
  repos (2026-06).** A cluster of proposals converging on the primitive
  Vaara shipped earlier: AutoGen Cryptographic Action Receipts
  (microsoft/autogen#7360, Ed25519-signed, SHA-256 input/output hashes,
  offline-verifiable); a minimal Ed25519 + RFC-9421 per-message signing
  extension for A2A (a2aproject/A2A#1829); A2A agent-identity
  verification with AgentCard revocation (a2aproject/A2A#1497); signed
  MCP tool manifests for tool-poisoning / rug-pull defense
  (modelcontextprotocol#2913); and runtime tool-drift detection
  (modelcontextprotocol#2826). Vaara shipped the signed per-action
  execution-receipt envelope at v0.42.0 (2026-05-29) and the DID-bound
  receipt identity with revocation at v0.52.0 to v0.55.0 (2026-06-02 to
  2026-06-05), ahead of these proposals; listing them records the
  convergence, not a competitive claim.
- **Agent Authorization Envelope (AAE).** Kroehl,
  draft-kroehl-agentic-trust-aae-00 (IETF Internet-Draft, 21 May
  2026). A structured authorization container for autonomous agents,
  built from MANDATE, CONSTRAINTS, and VALIDITY blocks bound to W3C
  DIDs and Verifiable Credentials, with SHA-256 delegation-chain
  linking computed over the parent envelope's raw JWS serialization
  (canonicalization explicitly excluded). AAE specifies the
  authorization a gate evaluates against; a Vaara receipt records the
  decision the gate reached and binds it to a recomputable evidence
  record. Adjacent at the grant layer and interoperable: a Vaara
  authorization-decision receipt can name the AAE it evaluated.

### Same-lane open projects

- **Determs (Verifiable Decision Record).** determs.com,
  github.com/determs-com, PyPI `determs`; surfaced 2026-06-07. An open
  spec plus reference implementation for per-action machine-generated
  records (Article 12) whose digests are anchored to public
  infrastructure (Article 19). Anchor-centric: no signing key in the
  trust path, with verification from the record JSON, RFC 8785,
  SHA-256, and public-blockchain anchor data (OpenTimestamps to
  Bitcoin). Vaara's comparable evidence story combines a signature, an
  independent eIDAS-qualified RFC-3161 time anchor (v0.48.0, 2026-05-31),
  transparency-log consistency proofs (v0.54.0, 2026-06-05), and a
  revocation registry (v0.55.0, 2026-06-05).
- **Interlock (getinterlock.dev).** Runtime tool-drift detection at the
  MCP layer (modelcontextprotocol discussion #2826). Notable to this
  document because Interlock has begun recomputing Vaara's published
  SEP-2828 evidenceRef drift vectors, the first independent exercise of
  the keyless conformance verifier shipped at v0.60.0 (2026-06-07).
  Interlock detects drift; Vaara emits the signed, content-addressed
  record the drift check resolves against. Listed as adjacent and
  interoperating.
- **TRACE (Opaque Systems).** Distributed with the open
  `microsoft/agent-governance-toolkit`; surfaced 2026-06-21, announced for
  launch at the Confidential Computing Summit on 2026-06-23. An Entity
  Attestation Token (EAT) governance record for agent actions, with
  completeness scoped to a per-session call count and a planned SCITT
  transparency log. Its own `LIMITATIONS.md` states that the software-only
  tier does not by itself meet EU AI Act Article 12 or DORA, and that
  compliance-grade trust requires a hardware TEE. Vaara's comparable evidence
  is root-agnostic: an Article-12 record is provable with or without a TEE and
  re-expressible as an IETF RATS EAR carrying an AR4SI vector (v0.71.0,
  2026-06-16), externally time-anchored over RFC 3161 / eIDAS (v0.48.0,
  2026-05-31; self-hosted v1.2.0, 2026-06-19), with gap-evident completeness
  verifiable from the held receipts alone (v1.4.0, 2026-06-21). A TRACE
  attestation entry can reference a Vaara receipt as the content-addressed
  artifact its `type` names, so the two interoperate at the record level
  without either format forking. Listed as adjacent and interoperating.

### Classical foundations

- **Conformal prediction.** Vovk, Gammerman, Shafer. *Algorithmic
  Learning in a Random World* (Springer). Vaara implements
  split-conformal prediction with a distribution-free coverage
  guarantee, as documented in `docs/formal_specification.md` and
  explained in plain language in `docs/conformal-prediction.md`.
- **Linear Temporal Logic and runtime verification.** Pnueli (1977),
  Bauer, Leucker, Schallhart (2011). Background for the runtime-monitor
  literature cited above.

## What this document is not

This document is not a competitive matrix. It deliberately omits
vendor comparisons, feature checklists against named peers, and any
"first to ship" claims framed as authority rather than chronology.
Inclusion of a paper here means the work is in the same lane and
worth reading, not that it is positioned as inferior or superior to
Vaara.

For vendor positioning relative to commercial peers, see the
discussion in `COMPLIANCE.md` and the framework integrations under
`src/vaara/integrations/`.

## How to keep this current

When a tagged release adds a load-bearing concept (a new audit
primitive, a new evidence shape, a new public surface, or a new
formal property), add a row to the chronology table above with the
version, date, and a path into the codebase or docs. When a relevant
paper or peer specification appears in the wider literature, add it
to the related-work section with a neutral one-paragraph summary and
the publication date. The goal is a paper trail that a reader can
verify against PyPI, the git tags, and the cited URLs.
