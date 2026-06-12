# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.70.0] - 2026-06-12

The hardware-rooted continuous-governance layer. A signed SEP-2828 execution
record is bound to a TPM 2.0 quote and the kernel's IMA measurement log, and then
to an ordered chain of quotes that shows the measured platform held continuously
across a window. A regulator recomputes every link offline, without trusting the
operator and without a vendor-specific confidential VM. This is the
commodity-hardware floor beneath the existing SEV-SNP enforcement path: the same
governance record, now anchored to whatever silicon a box actually has. The
verifiers and document formats are pure and fully tested with no hardware
present; the capture scripts feed them real evidence on a box with a TPM.

### Added
- `vaara verify-tpm-chain`: Phase 1 of the hardware-governance binding, the
  continuous-attestation loop. Where `verify-tpm-binding` binds one TPM 2.0 quote
  to a record at a point in time, this binds an ordered sequence of quotes over a
  window (the Keylime-style loop): each tick re-quotes the TPM and folds the grown
  IMA log into a hash-linked chain. Each link's `extraData` is
  `SHA-256(jcs(record) || prev_digest || seq)`, so dropping, reordering, or
  splicing a link fails its successor's binding. A `continuous` verdict requires,
  on top of every link passing the Phase-0 check: the TPM clock strictly
  increasing, `resetCount`/`restartCount` constant (no reboot, no unmeasured gap),
  and the IMA log growing append-only across the window. This is what moves
  freshness from `not_established` (a lone quote carries no challenge) to
  `chain_continuity`, an ordered single-boot monotonic-clock window; it is not a
  live verifier challenge, so the chain stays offline-verifiable, and anchoring its
  head and tail to a trusted timestamp is what would bound it to wall-clock. Tiers
  `unverified` / `linked` / `continuous`; the Phase-0 honesty fields carry over
  unchanged (the AK is trusted as supplied, IMA measures files not decisions, so
  `--strict` stays unreachable). New `vaara.tpm-evidence-chain/v0` document and
  `vaara[attestation]` surface: `verify_tpm_chain`, `verify_tpm_chain_bundle`,
  `build_tpm_chain_document`, `bind_record_to_chain_extra_data`, `TPMChainLink`,
  `TPMChainVerdict`. A real-hardware loop-capture script lives at
  `scripts/tpm/capture-tpm-chain.sh`; the verifier is pure and tested
  (`tests/test_tpm_chain.py`) with no hardware present.
- `vaara verify-tpm-binding`: the commodity-hardware, vendor-neutral twin of
  `verify-enforcement` (Phase 0 of the hardware-governance binding). Where the
  SEV-SNP path binds a confidential-VM report, this binds an ordinary **TPM 2.0
  quote** plus the kernel's **IMA** runtime-measurement log to a signed SEP-2828
  record. One `vaara.tpm-evidence-bundle/v0` document carries the record, the
  quote, its AK signature, the quoted PCR values, and the IMA log; the offline
  verifier checks four links a regulator can reproduce without trusting the
  operator: the AK signature over the quote, `extraData == sha256(jcs(record))`,
  the supplied PCR values recomputing the signed `pcrDigest`, and the IMA log
  replaying to the quoted PCR 10. The verdict tiers `unverified` / `bound` /
  `pcr_pinned` mirror the enforcement model, with the same honesty fields: the AK
  is trusted as supplied (its EK chain to a TPM vendor root is the deferred
  `attested` tier, so `--strict` is honestly unreachable), IMA measures files not
  decision semantics (`decision_logic_basis` is always `not_established`), and a
  captured quote carries no verifier challenge so freshness is not asserted. The
  binding preimage is the full record including its signature, closing the same
  signature-malleability gap as the SEV-SNP binding. New `vaara[attestation]`
  surface: `verify_tpm_binding`, `verify_tpm_bundle`, `build_tpm_bundle_document`,
  `bind_record_to_extra_data`, `MockTPMQuoter`. A real-hardware capture script
  lives in `scripts/tpm/` (needs tpm2-tools and TPM access); the verifier and bundle
  format are pure and tested with no hardware present.
- A negative conformance vector for an unsupported `alg` identifier in the
  SEP-2787 attestation set (`neg_unknown_alg`): a well-formed HS256 envelope
  re-labelled `alg: "none"`, the classic algorithm-confusion shape. A conformant
  verifier must refuse it rather than treat the envelope as unsigned. The
  Vaara-free walker reaches `signature_ok` false (no `alg` branch); the Vaara
  reference refuses earlier, at the parse boundary (`alg` not in `VALID_ALGS`).
  This completes the negative coverage promised for this surface alongside the
  receipt replay-substitution case already shipped.

### Fixed
- The TPM quote binding nonce is SHA-256 (32 bytes), not SHA-512. A quote's
  `qualifyingData` is a `TPM2B_DATA` whose buffer ceiling is the TPM's largest
  *implemented* hash, so a 64-byte SHA-512 nonce is rejected `TPM_RC_SIZE` on the
  many fTPMs that cap at SHA-384 (confirmed against an AMD fTPM). SHA-256 is
  mandatory on every TPM 2.0 and fits the slot. SEV-SNP `REPORT_DATA` is a flat
  64-byte field with no such constraint and is unchanged.
- The capture scripts flush stale transient objects and sessions before creating
  the EK/AK. fTPMs implement only a few object slots, so a prior interrupted run
  could leave keys resident and make the next `tpm2_createak` fail
  `TPM_RC_OBJECT_MEMORY`.

## [0.69.0] - 2026-06-11

### Added
- Post-quantum hybrid signing for execution receipts (Track E1). A receipt is a
  durable Article 12 record kept for years, so the live threat is "trust now,
  forge later": its classical ES256 / RS256 signature is forgeable by a future
  quantum adversary who can then backdate a forgery into the retention window. A
  receipt may now carry a parallel **ML-DSA-65 (FIPS 204)** signature over the
  same JCS preimage the classical signature covers; both must verify. The new
  `receiptAsserted.sigSuite` names an allowlisted hybrid suite
  (`ES256+ML-DSA-65` / `RS256+ML-DSA-65`) **inside** the signed preimage, so a
  stripped `pqSignature` is a detectable downgrade rather than a silent loss of
  protection, and the new top-level `pqSignature` block rides outside it. A new
  `pq_verdict` reports a quantum-resistance tier orthogonal to
  verifiable/corroborated: `hybrid-verified` (quantum- and downgrade-resistant),
  `pqc-present` (a valid PQC signature but no committed suite, so strippable),
  `classical-only` (verifiable today, not quantum-resistant), and the fail-closed
  `hybrid-downgraded`. Pre-quantum and classical-only receipts are byte-for-byte
  unchanged and verify exactly as before. ML-DSA public keys ride in the DID
  document as an `AKP` JWK, tracking the JOSE/COSE post-quantum drafts. ML-DSA is
  the pure-Python `dilithium-py`, imported lazily through the `pq` extra
  (`pip install 'vaara[attestation,pq]'`); the base install and every classical
  path stay standard-library. `dilithium-py` is a reference implementation (not
  constant-time, not side-channel-hardened per its own docs): used for the
  independent verifier and the vectors, while production signing of real
  long-lived keys should use a hardened or FIPS-validated ML-DSA behind the same
  signer boundary. v0 reports the quantum-resistance tier through `pq_verdict`
  but does not yet gate `verify_receipt_retained`, the evidence bundle, or the
  conformance `conforms` verdict on it (an E1b follow-on); a committed-downgrade
  record is surfaced as a conformance advisory. New `pq_hybrid_v0` vectors (ten
  cases) plus a Vaara-free independent checker reproduce every verdict. See
  `docs/design/pq-hybrid-signing-spec.md`.
- Execution-receipt parsing now rejects an unrecognized field under any signed
  block (`receipt`, `receiptAsserted`, `backLink`, `outcomeDerived`) instead of
  silently dropping it, so the modeled signing preimage stays byte-exact to the
  wire and a verifier that re-derives the preimage from the model cannot call a
  record signed when injected bytes were never covered by either signature.

## [0.68.0] - 2026-06-09

### Added
- `examples/skills/vaara-governed-tool-call/`: an agent skill (the portable
  `SKILL.md` + script format) that puts an EU AI Act Article 14 human-oversight
  checkpoint and an Article 12 record in front of a high-risk tool call. The
  agent gates a proposed call to allow / escalate / deny, escalations route to
  the `vaara review` queue for a human, and the decision, escalation,
  resolution, and outcome append to a hash-chained trail that exports to a
  signed Article 12 package and verifies offline. Drops in next to capability
  skills (for example clinical or genomic database skills) that ship the action
  but not the oversight.

### Changed
- `vaara.attestation.decision.superseding_decision` now reports an
  equal-`decidedAt` tie between distinct decision records as ambiguous, raising
  `AmbiguousSupersessionError`, instead of breaking it by lowest lexicographic
  `issuerAsserted.nonce`. The nonce is unique per record, not an ordering field,
  so the old rule named a record that was not the genuinely-later decision and
  could mask a producer that emitted two records which should never have tied.
  Byte-identical records still resolve to one decision. A future ordering field
  (a sequence or revision number) restores a deterministic winner. The SEP-2828
  clause and the `supersession_equal_decidedat_tie` vector verdict (`winner:
  d5a` becomes `supersession: ambiguous`) are updated to match. Raised by
  rpelevin on modelcontextprotocol#2852.
- The no-SEP-2787 fallback back-link now binds a **named, versioned projection**
  of the request envelope, not the whole observed `_meta`. The preimage is an
  allowlist: exactly the `tools/call` `name`+`arguments` plus the
  `_meta.authorization_binding` block (the per-call `nonce`), and nothing else.
  Every other `_meta` field is excluded by construction, so a gateway view and a
  provider view of the same call project to the same digest; changing the bound
  params or the binding block breaks the back-link; an absent binding block fails
  closed instead of widening the preimage to the whole `_meta`. The signed record
  names the projection version it used in a new `backLink.fallbackProjection`
  field, so a verifier reconstructs the same projection deterministically from
  trusted data and a later projection revision is an explicit new version rather
  than a silent reinterpretation. `vaara.attestation.decision` adds
  `fallback_projection`, `MalformedFallbackBindingError`, and
  `FALLBACK_PROJECTION_V1`; `request_envelope_digest`,
  `verify_decision_fallback_binding`, and the shared `BackLink` carry the version
  and the fail-closed contract in the reference verifier. The `decision_pairing_v0`
  corpus carries the provider, gateway, replayed, tampered-binding, and
  no-binding envelopes as first-class fixture inputs, and the Vaara-free checker
  reproduces every verdict. Refines the no-2787 fallback shape as it converged on
  modelcontextprotocol#2867.

### Fixed
- `vaara review resolve --audit-db` no longer forks the audit hash chain. It
  was writing the `ESCALATION_RESOLVED` record (Article 14(4)(d)) through a
  fresh `AuditTrail` whose `previous_hash` started empty, so resolving an
  escalation into an audit DB that already held the action's lifecycle broke
  chain continuity and `verify_chain()` flagged the trail. The resolution now
  appends to the loaded trail and continues the chain. Surfaced by the
  `vaara-governed-tool-call` example skill.

## [0.67.0] - 2026-06-09

### Added
- The Article 12 regulator pack now folds in cross-org handoff (Article 26(6))
  and confidential-VM enforcement evidence, and verifies a full directory of
  records in one pass. One deployer command produces one package coherent across
  Article 12 (the signed record-keeping trail), Article 19 (existence in time
  under the eIDAS anchor), and Article 26(6) (the records a deployer hands its
  own regulator), with an optional "where it ran" confidential-VM attestation
  alongside. It is a more complete pack, not a certificate: the eIDAS time
  anchor stays the only un-forgeable component, a handoff is corroborated only
  with a verified anchor and pinned only against a trusted DID document, and an
  enforcement binding is never "attested" (the VCEK chain to AMD's root is not
  validated in this release).
- `vaara trail export-article12 --handoffs <dir> --enforcements <dir>` (also
  `--handoff <pkg.json>`, repeatable): fold verified handoff packages and
  confidential-VM enforcement bindings into the regulator package as sidecars
  under `evidence/`, with the roll-up in `evidence/attestations_summary.json`
  and a report section mapping them to Article 26(6) custody and the
  confidential-VM "where enforcement ran" evidence. Each attachment is verified
  at export; one that does not verify fails the export closed, so the package
  never ships evidence it cannot back. `--trusted-did-document` pins handoff
  producer identity out of band, `--expected-measurement` pins enforcement
  launch images. New `article12_fold_v0` conformance vectors with a Vaara-free
  checker that reproduces every folded verdict from the bytes inside the package.
- `vaara verify-handoffs` and `vaara verify-enforcements`: the set-level forms of
  `verify-handoff` and `verify-enforcement`, for an auditor holding a directory
  of evidence rather than one file. `verify-handoffs` runs the handoff lens over
  every package and rolls up how many records verify under their rotated-out
  keys, how many are anchor-corroborated rather than resting on the signature
  alone, and how many had their producer identity pinned. `verify-enforcements`
  binds a directory of records to their SEV-SNP reports, discovering each triple
  by stem (`NAME.record.json` with `NAME.report.bin` and `NAME.vcek.pem`), and
  rolls up how many bind to a confidential VM, the per-tier tally, and whether
  any pinned a vetted launch image. Each set is `ok` only when every item
  verifies for the chosen mode; a coverage note (no producer pinned, no image
  pinned) is advisory and does not gate, mirroring `verify-bundles`. New
  `check_handoff_set` / `check_enforcement_set` with `handoff_set_v0` and
  `enforcement_set_v0` conformance vectors, each with a Vaara-free checker that
  reproduces the roll-up by composing the single-verb suite's own evaluator.

## [0.66.0] - 2026-06-09

### Added
- `vaara verify-enforcement`: verify that a signed SEP-2828 execution record was
  emitted by an enforcement point running in an AMD SEV-SNP confidential VM. The
  enforcement point, inside the VM, requests a hardware attestation report whose
  64-byte `REPORT_DATA` carries `sha512(jcs(record))`; the verifier checks the
  report parses, its ECDSA-P384 signature verifies against a supplied VCEK, and
  `REPORT_DATA` binds to this exact record. The binding is over the full record
  including its signature, so a report for one record never verifies another and
  a signature-stripped variant never rides a genuine report. The verdict is
  honest about its limits: it does not validate the VCEK chain to AMD's ARK (the
  KDS fetch is deferred), so `vcek_chain_basis` stays `caller_supplied_unverified`
  and a report with no AMD provenance passes the same check; it does not prove
  the decision logic ran in the enclave (`enforcement_logic_basis` is always
  `not_established`). Tiers are `unverified`, `bound`, and `measurement_pinned`
  (with `--expected-measurement` pinning the launch image); `attested` and a
  `--strict` pass are reserved for the chain-rooted future tier and are
  unreachable in v0. New `verify_enforcement` / `bind_record_to_report_data` /
  `EnforcementVerdict` reusing the SEV-SNP primitives in
  `vaara.attestation.tee`, `enforcement_attestation_v0` conformance vectors with
  a Vaara-free checker, and `docs/design/enforcement-attestation-spec.md`.

## [0.65.0] - 2026-06-09

### Added
- `vaara build-handoff` / `vaara verify-handoff`: package one organisation's
  signed execution record so a different organisation's regulator can verify it
  offline, years later, under a key that has since rotated out. A provider
  (vendor A) signs a record; a deployer (customer B) who runs A's system keeps
  the logs (Article 26(6)) and is audited by B's own regulator, with no live
  channel back to A. `build-handoff` stitches the record, the archived DID
  document, the key history, revocations, and an optional eIDAS RFC 3161 anchor
  into one self-contained file, pinning each component by content digest (model
  digests for the key history and revocations, `sha256(jcs(...))` for the rest).
  `verify-handoff` recomputes every digest, routes the record through the same
  rotated-key lens as `verify-retained` (the `verifiable` / `corroborated`
  tiers, unchanged), and confirms an enclosed anchor's imprint is
  `sha256(jcs(record))`, so an anchor taken over a different record never
  corroborates this one. The verdict is honest about trust: content addressing
  proves only that the package is internally consistent, since the holder
  controls both the components and the manifest that pins them. The record's
  authenticity rests on the provider's signature against an identity the
  regulator establishes out of band; `producer_identity_basis` stays
  `self_asserted_unpinned` until `--trusted-did-document` pins it against a key
  set the regulator already trusts. The eIDAS anchor is the one component a
  holder cannot forge. `--strict` passes only a corroborated record with a
  recorded validity window, an affirmative revocation source, and a pinned
  identity. An optional holder custody signature is reported separately and
  never changes the record verdict. New `cross_org_handoff_v0` conformance
  vectors with a Vaara-free checker that reproduces every verdict using only
  `cryptography` and a JSON canonicalizer. Design in
  `docs/design/cross-org-handoff-spec.md`.

### Changed
- `verify-handoff` and the `KeyHistory` / `RevocationRegistry` loaders fail
  closed with a named `ValueError` on a malformed package rather than letting a
  parser exception escape as a traceback, hardening the unhappy path against
  hostile input.

## [0.64.0] - 2026-06-09

### Added
- `vaara verify-retained`: verify an execution record under a signing key that
  has since rotated out, over the Article 12 retention window (the 7-year
  problem). It binds the signature to a key the archived DID document lists,
  then checks the claimed signing time falls inside that key's validity window
  (`validFrom` / `validUntil` on the verification method) and that the key was
  not revoked before issuance. A retired key still verifies a signature it made
  while valid: retirement is graceful end-of-life, distinct from revocation. A
  naive check against the current document fails on genuine old records once a
  key rotates, which over a multi-year window is every key. With a verified
  eIDAS RFC 3161 time anchor the verdict is corroborated: the record provably
  existed before the key's end of life, so it cannot be a later forgery made
  with a stolen retired key. Without an anchor the verdict rests on the
  record's self-asserted time and names that basis. New source-agnostic
  `KeyHistory` validity-window model (built from a DID document, an out-of-band
  list, or directly; carries a canonical digest to pin into a signed export)
  and `verify_receipt_retained`, composing the existing identity, revocation,
  and time-anchor lenses. Ships with the `key_rotation_v0` conformance vectors
  and a Vaara-free independent checker. The window and revocation checks run in
  the base install; binding needs the attestation extra.

## [0.63.0] - 2026-06-09

### Added
- `vaara conformance-statement`: the self-test a producer runs to prove SEP-2828
  conformance against the published corpus instead of asking to be trusted. It
  confirms the corpus bytes match `MANIFEST.json`, re-runs this implementation's
  keyless conformance check over every corpus fixture and confirms it reproduces
  the verdict the corpus records, optionally runs the producer's own records
  through the same set check, and prints one statement that names the exact
  corpus version and corpusDigest it was checked against. Keyless and
  deterministic: no signing key and no clock, so anyone holding the same corpus
  re-runs the command and reaches the same verdict. Ships with the
  `conformance_statement_v0` vectors and a Vaara-free independent checker that
  re-derives every claim and confirms the statement states it faithfully.
  Markdown or `--json`; runs in the base install.

## [0.62.0] - 2026-06-08

### Added
- `vaara normalize`: read an adjacent MCP record and map it onto the SEP-2828
  execution-record evidence model. A SEP-2643 authorization denial becomes a
  refused outcome, a SEP-2787 tool-call attestation becomes the
  decision-attested back-link a conformant receipt must pin, and a SEP-2817
  invocation audit context becomes advisory decision-input. For each record the
  command reports which evidence plane it fills, which SEP-2828 fields it
  populates, and what is still missing for a complete signed record. It promotes
  nothing: an unsigned client claim stays advisory, and when the source flags an
  intent as redacted the cleartext is withheld. SEP-2643 and SEP-2817 run in the
  base install; the SEP-2787 back-link digest needs the attestation extra. Ships
  with the `normalize_v0` vectors (verbatim spec examples plus an
  extension-field attestation) and a Vaara-free independent checker that
  reconstructs the modeled envelope and reproduces every case.

## [0.61.3] - 2026-06-08

### Added
- README: documented the set-level auditor commands shipped in 0.61.0
  (`verify-records`, `verify-bundles`, `audit-summary`), so the evidence
  walkthrough now covers a whole directory of records or bundles, not just a
  single file.

## [0.61.2] - 2026-06-08

### Changed
- Documentation and repository housekeeping: prose tidied across the shipped
  docs, an internal README reference corrected, and an unused log directory
  removed.

## [0.61.1] - 2026-06-08

### Fixed
- `vaara audit-summary` no longer emits an em-dash in the findings line of the
  rendered report; it uses a colon. Cosmetic only: the verdict, the counts, and
  the findings are unchanged, and the page stays byte-deterministic.

### Verification
- Added a standalone checker for the audit-summary golden pages
  (`tests/vectors/audit_summary_v0/_check_independent.py`) that parses each
  rendered page with no Vaara import and asserts the verdict, counts, and
  findings it states equal what the `record_set_v0` conformance vectors
  independently compute. The regulator page is confirmed to state the same
  verdict a neutral party derives from the records, not only to be byte-stable.

## [0.61.0] - 2026-06-08

**Theme: the auditor's workbench. 0.60 made Vaara the neutral checker of one
record. This release adds the receiving side an auditor works from: point it at
a whole directory of records or bundles and get the set-level answer, then read
it as a page a regulator can follow.**

### Added
- `vaara verify-records DIR [--glob] [--json]`: set-level conformance over a
  directory of SEP-2828 records, possibly from more than one emitter. Each
  record is classified (decision or outcome) and checked against its type's
  schema, then the set is checked for cross-record properties: a call recorded
  twice (required), an allow or escalate decision with no matching outcome or an
  outcome with no matching decision (advisory pairing gaps), and an executed
  action that committed no result (the Article 12 hole, advisory). Keyless. New
  `check_record_set` on the public `vaara.attestation.receipt` surface;
  `record_set_v0` vectors with a Vaara-free checker.
- `vaara verify-bundles DIR [--glob] [--json]`: the full lens stack over a
  directory of evidence bundles. Rolls up how many bundles verify, how many
  establish their signature, and per-lens coverage, with an advisory gap naming
  the lenses no bundle in the set exercised. New `check_bundle_set`;
  `bundle_set_v0` vectors reusing the `bundle_doc_v0` documents with a Vaara-free
  checker. Requires the attestation extra.
- `vaara audit-summary DIR [--glob] [--out FILE]`: the human-readable face of
  `verify-records`. Renders the verdict, the record counts, and the findings as
  a one-page Markdown report a regulator reads, deterministic and keyless (no
  timestamp, no signing key). New `render_record_set_summary`;
  `audit_summary_v0` golden pages.

## [0.60.0] - 2026-06-07

**Theme: verify anyone's record, not just your own. The trust plane could
already issue and check Vaara's own evidence end to end. This release turns
Vaara into the neutral checker of the format itself: a keyless conformance
check that any party can run on any SEP-2828 execution record, including one
Vaara did not produce.**

### Added
- `vaara verify-record PATH [--attestation ATT.json] [--json]`: a
  producer-agnostic conformance check on a candidate SEP-2828 execution
  record. It is keyless by design. It validates the wire schema (required
  fields and types, supported `alg`, valid `status`, `sha256:<hex>` digest
  formats, `receiptAsserted.alg` matching the envelope) and the one binding a
  record proves about itself: `resultCommitment.projectionDigest` equals the
  SHA-256 of the projection bytes beside it, recomputable from the record alone
  with nothing but a hash function. With `--attestation` the back-link to the
  request the record answers is verified too, still without a key. The
  signature check (which needs the signer's key) stays in `vaara receipt
  verify`. Exit 0 iff the record conforms.
- `check_record_conformance` on the public `vaara.attestation.receipt` surface,
  returning a `ConformanceReport` that lists every check with its severity
  (`required` gates conformance; `advisory` is surfaced but does not). The
  conformance module is pure standard library, so it runs in the base install
  with no extras.
- `record_conformance_v0` conformance vectors: ten records (conforming,
  non-conforming with one isolated failure each, and an advisory case) plus a
  stdlib-only `_check_independent.py` that re-implements the rules with no Vaara
  import, the same second-implementation discipline as the other vectors.

### Security
- Bumped `gradio` in the Hugging Face Space example
  (`examples/huggingface-space/requirements.txt`) from `>=5.0` to `>=6.16.0`,
  clearing 10 OSV advisories (path traversal, SSRF, CORS bypass, open redirect)
  that OpenSSF Scorecard flagged. Example-only: `gradio` is not a dependency of
  the Vaara package or the published wheel, so package users were never exposed.

## [0.59.0] - 2026-06-07

**Theme: the regulator package now proves when, not just what. The Article 12
export already produced a signed, self-describing record-keeping package. This
release folds an external RFC 3161 time anchor over the signed trail head into
that package, so a regulator can confirm the logs existed at a point in time
independently of the operator's signing key. Pinned to an eIDAS-qualified
Time-Stamp Authority, that timestamp is the form an EU regulator already
recognises, which is the Article 19 existence-in-time property.**

### Added
- `vaara trail export-article12 --anchor-tsa URL` (or `--anchor-file PATH`):
  fold an external RFC 3161 time anchor over the signed trail head into the
  Article 12 package as `time_anchor.json`. The anchor binds to the trail it
  ships with; the export verifies the chain-head position and the RFC 3161
  token over that exact `record_hash` before writing it, so a package never
  claims an anchor it cannot back. The report gains an "External time anchor
  (Article 19)" section and an `art19_logs` obligation row, and
  `article12_summary.json` carries a `time_anchor` block.
- `vaara trail verify-anchor --zip PACKAGE`: verify that anchor offline. It
  checks the anchored `chain_head_hash` equals the last `record_hash` in
  `trail.jsonl` and that the RFC 3161 token verifies. Pin the TSA certificate
  to enforce an eIDAS-qualified authority.

### Changed
- Dropped the superseded classifier bundles `adversarial_classifier_v1`,
  `v2`, `v3`, `v5`, `v6`, `v7` from the repository (about 5.5 MB). The
  published wheel already shipped only the production bundle (`v9`), so this is
  a repository-size cleanup with no effect on the package. `v8` stays in tree
  because the cross-evaluation scripts (`eval_v039_v9.py`,
  `eval_v039_bipia.py`, `eval_v038_phase1.py`) compare against it; `v9` is the
  runtime bundle. `eval_pipeline_attribution.py` now defaults to `v9`. Historic
  per-version numbers remain recorded in `bench/`.

### Fixed
- `vaara build-bundle`: a receipt that is valid JSON but the wrong shape (for
  example missing a required field) now exits 1 with `cannot assemble bundle:
  ...` instead of letting an `AttestationError` traceback escape. The handler
  around `build_bundle_document` caught only `ValueError`; the validation path
  can also raise `AttestationError`, `KeyError`, or `TypeError`, so it now
  catches all four, matching the other attestation CLI commands.

## [0.58.0] - 2026-06-05

**Theme: one command for the party producing evidence, the mirror of
`verify-bundle`. v0.57.0 gave a verifier one command to check an evidence
bundle on disk. This release gives the issuer one command to produce it.
`vaara build-bundle` assembles the receipt plus whatever identity, signature,
back-link, inclusion, consistency, and revocation material the issuer holds
into the single document `verify-bundle` reads, byte for byte. Producing and
checking evidence become one closed loop over one file.**

### Added
- `vaara build-bundle`: assemble a complete evidence bundle on disk from the
  issuer's pieces. Supply them with `--from-dir DIR` (each piece in a
  conventional file: `receipt.json`, `attestation.json`, `did_document.json`,
  `verifying_jwk.json`, `inclusion.json`, `consistency.json`, `registry.json`,
  plus the scalars `expected_keyid.txt` and `inclusion_leaf_hex.txt`) or with
  explicit per-piece flags. Writes the bundle to `--out` or stdout, then loads
  it back and reports the `verify-bundle` verdict. A malformed piece fails the
  build with the offending field named.
- `vaara.attestation.build_bundle_document`: the issuer-side mirror of
  `evidence_bundle_from_json`. Stitches the pieces into the one on-disk
  document and validates it by loading it straight back, so a bad piece is
  caught at assembly time. The output is byte-for-byte the shape the
  `bundle_doc_v0` vectors commit and `verify-bundle` reads.
- `vaara.attestation.load_bundle_pieces_from_dir`: discover an issuer's pieces
  in a directory by the conventional file names, returning the kwargs
  `build_bundle_document` takes.
- `tests/vectors/build_bundle_v0/`: each evidence bundle split into the
  issuer's separate pieces, with a Vaara-free independent checker that
  re-assembles them, asserts the result is byte-for-byte the document the
  verifier reads, and reproduces the verdict, with no Vaara import.

### Notes
- Purely additive over v0.57.0. No change to the receipt envelope or any
  canonicalization; `build-bundle` assembles the existing on-disk shape and
  defines no new wire format. Envelope version stays 1. Round-trip property:
  a bundle from `build-bundle`, fed to `verify-bundle`, verifies. 1308 passed.

## [0.57.0] - 2026-06-05

**Theme: one command to verify an evidence bundle from disk. v0.56.0 added
`verify_evidence_bundle`, one call that runs every applicable lens over an
in-memory bundle and returns one verdict. But to check evidence someone handed
you, those objects first have to be rebuilt from files. This release adds the
on-disk bundle format and the `vaara verify-bundle` command, so a regulator
handed a single file runs one step and gets the verdict, without writing any
code or trusting the issuer's tooling.**

### Added
- `vaara verify-bundle PATH`: verify a complete evidence bundle in one command.
  PATH is a bundle JSON file, or a directory holding `bundle.json`. The command
  runs every lens whose evidence is present (identity, signature, back-link,
  inclusion, consistency, revocation), prints a per-lens summary, and exits 0
  only when the bundle is `ok`. `--json` emits the full verdict.
- `vaara.attestation.evidence_bundle_from_json`: load an `EvidenceBundle` from
  its on-disk JSON document. The shape is exactly the `bundle` object the
  `evidence_bundle_v0` vectors already commit, so every committed vector is a
  valid bundle file. The loader is strict on what is present: a malformed block
  raises `ValueError` naming the field, rather than dropping a lens and
  returning a falsely narrow verdict.
- `BundleVerdict.to_dict` and `LensResult.to_dict` for JSON output.
- `tests/vectors/bundle_doc_v0/`: the eight evidence-bundle cases as standalone
  on-disk files, with a Vaara-free independent checker that reads each file on
  its own and reproduces its verdict, so a single bundle file is verifiable
  without Vaara.

### Notes
- Purely additive over v0.56.0. No change to the receipt envelope or any
  canonicalization; the loader parses the existing bundle shape off disk and
  defines no new wire format. Envelope version stays 1. 1278 passed.
- The verifiable trust plane, the 0.6 milestone, is shipped end to end: the
  receipt, each verification lens, the one-call verdict, and now a command a
  non-developer can run against a file.

## [0.56.0] - 2026-06-05

**Theme: one call to verify an evidence bundle. The 0.52 to 0.55 line built
six verification lenses, each on its own call: resolvable identity, the
receipt signature, the back-link to the request attestation, transparency-log
inclusion, append-only consistency, and cross-stack revocation. A consumer
holding a full bundle had to invoke all six, track which applied, and combine
the answers, and the combination has a sharp edge: proving a receipt is in a
log and not revoked says nothing about who issued it unless the signature was
also checked. This release adds `verify_evidence_bundle`, a single entrypoint
that runs every applicable lens and returns one verdict with that edge
enforced.**

### Added
- `vaara.attestation.verify_evidence_bundle` and `EvidenceBundle`: one
  receipt plus whatever evidence the holder has. Each evidence field feeds one
  lens; a field left unset makes that lens not applicable, so a partial bundle
  is verified for what it carries rather than rejected for what it lacks.
- `BundleVerdict` and `LensResult`: the verdict reports each lens (applicable,
  ok, reason), whether authenticity was established, the identity-resolved
  keyid, and one overall `ok`. `verdict.lens(name)` looks up a single result.
- Fail-closed authenticity. `ok` is true only when the receipt signature was
  established (the identity lens bound it to a document key, or the signature
  lens verified it under supplied key material) and every applicable lens
  passed. A receipt that is merely included in a log, with no signature ever
  checked, is not `ok`: an unauthenticated record proves nothing about who
  issued it.
- Lens coupling. Identity runs first so the keyid it resolves sharpens the
  revocation lens: a key-scope revocation bites under a resolved identity and
  correctly cannot match when the receipt is verified by signature alone.
- `evidence_bundle_v0` conformance vectors: eight bundles covering every
  outcome (all lenses passing, each lens failing in turn, signature-only
  authenticity, and the fail-closed included-but-unauthenticated case), with a
  Vaara-free independent checker that reproduces every verdict.

Purely additive over 0.55; the envelope version stays 1. The entrypoint
composes the existing lens functions unchanged and issues no new crypto.
1255 passed, 13 skipped, ruff clean, mypy clean on the gated set. See
`docs/design/evidence-bundle-spec.md`.

## [0.55.0] - 2026-06-05

**Theme: cross-stack revocation. Revocation used to live in exactly one
place, the level-3 live identity check: a signing key revoked at or before a
receipt was issued no longer yielded a trusted verdict. The same receipt
checked through the receipt verifier, the transparency log, or an Article-12
export ignored revocation entirely, so one receipt could get three different
answers. This release lifts the revocation-in-time rule into a
source-agnostic `RevocationRegistry` that every lens consults, so a receipt
whose issuer was revoked-in-time fails consistently whichever lens looks.**

### Added
- `vaara.attestation.RevocationRegistry`: a set of revocation facts with one
  predicate, `status(iss, issued_at, keyid=None)`. A receipt is
  revoked-in-time iff a matching entry (identity-scope on the issuer, or
  key-scope on the bound keyid) has a `revoked_at` at or before issuance.
  Build it from a DID document (`from_did_document`, the same data level 3
  reads, so the two agree by construction), from an operator's out-of-band
  list, or from a dict. `digest()` gives a stable `sha256:` over the RFC 8785
  canonical bytes.
- `RevocationEntry` and `RevocationStatus`: the fact and the verdict.
  `RevocationStatus` reports `revoked`, `matched_by`, `revoked_at`, and
  `issued_at`, so a verifier with a stronger time anchor than the receipt's
  self-asserted `iat` can re-decide.
- `vaara.attestation.check_receipt_revocation`: the receipt-verifier lens,
  the offline counterpart of the level-3 rule, with no DID fetch.
- `vaara.attestation.verify_logged_receipt` and `LoggedReceiptVerdict`: the
  transparency-log lens, checking inclusion and revocation in one call. `ok`
  is true only when the receipt is both included and not revoked-in-time.
- `export_signed(..., revocation=registry)`: the Article-12 export lens. The
  registry's canonical bytes are written to a `revocation.json` member and
  its digest is pinned into the signed manifest
  (`revocation.registry_sha256`), so the revocation state at export time is
  part of the tamper-evident bundle. The standalone
  `scripts/verify_vaara_trail.py` checks the member against the pinned digest.
- A `cross_stack_revocation_v0` conformance vector set (one receipt, four
  registries) asserting the receipt-verifier, transparency-log, and
  export-digest lenses reach the same verdict, with a Vaara-free,
  standard-library-plus-`rfc8785` independent checker that reproduces it.

### Changed
- The level-3 live identity check (`verify_receipt_identity_live`) now applies
  revocation through the shared `revoked_in_time` rule rather than its own
  copy, so the identity lens and the registry cannot drift. No behavior
  change; the existing `agent_identity_v0` verdicts are unchanged.
- Purely additive over 0.54. No change to the receipt envelope,
  canonicalization, inclusion- or consistency-proof formats, or signature
  verification; the envelope version stays 1. `export_signed` with no
  `revocation` argument produces a byte-identical manifest to 0.54.

## [0.54.0] - 2026-06-05

**Theme: append-only consistency proofs for the transparency log. The log
already issued inclusion proofs: evidence that a given receipt is in the log.
An inclusion proof says nothing about whether earlier history was rewritten.
A consistency proof does: it shows the log at an earlier size is a verifiable
prefix of the log at a later size, so a fork or a quiet rewrite of past
entries is detectable even when every individual inclusion proof still
verifies. A monitor that pins the log's root over time and checks a
consistency proof between successive roots gets the append-only guarantee a
transparency log exists to provide. RFC 9162 (RFC 6962-bis) section 2.1.4.**

### Added
- `vaara.attestation.verify_consistency`: verifies an RFC 9162 consistency
  proof between two tree sizes and their roots, returning whether the smaller
  tree is a verifiable prefix of the larger one. Recomputes both roots from
  the proof; a single returned `bool`.
- `ConsistencyProof`: the proof object (`first_size`, `second_size`, and the
  ordered sibling `hashes`), shaped to match what a sigstore Rekor-backed
  adapter would expose at the same call site.
- `InProcessTransparencyLog.consistency_proof(first_size, second_size)`:
  produces a proof between any two sizes the log has reached, empty for the
  trivial cases (an empty prefix, or identical sizes).
- `InProcessTransparencyLog.root_at(tree_size)`: the Merkle root over the
  first `tree_size` leaves, for pinning a historical signed tree head before
  requesting a consistency proof against a later one.
- A `transparency_consistency_v0` conformance vector set (nine cases over a
  twelve-leaf log, covering power-of-two and non-power-of-two prefixes plus a
  tampered-proof and a forked-history negative) with a Vaara-free,
  standard-library-only independent checker that reproduces every verdict.

### Changed
- Purely additive over 0.53. No change to the receipt envelope,
  canonicalization, inclusion-proof format, or signature verification; the
  envelope version stays 1.

## [0.53.0] - 2026-06-03

**Theme: resolvable agent identity, level 3. Level 2 confirmed a receipt was
signed by a key a DID document lists, given the document. Level 3 fetches that
document over HTTPS at audit time, records the resolution so it is
reproducible, and adds revocation in time: a signing key revoked at or before
the receipt was issued no longer yields a trusted verdict, even when the
signature still verifies. A key revoked afterwards still binds, because
revocation is not retroactive.**

### Added
- `vaara.attestation.verify_receipt_identity_live`: level-3 check that
  resolves a `did:web` issuer over HTTPS (or from cache), runs the level-2
  binding, then applies deactivation and revocation-in-time. Returns
  `LiveIdentityResult` with a single `trusted` verdict plus `issued_at` and
  `revoked_at`, so a verifier holding a stronger time anchor than the
  self-asserted `iat` (the audit-trail hash chain) can re-decide.
- `ResolutionMeta`: an auditable record of each resolve (`did`, `url`,
  `fetched_at`, `document_digest` over the exact bytes, `from_cache`).
- `DidDocumentCache`: in-memory TTL cache keyed by DID, caller-supplied clock
  so it is deterministic under test.
- `https_fetch`: the default size-capped, HTTPS-only DID-document fetcher.
  Redirects are refused (an SSRF vector); deployers inject their own fetcher
  for allowlisting, pinning, or proxy egress, or to verify offline against a
  captured document.
- A third `agent_identity_v0` conformance vector, `revoked.json` (bound but
  not trusted, key revoked before issuance), with the Vaara-free independent
  checker extended to reproduce the revocation verdict offline. `expected.json`
  now carries `revoked` and `trusted` per case.

### Changed
- Purely additive over 0.52. The receipt envelope, canonicalization, and the
  level-2 verifier are unchanged; existing receipts and the `bound`/`unbound`
  vectors verify exactly as before.

## [0.52.0] - 2026-06-02

**Theme: resolvable agent identity (level 2). A receipt no longer just proves it
was signed by some key; when its `receiptAsserted.iss` is a `did:web` identity it
can prove the signing key is one the named agent actually publishes. The check is
pinned-resolvable: the verifier supplies the DID document, so the property holds
offline with no network fetch. It composes over the unchanged signature verifier,
so every existing receipt and vector verifies byte for byte.**

### Added
- `vaara.attestation.verify_receipt_identity`: confirms a receipt whose
  `receiptAsserted.iss` is a `did:web` identity was signed by a key listed in the
  supplied DID document. Runs after, and on top of, ordinary signature
  verification. `ES256`/`RS256` resolve against the document; `HS256` (symmetric)
  never does, since a shared secret cannot bind a public identity.
- `agent_identity_v0` conformance vector family (`bound.json`, `unbound.json`,
  `expected.json`) plus a Vaara-free independent checker
  (`_check_independent.py`), so an external party reproduces the bound/unbound
  verdicts from the committed wire bytes with no Vaara import.
- Design spec: `docs/design/resolvable-agent-identity-spec.md`.

### Changed
- Purely additive over 0.51. The wire `version` is unchanged; receipts without a
  `did:web` issuer verify exactly as before. Level 3 (live HTTPS DID resolution
  with caching and revocation) is the planned follow-up.

## [0.51.0] - 2026-06-02

**Theme: SEP-2828 Check B. A decision and an outcome now pair on two checks, not
one. Instance binding (the shared attestation back-link) stays the anchor, and
the outcome record additionally commits to a digest of the exact decision it ran
under. Instance binding alone could not tell which decision a call answered when
several shared an attestation, for example an `escalate` and the human verdict
that superseded it; the content digest closes that gap.**

### Added
- `outcomeDerived.decisionDigest` on the execution receipt: `sha256:<hex>` over
  the full signed decision-record wire bytes the outcome was produced under. The
  field is additive (wire `version` stays `1`); a v0.51 emitter sets it and
  pairing fails without it.
- `vaara.attestation.decision.decision_digest`: the Check B digest input.
- `vaara.attestation.decision.superseding_decision`: resolves the effective
  decision among records sharing a back-link (latest `decidedAt`, ties broken by
  lowest lexicographic `issuerAsserted.nonce`, so every verifier agrees with no
  clock authority).
- A seventh conformance vector, `substituted_decision_under_shared_attestation`,
  isolating the case Check A cannot catch, plus the resolved supersession-tie
  winner. The standard-library walker reproduces both from the wire bytes.

### Changed
- `records_paired` now requires both Check A (instance anchor) and Check B
  (outcome-to-decision digest). A receipt with no `decisionDigest` does not pair.

## [0.50.0] - 2026-06-01

**Theme: the verifiable evidence plane. Trail exports can be threshold-signed so
no single custodian can issue or forge them, and custodian changes are pinned in
the chain. Execution receipts serialize losslessly as W3C Verifiable Credentials
for interoperability without a second trust surface. One command turns a trail
into a signed, self-explaining EU AI Act Article 12 regulator package.**

### Added
- `vaara.audit.export.export_signed_threshold` and the `vaara trail
  export-threshold` CLI: k-of-n threshold signing for audit exports. n
  custodians each hold an independent Ed25519 key; the export carries one
  detached signature per available custodian and verification requires at
  least `threshold_k` valid signatures from the authorized set. No single
  key-holder can issue or forge a trail. `threshold_k`, `signers_n`, and the
  authorized fingerprint set are written inside the signed manifest, so the
  quorum cannot be downgraded without invalidating every member signature.
  Each stored public key is bound to its manifest fingerprint, so a
  substituted key is rejected and an unauthorized extra signature is ignored
  rather than counted. The standalone verifier
  (`scripts/verify_vaara_trail.py`) verifies threshold exports with only the
  `cryptography` package. See `docs/design/threshold-signing-spec.md`.
- **Chain-anchored key-lifecycle markers** (the second half of threshold
  signing). `AuditTrail.record_key_lifecycle(action, fingerprint, ...)` records
  custodian rotation, revocation, and addition as ordinary audit records, so
  they inherit the v0.47 hash chain and the v0.48 external time anchor. A
  `revoked` marker anchored before a compromise window pins the revocation in
  time: a compromised key can re-sign but cannot re-anchor past chain heads.
  `verify_signed` now returns any lifecycle records found
  (`VerifyResult.key_lifecycle`), and the standalone verifier prints them, so a
  reviewer sees the custodian set's history inline with the evidence. New
  `EventType.KEY_LIFECYCLE` maps to EU AI Act Articles 15(1) and 12(1).
- **W3C Verifiable Credential serialization for execution receipts (opt-in).**
  `vaara.attestation.receipt.receipt_to_vc` / `receipt_from_vc` present an
  `ExecutionReceipt` as a VCDM 2.0 credential and recover it losslessly
  (`receipt_from_vc(receipt_to_vc(r)) == r`). The VC is a view, not a second
  trust surface: `credentialSubject` carries the native receipt blocks verbatim,
  `proof.proofValue` carries the existing SEP-2787 detached signature, and
  verification routes through the unchanged `verify_receipt_signature` stack. No
  new crypto, no re-canonicalization. The `@context` is vendored
  (`load_receipt_context`), so verification needs no network.
- **Article 12 one-command regulator export.** `vaara trail export-article12`
  (and `vaara.audit.article12_export.export_article12`) writes a signed
  evidence zip plus a generated, human-readable report that maps the trail to
  the EU AI Act Article 12 / 26(5) record-keeping obligations: a cover from
  operator `--system-meta`, an obligation table driven by the event types
  present and the chain's `regulatory_articles` tags, an event inventory, an
  integrity statement, and verify instructions a regulator runs without a
  Vaara install. It composes the existing signed export (single-signer or
  k-of-n threshold), it does not duplicate it. The report is built from the
  signed trail bytes and bound to them by the manifest `trail_sha256`; it is a
  derived view, not a second signed surface, and the package says so. `--period`
  is a report lens that narrows the summary counts only; the signed trail stays
  whole. `--format md|html`. See `docs/design/article12-export-spec.md`.

### Fixed
- `scripts/verify_vaara_trail.py` hash-chain re-check now binds `tenant_id`
  and `chain_version` for chain v2 records (v0.47+), matching the in-package
  verifier. The standalone script previously failed every v0.47+ trail.

## [0.49.0] - 2026-05-31

**Theme: decision records. The evidence chain now covers the full call lifecycle:
policy verdict before execution (decision record), attestation at request time
(SEP-2787), and outcome after execution (execution receipt). A verifier can prove
the governing server committed to allow-or-block before the side effect ran.**

### Added
- `vaara.attestation.decision` module (`pip install 'vaara[attestation]'`).
  `emit_decision_record` signs a `DecisionRecord` envelope that binds the
  governing server's policy verdict and risk basis to the SEP-2787 attestation
  via a digest back-link, before the tool call executes. Verification is two
  composable checks: `verify_decision_signature` (crypto) and
  `verify_decision_back_link` (binding to the attestation instance).
  `records_paired` joins a decision record to its execution receipt.
  Canonicalization and signing (HS256 / ES256 / RS256) reuse the same path as
  the receipt and SEP-2787 modules; no new crypto is required.
- `AuditTrail.enable_auto_anchor(client, *, every_records)` for automatic
  cadence anchoring. Once enabled the trail anchors its own chain head every N
  records without a manual call. Fail-open: a failed anchor attempt writes a
  chained `ANCHOR_GAP` marker so the gap is auditable and the trail continues.
- Negative test vector `neg_replay_substituted_field`: verifies that replaying a
  receipt with any field substituted fails verification.

### Fixed
- MCP manifest `description` fields trimmed to the 100-character registry cap
  (both `server.json` and `server-vaara-server.json`).

## [0.48.0] - 2026-05-31

**Theme: external time anchoring. The audit chain head can now be timestamped by
a third-party authority, so the chain's existence is provable against an external
clock even if the signing key is later compromised. This is the anti-backdating
property the server-side signed execution-record SEP relies on.**

### Added
- External time anchoring for the audit hash chain (`vaara.audit.timeanchor`, new
  `timeanchor` extra). `AuditTrail.anchor_head(client)` takes the current chain
  head (already a SHA-256 digest) and obtains an RFC 3161 trusted timestamp over
  it from an external Time-Stamp Authority. RFC 3161 underpins eIDAS qualified
  electronic timestamps, so a qualified TSA makes this regulator-grade evidence
  under EU AI Act Article 12. The token is verified on receipt and kept as a
  `TimeAnchor`; verification is offline (`verify_anchor`,
  `verify_anchor_over_records`) and binds the anchor to a specific record, so a
  rewritten chain or a token over a different digest is rejected. The HTTP round
  trip uses the standard library; only the ASN.1 and signature checks need the
  extra. See `docs/sep/sep-server-execution-record.md`.

### Changed
- Public framing leads with EU AI Act runtime evidence and data sovereignty (runs
  in your own environment, no SaaS, no telemetry) across the README, package
  descriptions, MCP manifests, and vaara.io. The tamper-evident receipt stays the
  mechanism, not the headline.

## [0.47.0] - 2026-05-31

**Theme: tenant isolation across the evidence path. The reference server can no
longer leak one tenant's audit chain to another, and tenant identity is now part
of the tamper-evident hash chain itself.**

### Security
- Tenant identity is now bound into the audit hash chain. `AuditRecord` carries
  a `chain_version`; records written from this release on (chain v2) fold
  `tenant_id` into `compute_hash`, so re-attributing a record to another tenant
  after the fact breaks `verify_chain()` instead of passing silently. Pre-v0.47
  records (chain v1) keep `tenant_id` out of the hash and re-verify byte for
  byte, so existing trails and signed exports stay valid. The SQLite store gains
  a `chain_version` column (schema v4) with a migration defaulting legacy rows to
  v1. The standalone verifier mirrors the same rule.
- The reference HTTP server's audit-chain read (`GET /v1/audit/actions/{id}/chain`)
  is now tenant-scoped: a caller can no longer read another tenant's action chain
  by guessing an `action_id`. Unknown and cross-tenant actions both return 404
  with an identical body, so the response is not an existence oracle. The scoped
  read also resolves chain positions in one pass, removing an O(n^2) lookup.
- SSE notification broadcast is now tenant-scoped: upstream-pushed notifications
  on a shared upstream no longer fan out across tenants. Unattributable log
  notifications (no progressToken) broadcast only within a single tenant scope.

## [0.46.0] - 2026-05-31

**Theme: multi-tenant runtime governance, made real. A hardening release that
makes the concurrent-multi-tenant claim true and safe, with no new features.**

### Security
- SEP-2787 attestation verification now rejects a future-dated `iat`. The TTL
  check had only an upper bound (`iat + exp_seconds + clock_skew`), so an
  attestation stamped with an issuance time in the future kept its validity
  window open indefinitely. Verification now also enforces the lower bound
  `now >= iat - clock_skew_seconds`, tolerating only the configured skew of
  forward drift. The conformance test set gains explicit future-dated cases.

### Fixed
- Race in the audit trail's action-to-tenant map under concurrent writers.
  `record_action_requested` mutated the map (length check, eviction, insert)
  and `_tenant_for` read it without a lock, so concurrent multi-tenant traffic
  could raise `dictionary changed size during iteration` during eviction or
  hand one lifecycle another tenant's scope. The map now has a dedicated lock,
  separate from the hash-chain lock. New tests run 16 tenants through full
  lifecycles concurrently and assert chain integrity plus per-tenant scope.

### Changed
- Wheel slimmed from ~8MB to ~0.8MB. Only the production classifier bundle
  (`adversarial_classifier_v9.joblib`) is loaded at runtime, but the wheel was
  shipping all of v1-v8 (~7MB of dead weight) via the default
  `include-package-data` file finder. Packaging now ships only the v9 bundle;
  the older bundles stay in the repo for bench and cross-eval.

### CI / tooling
- mypy now runs in CI as a build-failing gate on the strict module set
  (`vaara.policy.*`), pinned to mypy 1.20.2. Fixed the one outstanding
  `no-any-return` in `policy/modes.py`.
- `.gitignore` now covers SQLite WAL sidecar files (`*.db-wal`, `*.db-shm`,
  `*.db-journal`).
- `scripts/RELEASE.md` step 3 corrected to match `release_merge_and_tag.sh`,
  which tags `origin/main` directly rather than checking out and pulling main.

### Bench
- `bench/vaara-bench-v0.46.md`: concurrency and governance-overhead evidence.
  Per-call governance overhead is sub-2ms p50 and flat across 1-8 upstream
  fan-out; raw numbers in `bench/v046_fanout.json`.

## [0.45.1] - 2026-05-30

**Theme: audit-finding fixes on the remote HTTP connector, the HTTP transport, and the public numbers.**

### Security
- SSRF egress floor on the `--upstream-url` connector. The remote HTTP connector
  handed a user-supplied upstream URL straight to `urllib` and followed
  redirects with the static `Authorization` header attached, so a hostile or
  compromised upstream (or an attacker controlling a redirect target) could aim
  the proxy at the cloud instance-metadata service or an internal host and have
  it fetch the target with the operator's bearer token. The new `_egress_guard`
  resolves the host and refuses loopback, link-local, RFC1918, IPv6 ULA, and the
  cloud-metadata address (including its dotless and IPv4-mapped encodings) before
  any socket opens; a guarded opener caps redirects, re-applies the floor to each
  hop, and drops the auth header on a cross-origin redirect. Default is SAFE; a
  trusted internal upstream is opted in via `--allow-private-upstream-hosts`,
  the `allow_private_hosts` constructor arg, or the
  `VAARA_MCP_ALLOW_PRIVATE_UPSTREAM` env flag. The metadata address stays refused
  even with the opt-in.
- DNS-rebind closure on that egress floor. Resolving the host and then handing
  the name back to `urllib` left a gap: `urllib` re-resolved at socket-connect,
  so a name that answered with a public address at the check and a blocked one a
  moment later (a time-split rebind) reached the blocked target with the auth
  header attached. The connector now validates and pins the address at connect
  time and dials the IP literal, so the address that passed the floor is the
  exact address the socket reaches; HTTPS still verifies the certificate against
  the original hostname. The pin is re-applied on every redirect hop. An absent
  `--allow-private-upstream-hosts` flag now leaves the
  `VAARA_MCP_ALLOW_PRIVATE_UPSTREAM` env opt-in live instead of silently
  shadowing it with a `False`.
- Egress opt-in narrowed to the private classes only. The opt-in previously
  bypassed the whole floor, so trusting internal hosts also re-opened
  `0.0.0.0`, multicast, and reserved ranges. The never-routable classes
  (cloud-metadata, unspecified, reserved, multicast) are now refused regardless
  of the opt-in; only loopback, link-local, and private (RFC1918 and IPv6 ULA)
  are relaxed by it.

### Fixed
- HTTP transport no longer serialises concurrent requests. The POST `/mcp`
  endpoint ran the blocking `_handle_request` inline on the event loop, so one
  slow upstream stalled every other POST, SSE drain, and `/health` (real
  concurrency 1). It now runs on a worker thread via `asyncio.to_thread`, with
  the per-request ContextVars preserved across the hop through
  `contextvars.copy_context()`. The JSON-RPC notification branch is offloaded
  the same way, so a slow upstream `notify()` no longer parks the event loop
  either.
- `vaara trail receipt`, `compliance dashboard`, and `compliance report` leaked
  the SQLite audit connection: each opened `SQLiteAuditBackend` and never closed
  it, locking the DB file under in-process invocation. All three are now
  context-managed.
- SSE reconnect race that dropped notifications for the live session. On
  reconnect under the same `Mcp-Session-Id`, the old stream's teardown
  unregistered the NEW session. `unregister_session` is now identity-checked and
  only removes the entry when it is still the tearing-down stream's own state.
- README mislabelled the rule-scorer latency as classifier latency. The
  140 µs / 210 µs figure is the hot-path rule scorer; the MiniLM classifier is
  opt-in (`vaara[ml]`) and not in that path. Also surfaces the cross-model
  held-out recall (66.8%) and its weakest sub-cell (38.9%) the bench docs
  already disclose.
- `llms.txt` advertised a two-generations-stale classifier (5,955-entry corpus,
  97.1% at threshold 0.55). Regenerated from the current v9 numbers and switched
  the lede to the tamper-evident runtime evidence framing.

## [0.45.0] - 2026-05-30

**Theme: reach remote MCP upstreams over HTTP, and make the proxy's Streamable HTTP handling conform to the spec.**

### Added
- `--upstream-url NAME=URL` on `vaara-mcp-proxy`: front a remote MCP server
  over Streamable HTTP instead of a local stdio subprocess. A bare
  `--upstream-url URL` lands under the `default` slot. The connector speaks the
  2025-03-26 and 2025-06-18 protocol revisions: it POSTs JSON-RPC and reads
  either `application/json` or `text/event-stream` replies, captures and echoes
  the `Mcp-Session-Id`, sends the negotiated `MCP-Protocol-Version`, and holds a
  standing GET SSE channel for server-initiated notifications with
  `Last-Event-ID` resume and bounded reconnect. Built on the standard-library
  `urllib` only, so the zero-dependency core is preserved (httpx is not a
  dependency; only fastapi and uvicorn ship behind the `server` extra). The
  deprecated 2024-11-05 two-endpoint transport and interactive OAuth are out of
  scope; remote auth is static-header only.
- `--upstream-header NAME=HEADER` on `vaara-mcp-proxy`: attach a static request
  header such as a bearer token to a URL upstream. The header name splits on the
  first `=`, so a base64 token's trailing `=` survives. Startup rejects headers
  aimed at an unknown slot and stdio/url slot-name collisions.
- In-repo SEP-2787 attestation conformance vectors at
  `tests/vectors/sep2787_attestation_v0/`: pinned HS256, ES256, and RS256 keys
  and six cases (`hs256_digest_identity`, `es256_projection_identity`,
  `rs256_signature_ttl_only`, `neg_bad_signature`, `neg_expired`,
  `neg_args_mismatch`) spanning the signature, TTL, and args-commitment
  dimensions, with a standard-library-only independent checker that imports no
  Vaara code, a generator script, and a pytest cross-check against the library
  verifier. `docs/sep2787-conformance.md` now points at these in-repo vectors
  rather than a planned follow-up.

### Fixed
- Streamable HTTP conformance in the proxy's HTTP transport. The `Mcp-Session-Id`
  is now validated as visible ASCII (0x21 to 0x7E) on both POST and GET alongside
  the existing 128-character cap. The `MCP-Protocol-Version` header is read and
  validated against the supported set (`2025-03-26`, `2025-06-18`); an absent
  header is treated as `2025-03-26`, an unsupported value returns 400. The POST
  `Accept` header must offer both `application/json` and `text/event-stream`; the
  check is wildcard-aware, so `*/*` and an absent header still pass and existing
  clients are not broken, and a violation returns 406.

## [0.44.0] - 2026-05-30

**Theme: a runnable reference verifier. Generate the attestation key, then verify attestations and receipts offline from the command line.**

### Added
- `vaara keygen --attest --out PATH`: generate an EC P-256 (ES256) keypair
  for SEP-2787 attestation signing, compatible with `vaara-mcp-proxy
  --attest-signing-key`. Writes a PKCS8 PEM private key (0600) and a
  SubjectPublicKeyInfo PEM public key. Prints the `secretVersion` (first 8 hex
  of SHA-256 over the public-key DER) that appears in every attestation the
  key signs, so a signed envelope can be matched to the keypair without the
  private key. Replaces the documented `openssl ecparam | pkcs8` pipe. Unlike
  the Ed25519 trail-signing keygen it does not gate on `--dev`, since this is
  the documented operator path for the proxy.
- `vaara attest verify ENVELOPE.json (--pubkey-file PUB.pem |
  --hs256-secret-file SECRET) [--enforce-ttl]`: verify a SEP-2787 attestation
  envelope's signature and report whether its TTL has expired. TTL is reported
  but not enforced by default, because a saved attestation is durable evidence
  and its short TTL is normally long past at verification time; `--enforce-ttl`
  makes expiry a failure. Emits a JSON verdict and exits non-zero on a failed
  signature.
- `vaara receipt verify RECEIPT.json --attestation ATT.json (--pubkey-file
  PUB.pem | --hs256-secret-file SECRET) [--result RESULT.json]`: verify an
  execution receipt against its attestation in three composable checks: the
  receipt signature, the attestation signature (TTL ignored), and the
  `backLink` binding the two. When the receipt carries a result commitment,
  `--result` verifies it against the runtime result object. Emits a JSON
  verdict and exits non-zero if any check fails.
- `docs/sep2787-conformance.md`: the conformance surface the verifier commands
  cover, keyed to the spec revision Vaara aligns to (`dd030d5b`, tag
  `sep2787-ref-v2`), with the verification steps Vaara checks versus the ones
  that remain the runtime's responsibility (nonce replay, tool-call match).
- 18 tests in `tests/test_cli_attest_receipt_verify.py` covering ES256 and
  HS256 paths, wrong-key rejection, alg/material mismatch, TTL reporting and
  enforcement, back-link failure, result-commitment match and mismatch, and a
  `keygen --attest` to `attest verify` round-trip.

### Changed
- The PyPI summary now leads with the runtime-evidence framing
  ("Tamper-evident runtime evidence layer for AI agents: ..."), aligning the
  package metadata with the README hero and vaara.io.

## [0.43.0] - 2026-05-29

**Theme: proxy pairing -- SEP-2787 request attestation and execution receipt emitted per tools/call.**

### Added
- `src/vaara/integrations/_mcp_attest.py`: `AttestPairEmitter`, the paired
  SEP-2787 attestation and execution-receipt emitter for the MCP proxy. Each
  allowed `tools/call` writes two JSON files to a configurable receipts
  directory: `{counter}-{nonce[:8]}-attest.json` (request attestation) and
  `{counter}-{nonce[:8]}-receipt.json` (execution receipt). The pair is
  cryptographically linked: the receipt carries a `backLink` digest over the
  full attestation wire bytes, so a verifier can confirm they belong together.
- `--attest-signing-key PATH` and `--attest-receipts-dir DIR` flags on
  `vaara-mcp-proxy`. Off by default. Key type is auto-detected: EC P-256 PEM
  uses ES256, RSA PEM uses RS256, raw bytes file uses HS256. For ES256 and
  RS256 a `pubkey.pem` is written to the receipts directory so external
  verifiers need only the public key.
- `serverFingerprint` in each attestation starts as a SHA-256 of the upstream
  command string (`cmd:sha256:{hex}`) and upgrades to a SHA-256 of the
  canonical JSON of the tools list (`manifest:sha256:{hex}`) on the first
  `tools/list` response, binding the exact capability set the proxy presented
  to the agent.
- `X-Vaara-Intent` HTTP request header: operators can supply a richer intent
  label per call. stdio transport falls back to the derived
  `tools/call/{tool_name}` string.
- `issuerAsserted.iss` is always `"vaara-mcp-proxy"`. `sub` is
  `"{tenant_id}/{upstream_name}"` when a tenant is set, else
  `"{upstream_name}"`. Reuses the SEP-2787 and receipt signing stack unchanged
  (HS256 / ES256 / RS256, RFC 8785 JCS canonicalization): a verifier that
  already checks SEP-2787 signatures needs no new crypto for the paired
  receipts.
- 17 tests in `tests/test_integrations_mcp_proxy_attest.py` covering pairing,
  SEP-2787 signature verification, back-link integrity, manifest fingerprint
  upgrade, intent override via ContextVar, errored-receipt pairing when the
  upstream raises, and `AttestConfigError` handling.

## [0.42.0] - 2026-05-29

**Theme: execution receipts, the post-execution sibling of SEP-2787.**

### Added
- `vaara.attestation.receipt`: a signed execution-receipt envelope that
  binds the outcome of one attested `tools/call` and links back to the
  SEP-2787 request attestation it answers. SEP-2787 attests the request
  before it runs; the receipt covers the deferred half, what happened to
  it. The two together give end-to-end accountability for one action.
- Three-block envelope (`backLink`, `receiptAsserted`, `outcomeDerived`)
  plus a signature, reusing the SEP-2787 canonicalization (RFC 8785 JCS)
  and signing stack (HS256 / ES256 / RS256) unchanged. A verifier that
  already checks SEP-2787 signatures needs no new crypto for receipts.
- `backLink` pins the attestation by nonce and by a digest over its full
  wire bytes. `outcomeDerived` carries the status (`executed` /
  `refused` / `errored`), completion time, and an optional result
  commitment that reuses the SEP-2787 argument-commitment shapes via
  `make_result_digest` (payload stays local) and `make_result_projection`.
- Three composable verification checks: `verify_receipt_signature`,
  `verify_back_link`, and the existing `verify_args_commitment` for the
  result binding. A receipt is a durable record rather than a
  time-bounded capability, so there is no TTL.
- v0 conformance vectors under `tests/vectors/execution_receipt_v0/`
  with pinned keys, five cases (positive across all three algorithms,
  refused-without-commitment, result-mismatch, broken back-link), and a
  stdlib-only independent walker that verifies them without importing
  Vaara.
- `docs/execution-receipts.md` documents the format, emission,
  verification, and the relationship to OVERT 1.0 Part 3.

### Fixed
- `AdversarialClassifier.score` now applies a deterministic floor for
  cloud instance-metadata endpoints (AWS IMDS, GCP metadata server, ECS
  task-role). The v9 model underweighted a bare `http_post` to these
  known credential-theft destinations, scoring them below the calibrated
  threshold. The floor lifts them above it regardless of the model
  output, defense-in-depth on top of the learned score. The match list
  also covers the parser-confusion encodings that slip a literal-string
  check: the IPv6 link-local address AWS serves IMDS on, and the dotless
  32-bit decimal and hex forms of the IMDS IPv4 address, each bounded so
  it cannot fire on a longer digit or hex run. The list is intentionally
  not exhaustive: IMDSv2 token flows, DNS rebinding, and arbitrary octal
  or mixed encodings still fall back to the model. The supportable claim
  is that Vaara flags the well-known cloud instance-metadata endpoints.

### Notes
- Library surface only this release. Pairing both the SEP-2787 request
  attestation and the receipt into the MCP proxy emission path is the
  next increment; a receipt without its attestation persisted is not a
  coherent artifact, so the attestation side lands properly rather than
  bolted on.

## [0.41.0] - 2026-05-28

**Theme: server-initiated notifications and cancellation routing under fan-out.**

### Added
- `GET /mcp` Server-Sent Events endpoint on the Streamable HTTP
  transport. Upstream-initiated notifications (progress events, log
  messages) reach the client over a long-lived SSE stream keyed by
  the standard `Mcp-Session-Id` header. The endpoint requires the
  session header, refuses values longer than 128 characters, returns
  `404` for unknown `X-Vaara-Upstream` values, and returns `400`
  when fan-out is configured and the upstream header is missing
  instead of guessing the slot.
- Reconnect-with-resume via `Last-Event-ID`. The server keeps a
  bounded replay buffer of the most recent 100 events per session
  and replays any events newer than the resumption cursor when the
  client reconnects. Event ids are strictly monotonic per session,
  so a gap from eviction is observable on the client.
- 15-second SSE heartbeat (`: keepalive` comment) and a 5-second
  `retry:` hint so EventSource clients reconnect on a fixed cadence
  after transient socket loss.
- `notifications/cancelled` routing across fan-out. The proxy tracks
  every in-flight `tools/call` request id against the upstream
  serving it and forwards a client cancellation to that upstream
  regardless of what `X-Vaara-Upstream` the cancel POST carries.
  Under fan-out the cancel reaches the subprocess that owns the
  long-running call rather than the slot named by the cancel header.
- New `vaara.integrations._mcp_notify` module: `NotificationRouter`
  protocol with `StdioRouter` (writes through the proxy's stdout
  lock) and `HttpRouter` (per-session asyncio queues, bounded replay
  buffer, broadcast across an upstream's sessions when no session
  scope is known). The proxy holds one router instance for the
  lifetime of the serve loop and delivers every upstream-initiated
  notification through it.

### Changed
- `VaaraMCPProxy._inflight_progress` now carries the originating
  `Mcp-Session-Id` alongside action id, agent id, tool name, and
  tenant. The audit and OVERT paths discard the session id so the
  perimeter event schema stays unchanged.
- `UpstreamMCPClient.on_notification` is wrapped per upstream so the
  reader-thread callback carries the upstream's name. Closes a
  late-binding bug that would have pinned every reader thread to
  the last name in the `_upstreams` dict.
- `run_http` split into `_build_http_app` + uvicorn launch so tests
  drive the FastAPI app via `fastapi.testclient.TestClient` without
  standing up a server.
- `server.json`, `server-vaara-server.json`, `pyproject.toml`,
  `src/vaara/__init__.py`, and `clients/ts/package.json` version
  fields bumped to 0.41.0.
- `.claude-plugin/marketplace.json` `ref` bumped from `v0.40.4` to
  `v0.41.0`.

## [0.40.4] - 2026-05-28

**Theme: policy mode presets + plugin shakedown fix delivery.**

### Added
- `vaara mode` CLI subcommand with three actions:
  - `vaara mode list` prints the four built-in preset operating points
    (`eco`, `balanced`, `performance`, `strict`) with their thresholds
    and one-line descriptions.
  - `vaara mode show NAME` prints thresholds, description, and watt
    profile for a single preset.
  - `vaara mode emit NAME [--format json|yaml] [--output PATH]` emits
    a minimal valid Vaara policy document for the chosen preset,
    ready for the deployer to add action classes, sequences, and
    escalation routes. Output round-trips through
    `vaara.policy.from_dict`, `from_json`, and `from_yaml`.
- New `vaara.policy.modes` module exposing `Mode`, `available_modes`,
  `get_mode`, `to_policy_dict`, `emit_json`, and `emit_yaml`. Presets
  are shaped like CPU power profiles: `eco` (0.40 / 0.60) cuts agent
  loops short on borderline risk, `balanced` (0.55 / 0.85) is the
  default, `performance` (0.70 / 0.92) is for high-throughput
  pipelines with tight action-class overrides, `strict` (0.30 / 0.55)
  escalates on doubt for incident response and audit prep.

### Changed
- `.claude-plugin/marketplace.json` `ref` bumped from `v0.40.3` to
  `v0.40.4`. Delivers the session_start audit-DB-creation fix from
  PR #161 to marketplace users.
- `plugins/claude-code-vaara-governance/.claude-plugin/plugin.json`
  bumped from `0.1.0` to `0.1.1`. Picks up the session_start fix.
- `server.json` and `server-vaara-server.json` version fields bumped
  to 0.40.4.

## [0.40.3] - 2026-05-28

**Theme: registry completion + supply-chain cleanup.**

### Added
- `mcp-name: io.github.vaaraio/vaara-server` ownership marker in
  `README.md`, alongside the existing `mcp-name:
  io.github.vaaraio/vaara` marker. Required by the MCP Registry to
  verify that the PyPI `vaara` package owns both the proxy and the
  standalone-server slots before publishing to either. v0.40.2 added
  the `server-vaara-server.json` manifest but the marker was missing,
  so the second-slot registry submission could not complete. This
  release ships the marker so the second slot publishes cleanly.

### Changed
- `server.json` and `server-vaara-server.json` version fields bumped
  to 0.40.3.
- `.claude-plugin/marketplace.json` `ref` bumped from `v0.40.2` to
  `v0.40.3`. Closes the self-reference loop where v0.40.2's
  marketplace.json pinned to its own pre-fix tag.

## [0.40.2] - 2026-05-28

**Theme: vaara-mcp-server packaging + fan-out latency numbers.**

### Added
- `vaara-mcp-server` console script in `[project.scripts]`, wired to
  the existing `vaara.integrations.mcp_server:main` entry point. The
  standalone MCP server has shipped in the package for several
  releases but only ran via `python -m vaara.integrations.mcp_server`.
  It now installs as a console script alongside `vaara-mcp-proxy` and
  exposes the `vaara_check`, `vaara_intercept`, and
  `vaara_report_outcome` tools plus the `vaara://status` and
  `vaara://compliance` resources to any MCP host.
- `bench/latency_fanout.py` and `bench/v040_fanout.json`. Measures
  Vaara's per-call overhead on the v0.40 streamable-HTTP transport
  across N upstream slots. Result: 1.2 ms mean, 1.4 ms p99, flat from
  N=1 to N=8. Closes the honest-scope caveat from the v0.40 PR body.
  The upstream subprocess is mocked at the `UpstreamMCPClient` boundary
  so the number isolates added governance cost (HTTP parse, tenant and
  upstream header resolution, `Pipeline.intercept`, dispatch).
- `server-vaara-server.json` for a second MCP Registry listing under
  `io.github.vaaraio/vaara-server`, alongside the existing
  `io.github.vaaraio/vaara` proxy entry. Two separate registry slots
  for two separate MCP servers: the proxy that wraps upstream MCP
  servers, and the standalone server that exposes Vaara governance as
  MCP tools.
- `.claude-plugin/marketplace.json` at the repo root. Registers a local
  Claude Code plugin marketplace so `claude plugin install
  vaara-governance@vaara` works after `claude plugin marketplace add
  /path/to/vaara`. Same file doubles as the seed for a future
  submission to `anthropics/claude-plugins-official`.

### Changed
- `server.json` version bumped to 0.40.2 to track the PyPI package.

## [0.40.1] - 2026-05-28

**Theme: registry-prep + README trim.**

### Added
- `server.json` at the repo root, conformant to the
  `2025-12-11/server.schema.json` schema. Registers Vaara as
  `io.github.vaaraio/vaara` against the official MCP Registry, points
  at the PyPI `vaara` package, and describes the `vaara-mcp-proxy`
  stdio invocation through `uvx --from vaara vaara-mcp-proxy
  --upstream NAME=CMD`.
- `mcp-name: io.github.vaaraio/vaara` ownership marker (HTML comment)
  in `README.md`, required by the MCP Registry to verify the PyPI
  package belongs to the publishing GitHub identity.
- `mcpName: io.github.vaaraio/vaara` in `clients/ts/package.json`,
  the npm-side equivalent of the same ownership check.

### Changed
- README trimmed for shape. Buyer-persona "Who reaches for Vaara"
  section removed. Per-article verdict drill-down details collapsible
  inlined as a single paragraph. Upstream-signal-adapter intro no
  longer enumerates AI Act article numbers (the mapping table in
  COMPLIANCE.md is the authority). Composite-scorer aside removed
  from the HTTP API section. MCP-proxy OVERT collapsible removed
  (the OVERT 1.0 attestation section directly below covers the same
  ground). MCP-proxy streaming-notifications collapsible compressed
  to one line. OVERT trailing IAP / S3P / TEE paragraphs collapsed
  into one. Vendor-namedrop tail after the worked examples removed.

## [0.40.0] - 2026-05-28

**Theme: deployment shape. One Vaara process now serves a fleet of
upstream MCP servers, with multi-tenant policy, audit, and attestation
on the same substrate.**

The v0.39 sidecar shape ran one Vaara process per upstream. v0.40
turns that into a single process that speaks Streamable HTTP, holds
N upstream MCP-server connections, picks the upstream per request
from a header, scopes every score, audit record, and OVERT envelope
to a tenant, and reloads per-tenant policy in place.

### Added
- `vaara-mcp-proxy --transport http --http-host H --http-port P`:
  Streamable HTTP transport at `POST /mcp`, backed by FastAPI /
  uvicorn (the `vaara[server]` extra already shipped in v0.39 for
  `vaara serve`). The endpoint reads `X-Vaara-Tenant` and
  `X-Vaara-Upstream` per request, pushes them into ContextVars, and
  dispatches into the existing `_handle_request` path so the policy,
  perimeter, OVERT, and progress-notification handling all light up
  unchanged. Notifications (no JSON-RPC `id`) return 202 Accepted.
  Bodies above 1 MiB return 413.
- `vaara-mcp-proxy --upstream NAME=CMD` (repeatable) for fan-out.
  One Vaara process holds N `UpstreamMCPClient` instances in a name
  -> client map. Bare `--upstream CMD` keeps the v0.39 single-
  upstream contract; it lands in the "default" slot. Commands that
  themselves contain `=` (e.g. `python -m foo --bar=baz`) stay
  intact because the name-side regex only matches short alphanumeric
  slugs. When more than one upstream is configured, a request with
  no `X-Vaara-Upstream` header returns 400 with the list of valid
  slots; silent fallback to whichever slot won the sort would be a
  failure mode that surfaces only in production. Single-upstream
  deployments keep the silent-default contract.
- `tenant_id` is first-class through the request, decision, audit,
  and attestation layers:
  - `ScoreRequest`, `AuditEventRequest`, and `PolicyReloadRequest`
    accept a `tenant_id` body field, with `X-Vaara-Tenant` as the
    HTTP-header alternative. Body wins over header.
  - `AuditRecord` gains a `tenant_id` field, excluded from
    `compute_hash()` so pre-v0.40 chains still re-verify on load.
  - `AuditTrail` keeps an `action_id -> tenant_id` map seeded by
    `record_action_requested`, so every follow-up record
    (`risk_scored`, `decision`, `execution`, `escalation`,
    `outcome`, `policy_override`) inherits the same scope without
    every caller threading `tenant_id` through every signature.
    The map is soft-capped (50k entries, 12.5% eviction on
    pressure) so long-running deployments cannot leak memory.
  - `SQLiteAuditBackend.write_record` prefers the per-record
    `tenant_id` when set, with the instance-scoped `tenant_id`
    (legacy CLI tooling path) as fallback. A single backend
    instance can now serve a multi-tenant runtime.
  - OVERT envelopes carry `tenant_id` as a `non_content_metadata`
    claim when present.
- `vaara.policy.registry.PolicyRegistry`: one `PolicyController` per
  tenant, with the empty string slot reserved as the default
  fallback for unmatched lookups.
- `vaara serve --policy-dir DIR`: loads one YAML/JSON policy per
  file. Filename stem = `tenant_id`; `default.yaml` lands in the
  fallback slot. Mutually exclusive with `--policy`.
- `POST /v1/policy/reload` accepts a `tenant_id` body field (or
  `X-Vaara-Tenant` header) and routes to the right registry slot;
  creates the slot on first reload.

### Changed
- `Pipeline.intercept` takes a `tenant_id` keyword that flows onto
  the `ActionRequest` and into the audit trail. Default `""` keeps
  the v0.39 single-tenant contract.
- `AdaptiveScorer.evaluate` dispatches allow / deny thresholds per
  tenant at call time. A new `policy_lookup` constructor arg (and
  `set_policy_lookup` setter for late binding from `ServerState`)
  takes a `Callable[[str], Optional[Policy]]`; on every evaluate, the
  scorer asks the registry for the calling tenant's policy and uses
  its thresholds. An unknown or unmapped tenant falls back to the
  scorer-bound defaults that the default-slot listener keeps fresh on
  reload. The backend decision dict surfaces the applied
  `threshold_allow` and `threshold_deny` so operators can confirm
  which tenant's policy ran. MWU expert state, the conformal
  calibrator, agent profiles, and sequence patterns stay shared
  across tenants; only threshold application is per-tenant in v0.40.

### Scope notes
- The HTTP transport on the proxy is POST-only. GET-SSE for
  server-initiated notifications (sampling, server-pushed progress)
  is v0.41. The audit + OVERT emission path for upstream-originated
  notifications still works unchanged on stdio.
- Classifier bundle and conformal-calibrator hot-reload remain a
  restart operation in v0.40. Per-tenant policy reload IS hot; that
  is the configuration plane that needed to be live across tenants.
  Classifier reload waits on a shared singleton lifecycle plus
  per-tenant scoping question (v0.41 candidate).

## [0.39.2] - 2026-05-27

**Theme: SEP-2787 envelope v2 shape, full wire round-trip, versioned
audit-event schema, and Qi-survey coverage mapping.**

The four mechanical alignments Vaara committed to in
`modelcontextprotocol/modelcontextprotocol#2787` after the
trust-surface grouping was incorporated into the SEP draft on
soup-oss commit `dd030d5b` ship as the v2 envelope shape:

1. `toolCalls` lives under `payloadDerived`, not `plannerDeclared`.
   Tool bindings (name, server fingerprint, args commitment) are
   facts derived from the request payload, not planner declarations.
2. `argsProjection` serialises with a JSON-stringified `projection`
   field carrying the JCS-canonical encoding of the projection
   object. The digest is taken over those bytes.
3. The v1 `kind`-discriminated union is dropped. `ArgsRef` (ref +
   digest) and `ArgsProjection` (projection + projectionDigest)
   self-discriminate by which fields are present.
4. Commitment-only audit composes on `ArgsProjection` as a
   hash-only-identity projection of the form `{"digest":
   "sha256:..."}`. No separate `ArgsDigest` type ships in the spec.

### Added
- `parse_attestation(d)` (and `sep2787_parse_attestation` from the
  package root): inverse of `Attestation.to_dict()`. Reconstructs the
  Python dataclass tree from a wire JSON dict so third-party
  consumers of the v0 test vectors can parse, verify, and re-emit
  envelopes byte-identically. Strict field-presence validation and
  alg allowlisting on the boundary.
- `docs/audit_event_schema.md`: AUDIT-EVENT-SCHEMA-1.0, versioned
  wire/storage contract for the audit events Vaara emits.
  Independent of code version so third-party consumers can pin
  without coupling to a Python runtime version.
- `docs/qi_survey_mapping.md`: Vaara surface coverage against the
  taxonomy in Qi et al., *Towards Trustworthy Agentic AI*
  (arXiv:2605.23989, 2026-05-17). Direct, partial, and
  out-of-scope rows by Perceive / Plan / Act / Reflect / Learn /
  Multi-agent / Long-horizon stage under both top-level dimensions.
- `tests/test_attestation_sep2787_wire.py`: 13 wire round-trip tests
  covering `emit -> JCS bytes -> parse -> verify` across HS256,
  ES256, RS256 for both `ArgsRef` and `ArgsProjection`, plus parse
  rejection on missing-field and unsupported-alg inputs and a
  byte-identical re-emit check.

### Changed
- `vaara.attestation.sep2787` emits the v2 envelope shape.
- `docs/sep2787-overt-mapping.md` field table updated to v2.
- `COMPLIANCE.md` "Position relative to open runtime-attestation
  standards" gains a SEP-2787 v2 subsection alongside the OVERT 1.0
  positioning.
- `vaara.attestation` package docstring covers both OVERT 1.0 and
  SEP-2787 v2 surfaces (previously OVERT-only by omission).

### SEP-2787 reference implementation tag
- `sep2787-ref-v2`: v2 envelope shape with the four post-soup-oss
  alignments and full wire round-trip. Pinned for cross-repo
  provenance citation against
  `modelcontextprotocol/modelcontextprotocol#2787` and the v0 test
  vector PR (`#2789`).
- `sep2787-ref-v1` (preserved at commit `a61e87c`): camelCase
  envelope, the prior proposed-shape artefact.
- `sep2787-ref-v0` (preserved at commit `3d7af54`): snake_case
  envelope, the historical proposed-shape artefact.

## [0.39.1] - 2026-05-27

**Theme: SEP-2787 reference impl follows the spec into camelCase.**
soup-oss adopted MCP camelCase convention for SEP-2787 envelope field
names in `modelcontextprotocol/modelcontextprotocol@48c739b1`. Vaara's
proposed-shape reference implementation now emits camelCase JSON keys
on the serialisation boundary while keeping Python-side attributes in
snake_case.

### Changed
- `Attestation.to_dict()` and the JCS-canonical signing payload emit
  `plannerDeclared`, `issuerAsserted`, `payloadDerived`, `toolCalls`,
  `serverFingerprint`, `secretVersion`, `expSeconds`,
  `requestedCapability`, `projectionDigest`. The proposed-shape
  reference now matches MCP camelCase convention; Python dataclass
  attributes stay snake_case so user code is unchanged.
- `docs/sep2787-overt-mapping.md` field-by-field table updated to the
  camelCase shape.
- New `issuer_to_dict` helper in `vaara.attestation._sep2787_types`
  replaces the prior `asdict()` call so the issuer block sorts and
  renames deterministically without leaking Python-internal names.

### SEP-2787 reference implementation tag
- `sep2787-ref-v1`: camelCase envelope. Pinned for cross-repo
  provenance citation against
  `modelcontextprotocol/modelcontextprotocol#2787` and the v0 test
  vector PR (`#2789`, head `2a9360f`).
- `sep2787-ref-v0` (preserved at commit `3d7af54`): snake_case envelope,
  the historical proposed-shape artefact.

## [0.39.0] - 2026-05-27

**Theme: v9 classifier retrain on a BIPIA-augmented corpus, with one
auto-labeller bug found and fixed in tree.** The v0.38 BIPIA pilot at
n=200 on `claude-haiku-4-5` flagged 50.5% of benign tool calls under
`context.source=injected_via_bipia_<class>`. v0.39 scales the
measurement to four agent backends, exposes a 30 to 50% v8
false-positive rate on benign-under-pressure tool calls, and ships v9
to close it. v9 shares v8's feature schema and XGBoost
hyperparameters. The training diff is the BIPIA-derived rows added to
TRAIN with `sample_weight=8.0` on true-follow rows. Under BIPIA
pressure, v9 FPR drops to 1.2% [0.4, 3.6] across the four backends.
In-distribution recall moves within Wilson intervals (v035 TEST -1.9
pp inside the 95% CI, Phase 1 PAIR -0.3 pp). v9 calibrates to T=0.9150
against v035 VAL at a 5% FPR target.

Auto-labeller bug, found and fixed: the v0.38 pilot's
`looks_like_injection_follow` rule fired on any `http_request` URL
containing `example.com`, regardless of injection content. Agents use
`example.com` as a benign placeholder hostname. The audit script
counted 67% of pre-fix flagged follows as placeholder false positives.
After fix: 42 follows across four backends drop to 14. `claude-haiku-4-5`
produced zero true injection-follows under the post-fix rule across
the full BIPIA run.

LLMail-Inject corpus and the IPI fourth attacker family carry to
v0.40. The v0.40 follow-recall headline lands on top of LLMail-Inject's
larger labelled-positive pool.

### Added
- `src/vaara/data/adversarial_classifier_v9.joblib` (sha256
  `2566da22`): v9 production bundle, calibrated T=0.9150, MiniLM rev
  `c9745ed1d9f2`.
- `tests/adversarial/v039_split.json` (sha256 `73e7730d`): v0.39 split
  manifest covering v8-corpus carry-over plus 1,181 BIPIA-derived
  rows.
- `tests/adversarial/v039_bipia/bipia-s43-{haiku,llama31-8b,mistral-7b,qwen25-7b}.jsonl`:
  post-relabel BIPIA-derived training entries, one file per backend.
- `tests/adversarial/external/bipia/`: unmodified Microsoft BIPIA
  release ([github.com/microsoft/BIPIA](https://github.com/microsoft/BIPIA), MIT
  licensed), vendored for eval reproducibility.
- `scripts/_v039_common.py`: shared `BipiaTrial`, `HarnessTrace`,
  `ToolCallEmit`, tool catalog, JSONL read/write.
- `scripts/run_v039_bipia.py`: BIPIA driver with two provider
  adapters (Anthropic SDK, OpenAI-compatible for Ollama / vLLM /
  llama.cpp).
- `scripts/build_v039_entries.py` and `scripts/build_v039_split.py`:
  trace-to-entries assembly and split allocation.
- `scripts/audit_bipia_labels.py`: labeller audit. Prints the
  before/after breakdown across all four trace files.
- `scripts/relabel_v039_traces.py`: idempotent replay of the post-fix
  rule across `tests/adversarial/traces/bipia-s43-*.jsonl`.
- `scripts/train_v9_upweighted.py`: v9 trainer with
  `--follow-weight` and `--max-bipia-benign` flags.
- `scripts/eval_v039_v9.py`: four-surface eval (v035 VAL, v035 TEST,
  v0.38 Phase 1, v0.39 BIPIA holdout), v8 vs v9 side-by-side, writes
  the bench artifact.
- `bench/v039_v9_eval.json`: every number in the v0.39 bench doc.
- `bench/v039_bipia_full_eval_v8_{haiku,llama,mistral,qwen}.json` and
  `bench/v039_bipia_pilot_eval_v8.json`: v8 baseline numbers under
  BIPIA pressure (the gap that motivated v9).
- `bench/vaara-bench-v0.39.md`: v0.39 methodology, the labeller-bug
  arc, chain of custody, and ship-gate record.

### Changed
- `src/vaara/adversarial_classifier.py`: `_DEFAULT_BUNDLE` bumped from
  `adversarial_classifier_v7.joblib` to `adversarial_classifier_v9.joblib`.
  The default had last been bumped in v0.36. v0.37 shipped v8 and
  v0.38 carried v7 forward. Neither release updated the runtime
  default, so PyPI `vaara==0.37.x` and `0.38.x` installs loaded v7 by
  default while bench docs quoted v8 numbers. Bench numbers themselves
  were honest because eval scripts pass `--bundle` explicitly, but the
  library default was two versions stale. v0.39 catches up to v9.
- README headline numbers updated to v9 on v035 TEST + Phase 1 + the
  BIPIA-pressure FPR row. Bench pointer swapped from v0.38 to v0.39.

## [0.38.0] - 2026-05-27

**Theme: Phase 1 PAIR scale-up to n=300 per attacker family on the
Llama-3.3-70B leg.** 900 fresh adversarial entries generated by
`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` on AMD-backed MI300X
SR-IOV under `rocm/vllm:latest` at seed 43. The v8 production
classifier is carried forward unchanged from v0.37 and evaluated at
calibrated T=0.9006 against the new corpus. Overall recall lands at
88.4% [86.2, 90.4], a 2.6 pp lift over the v0.37 Llama-3.3 leg
(85.8%). The biggest move is on `data_exfil` (69.0% to 75.3%, +6.3
pp), with `tool_misuse` at 93.7% and `privilege_escalation` at 96.3%.
The Phase 1 entries are content-distinct from the v0.37 Llama-3.3 leg
because the new seed produces fresh samples.

External-corpus eval (BIPIA, LLMail-Inject) and the IPI fourth attacker
family both move to v0.39. Neither external corpus pre-extracts the
tool calls that v8 classifies, so an honest eval requires an LLM-agent
harness rather than direct classifier inference. IPI fits the same
release window as a different attack class.

### Added
- `tests/adversarial/generated/{TM,PE,DE}-v038-llama33-s43.jsonl`:
  900 Phase 1 entries (300 per category) generated at seed 43,
  schema-valid, fingerprint-deduplicated against v037.
- `scripts/eval_v038_phase1.py`: reads the three Phase 1 jsonls
  directly and runs the production v8 bundle at T=0.9006. Reports
  overall recall, per-category recall, per-severity recall, and
  Wilson confidence intervals. Writes the eval artifact to
  `bench/v038_phase1_eval_v8.json`.
- `scripts/v038_droplet_run.sh`: droplet driver mirroring the v0.37
  shape with the `--quantization fp8` argument removed. The current
  `rocm/vllm:latest` image refuses the explicit quantization flag
  when the model config already declares `compressed-tensors`. vLLM
  auto-detects on that path.
- `scripts/v038_local_watcher.sh`: 60-second rsync-back loop for
  continuous recovery of entries and logs during long droplet runs.
- `bench/v038_phase1_eval_v8.json`: Phase 1 eval artifact.
- `bench/vaara-bench-v0.38.md`: v0.38 methodology, chain of custody,
  ship gate, and the explicit scope note on the v0.39 external-corpus
  and IPI threads.

### Changed
- README bench pointer swapped from v0.37 to v0.38.

## [0.37.1] - 2026-05-27

**Theme: SEP-2787 verifier step 5, argument commitment verification.**
Closes the missing piece in Vaara's SEP-2787 reference implementation
after the upstream SEP draft was rewritten to contain Vaara's four
schema proposals plus a verifier-side `args_commitment_mismatch`
rejection path. The new `verify_args_commitment` function covers all
three commitment kinds in Vaara's three-way args shape. `ArgsDigest`
recomputes the JCS-canonical hash of the runtime arguments and
compares it to the bound digest. `ArgsRef` resolves the URI through a
caller-supplied resolver, hashes the content, and matches both the
stored digest and the canonicalized runtime arguments. `ArgsProjection`
recomputes its own digest and reports whether the projection is an
identity match for the runtime arguments (identity projection) or a
signed redaction (no completeness claim, per spec). Step 5 is exposed
as a separate function so callers compose it after the existing
signature and TTL checks once the `tools/call` arguments are in hand.

### Added
- `src/vaara/attestation/_sep2787_verifier.py`: `verify_args_commitment`
  and the `ArgsCommitmentResult` dataclass. Returns `ok`, an optional
  `reason` matching the spec's `args_commitment_mismatch` enum value,
  and a tri-state `projection_match` field for projection-kind
  commitments.
- Public re-exports of `verify_args_commitment` and
  `ArgsCommitmentResult` from `vaara.attestation.sep2787`.
- 11 new tests in `tests/test_attestation_sep2787.py` covering digest
  match and mismatch, key-reorder canonicalization stability, ref
  resolver missing or raising, ref digest mismatch, ref content not
  matching runtime args, identity projection, redacted projection, and
  projection-digest tampering.

### Changed
- `vaara.attestation.sep2787` module docstring documents verifier
  coverage: steps 1 (signature) and 3 (TTL) inside `verify_attestation`,
  step 5 via the new `verify_args_commitment`, steps 2 (nonce replay)
  and 4 (tool call match) left to the runtime caller.

## [0.37.0] - 2026-05-27

**Theme: third attacker family added to cross-model held-out, v8
classifier closes the worst v0.36 sub-cell.** 900 adversarial entries
generated by `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` on
AMD-backed MI300X via `rocm/vllm:latest` extend the cross-model fold
to a third family. v8 retrains on v035 TRAIN plus v036 Mixtral and
Claude TM/PE folded in, with v036 DE both legs and the full v037
Llama-3.3 leg held out. v8 holds in-distribution at 86.6% recall on
v035 TEST (vs v7's 85.3%, +1.3pp) with 5.0% FPR. On the new v037
holdout (2,277 entries), overall recall lands at 66.8%, a 7.6 pp lift
on the comparable v036 number. The Llama-3.3 leg covers a third
attacker family at 85.8% overall. The worst v0.36 sub-cell (data_exfil
× Claude) lifts from 26.0% (v7) to 38.9% (v8) on the same 700 entries,
+12.9 pp, confirming the v0.36 diagnosis that the constraint was
training-corpus distribution and not feature space. Folding v036
TM/PE into TRAIN reweights the existing destination features against
the closed-weight attacker patterns the v7 fold was missing.

### Added
- `tests/adversarial/generated/{TM,PE,DE}-v037-llama33.jsonl`: 900
  third-family entries (13 schema-invalid TM dropped, 887 valid).
- `tests/adversarial/v037_split.json`: v0.37 split manifest. Inherits
  every v035_split assignment unchanged, folds v036 Mixtral and Claude
  TM/PE into the train fold, marks v036 Mixtral and Claude DE plus the
  full v037 Llama-3.3 leg as held out.
- `src/vaara/data/adversarial_classifier_v8.joblib`: new production
  bundle, 638 features (254 hand plus 384 MiniLM embedding), trained
  on the 11,287-entry union fold at threshold 0.9006.
- `scripts/_v037_common.py`, `scripts/generate_targeted_v037.py`,
  `scripts/v037_droplet_run.sh`, `scripts/v037_local_watcher.sh`:
  Llama-3.3 generation pipeline mirroring the v0.36 shape with the
  third-family swap and a continuous-monitoring local watcher.
- `scripts/build_v037_split.py`: builds `v037_split.json` from the
  v035 inheritance plus the v036 and v037 generated entries.
- `scripts/validate_v037.py`, `scripts/eval_v037_holdout.py`: schema
  check and three-leg holdout eval (mixtral, claude, llama33).
- `bench/v037_eval_v8_holdout.json`: full per-category and
  per-category-per-leg eval results.
- `bench/vaara-bench-v0.37.md`: ship-gate record, chain of custody,
  reproduction recipe, named limits.
- `vaara.attestation.sep2787`: reference implementation of the SEP-2787
  Tool Call Attestation envelope (MCP spec PR
  `modelcontextprotocol/modelcontextprotocol#2787`), proposed shape.
  Implements the four schema changes Vaara raised in the v1 draft
  thread: fact-source labels (three trust-surface blocks
  `planner_declared` / `issuer_asserted` / `payload_derived`),
  explicit three-way args shape (`ArgsDigest` / `ArgsRef` /
  `ArgsProjection`), RFC 8785 (JCS) canonicalization with IEEE-754
  float rejection at the boundary, and request-attestation scope only
  (the v1 optional `ack` field is excluded and belongs in a separate
  extension). Supports HS256, ES256, RS256 signing per the v1 draft.
  Coexists with the existing OVERT 1.0 implementation in
  `vaara.attestation.overt`. See `docs/sep2787-overt-mapping.md` for
  field-level mapping.
- `docs/sep2787-overt-mapping.md`: field-level mapping between OVERT
  1.0 Base Envelope (CBOR) and the SEP-2787 proposed shape (JSON).
- `tests/test_attestation_sep2787.py`: 16 unit tests covering all
  three signing algorithms, all three args-commitment shapes,
  tampering rejection, canonicalization invariants, and TTL handling.

### Changed
- Production classifier: v7 to v8. v7 retained on disk for cross-eval
  reproducibility. Threshold unchanged at 0.9006.
- `attestation` optional extra: adds `rfc8785>=0.1.4` for JCS
  canonicalization.

## [0.36.0] - 2026-05-25

**Theme: first public cross-model held-out evaluation. v7 classifier
with destination-aware features ships as the new production default.
Methodology release with honest training-corpus diagnosis.** 4,176
adversarial entries generated by Mixtral-8x7B-Instruct (FP16 on
AMD-backed MI300X) and Claude Sonnet 4.6 (Anthropic API) on the three
weakest v0.35 categories, never folded into TRAIN. The v6 classifier
on this held-out fold scores 58.4% overall recall versus 85.0% on the
v035 in-distribution TEST set, with the worst sub-cell (data_exfil
generated by Claude) at 22.0%. Root cause is a destination-signal axis:
v6 reads tool_name strongly and misses parameter-level destination
patterns (external URLs, personal email domains, public-read buckets,
no-expiry pastebins). The v7 retrain adds 18 destination-aware features
prefixed `dst__`, holds the in-distribution number at 85.3% on v035 TEST
(no regression), and lifts the worst cell from 22.0% to 26.0%. Marginal
overall lift (+0.8 pp) confirms the constraint is training-corpus
distribution, not feature space: features fire correctly but XGBoost
weights them weakly because the v035 TRAIN fold lacks adversarial
examples where those features are the discriminating signal. The
v0.37 scope shifts to corpus augmentation accordingly.

### Added
- `tests/adversarial/generated/{TM,PE,DE}-v036-mixtral.jsonl` and
  `tests/adversarial/generated/{TM,PE,DE}-v036-claude.jsonl`: 4,200
  cross-model held-out entries (24 schema-invalid dropped, 4,176 valid).
- `tests/adversarial/v036_holdout.json`: held-out fold manifest, keyed
  `<relative_path>#L<line_index>`. Pure eval fold, never folded into
  TRAIN.
- `tests/adversarial/MANIFEST.sha256` regenerated to 302 entries
  anchoring the new files.
- `src/vaara/data/adversarial_classifier_v7.joblib`: new production
  bundle, 638 features (620 base + 18 dst), trained on v035 TRAIN.
- `bench/vaara-bench-v0.36.md`: canonical bench doc for this release.
  Three-contribution structure with CKD-paper citation in the
  why-cross-model intro.
- `bench/v036_eval_v6_holdout.json`, `bench/v036_eval_v7_holdout.json`,
  `bench/v036_eval_v7_v035test.json`: raw eval artefacts.
- `scripts/generate_targeted_v036.py`: Mixtral leg, OpenAI-compatible
  vLLM client with cross-version dedup-prior flag.
- `scripts/generate_targeted_v036_claude.py`: Claude leg, Anthropic SDK
  driver with identical prompt structure.
- `scripts/_v036_common.py`: shared schema, prompt, fingerprint,
  prior-corpus dedup loader.
- `scripts/v036_droplet_run.sh` and `scripts/v036_local_watcher.sh`:
  droplet-side and local-side runners with served-model identity check
  and no destructive EXIT trap (lessons baked in from the v0.31 bench
  incident).
- `scripts/eval_v036_holdout.py`: held-out eval with per-category and
  per-leg cuts (Mixtral vs Claude).
- 18 destination-aware features in `_DST_PATTERNS` covering
  personal-email domains, webhook relays, paste services, free
  file-hosts, public-bucket markers, no-expiry markers, non-internal
  share schemes, external-bucket-name patterns, PII column patterns,
  bulk-SELECT-LIMIT, SELECT-star-FROM, export/dump destination keys,
  suspicious TLDs, webhook parameter keys, public-path markers, share
  platforms, recipient-to-personal-email, attachment-with-external-
  recipient.
- Adjacent-work entries in `PRIOR_ART.md` for arxiv:2605.21566 (CKD
  calibration / deployment readiness) and arxiv:2605.22004
  (FCR-controlled informative conformal sets, v0.37+ combiner pointer).

### Changed
- `src/vaara/data/adversarial_classifier_v6.joblib` is retained on disk
  for cross-eval reproducibility but is no longer the default bundle.
  `_DEFAULT_BUNDLE` now points to `adversarial_classifier_v7.joblib`.
- `src/vaara/adversarial_classifier.py` schema check adapts to the
  bundle's `feature_names`: detects `dst__` presence and validates the
  static-feature tail against `_STATIC_FEATURES` or
  `_STATIC_FEATURES + _DST_STATIC` accordingly. v6-style bundles still
  load.
- `scripts/train_adversarial_classifier.py:build_features` emits the
  new `dst__*` features. Output dimension grows from 620 to 638
  (with embeddings).

### Notes
- AWQ quantization is not supported on AMD ROCm (verified against vLLM
  docs). The original v0.36 plan to use Mixtral-8x22B-Instruct with
  AWQ-4bit was substituted with Mixtral-8x7B-Instruct in FP16. Adding a
  third attacker family (e.g. DeepSeek-V3, Llama-3-Instruct) is v0.37+
  scope.
- Total Claude API spend for the held-out generation was under three
  dollars at Sonnet 4.6 pricing. Mixtral compute ran on an AMD-backed
  MI300X droplet under `rocm/vllm:latest`, shutdown issued post-rsync.

## [0.35.0] - 2026-05-25

**Theme: matched-benign corpus extension closes the v0.34 class-balance
gap and lifts the production classifier.** 2,100 new benign entries
(700 each for `tool_misuse`, `privilege_escalation`, `data_exfil`),
generated via Qwen-72B-Instruct on AMD-backed MI300X, share tool_name
and parameter shape with the v0.34 adversarial entries while encoding
safe operations. Corpus grows from 10,055 to 12,155 entries, train
positive_rate corrects from 0.701 back to 0.579, and a v6 classifier
trained on the new v035 split clears the ship gate. Held-out TEST
recall lifts to 85.0% [82.8, 87.1] at FPR 4.6% [3.3, 6.3] on n=1,827,
calibrated threshold 0.9006. Cross-eval shows v6 lifts every category
versus the v3 production classifier on the v035 TEST fold, with mean
recall gain +44.7 pp and no regressions. The v6 bundle becomes the
default production classifier. v3 remains in `src/vaara/data/` as the
historical anchor.

### Added
- `tests/adversarial/benign_generated/BT-v035-TM.jsonl`,
  `tests/adversarial/benign_generated/BT-v035-PE.jsonl`,
  `tests/adversarial/benign_generated/BT-v035-DE.jsonl`: 700 matched-benign
  entries each, generated via Qwen2.5-72B-Instruct on AMD MI300X. Each
  entry shares tool_name and parameter shape with a v0.34 adversarial
  anti-seed while encoding a safe operation a real operator would run.
- `tests/adversarial/MANIFEST.sha256` regenerated to 296 entries
  anchoring the new files.
- `tests/adversarial/v035_split.json`: deterministic stratified
  70/15/15 over the rebalanced 12,155-entry corpus, n=8,501 TRAIN /
  1,827 VAL / 1,827 TEST.
- `src/vaara/data/adversarial_classifier_v6.joblib`: production
  classifier bundle. 620-dim feature (236 hand + 384 MiniLM, revision
  pinned `c9745ed1d9f2`), XGBoost lr=0.10, trained on v035 TRAIN fold.
- `bench/vaara-bench-v0.35.md`: canonical bench doc for this release.
  Methodology delta against v0.34, per-category recall vs v3 cross-eval,
  ship-gate verdict.
- `bench/v035_eval_v6.json`, `bench/v035_eval_v3_cross.json`,
  `bench/v035_per_category_v6.json`, `bench/v035_per_category_v3_cross.json`:
  raw eval artefacts.
- `scripts/generate_matched_benign_v035.py`: matched-benign generator,
  shares anti-seeds with v0.34 adversarial entries on the same tool surface.
- `scripts/v035_droplet_run.sh`: one-shot droplet driver for vLLM serve
  + three parallel generators.

### Changed
- `src/vaara/adversarial_classifier.py`: `_DEFAULT_BUNDLE` now resolves
  to `adversarial_classifier_v6.joblib`. v3 stays on disk as the
  historical anchor.
- `README.md`: headline recall, FPR, TEST n, and calibrated threshold
  updated to the v6 numbers. Bench pointer flips from v0.34 to v0.35.
- `Makefile`: `bench` target now evaluates v6 on the v035 split with v3
  cross-eval as the regression control.

## [0.34.0] - 2026-05-25

**Theme: adversarial corpus extension + a methodology lesson honestly
recorded.** The benchmark grows from 7,955 to 10,055 entries via 2,100
targeted Qwen-72B-generated entries on the three weakest v0.32 TEST
categories. A retrained v5 classifier did not clear the ship gate at
matched FPR (one targeted category lifted +13.3 pp, two were flat,
three untargeted categories regressed 7-13 pp), so the production
classifier stays at v3 from v0.33. The corpus extension itself is the
durable contribution. v0.35 will add matched benign coverage before
retraining.

### Added
- `tests/adversarial/generated/TM-v034.jsonl`,
  `tests/adversarial/generated/PE-v034.jsonl`,
  `tests/adversarial/generated/DE-v034.jsonl`: 700 entries each
  generated via Qwen2.5-72B-Instruct on AMD MI300X. Schema-validated,
  deduplicated within and across files.
- `tests/adversarial/MANIFEST.sha256` regenerated to 293 entries
  (290 + 3) anchoring the new files.
- `tests/adversarial/v034_split.json`: deterministic stratified
  70/15/15 over the extended corpus, n=7,033 TRAIN / 1,511 VAL /
  1,511 TEST.
- `src/vaara/data/adversarial_classifier_v5.joblib`: experimental A/B
  bundle trained on v034 TRAIN. Not loaded by default. Preserved for
  reproducibility of the negative ship-gate result.
- `bench/vaara-bench-v0.34.md`: methodology delta, chain of custody,
  honest record of why the v5 swap did not ship.
- `bench/v034_eval_v3_cross.json`, `bench/v034_eval_v5.json`,
  `bench/v034_per_category_v3_at_T080.json`,
  `bench/v034_per_category_v5_at_T095.json`: cross-eval artefacts
  at matched FPR thresholds.
- `bench/v034_droplet_logs/`: vLLM session log and per-category
  generator logs (infrastructure IPs redacted before commit) for
  the v0.34 bench-doc reproducibility table.
- `scripts/generate_targeted_v034.py`: generalised over
  `e1_generate.py` to accept `--category` and pull few-shot seeds
  from `tests/adversarial/<category>.jsonl`.
- `make bench` Makefile target: version-agnostic reproduction of the
  current bench doc, replacing the v0.31-specific historical target
  (which remains available as `make repro-v031-bench`).

### Changed
- `README.md` benchmark section collapses to a single canonical
  pointer at the current bench doc. Historical per-version bench
  links live under `bench/` for chain of custody but no longer
  accumulate in the README.
- README OVERT example reads `vaara.__version__` at runtime instead
  of hardcoding a stale version string.

### Not in this release
- A production classifier swap to v5. The v5 bundle is documented
  as a negative-result A/B, same pattern as bge-base in v0.33. The
  production loader continues to use the v3 bundle shipped in v0.33.
- Matched benign generation for the three weakest categories. That
  is the scope correction for v0.35: matched benign first, then a
  v6 retrain on the balanced corpus.

## [0.33.0] - 2026-05-25

**Theme: pin the embedding model by HuggingFace commit SHA so v0.32's
recall numbers stay reproducible.** v0.32 loaded
``sentence-transformers/all-MiniLM-L6-v2`` by ID without a revision
pin, which made the classifier silently sensitive to upstream model
updates. v0.33 pins the revision
(``c9745ed1d9f207416be6d2e6f8de32d1f16199bf``) and records it in the
classifier bundle metadata. No change to corpus, split, hparams, or
trained weights. Headline TEST recall stays at 84.3% [81.5, 86.7] at
FPR 4.6% [3.1, 7.0].

### Added
- ``src/vaara/embeddings.py``: ``EMBED_MODEL_REVISION`` constant.
  The lazy MiniLM singleton now passes ``revision=`` to
  ``SentenceTransformer`` so the embedding is reproducible across
  upstream updates.
- ``scripts/save_classifier_bundle.py``: writes
  ``embedding_model_id`` and ``embedding_model_revision`` into the
  bundle metadata when ``--embeddings`` is set.
- ``scripts/eval_per_category.py``: per-category TEST recall / FPR
  breakdown for any classifier bundle. Sorted ascending by recall,
  surfacing categories where corpus extension would help most.
- ``scripts/train_and_eval_bge_base.py``: standalone A/B harness for
  ``BAAI/bge-base-en-v1.5`` against the MiniLM baseline. Used in
  v0.33 ship-or-skip; result was skip, recorded below.
- ``bench/vaara-bench-v0.33.md``: chain-of-custody anchor for the HF
  revision SHA; documents the bge-base A/B negative result and the
  per-category recall floor that motivates v0.34.
- ``bench/v032_per_category_test.json``: per-category TEST recall
  at the v0.32 calibrated threshold. ``tool_misuse`` (47.1%),
  ``privilege_escalation`` (48.3%), and ``data_exfil`` (65.0%) are
  the weakest.
- ``bench/v033_bge_base_eval.json``: bge-base A/B record. TEST
  recall 82.5% at FPR 5.1% (Wilson [79.6, 85.1] / [3.4, 7.5]),
  -1.8 pp vs MiniLM at +440 MB extra wheel weight.
- ``bench/v033_eval_final.json``: retrained-v3 evaluation confirming
  headline reproduction.

### Changed
- ``src/vaara/data/adversarial_classifier_v3.joblib`` retrained on
  the same v0.31 TRAIN fold with the embedding revision now pinned
  in source. Determinism verified: bit-equivalent TEST recall and
  FPR to v0.32.

### Not in this release
- bge-base-en-v1.5 as the classifier embedding. The A/B showed
  -1.8 pp TEST recall vs MiniLM at higher cost. MiniLM stays.
- Targeted corpus extension on the three weakest categories.
  Scheduled for v0.34.
- FPR-bounded three-stage ensemble combiner. Scheduled for v0.35.

## [0.32.0] - 2026-05-25

**Theme: classifier recall lift from 53.9% to 84.3% at the same 4.6%
FPR.** The v0.31 hand-features hit a ceiling because regex patterns
catch known surface forms and miss semantic variation. v0.32 adds a
384-dim ``sentence-transformers/all-MiniLM-L6-v2`` embedding of the
parameter blob to the existing 236 hand-features and retrains XGBoost
on the concatenated 620-dim input. Threshold is calibrated on the
v0.31 VAL fold at target FPR 5%. Same 70/15/15 split as v0.31, so the
TEST result is directly comparable.

### Added
- ``src/vaara/embeddings.py``: lazy-loaded MiniLM singleton exposing
  ``embed(text)`` and ``embed_batch(texts)``. 384-dim L2-normalized
  float32. CPU backend. First call pays a one-time ~5s model load,
  subsequent calls are sub-millisecond.
- ``adversarial_classifier_v3.joblib`` (922 KB): retrained on the
  v0.31 TRAIN fold (5,563 entries) with embeddings + hparams
  ``n_estimators=400, max_depth=6, learning_rate=0.10``.
- ``scripts/eval_v032.py``: calibrate threshold on VAL at target FPR,
  report TEST recall and FPR with Wilson 95% intervals.
- ``scripts/sweep_v032_hparams.py``: XGBoost hparam grid search with
  per-config VAL calibration.
- ``--embeddings`` flag on ``scripts/save_classifier_bundle.py``,
  plus ``--n-estimators``, ``--max-depth``, ``--learning-rate``,
  ``--min-child-weight``.

### Changed
- Runtime classifier switched from ``adversarial_classifier_v2.joblib``
  to ``adversarial_classifier_v3.joblib``. The runtime detects
  ``embed__*`` feature names in the bundle and calls ``embed()`` at
  inference time. v2-format bundles still load unchanged.
- ``detect_injection()`` default threshold moves from 0.90 to 0.9226,
  the VAL-calibrated value for the new bundle. Explicit-threshold
  callers are unaffected.
- ``vaara[ml]`` extras now include ``sentence-transformers>=2.6``.
- README headline numbers swap to v0.32 values: TEST recall 84.3%
  [81.5, 86.7] at FPR 4.6% [3.1, 7.0] on held-out n=1,196.

### Notes
- 771 tests pass.
- Same v0.31 split manifest, same TEST set. The +30.4 percentage
  point recall lift is at the same FPR band, so the trade is a
  recall gain at no FPR cost.
- First-call latency now includes embedding model load (~5s on
  CPU). After warm-up, per-call adversarial scoring stays in the
  sub-millisecond band.

## [0.31.0] - 2026-05-25

**Theme: industrial-grade adversarial benchmark.** v0.31 retires the
cross-validated headline numbers and replaces them with a 70/15/15
stratified split, threshold picked on VAL only, and Wilson 95%
intervals on every figure cited in the release. The classifier
bundle is retrained on TRAIN entries only. A single command
(`make repro-v031-bench`) reproduces every number from a fresh clone
against the four chain-of-custody hashes printed by every script.

### Added
- `bench/vaara-bench-v0.31.md`: methodology spec parallel to
  `vaara-bench-v1.md`. Chain of custody, split methodology,
  threshold pick by Youden's J and balanced accuracy on VAL,
  headline numbers with Wilson 95% intervals, named limits,
  reproduction recipe.
- `scripts/build_train_val_test_split.py`: deterministic 70/15/15
  split stratified by `(category, source)`, per-stratum seeded RNG,
  key format `<rel_path>#L<line>` because raw `id` has 1,855
  cross-file collisions.
- `scripts/eval_pipeline_attribution.py`: per-entry attribution
  audit running every corpus entry through a fresh `Pipeline`
  (cold-start methodology) plus the classifier side-by-side.
- `scripts/three_way_variants.py`: rules-only / classifier-only /
  both-OR comparison on any fold with per-category and per-source
  breakdowns.
- `scripts/threshold_sweep_val.py`: classifier threshold sweep on
  VAL only.
- `scripts/wilson_intervals.py`: Wilson 95% intervals on every
  headline metric. PAIR ASR upper bound at 0/25 = 13.3%.
- `Makefile` with `repro-v031-bench` target: eight-step end-to-end
  reproduction.
- 2,000-entry v0.31 corpus extension: `JB-roleplay-v031.jsonl`,
  `JB-hypothetical-v031.jsonl`, `JB-fakemode-v031.jsonl`,
  `BT-canonical-v031.jsonl`. Qwen2.5-72B-Instruct on MI300x with
  documented parameters.

### Changed
- Runtime classifier switched from `adversarial_classifier_v1.joblib`
  (v0.5.3-trained, cross-validation point estimates) to
  `adversarial_classifier_v2.joblib` (v0.31-trained on TRAIN-only).
- `detect_injection()` default threshold raised from 0.55 to 0.90.
  The 0.55 was the v1 bundle's escalation band against a
  cross-validated corpus. 0.90 is the VAL-pick threshold against
  the v2 bundle, chosen by Youden's J and balanced accuracy.
  Callers passing an explicit threshold are unaffected.
- `README.md` headline numbers swap to v0.31 honest values: recall
  53.9% [50.3, 57.5] at FPR 4.6% [3.1, 7.0] on held-out TEST
  n=1,196. The earlier 97.1% number was a contaminated read because
  the threshold tuning saw the test fold.

### Notes
- 771 tests pass under the runtime switch + threshold change.
- The chain of custody is the v0.31 contribution. Reviewers cannot
  tear down the methodology, only the numbers, and the numbers are
  intervals not point estimates.

## [0.30.0] - 2026-05-24

**Theme: matrix pre-work. Vaara's coverage of OWASP Top 10 for
Agentic Applications 2026 and OVERT 1.0 Part 3 controls, written as
two standalone documents so an enterprise reader can see the honest
mapping without having to read the rest of the repository.** Carlos
Hernandez (Bosch AI Officer, awesome-eu-ai-act curator) offered a
post-2026-06-18 side-by-side mapping of Vaara, Watcher, MS Agent
Governance Toolkit, and OWASP controls on the GenAI Gurus platform.
The Vaara column for that matrix needs to be drafted under the
assumption of expert-review by the maintainers of the other
columns. This release ships the two reference documents that anchor
the Vaara entries for the matrix.

### Added
- `OWASP_AGENTIC.md`: Vaara mapping to OWASP Top 10 for Agentic
  Applications 2026 (ASI01 through ASI10). Per-risk coverage with
  ✅ / ◐ / ◯ status badge, the OWASP mitigation list quoted under
  CC BY-SA 4.0, the Vaara primitive that satisfies each mitigation,
  and an explicit "deployer-owned" note per risk. Includes a
  cross-mapping summary table and the source citation (genai.owasp.org).
- `OVERT_CONTROLS.md`: standalone OVERT 1.0 Part 3 (Agentic AI
  Controls) mapping, extracted from `COMPLIANCE.md`. Same
  ✅ / ◐ / ◯ status convention. Covers TOOL-*, MCP-*, MULTI-*,
  CAP-*, DISC-*, HITL-*, DRIFT-* control families plus the S3P
  measurement primitive in Section 9, MEA-2.
- `OWASP_AGENTIC.md` and `OVERT_CONTROLS.md` cross-linked from the
  README "Where things live" table.

### Changed
- `COMPLIANCE.md` "OVERT 1.0 Part 3 (Agentic AI Controls) mapping"
  section now points readers to the new `OVERT_CONTROLS.md` and
  `OWASP_AGENTIC.md` documents. The full inline mapping content is
  retained in `COMPLIANCE.md` for backward compatibility with
  existing anchors and references.

### Unchanged
- Hash chain format, OVERT envelope schema, MCP proxy semantics,
  CLI surface, HTTP API, release workflow. This release ships
  documentation surfaces only. The compliance engine, scorer,
  audit, OVERT, and policy paths are byte-identical to v0.29.0.

### Notes for deployers
- The OWASP Top 10 for Agentic Applications 2026 is published by
  the OWASP GenAI Security Project, Agentic Security Initiative,
  December 2025, under Creative Commons CC BY-SA 4.0. The mapping
  in `OWASP_AGENTIC.md` is referenced under that license with
  attribution.
- The OVERT 1.0 Part 3 mapping references controls defined by Glacis
  Technologies (overt.is). The status assessments in
  `OVERT_CONTROLS.md` are Vaara's own reading of which controls the
  shipped code satisfies, not an OVERT-issued certification.

## [0.29.0] - 2026-05-24

**Theme: chronology anchor for Vaara's load-bearing concepts and a
neutral list of adjacent published work.** As runtime-evidence,
hash-chained audit, and conformal-interval framings appear in
recent pre-prints (Protocol-Driven Development, Subjective Logic
runtime confidence updates, formal-methods runtime monitors), a
reader comparing Vaara against newer proposals deserves a single
file that anchors when each Vaara concept first shipped in a tagged
public release and lists adjacent peer-reviewed and pre-print work
without competitive framing. This release ships that file and adds a
short related-work section to the conformal-prediction explainer.

### Added
- `PRIOR_ART.md`: per-concept chronology table mapping each
  load-bearing Vaara concept to its first shipped version and a path
  into the codebase or docs. Includes a related-work section listing
  adjacent pre-prints (Protocol-Driven Development, Formal Methods
  Meet LLMs, Subjective Logic runtime confidence updates,
  mechanistic interpretability for EASA learning-assurance,
  backchaining LoC mitigations from national security benchmarks)
  and classical foundations (conformal prediction, Linear Temporal
  Logic runtime verification). Inclusion is neutral attribution, not
  ranking.
- `PRIOR_ART.md` cross-link in the README "Where things live" table.

### Changed
- `docs/conformal-prediction.md`: appended a "Related
  runtime-confidence work" section citing the Subjective Logic
  safety-arguments pre-print (arXiv:2605.22530v1) as a complementary
  research line, with a cross-link to `PRIOR_ART.md` for the broader
  chronology and related-work list. The plain-language explanation
  and the Article 15(1) mapping are unchanged.

### Unchanged
- Hash chain format, OVERT envelope schema, MCP proxy perimeter
  semantics, CLI surface, HTTP API, release workflow. This release
  ships documentation surfaces only. The compliance engine,
  scorer, audit, OVERT, and policy paths are byte-identical to
  v0.28.0.

### Notes for deployers
- The chronology table in `PRIOR_ART.md` is reconstructed from
  `CHANGELOG.md` and the git tags. Both can be verified
  independently against PyPI release history and the GitHub tag
  list. Where a deployer or auditor wants a paper trail for "since
  when has Vaara done X," `PRIOR_ART.md` is the file to cite.

## [0.28.0] - 2026-05-22

**Theme: making the evidence-sufficiency rules and the conformal
interval readable by the people who use them.** The compliance engine
has been producing `verdict_inputs` (the threshold-vs-observed snapshot
that drove each article's verdict) and `verdict_reasons` (human-readable
rationale lines) since v0.26.0, and conformal intervals on every risk
score since the first public release. What was missing was a public,
linkable reference for the threshold values per article and a
plain-language explainer of the conformal interval for compliance
reviewers and legal counsel who are not statisticians. This release
ships both.

### Added
- `VERDICTS.md`: per-article evidence sufficiency reference. Documents
  the EU AI Act (Articles 9, 11-15, 17, 61) and DORA (Articles 10, 12,
  13) defaults the engine ships with: minimum record count, staleness
  window, strong-strength count and freshness bounds, future-timestamp
  downgrade, chain-integrity pin, and the status/strength decision
  tree. Includes worked JSON examples of `verdict_inputs` for a
  sufficient article and the same article after a chain break.
  Cross-linked from `COMPLIANCE.md` and from the README "Where things
  live" table.
- `docs/conformal-prediction.md`: plain-language companion to
  `docs/formal_specification.md`. Explains why a point risk score is
  not enough, what the interval gives a reader, and how the
  distribution-free coverage guarantee maps to Article 15(1)
  ("appropriate level of accuracy"). Cross-linked from the README
  "Where things live" table.

### Changed
- `COMPLIANCE.md` "EU AI Act Article Mapping" intro now points readers
  to `VERDICTS.md` for the threshold rules behind status assignment.

### Notes for deployers
- The threshold defaults in `VERDICTS.md` are the values shipped in
  `EU_AI_ACT_REQUIREMENTS` and `DORA_REQUIREMENTS` in
  `vaara/compliance/engine.py`. A deployer can override any of them
  by registering a custom `RegulatoryRequirement` with the
  `ComplianceEngine`. The document is the public starting point, not
  a constraint.

## [0.27.0] - 2026-05-22

**Theme: continuous fuzzing on the parsers that ingest attacker-controlled
input.** The OVERT envelope decoder, the audit-record `from_dict`
deserialiser, and the policy YAML/JSON loader all sit at the boundary
between Vaara and untrusted bytes. A crash, hang, or unhandled exception
in any of them is a denial-of-service vector at minimum, and a
deserialisation hazard at worst. This release wires ClusterFuzzLite into
CI so those three parsers get continuously fuzzed on every PR and nightly
in batch, and ships the first finding the local smoke test caught
(`from_yaml` leaking `OSError(ENAMETOOLONG)` past the `PolicyError`
contract).

### Added
- `fuzz/fuzz_overt_envelope.py`: atheris target that decodes attacker
  CBOR bytes, validates the closed 9-field schema, reconstructs a
  `BaseEnvelope`, and signature-checks against a dummy pubkey. Mirrors
  the attack surface of `vaara overt verify`.
- `fuzz/fuzz_audit_from_dict.py`: atheris target for `AuditRecord.from_dict`
  + `compute_hash()` + the `narrative` property. Models the JSONL-replay
  path where trail records get reloaded from disk.
- `fuzz/fuzz_policy_loader.py`: atheris target for `from_json` and
  `from_yaml`. Exercises both text paths with attacker-controlled strings.
- `.clusterfuzzlite/Dockerfile` and `.clusterfuzzlite/build.sh`: builder
  image based on `gcr.io/oss-fuzz-base/base-builder-python`, installs
  Vaara with `[attestation,yaml]` extras, compiles each `fuzz/fuzz_*.py`
  target with `compile_python_fuzzer`.
- `.github/workflows/cflite_pr.yml`: PR-triggered fuzzing for 300s under
  both address and undefined sanitizers, code-change mode.
- `.github/workflows/cflite_batch.yml`: nightly cron, 3600s batch fuzzing
  under both sanitizers.
- `.github/workflows/cflite_cifuzz.yml`: build-sanity on push to main, so
  a broken Dockerfile or build.sh surfaces immediately rather than at
  next PR.
- `tests/test_policy.py`:
  `test_from_yaml_oversize_single_line_treated_as_content_not_path`
  pins the regression behaviour of the loader fix below.
- SECURITY.md: brief note on the continuous-fuzzing posture so reporters
  know the parsers are under active fuzz coverage.

### Fixed
- `vaara.policy.loader.from_yaml`: a single-line YAML string longer than
  the OS path limit (~255 bytes on most filesystems) previously caused
  `Path(source).is_file()` to raise `OSError(ENAMETOOLONG)` directly,
  bypassing the loader's `PolicyError` contract. Any caller that loads
  YAML from attacker-controlled config could be DoS'd by an oversized
  single-line payload. The is_file probe is now wrapped: any stat
  failure is interpreted as "not a path" and the input falls through to
  YAML parsing, where it surfaces as a normal `PolicyError`. Found by
  the local smoke run of `fuzz_policy_loader.py` before any atheris
  fuzzing ran.

### Unchanged
- Hash chain format, OVERT envelope schema, MCP proxy perimeter semantics,
  CLI surface, HTTP API, release.yml SLSA provenance. The fuzz targets,
  Dockerfile, and CFLite workflows are additive supply-chain
  infrastructure; nothing in the runtime kernel moves.

## [0.26.0] - 2026-05-21

**Theme: per-article verdict drill-down inside the compliance report.** A
reviewer reading a v0.25 evidence report could see that Article 9(1) was
`evidence_sufficient` with `strength: strong`, but had to re-run the engine
or open the raw audit DB to learn which records the verdict sat on or which
threshold pushed the strength up. v0.26.0 attaches that reasoning to every
article so a regulator can trace status to threshold delta to concrete event
in one read. The change extends existing report machinery and stays
backwards compatible at the wire level. Existing fields keep their shape.
Two new fields appear next to them.

### Added
- `ArticleEvidence.verdict_inputs`: dict of the threshold-vs-observed
  inputs the engine compared against to produce status and strength.
  Surfaces `min_evidence_count`, `staleness_hours`, `evidence_event_types`,
  `evidence_count_observed`, `freshest_evidence_age_hours`,
  `oldest_evidence_age_hours`, `future_timestamp_count`, `chain_intact`,
  a `strength_thresholds` sub-dict (the 2x / staleness/4 bounds used for
  `STRONG`), and a `verdict_reasons` list of human-readable rationale lines
  explaining why this article landed at its current status and strength.
  JSON-strict: no inf/NaN at the dict boundary.
- `ArticleEvidence.contributing_events`: list of the most recent (up to 5)
  qualifying audit records the verdict sits on. Each entry carries
  `record_id`, `action_id`, `event_type`, `timestamp_iso`, `age_hours`,
  `agent_id`, `tool_name`, the record's `narrative`, and a curated
  `drill_down` dict containing only the data fields that directly fed
  risk/decision/outcome reasoning (point estimate, conformal interval,
  decision, reason, outcome severity, escalation target, resolution,
  reviewer, override reason). Free-form `data` keys are not inlined so a
  caller-supplied secret cannot leak into a regulator-facing summary.
- Renderers updated end to end. `render_markdown` adds a `Verdict inputs`
  table and `Contributing events` table per article. `render_narrative`
  appends a `Why:` rationale line and an `Event:` line per article.
  `render_pdf` adds the same two tables to each per-article section.
  `compliance.dashboard.render_html` adds the same drill-down to each card
  with HTML-escaped values. Still no JavaScript and no external assets.
- Broken-chain handling extends to drill-down: when `verify_chain()`
  fails, every article's `verdict_inputs.chain_intact` flips to `False`
  and the rationale list prepends a chain-integrity warning, matching
  the existing status / strength downgrade.
- Tests: 11 new tests across `tests/test_compliance.py`,
  `tests/test_compliance_render.py`, and `tests/test_compliance_dashboard.py`
  covering verdict_inputs shape, contributing-events ordering, drill-down
  key filtering (free-form data must not leak), broken-chain marking,
  empty-trail INSUFFICIENT pinning, JSON strict-mode round-trip, and
  renderer output across markdown / narrative / dashboard / PDF.

### Changed
- `.github/workflows/release.yml`: the `build` job now generates SLSA
  build provenance via `actions/attest-build-provenance@v4.1.0` after the
  wheel and sdist are built. The attestation binds workflow + commit SHA
  + builder identity to the artefact digests and ships alongside the
  Release for downstream `slsa-verifier` / `gh attestation verify` use.
  Required new permissions on the build job: `id-token: write`,
  `attestations: write`.

### Unchanged
- Hash chain format, OVERT envelope schema, MCP proxy perimeter semantics,
  CLI surface (`vaara compliance report` still takes the same flags),
  HTTP API. The `to_dict()` boundary adds two keys but does not rename or
  remove existing ones, and v0.25 consumers ignore the additions cleanly.

## [0.25.0] - 2026-05-21

**Theme: streaming notifications inside the audit boundary.** Long-running
upstream MCP tools emit `notifications/progress` and `notifications/message`
between the request and the final response. Until v0.25.0 those flowed
through the proxy untouched: forwarded to the client, but invisible to the
audit chain and the OVERT receipt directory. A regulator reconstructing the
session from receipts only could see the request and the result, never the
events the upstream emitted while working. v0.25.0 brings each notification
inside the same audit + attestation perimeter that already governs
`tools/call`, `resources/read`, and `prompts/get`. The notification still
forwards to the client unchanged. Observation never blocks streaming.

### Added
- Streaming-notification interception in `vaara.integrations.mcp_proxy`.
  When a `tools/call` arrives with `params._meta.progressToken` set, the
  proxy records the token in an in-flight map alongside the originating
  `action_id`, `agent_id`, and tool name. The map is cleared in a
  `finally` block when the call returns or raises.
- Every `notifications/progress` arriving from the upstream is recorded
  as a perimeter audit pair (`tool_name="mcp.notification.progress"`,
  `decision="observed"`) carrying the parent `action_id` and tool name
  pulled from the in-flight map. When OVERT is configured, the
  notification additionally emits a dedicated Base Envelope with action
  class `mcp.notification.progress` and `parent_action_id` /
  `parent_tool` / `progress_token` in `non_content_metadata`.
- Every `notifications/message` arriving from the upstream is similarly
  recorded as `mcp.notification.message` and, when OVERT is configured,
  emits a Base Envelope carrying the `level` and `logger` fields.
- Orphan progress notifications (token with no matching in-flight call)
  still audit and emit, with an empty `parent_action_id`, so the
  regulator can spot dangling events rather than have them disappear.
- Tests: 10 new tests across `tests/test_integrations_mcp_proxy.py` and
  `tests/test_integrations_mcp_proxy_overt.py`. Coverage includes
  correlation to in-flight `tools/call`, map cleanup on success and on
  raise, audit-failure-still-forwards (observation must not block
  streaming), orphan-progress, and OVERT envelope contents for both
  progress and message surfaces.

### Changed
- `.github/workflows/release.yml`: npm publish step reverts to OIDC-only
  (no `NODE_AUTH_TOKEN`, no `NPM_TOKEN`) now that the `@vaara/client`
  package has a Trusted Publisher configured. Provenance flag preserved.

### Unchanged
- Pipeline, audit chain hash format, perimeter allow/deny semantics, MCP
  wire format, OVERT envelope closed-schema fields. Notifications still
  forward to the client byte-identically. A v0.24.0 deployment behaves
  the same when it does not see streaming traffic.

## [0.24.0] - 2026-05-20

**Theme: MCP proxy emits OVERT 1.0 Base Envelopes per interaction.** With
the v0.21-v0.23 surface coverage in place across tools, resources, and
prompts, the next step is making each interaction independently verifiable
by a regulator or auditor. The MCP proxy can now sign every governed
interaction as an OVERT 1.0 Protocol Profile 1.0 Base Envelope and write
it to disk in canonical CBOR, ready for `vaara overt verify` or any other
conformant OVERT verifier. The OVERT substrate (emitter, verifier, schema,
CLI) shipped in earlier releases. This release wires the emitter into the
MCP proxy as the first concrete production-shape OSS OVERT 1.0 integration.

### Added
- `vaara-mcp-proxy` learns three new flags: `--overt-signing-key PATH`
  (Ed25519 PEM private key), `--overt-operator-key PATH` (raw bytes for
  the HMAC request_commitment, minimum 16 bytes; also accepts
  `VAARA_OVERT_OPERATOR_KEY_HEX` from the environment), and
  `--overt-receipts-dir DIR` (where canonical-CBOR envelopes land). All
  three are off unless set. When set, every allowed and every
  perimeter-blocked `tools/call`, `resources/read`, and `prompts/get`
  produces one Base Envelope on disk.
- `vaara.integrations._mcp_overt`: internal `OVERTReceiptEmitter` with
  per-process monotonic counter under a lock, per-process
  `arbiter_instance_identifier`, `encoder_binary_identity` derived from
  the Vaara version plus SHA-256 of the canonical perimeter config
  (allow/deny lists across all three primitives), and a pinned
  `pubkey.bin` written into the receipts directory on emitter start.
- Receipt layout: `DIR/{nanosecond_timestamp}-{counter:010d}.cbor` plus
  `DIR/pubkey.bin` (32 raw Ed25519 public-key bytes).
- `non_content_metadata` carries structural fields only: action class
  (`mcp.tool.call` / `mcp.resource.read` / `mcp.prompt.get`),
  tool/resource/prompt identifier, decision, reason, agent_id, action_id.
  Request content never leaves the operator environment. Only its
  HMAC-SHA256 commitment crosses the trust boundary, per OVERT Annex B.4.
- Tests: `tests/test_integrations_mcp_proxy_overt.py`. Coverage includes
  emitter-disabled-by-default, one envelope per call site (allowed and
  filtered), monotonic counter strictly increases, envelopes verify
  under the pinned public key via `verify_base_envelope`, and the CLI
  rejects misconfigurations.

### Unchanged
- Pipeline, audit chain, perimeter allow/deny semantics, MCP wire format.
  When `--overt-signing-key` is absent, proxy behavior is byte-identical
  to v0.23.1.
- OVERT envelope schema (closed 9-field Annex B.6) and `vaara overt
  verify` semantics. The verifier reads canonical-CBOR files one at a
  time, as it did since v0.17.0.

## [0.23.1] - 2026-05-20

**Theme: CI fix.** The v0.23.0 release workflow flipped `npm publish` to
OIDC-only trusted publishing, but the npmjs.com side of the
`@vaara/client` package was not yet configured for a trusted publisher.
The npm leg of the v0.23.0 release therefore failed with a 404 on PUT,
even though PyPI shipped cleanly and the GitHub Release was signed and
attached. This patch restores the token + provenance model that v0.22.0
used.

### Fixed
- `.github/workflows/release.yml`: re-add `NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}`
  env on the npm publish step. The `--provenance` flag is preserved, so
  the npm package still ships with a Sigstore-attested provenance
  statement from the GitHub Actions OIDC token. Auth flows via the
  token, provenance flows via OIDC. OIDC-only publishing can return
  later once trusted publishing is configured on the npm package
  settings page.

### Unchanged
- Library code, public API, audit chain format, OVERT envelope schema,
  MCP proxy behavior. v0.23.1 is a pure CI fix.

## [0.23.0] - 2026-05-20

**Theme: MCP proxy closes the surface-coverage gap.** Resources and
prompts join tools as governed MCP primitives. The same operator
allow/deny perimeter applies symmetrically to `resources/list` /
`resources/read` and `prompts/list` / `prompts/get`, and every allowed
read or prompt retrieval writes a request+decision audit pair to the
hash chain. Vaara now covers all three MCP primitives, not one.

### Added
- `src/vaara/integrations/mcp_proxy.py`: `VaaraMCPProxy.__init__`
  accepts `resource_allowlist`/`resource_denylist` and
  `prompt_allowlist`/`prompt_denylist`. New `_handle_list` helper is
  shared by `tools/list`, `resources/list`, and `prompts/list`. New
  `_handle_resources_read` and `_handle_prompts_get` enforce the
  perimeter and write a request+decision audit pair on allow.
  Perimeter blocks for resources/prompts surface as JSON-RPC errors
  (code -32000) with a `data` payload mirroring the tool-call block
  shape (`vaara_blocked: true`, `decision: "FILTERED"`, `reason`).
- CLI: `--allow-resource URI`, `--deny-resource URI`,
  `--allow-prompt NAME`, `--deny-prompt NAME`, all repeatable.
  Backward compatible: no flags = passthrough.
- `tests/test_integrations_mcp_proxy.py`: twelve new tests covering
  resources/list filtering (denylist, allowlist, no-policy), prompts/
  list filtering, blocked resources/read returns JSON-RPC error,
  blocked prompts/get returns JSON-RPC error, allowed resources/read
  writes the audit pair without invoking the risk scorer, and
  allowed prompts/get writes the audit pair.

### Design
Resource reads and prompt gets are read-oriented MCP surfaces. They
need audit coverage so regulators can reconstruct what an agent
accessed, but they do not run through the risk scorer (the scorer is
for actions). The operator perimeter is the gate. The audit chain
captures the evidence. Tool calls keep the full pipeline (classify,
score, decide, audit).

### Fixed
- `src/vaara/__init__.py`: `__version__` was stale at `0.21.0` after
  v0.22.0 shipped. Bumped to `0.23.0` together with `pyproject.toml`
  and `clients/ts/package.json`.

## [0.22.0] - 2026-05-20

**Theme: MCP proxy operator-side tool filtering.** Adds two repeatable
CLI flags to the MCP proxy that let an operator shape the upstream
tool surface visible to the AI client: `--allow-tool NAME` (if any are
given, only those tools pass through) and `--deny-tool NAME` (those
tools are filtered, wins on overlap with allowlist). Filtered tools
are dropped from `tools/list` responses before the client sees them,
and any `tools/call` to a filtered tool is rejected at the proxy
perimeter with an MCP `isError: true` payload (`decision: "FILTERED"`,
`reason: "Tool filtered by operator policy"`) without forwarding to
the upstream or invoking the risk pipeline.

### Added
- `src/vaara/integrations/mcp_proxy.py`: `VaaraMCPProxy.__init__`
  accepts `allowlist: Optional[set[str]]` and `denylist:
  Optional[set[str]]`. New `_tool_filtered(name)` helper applied to
  both `tools/list` (filters the `result.tools` array) and
  `tools/call` (returns a `FILTERED` block payload before the
  pipeline runs).
- CLI: `--allow-tool NAME` and `--deny-tool NAME`, both repeatable.
  Backward compatible: no flags = passthrough.
- `tests/test_integrations_mcp_proxy.py`: eight new tests covering
  denylist drops, allowlist restricts, denylist-wins-on-overlap,
  no-policy passthrough, filtered tools/call returns block, and
  allowlisted tools/call still routes through the pipeline.

### Verified
- End-to-end smoke against `github/github-mcp-server` (42 real tools):
  `--deny-tool` drops named entries from `tools/list` and rejects
  matching `tools/call`; `--allow-tool` restricts `tools/list` to the
  allowlist and routes allowlisted calls through the pipeline as
  before; no-flag run is identical to v0.21.0 behaviour.

### Use case
Hide write/delete tools (e.g. `delete_file`, `merge_pull_request`)
when the upstream MCP server exposes more capability than the
deployment policy allows. The LLM client never learns the tool
exists, which is materially different from instructing the model
not to call it.

## [0.21.0] - 2026-05-19

**Theme: MCP-aware proxy.** Adds
`vaara.integrations.mcp_proxy.VaaraMCPProxy`, a transparent MCP proxy
that sits between an MCP client (Claude Code, Cursor, any MCP-capable
host) and an upstream MCP server (SAP ADT MCP, SAP Graph API MCP, SAP
Cloud ALM MCP, any community-built MCP server). Every `tools/call`
request from the client routes through Vaara's interception pipeline
before reaching the upstream. Allowed calls forward transparently and
report the upstream outcome back to the scorer. Blocked calls return
an MCP `isError: true` response with the block reason. Other MCP
methods (initialize, tools/list, resources, notifications) forward
unchanged.

### Added
- `src/vaara/integrations/mcp_proxy.py`: `VaaraMCPProxy` and CLI entry
  point. Invoke as `python -m vaara.integrations.mcp_proxy --upstream
  <cmd> [--upstream-arg ...]`.
- `src/vaara/integrations/_mcp_upstream.py`: `UpstreamMCPClient`,
  subprocess lifecycle plus JSON-RPC id demultiplexing on a background
  reader thread. Internal module, not part of the public surface.
- `tests/test_integrations_mcp_proxy.py`: six smoke tests covering
  blocked tool calls, allowed forward, severity mapping, the
  `_vaara_agent_id` strip, non-tools/call passthrough, and invalid
  request handling.

### Strategic frame
The community SAP MCP servers shipped at SAP Sapphire 2026 plus the
Anthropic-SAP partnership announcement put SAP ABAP / Graph / Cloud
ALM behind Claude Code in enterprise developer workflows. None of the
parties (SAP, Anthropic, the community MCP server authors) ships the
runtime governance layer the EU AI Act high-risk obligations require
for those tool calls. The proxy is that layer in OSS today.

## [0.20.0] - 2026-05-18

**Theme: OSS guardrail adapters.** Adds four adapters that take findings
from NVIDIA NeMo Guardrails, Guardrails AI, LLM Guard, and Rebuff and
route them through Vaara's hash-chained audit trail and OVERT envelope
with EU AI Act article tags. Same pattern as v0.19.0's cloud adapters.
The OSS guardrail runs as an upstream signal in the deployer's
environment. Vaara records what the guardrail flagged, normalises it
to a shared vocabulary, and tags each finding against Art. 5, 10, 13,
15, and 53.

The mapping itself is the value. Each adapter is a thin SDK wrapper
around a published category-to-article table extended in
`_content_safety_articles.py`. A deployer can read the table, dispute
a row, and override mappings without touching adapter code.

### Added
- 41 new mapping rows in `_content_safety_articles.py` across the four
  OSS providers (7 NeMo, 10 Guardrails AI, 20 LLM Guard, 4 Rebuff),
  each tagged with EU AI Act articles and OWASP LLM Top 10 codes.
  Seven new Vaara categories: `secrets_leak`, `schema_violation`,
  `output_validation`, `bias`, `language`, `sentiment`,
  `resource_limit`.
- `src/vaara/integrations/nemo_guardrails.py`:
  `NemoGuardrailsAdapter` plus `parse_generation_response`. Maps input
  rails, dialog rails, output rails, and retrieval rails from the
  `GenerationResponse.log.activated_rails` payload.
- `src/vaara/integrations/guardrails_ai.py`: `GuardrailsAIAdapter` plus
  `parse_validation_outcome`. Maps `ValidationOutcome` summary shape
  to findings with PascalCase normalisation across hub validators.
- `src/vaara/integrations/llm_guard.py`: `LLMGuardAdapter` plus
  `parse_scan_result`. Wraps `scan_prompt` and `scan_output`
  callables. Scanner-callable injection for test isolation.
- `src/vaara/integrations/rebuff.py`: `RebuffAdapter` plus
  `parse_detect_response` and `parse_canary_leak`. Records all three
  injection-detection layers (heuristic, model, vector) and the
  canary-word leak on responses.
- Four test files exercising parsers and adapters with no SDK install
  required (29 new tests, 712 total).

### Pyproject
- New optional extras: `nemo-guardrails`, `guardrails-ai`, `llm-guard`,
  `rebuff`. Base install stays free of OSS guardrail dependencies.
- HuggingFace Space added as `Demo` URL in `[project.urls]`.

### README
- HuggingFace Space badge added to the badge row.

### Strategic frame
The OSS guardrails are inputs to Vaara, not Vaara's replacement.
Vaara stays the schema findings flow into: hash-chained audit trail,
OVERT envelope, EU AI Act article-level evidence. Add NeMo, Guardrails
AI, LLM Guard, or Rebuff to your stack and their findings land in the
same Vaara audit record as Bedrock, Azure, and GCP findings did in
v0.19.0.

## [0.19.1] - 2026-05-18

**Patch: audit DB upgrade safety.** Opening an existing audit DB at
any schema version older than the current one crashed on first
MCP-server boot with `no such column: tenant_id`. The init path ran
`SCHEMA_SQL` before migrations, and `SCHEMA_SQL` contains indexes on
columns that later migrations add.

Init now runs migrations from the stored version (or from v0 for
pre-versioned DBs that have no `audit_meta` row yet) BEFORE running
`SCHEMA_SQL` idempotently. Fresh DBs continue to use the existing
single-pass `SCHEMA_SQL` path.

Tests added for the v0 (pre-versioning), v1, and current-version
open paths.

## [0.19.0] - 2026-05-17

**Theme: Big Cloud guardrail adapters.** Adds three adapters that take
findings from AWS Bedrock Guardrails, Azure AI Content Safety, and GCP
Model Armor and route them through Vaara's hash-chained audit trail
and OVERT envelope with EU AI Act article tags. The cloud filter runs
as an upstream signal in the deployer's environment. Vaara records
what the filter flagged, normalises it to a shared vocabulary, and
tags each finding against Art. 5, 10, 13, 15, 53, and the
CSAM-specific obligation from the May 2026 Digital Omnibus political
agreement.

The mapping itself is the value. Each adapter is a thin SDK wrapper
around a published category-to-article table in
`_content_safety_articles.py`. A deployer can read the table, dispute
a row, and override mappings without touching adapter code.

### Added
- `src/vaara/integrations/_content_safety_articles.py`: canonical
  mapping rows for 27 provider categories across the three vendors
  (10 Bedrock, 9 Azure, 8 GCP), each tagged with EU AI Act articles
  and OWASP LLM Top 10 codes.
- `src/vaara/integrations/_content_safety_base.py`: shared
  `ContentSafetyFinding` and `FindingCategory` dataclasses,
  `ContentSafetyScorer` protocol, verdict and severity aggregators.
  Adapter output ships with `to_audit_context()` for
  `pipeline.intercept(context=...)` and `to_overt_metadata()` for OVERT
  envelope `non_content_metadata` (decimal-string severities only per
  Protocol Profile 1.0).
- `src/vaara/integrations/bedrock_guardrails.py`:
  `BedrockGuardrailsAdapter` plus `parse_apply_guardrail_response`.
  Maps Bedrock's five policy buckets (topicPolicy, contentPolicy,
  wordPolicy, sensitiveInformationPolicy, contextualGroundingPolicy).
  Normalises `ANONYMIZED` to the shared `REDACTED` action.
- `src/vaara/integrations/azure_content_safety.py`:
  `AzureContentSafetyAdapter` plus `parse_responses`. Wraps
  `analyze_text`, Prompt Shields, Protected Material, and Groundedness
  Detection behind a single `scan_prompt` / `scan_response` pair.
  Block threshold defaults to severity 4.
- `src/vaara/integrations/gcp_model_armor.py`: `GcpModelArmorAdapter`
  plus `parse_sanitize_response`. Wraps `sanitize_user_prompt` and
  `sanitize_model_response`. CSAM, prompt injection, malicious URIs,
  and SDP findings always block. Responsible-AI filters block at
  `MEDIUM_AND_ABOVE` by default.
- Three test files exercising the parsers and adapters with canned
  fixtures (44 new tests, 680 total). No hard dependency on boto3,
  azure-ai-contentsafety, or google-cloud-modelarmor.

### Pyproject
- New optional extras: `bedrock`, `azure-content-safety`,
  `gcp-model-armor`. Base install stays free of cloud SDK
  dependencies.

### Strategic frame
The cloud filters are inputs to Vaara, not Vaara's replacement.
Vaara stays the schema findings flow into: hash-chained audit trail
with explicit article tags, OVERT envelope metadata, EU AI Act and
OWASP vocabulary on every finding. Not "Vaara works with AWS / Azure
/ GCP." The other direction: AWS Bedrock Guardrails, Azure AI Content
Safety, and GCP Model Armor work inside Vaara's compliance kernel.

## [0.18.1] - 2026-05-17

**Release-bookkeeping patch.** The v0.18.0 release shipped Python to PyPI
successfully but the npm publish step failed because
`clients/ts/package.json` was not bumped from 0.17.0. The TS client is
unchanged in behaviour; this patch restores PyPI/npm version lockstep
established in v0.15.0. No Python code changes versus 0.18.0.

### Changed
- `clients/ts/package.json`: 0.17.0 to 0.18.1 (lockstep with PyPI).
- `pyproject.toml`, `src/vaara/__init__.py`: 0.18.0 to 0.18.1.

## [0.18.0] - 2026-05-17

**Theme: hardware TEE attestation hook (experimental).** Adds an optional
hardware-rooted attestation layer alongside the Ed25519 (or ML-DSA-65)
signature already on the OVERT 1.0 Base Envelope. Initial backend is AMD
SEV-SNP, the natural fit for the confidential-VM deployment model used
in agent runtimes. Intel TDX and SGX backends are tracked for later
releases. The OVERT envelope schema is unchanged (closed per spec); the
TEE report is a sibling artefact bound to a specific envelope by placing
`SHA-512(canonical_cbor(envelope))` into the report's 64-byte
`REPORT_DATA` field.

### Added
- `src/vaara/attestation/tee.py` module with:
  - `parse_sev_snp_report`: binary parser for the 1184-byte SEV-SNP
    attestation report (AMD SEV Secure Nested Paging Firmware ABI
    Specification rev. 1.55, Table 22).
  - `bind_overt_envelope_to_report_data`: computes the 64-byte
    `REPORT_DATA` value that binds a TEE report to a specific OVERT
    envelope. Hash covers all 9 envelope fields including the inner
    Ed25519 signature.
  - `verify_sev_snp_report_signature`: validates the ECDSA P-384
    over the report body against a caller-supplied VCEK PEM.
  - `verify_envelope_binding`: confirms `REPORT_DATA` matches
    SHA-512 of the supplied envelope.
  - `MockSEVSNPAttester`: deterministic in-memory attester for tests
    and CI, building byte-compatible report blobs.
  - `SEVSNPHostAttester`: skeleton that wraps `/dev/sev-guest` and
    raises a clear error when not on an SEV-SNP host.
- `vaara tee parse REPORT` CLI: dump key fields of a SEV-SNP report as
  JSON.
- `vaara tee verify REPORT --vcek VCEK.pem [--overt ENVELOPE.cbor]` CLI:
  signature check plus optional binding check against an OVERT envelope.
- 16 tests in `tests/test_attestation_tee.py` covering parse round-trip,
  size rejection, mock-attester emission, signature verification,
  tamper detection, wrong-VCEK rejection, envelope binding, wrong-curve
  rejection, non-EC-key rejection, unsupported-algo rejection, and the
  non-SEV-SNP host error path.

### Not in this release
- AMD KDS-based cert-chain validation (VCEK to ASK to ARK). Validating
  against AMD's Key Distribution Service requires a network fetch
  against `https://kdsintf.amd.com/` and is tracked for v0.19+.
- Live `/dev/sev-guest` ioctl emission. The `SNP_GET_REPORT` ioctl path
  is documented in `linux/sev-guest.h` and is tracked for v0.19+ once a
  tested SEV-SNP guest is available for integration testing.
- Intel TDX, Intel SGX backends. Same module shape will accommodate them
  via additional attester classes.

### Fixed
- `src/vaara/__init__.py` `__version__` had drifted to `0.15.0` while
  `pyproject.toml` had moved through 0.16.0 and 0.17.0. Both now read
  `0.18.0`.

### Notes
- The OVERT 1.0 envelope schema is closed and unchanged. TEE attestation
  is a strictly additive sibling artefact; an OVERT verifier without TEE
  awareness still validates Vaara envelopes exactly as before.
- 636 Python tests pass (was 620 + 16 new).

## [0.17.0] - 2026-05-17

**Theme: Vaara as the OVERT 1.0 reference verifier.** Vaara has emitted
OVERT 1.0 Protocol Profile 1.0 Base Envelopes since v0.10.0 and Phase 3
IAP attestations since v0.13.0. This release adds the CLI surface so any
operator, auditor, or Independent Attestation Provider can verify a Base
Envelope produced by any conformant emitter, not only Vaara. The verifier
is implementation-agnostic and reads the canonical CBOR wire format
defined in Annex B.6 verbatim.

### Added
- **`vaara overt verify RECEIPT.cbor --pubkey-file PUB.bin` CLI.**
  Validates an OVERT 1.0 Base Envelope from a canonical CBOR file against
  a supplied raw 32-byte Ed25519 public key. The schema is closed per
  the OVERT 1.0 spec, so envelopes carrying unknown fields are rejected.
  Exit 0 on valid, 1 on invalid envelope or signature mismatch, 2 on
  argument or I/O errors. Requires the existing `vaara[attestation]`
  extra (already present for emission). Accepts `--pubkey-hex` as an
  inline alternative to `--pubkey-file`.
- 10 new tests in `tests/test_overt_verify_cli.py` covering happy path,
  wrong pubkey, hex form, malformed hex, wrong-length pubkey, missing
  receipt, malformed CBOR, missing required fields, closed-schema
  rejection of extra fields, and tampered-signature rejection.

### Notes
- Anyone implementing OVERT 1.0 (overt.is) can route their conformance
  check through `vaara overt verify` without taking a runtime dependency
  on Vaara's emitter side. The CLI reads only the Annex B.6 wire format.
- 620 Python tests pass (was 610 + 10 new). TypeScript client bumped to
  0.17.0 for PyPI/npm lockstep, no code change.

## [0.16.0] - 2026-05-17

**Theme: auditor-shippable PDF.** The article-evidence report has rendered
as Markdown, JSON, and narrative since v0.13.0; auditors and Notified
Bodies consume PDFs. ``vaara compliance report --format pdf --out FILE``
now ships a styled, single-file PDF suitable for attaching to a conformity
submission or compliance binder. No new wire-format surface. The underlying
``ConformityReport`` is the same artefact in a different render.

### Added
- **`render_pdf(report, path)` in `vaara.compliance.render`.** Single-file
  PDF: cover with system metadata + integrity, per-domain article tables,
  per-article detail sections with status, strength, evidence age, gaps,
  recommendations, and sample record IDs. Layout mirrors the Markdown
  rendering. Returns bytes written.
- **`vaara compliance report --format pdf --out FILE` CLI option.**
  ``--out`` is required for ``pdf`` (binary output, no stdout path).
  Missing reportlab raises a clear ``ImportError`` pointing at
  ``pip install 'vaara[pdf]'``.
- **`vaara[pdf]` optional extra.** ``reportlab>=4.0``. Pure-Python, no
  native deps. Keeps the base install lean for deployers who only need
  Markdown/JSON.
- 4 new tests in ``test_compliance_render.py``: PDF magic bytes + EOF
  marker, HTML metachar escaping in ``system_name`` (so a hostile
  deployer name cannot smuggle ``<b>`` into a regulator-facing PDF),
  broken-chain rendering, and the missing-reportlab error path.

### Notes
- A3 commit-prove receipt pair is **already shipped** as of v0.13.0
  (``vaara.audit.receipts``, ``vaara trail receipt`` CLI); this release
  closes the remaining A-tier item from the post-v0.15.0 competitive
  differentiation move list.
- 610 tests pass (was 606 + 4 new).

## [0.15.0] - 2026-05-17

**Theme: Vaara reaches JavaScript.** First-party TypeScript HTTP client
for the v1 API so JS/TS agents (LangChain.js, Vercel AI SDK, MCP, any
Node service) can call Vaara without spawning a Python sidecar.

### Added
- **`@vaara/client` npm package (clients/ts).** Typed wrapper over
  every v1 endpoint: ``score`` / ``reportOutcome`` /
  ``appendAuditEvent`` / ``getActionChain`` / ``verifyAuditChain`` /
  ``reloadPolicy`` / ``detectInjection`` / ``detectPII`` /
  ``serverInfo`` / ``health``. ESM-only, Node 18+ (global ``fetch``),
  TypeScript declarations shipped. Server-side ``{error: {code,
  message}}`` bodies map to ``VaaraError`` (carries ``status`` +
  ``code`` for pattern matching against the spec); network failures
  map to ``VaaraTransportError``. Optional ``fetch`` injection makes
  the client trivially mockable in tests.
- **Release workflow gains a publish-npm job.** Guarded on the
  repository variable ``NPM_PUBLISH_ENABLED`` so the first publish
  stays manual. Once enabled and an ``NPM_TOKEN`` secret is set, every
  tag push publishes ``@vaara/client`` to npm with provenance.
- 6 new TypeScript tests covering URL construction, JSON body
  serialisation, the 4xx to ``VaaraError`` path with server-supplied
  code, the network-failure to ``VaaraTransportError`` path, the
  detector response shape, and constructor input validation.

### Notes
- Python package is unchanged in v0.15.0 beyond the version bump; this
  release ships the JS/TS surface alongside the existing PyPI artefact.

## [0.14.0] - 2026-05-17

**Theme: audit-retention horizon + composer position.** Two additions.
A pluggable signer abstraction lets operators sign the regulator-handoff
export envelope with ML-DSA-65 (FIPS 204) for retention horizons that
cross the credible quantum threshold. A composite-scorer module reuses
Vaara's own v0.10.0 /v1/score wire contract on the inbound side so
external scorers (NeMo Guardrails, another Vaara instance, any peer)
slot in alongside Vaara's adaptive scorer with a single object.

### Added
- **Pluggable signer abstraction (`vaara.audit.signer`).** New
  ``Signer`` / ``Verifier`` protocols with ``Ed25519Signer`` and
  ``MLDSA65Signer`` implementations. ``export_signed`` accepts a
  ``signer=`` keyword (mutually exclusive with the legacy
  ``signer_key`` Ed25519 PEM path). The export manifest carries
  ``signature_algorithm`` so verifiers dispatch automatically.
  Ed25519 exports still write ``signer_pubkey.pem`` for backward-
  compatible verification by older clients; ML-DSA-65 exports write
  ``signer_pubkey.bin`` (raw bytes, 1,952 bytes). ``verify_signed``
  reads the algorithm field and routes to the right verifier.
  ML-DSA-65 requires the new ``vaara[pq]`` extra
  (``pip install 'vaara[pq]'``), which pulls pure-Python
  ``dilithium-py``.
- **External-scorer composition (`vaara.scorer.composition`,
  `vaara.scorer.composite`).** ``ExternalScorer(url)`` calls a remote
  ``/v1/score`` endpoint (any service that implements the Vaara
  v0.10.0 wire contract) and returns the result in Vaara's internal
  scorer dict shape. Transport errors, malformed bodies, and non-JSON
  responses fail closed to ``deny`` with a structured reason in the
  ``raw_result``. ``CompositeScorer(scorers, combine="max"|"mean"|
  callable)`` runs Vaara's ``AdaptiveScorer`` alongside one or more
  external scorers and merges the results. The composite preserves
  the strongest decision (``deny`` > ``escalate`` > ``allow``) so one
  member firing high blocks the action. The composite's
  ``evaluate(context) -> dict`` shape matches what
  ``InterceptionPipeline`` already expects, so it drops into existing
  pipelines as a direct replacement.
- 20 new tests (10 signer + Ed25519 + ML-DSA-65 round-trip and zip
  tamper-detection; 10 composition + external-scorer error handling
  + composite combine modes).

## [0.13.0] - 2026-05-17

**Theme: operator surface + OVERT Phase 3 path.** Four additions that
close the most legible competitive gaps without diluting the kernel
position. Hot policy reload meets the Galileo Agent Control selling
point on its own ground. The OVERT 1.0 Phase 3 Independent Attestation
Provider (IAP) reference closes the AAL-3 to AAL-4 promotion path that
v0.11.0's Provisional Receipt opens, so Vaara owns the full path
without forcing dependence on an external IAP vendor. Named injection
and PII detectors expose existing scoring surface under buyer-visible
labels. A static HTML article-coverage dashboard adds the auditor-
facing visual artefact that the peer set has converged on.

### Added
- **Hot policy reload.** New `vaara.policy.controller.PolicyController`
  owns the live `Policy` and runs registered listeners under a write
  lock on `reload()`. `AdaptiveScorer.apply_policy(policy)` rebinds
  thresholds and sequence patterns atomically under the scorer's own
  RLock; an `evaluate()` call in flight on another thread either sees
  the old `(allow, deny)` pair or the new one, never a torn half.
  Conformal calibration, MWU expert state, and agent profiles are
  preserved across reloads. Malformed reloads are rejected with the
  previous policy left live. `POST /v1/policy/reload` accepts a
  server-side path or an inline body; `vaara serve --policy PATH`
  enables the endpoint; `vaara policy reload POLICY_PATH` triggers
  reload over HTTP from the operator's shell.
- **OVERT 1.0 AAL-4 Phase 3 IAP reference.** New
  `vaara.attestation.iap` ships a `Phase3Attestation` dataclass that
  wraps a Vaara `BaseEnvelope` with a notary Ed25519 signature (over a
  domain-separated prefix + canonical-CBOR of the inner envelope
  including its signature) and a transparency-log inclusion proof.
  Structural independence between the Arbiter key and the notary key
  is enforced at both emit and verify. New
  `vaara.attestation.transparency_log.InProcessTransparencyLog`
  implements an RFC 6962-style binary Merkle tree with domain-
  separated leaf and internal hashes; `append()` /
  `inclusion_proof()` / `root_hash` match the shape a sigstore Rekor
  adapter would expose, so a production deployment can swap in Rekor
  at the same call sites without changing the IAP contract.
- **Named injection + PII detector aliases.** `vaara.detect.detect_injection`
  routes free text through the same AdversarialClassifier behind
  vaara-bench-v1's published numbers (heuristic fallback when the ml
  extra is absent; the `backend` field reports which path served the
  call). `vaara.detect.detect_pii` is a zero-dependency regex extractor
  over six categories: email, phone, US SSN, IPv4, credit_card
  (Luhn-checked), IBAN (mod-97 checksum). `POST /v1/detect/injection`
  and `POST /v1/detect/pii` mirror the CLI. `vaara detect injection`
  and `vaara detect pii` read text from `--text`, `--file`, or
  `--stdin` and exit non-zero when the detector fires.
- **Static HTML article-coverage dashboard.**
  `vaara.compliance.dashboard.render_html` produces a single
  self-contained HTML page with embedded CSS, no JavaScript, no
  external assets, no network calls. Same content as the Markdown
  renderer (system metadata, audit-trail integrity, summary, critical
  gaps, per-domain article tables, detailed per-article sections) with
  status badges as colored pills and a print-friendly stylesheet.
  `vaara compliance dashboard --db PATH --out PATH` writes the page;
  a trailing slash or existing directory drops `index.html` inside.
- **OpenAPI spec coverage.** `docs/openapi.yaml` adds
  `/v1/detect/injection`, `/v1/detect/pii`, and `/v1/policy/reload`
  with full request and response schemas. The spec remains the
  authoritative integration surface.
- 53 new tests (14 PolicyController + reload HTTP; 11 IAP +
  transparency-log; 19 detect (injection + PII + HTTP); 9 HTML
  dashboard). Total 586 passing, 12 skipped.

## [0.12.0] - 2026-05-16

**Theme: agentic OVERT reference, published benchmark, product liability hook.**
v0.12.0 ships three additions that widen Vaara's category position.
First, an OVERT 1.0 MEA-2 Statistical Safety Signal Protocol (S3P)
emitter with exact Clopper-Pearson confidence intervals and a proposed
Protocol Profile extension that reports aggregate statistics over
Vaara's per-action conformal prediction intervals. Second, an explicit
control-by-control mapping of Vaara to OVERT 1.0 Part 3 (Agentic AI
Controls) in COMPLIANCE.md, covering Tool-Call Governance (Section 11),
MCP Server Trust Governance (Section 11.5), Multi-Agent System Controls
(Section 12), Capability-Based Access Control (Section 13), Agent
Disclosure (Section 14), Human-in-the-Loop Attestation (Section 15),
and Behavioural Drift Governance (Section 16). Third, vaara-bench-v1: a
versioned, reproducible adversarial-detection benchmark with frozen
corpus, methodology, and headline numbers under the v0.11.0 scorer
(soft TPR 100%, hard FPR 0% across 77 synthetic traces). Plus a forward
section on EU Product Liability Directive 2024/2853 (Article 9
rebuttable-presumption hook), transposition deadline 9 December 2026.

### Added
- **OVERT 1.0 MEA-2 S3P emitter (`vaara[attestation]` extra).** New
  module `vaara.attestation.s3p` provides exact Clopper-Pearson
  binomial confidence intervals via a pure-Python regularized
  incomplete beta function (no scipy dependency), plus a closed-schema
  S3P attestation per MEA-2.6 (Ed25519-signed, canonical CBOR
  encoding, decimal-string rates per Protocol Profile 1.0 numeric
  rules). Public API: `emit_s3p_attestation`, `verify_s3p_attestation`,
  `make_epoch_nonce_commitment`, `clopper_pearson_ci`,
  `regularized_incomplete_beta`, `S3PAttestation`,
  `ConformalExtension`.
- **`ConformalExtension`** Protocol Profile extension (proposed).
  Optional field in the signed S3P metadata that reports aggregate
  statistics over Vaara's per-action conformal prediction intervals
  alongside the standard binomial CI. Carries the same non-parametric
  coverage guarantee with no distributional assumption.
  Standard OVERT verifiers ignore the extension. Vaara-aware
  verifiers cross-check it against the per-action receipts.
- **vaara-bench-v1.** Versioned adversarial-detection benchmark with
  frozen corpus (`bench/adversarial_corpus.jsonl`, SHA-256
  `7a3219776e1c93a5127ab3b63832d73ba75f32fa044cabdbaa4e5d7088b33ff2`),
  frozen methodology (`bench/scorer_eval.py`), and frozen headline
  numbers under v0.11.0 (soft TPR 100%, soft FPR 20%, hard TPR
  28.85%, hard FPR 0%). Spec doc at `bench/vaara-bench-v1.md`.
  Machine-readable results at `bench/vaara-bench-v1-results.json`.
  Apache-2.0 licensed.
- 18 new tests in `tests/test_attestation_s3p.py` covering
  Clopper-Pearson against textbook values, the k=0 and k=n
  endpoints, round-trip emit-and-verify, status-band selection
  (compliant / threshold_exceeded / insufficient_sample), conformal
  extension attachment, tampered-field rejection, wrong-key rejection,
  and invalid-count rejection.

### Documentation
- **COMPLIANCE.md** gains two sections. **OVERT 1.0 Part 3 (Agentic AI
  Controls) mapping** walks Vaara's coverage of every control in
  Sections 11 through 16 with a ✅ / ◐ / ◯ marker for satisfied,
  partial, or gap-to-deployer / future-work. Covers TOOL-1 through
  TOOL-5, MCP-1/2/3, MULTI-1/2, CAP-1/2, DISC-1, HITL-1/2/3/4,
  SESS-1..5, STATE-1/2, IDENT-1, and DRIFT-1, plus a final S3P
  (Section 9, MEA-2) subsection. **EU Product Liability Directive
  2024/2853** is added as a new forward-looking section covering
  Articles 7, 8, and 9 (rebuttable presumption of defectiveness)
  and how Vaara's evidence stream supports the Article 9 rebuttal.
  Transposition deadline 9 December 2026 (Article 22).
- **README.md** updated to reference vaara-bench-v1 in the Numbers
  section and to mention the v0.12.0 S3P emitter and Part 3 mapping
  in the OVERT 1.0 attestation section. Sample code uses
  `arbiter_version="vaara/0.12.0"`.
- **bench/README.md** points to the vaara-bench-v1 spec doc as the
  canonical citation surface for external references to the
  benchmark.

## [0.11.0] - 2026-05-16

**Theme: first OSS Python reference implementation of OVERT 1.0.** v0.11.0
ships an emitter for the OVERT 1.0 Protocol Profile 1.0 Base Envelope
(Glacis Technologies, overt.is, March 2026) at AAL-3 Phase 2 (Provisional
Receipt). Vaara operates as the Arbiter in OVERT terms. Phase 3 notary
attestation and AAL-4 promotion are left to external Independent
Attestation Providers, per OVERT's structural-independence principle and
Vaara's always-OSS-kernel rule. Plus eIDAS NOTICE and OVERT position
statement in COMPLIANCE.md.

### Added
- **OVERT 1.0 Protocol Profile 1.0 Base Envelope emitter (`vaara[attestation]` extra).**
  First OSS Python reference implementation of the OVERT 1.0 AAL-3
  Phase 2 (Provisional Receipt) emission path. New module
  `vaara.attestation.overt` produces 9-field closed-schema Base
  Envelopes per Annex B.6, canonically CBOR-encoded per RFC 8949
  Section 4.2, Ed25519-signed per RFC 8032, with HMAC-SHA256 keyed
  request commitments per Annex B.4 and SHA-256 encoder-identity
  derivation. IEEE-754 floats are rejected at the canonical-encoding
  boundary per Protocol Profile 1.0 numeric rules. Vaara operates as
  the **Arbiter** in OVERT terms. External Independent Attestation
  Providers promote AAL-3 emission to AAL-4 by attaching Phase 3
  notary signatures and transparency-log inclusion proofs. Public API:
  `BaseEnvelope`, `emit_base_envelope`, `verify_base_envelope`,
  `make_request_commitment`, `encoder_binary_identity`, `canonical_cbor`.
- 13 new attestation tests (`tests/test_attestation_overt.py`).
  Round-trip verification, IEEE-754-float rejection at every nesting
  level, key-identifier binding, monotonic-counter binding, tampered-field
  rejection.

### Documentation
- **COMPLIANCE.md** gains two sections under the deployer-vs-Vaara
  ownership boundary: an explicit eIDAS NOTICE (Vaara hash chains and
  commit-prove receipts are technical evidence, not Qualified Electronic
  Attestations of Attributes under Regulation (EU) 910/2014 Article
  3(45), not a qualified trust service under Article 3(16)) and a
  position statement relative to OVERT 1.0 (Glacis Technologies). Vaara
  is structurally independent of the agent it governs and maps to
  OVERT AAL-3 operator-controlled attestation. Reaching AAL-4 requires
  pairing Vaara with an external Independent Attestation Provider. The
  design admits an external IAP layer without internal change.

## [0.10.0] - 2026-05-16

**Theme: Vaara as the kernel others build around.** v0.10.0 ships the
network-callable surface, the auditor-facing evidence artefact, and the
offline-verifiable receipt pair. Each of the three pieces is additive
and backward-compatible. Together they reposition Vaara from a Python
library to a runtime kernel that control planes, audit consumers, and
orchestration frameworks reference. The HTTP contract at
`docs/openapi.yaml` is versioned `/v1/` independently of the project
version, so the wire surface can stabilise without locking the
library cadence.

### Added
- **HTTP API reference server (`vaara[server]` extra).** Exposes the
  conformal scorer and hash-chained audit trail over HTTP per the
  contract in `docs/openapi.yaml`. Endpoints: `POST /v1/score`,
  `POST /v1/score/outcome`, `POST /v1/audit/events`,
  `GET /v1/audit/actions/{action_id}/chain`, `POST /v1/audit/verify`,
  `GET /v1/server`, `GET /v1/health`. The spec is authoritative. The
  reference server in `src/vaara/server/` is a FastAPI implementation
  suitable for local development and modest production loads.
- **`vaara serve`** CLI subcommand.
- **OpenAPI 3.1 contract at `docs/openapi.yaml`.** Stable v1 surface,
  intended as the integration point for control planes, orchestration
  frameworks, and audit consumers. Vaara defines the interface. The
  vendors call it.
- 11 new HTTP server tests (`tests/test_server.py`).
- **Auditor-facing evidence report rendering.** New module
  `vaara.compliance.render` with `render_markdown`, `render_json`, and
  `render_narrative` for the `ConformityReport` produced by
  `ComplianceEngine.assess`. Markdown output has per-domain article
  tables, per-article detail sections, evidence status badges,
  audit-chain integrity flagging, and a deployer-owns-the-decision
  disclaimer suitable for shipping to a regulator or attaching to an
  internal conformity submission.
- **`vaara compliance report --db PATH --format md|json|narrative
  [--out FILE]`** CLI subcommand. Loads an audit SQLite DB, runs
  `ComplianceEngine.assess`, renders to chosen format.
- 5 new compliance-render tests (`tests/test_compliance_render.py`).
- **Article 12 commit-prove receipt pair.** New module
  `vaara.audit.receipts` derives an offline-verifiable receipt from the
  existing audit chain: a `commit_hash` covering the gate-time decision
  (action_id, decision, risk_score, thresholds, decided_at) and an
  `outcome_hash` covering the post-execution outcome and embedding the
  commit_hash. Open-standards SHA-256 over canonical JSON, no external
  cryptography library required. Verification needs only `hashlib`,
  enabling per-action handoff to auditors without sharing the full chain
  or key material.
- **`vaara trail receipt --db PATH --action-id ID [--out FILE]`** CLI
  subcommand. Extracts and verifies the receipt pair, prints JSON.
- 11 new receipt tests (`tests/test_receipts.py`).

## [0.9.0] - 2026-05-15

**Theme: policy artifact validate + test framework.** v0.9.0 ships the
two CLI surfaces that turn the YAML / JSON policy from "a config file
the pipeline loads" into a policy-as-code artifact that compliance
teams can validate and test in CI, independently from the agent code
it governs.

### Added
- **`vaara.policy.validate` module.** `validate(Policy)` returns a
  `ValidationReport` with structured `PolicyIssue` records (level,
  code, path, message). Semantic warnings emitted: empty
  `action_classes`, threshold bands narrower than 0.05 (default and
  per-class merged), threshold overrides targeting an action class
  not declared, sequence-pattern steps that do not name a declared
  action class, escalation routes whose `if_articles` overlap no
  emitted regulatory tag, missing default escalation route.
  `validate_source(source, fmt="auto")` combines load and check so a
  single call yields `(policy, report)` or `(None,
  report-with-error)`. Stable JSON shape via `ValidationReport.to_dict()`.
- **`vaara.policy.test_cases` module - Conftest analog for Vaara
  policies.** `evaluate(policy, action_class, risk_score,
  matched_sequences=())` is the underlying primitive: applies any
  matched sequence pattern boosts (capped at 1.0), resolves the
  merged threshold for the action class, returns
  `EvaluationResult(verdict, boosted_risk, route)`. `PolicyTestCase`
  captures the inputs plus an expected verdict and (for
  `escalate`) an expected operator route. `run_test_cases(policy,
  cases)` runs the list, captures evaluation errors as failed cases
  rather than raising, and returns `PolicyTestResult` rows. The
  evaluator validates inputs at the boundary (risk score in `[0,1]`,
  action class declared, matched sequences known).
- **`vaara.policy.test_cases_io` module.** `load_test_cases(path)`
  reads a YAML or JSON cases document and returns a list of
  `PolicyTestCase`. Document shape: a top-level `cases:` list with
  `action_class`, `risk_score`, optional `matched_sequences`, and an
  `expect:` block carrying `verdict` and optional `route`.
- **`vaara policy validate POLICY_PATH [--json]`** and **`vaara
  policy test POLICY_PATH --cases CASES_PATH [--json]`** CLI
  subcommands. Both honour standard CI exit codes: validate returns
  1 on parse errors (warnings do not flip), test returns 1 on any
  failed case (and 2 if the policy itself fails to parse).
- **`examples/policies/test_cases.yaml`** - six worked test cases
  exercising thresholds, sequence-pattern boost, default and
  article-matched escalation routes against
  `examples/policies/full.yaml`.
- **48 new tests** (`tests/test_policy_validate.py`,
  `tests/test_policy_test_cases.py`,
  `tests/test_policy_test_cases_io.py`, `tests/test_policy_cli.py`)
  covering report shape, every warning code, evaluator edges
  (threshold-equal-escalate, boost cap at 1.0, unknown action class,
  unknown sequence, out-of-range risk), case-construction validation
  (bad verdict, route without escalate), YAML and JSON case files,
  the worked example end-to-end, and CLI smoke for each subcommand
  including `--json`. Full suite 472 / 472 pass.
- **COMPLIANCE.md** gains a *Policy artifact review* subsection under
  Article 14 documenting both CLI surfaces as the path to reviewing
  the policy artifact independently from the agent code.

### Note
Backwards-compatible. Pure addition. No existing module signatures
change. `Policy` and the load path are unchanged. The new modules
sit beside them under `vaara.policy.*`.

### Provenance note
The PyPI artifact for v0.9.0 was uploaded via `twine` (not via
trusted publishing) after a GitHub Actions infrastructure outage on
2026-05-15 caused the Release workflow's Build job to fail at the
artifact upload step, which skipped the downstream
trusted-publishing job. The wheel and sdist on PyPI are bit-identical
to what `python -m build` produced from commit `ea201fe`. Subsequent
releases use trusted publishing as standard.

## [0.8.0] - 2026-05-14

**Theme: Article 73 serious-incident export (interim).** Adds the export
surface a provider needs to satisfy EU AI Act Article 73 reporting
obligations. The Commission template promised by Article 73 paragraph 7
has not published, so the format is explicitly INTERIM. When the template
publishes, the schema_version bumps and new fields land additively.

### Added
- **`vaara.audit.incident_export` module.** `build_incident_report()` and
  `write_incident_report()` produce a standalone JSON document covering
  the fields Article 73 paragraphs 1 to 7 reference. Schema version
  `vaara-incident/1.0`. Reporting deadline is derived from the Article
  3(49) sub-category: general 15 days, death of a person 10 days,
  Article 3(49)(b) widespread or serious 2 days. Trigger record must have
  `event_type` in `{outcome_recorded, action_blocked, policy_override}`.
  Other event types describe normal operation and are rejected at
  build time.
  Paragraph 5 incomplete-initial plus complete-follow-up workflow is
  supported via `report_status` and `previous_report_id`. Audit records
  are referenced by `record_id` as evidence. The report does not
  duplicate their content. Pair with `vaara trail export-prov` for
  the full evidence bundle.
- **`vaara trail export-incident` CLI subcommand.** Reads a trail JSONL
  plus an operator-supplied incident metadata JSON, writes the report
  to the output path. Picks the most recent trigger-eligible record by
  default. Explicit `--trigger-record-id ID` overrides. No external
  template dependency, zero new runtime deps.
- **`tests/test_incident_export.py`** covers schema shape, deadline
  mapping per Article 3(49) sub-category, trigger-event validation,
  causal-link and reporter-role validation, initial-versus-complete
  report sequencing, and end-to-end trail-to-report flow.

### Note
Backwards-compatible. Pure addition. The `AuditRecord` schema is
unchanged. Severity is an incident-level concept, not a per-event
concept, so no `AuditRecord` column was added.

---

**Theme: human-in-the-loop review queue (Article 14).** Adds the
storage layer and operator surface that turn an `escalate` decision
into a substantive Article 14(4)(d) override path. The pipeline
already wrote `ESCALATION_SENT` for every escalated action. With a
queue wired in, those actions now wait in a queryable place with
their conformal interval, get claimed by an operator, and produce an
`ESCALATION_RESOLVED` audit record when resolved.

### Added
- **`vaara.audit.review_queue` module.** `ReviewQueue` is a
  SQLite-backed queue in its own DB file, separate from the audit DB
  (which keeps its append-only invariant clean). Statuses:
  `pending to claimed to resolved` happy path, `pending to expired`
  stale path. Resolutions: `allow`, `deny`, `abstain`. `enqueue`
  records each item with the conformal interval, risk signals,
  bucket category, and request parameters/context as JSON. The
  interval is what makes Article 14 oversight substantive rather
  than cosmetic (see `COMPLIANCE.md`). `claim` is optimistic -
  concurrent claim races resolve with one winner and
  `InvalidTransitionError` for the loser. `resolve` accepts an
  optional `trail` and writes `ESCALATION_RESOLVED` so the Article
  14(4)(d) evidence row lands on the hash chain. `expire_stale`
  marks pending items past a timeout. Claimed items are left alone
  since they are under active review.
- **`InterceptionPipeline(review_queue=...)`.** Optional constructor
  parameter. When supplied, every `escalate` decision is enqueued
  alongside the existing `ESCALATION_SENT` audit record. Default
  `None` preserves prior behaviour bit-for-bit. Queue write failure
  logs and continues - the action is already gated by the escalate
  verdict and the audit record stands.
- **`vaara review` CLI.** Subcommands `list`, `show`, `claim`,
  `resolve`, `expire`. `resolve --audit-db PATH` writes the
  `ESCALATION_RESOLVED` record into an audit DB at that path so a
  single operator command produces both the queue terminal state
  and the Article 14(4)(d) hash-chain evidence.
- **`tests/test_review_queue.py`** covers schema round trip,
  pending-only and cross-status listing, agent-id filter, claim
  race semantics, resolve from `pending` and from `claimed`,
  unknown-resolution rejection, terminal-state guard, trail
  write-through, expire-stale with `dry_run`, claimed-items-left-alone,
  positive-timeout guard, counts partitioning, file-backed
  persistence, oversized-blob capping, pipeline enqueue-on-escalate
  end-to-end, no-enqueue-on-allow, and CLI smoke for each
  subcommand including `resolve --audit-db` writing the audit row.

### Note
Backwards-compatible. Pure addition. No existing schemas migrate.
The queue lives in its own DB file with its own `review_queue_meta`
schema-version row.

## [0.7.0] - 2026-05-10

**Theme: class-conditional conformal calibration.** v0.7.0 adds Mondrian per-category conformal prediction on top of the marginal split conformal that has shipped since v0.5.x. The same coverage guarantee now holds independently per action category, so a 90% headline can no longer hide a 60% miss rate on `credential_exfil` behind a 99% pass rate on `benign_control`. Eval surfaces the per-category breakdown. PROV-DM exports surface the calibration context an external auditor needs to read each interval honestly.

### Added
- **Mondrian per-category conformal calibration in `ConformalCalibrator`.** Optional `category` argument on `add_calibration_point` and `predict_interval` routes residuals to a per-category bucket. Each bucket carries its own residual deque, FACI alpha trajectory, and conformal quantile, so a per-category coverage guarantee holds independently. New helpers: `calibration_size_for(category)`, `is_calibrated_for(category)`, `effective_alpha_for(category)`. Calls without a `category` argument land in a single shared bucket and behaviour is identical bit-for-bit to marginal split conformal. Reference: Vovk (2012), "Conditional validity of inductive conformal predictors". PR #60.
- **`AdaptiveScorer.mondrian_categories` constructor flag.** When set to `True`, each action's category (derived from the `tool_name` prefix via `_category_of`) routes through to the calibrator at all three call sites: `evaluate`, `dry_run_evaluate`, `record_outcome`. Default `False` preserves the marginal contract for existing callers. The 50-pair seed prior continues to populate the default bucket regardless, so a fresh Mondrian-mode scorer falls back to the conservative `point ± 0.3` interval per untouched bucket. PR #61.
- **`scripts/eval_adversarial.py --mondrian` flag.** Constructs `Pipeline(scorer=AdaptiveScorer(mondrian_categories=True))` so the adversarial corpus can be evaluated under both regimes for direct comparison. PR #61.
- **Empirical conformal coverage in `scripts/eval_adversarial.py`.** Per-category and overall: `coverage` (fraction of entries where `lower <= true_risk <= upper`) and `mean_interval_width` (mean of `upper - lower`). Per-entry rows now also carry `lower`, `upper`, and `actual_risk`. Result JSON gains an `overall` block alongside the existing `summary`. Eval-only change. PR #59.
- **Conformal calibration context in PROV-DM exports.** W3C PROV-JSON score Entities now surface `vaara:calibrationSize`, `vaara:effectiveAlpha`, and (in Mondrian mode) `vaara:bucketCategory`. An external auditor can now distinguish a wide interval from a cold calibrator, from FACI alpha drift, or from genuine uncertainty without re-running Vaara. `RiskAssessment` dataclass gains `effective_alpha` and `bucket_category` fields. `Pipeline` writes them through to the audit trail with defensive coercion. Pre-enrichment audit records produce identical PROV output (attributes are omitted when source data is missing or `None`). PR #62.

### Changed
- `vaara.scorer.adaptive.RiskAssessment` adds two optional fields: `effective_alpha: float = 0.10` and `bucket_category: Optional[str] = None`. Existing constructors that do not pass them keep working unchanged. PR #62.

### Note
Backwards-compatible release. All four PRs are additive. Mondrian is opt-in via `mondrian_categories=True`, and PROV enrichment fields are omitted when their source data is missing.

## [0.6.2] - 2026-05-05

**Theme: standards-track lineage.** v0.6.2 adds W3C PROV-DM as a second standards-track audit format alongside the JTC21-bound trajectory of v0.6.0. The audit record schema and event lifecycle are unchanged. Vaara now also emits PROV-JSON, so any PROV-aware consumer can ingest a trail without a Vaara-specific adapter.

### Added
- **W3C PROV-DM audit-trail export.** New module `vaara.audit.prov_export` and CLI subcommand `vaara trail export-prov --trail PATH --out PATH [--action-id ID] [--no-chain]`. Emits PROV-JSON (W3C Submission, 2013). Two layers: per-action bundles (lifecycle as Activities, request/score/decision/outcome as Entities, AI agent + Vaara pipeline + human reviewer as Agents) and an audit-record chain layer (each record as a `prov:Bundle`-typed Entity, consecutive records linked via `wasDerivedFrom`/`prov:Revision`). Regulatory articles surface as `aiact:satisfies` and `dora:satisfies` attributes on the Activity that generated them. Decision and outcome entity IDs are scoped by `record_id` so multi-override lifecycles preserve every prior decision. Chain edges are derived from `previous_hash` (not iteration order) so filtered slices and equal-timestamp records never assert false lineage. Cryptographic chain verification stays Vaara's job (`vaara trail verify`). Zero new runtime deps.

## [0.6.1] - 2026-04-27

**Theme: documentation sync to PyPI.** v0.6.0 shipped the functional changes (policy DSL, retention purge, transparency taxonomy, distribution-shift / stack-ablation / PAIR evals, lint sweep) but the README, library docstring, example file headers, and PyPI tagline stayed at the v0.5.0 framing. PyPI's package page kept publishing pre-rebalance numbers and the wrong default threshold. v0.6.1 ships only the documentation cleanup so new PyPI installs see the current state.

### Changed
- `README.md`: replaced the "Numbers" section with the v0.6 distribution-shift table (97.1% recall / 70.0% FPR hand-curated held-out. 95.2% / 87.5% LLM-generated in-sample) plus PAIR ASR 0.0% (0/25). Threshold default 0.5 -> 0.55, corpus description updated to the 5,955-entry rebalanced corpus, threshold-direction note corrected (recall drops as threshold rises). PR #44.
- `src/vaara/adversarial_classifier.py`: rewrote the module docstring. Removed all version-bound numbers. Readers now point at README + COMPLIANCE so the docstring does not go stale on every release. PR #44.
- `examples/adversarial_classifier.py`: ship-note threshold 0.8 -> 0.55. PR #44.
- `scripts/classifier_vs_heuristic.py`: clarified the script is the v0.5.0 historical reproducer, not the current production training path. PR #44.
- `pyproject.toml`: rephrased description for cleaner PyPI tagline rendering. PR #45.

### Note
No functional code changes. v0.6.0 users are on the same code. V0.6.1 only refreshes documentation surfaces visible to new PyPI installs and to anyone reading the package source.

## [0.6.0] - 2026-04-27

**Theme: standards alignment + legibility.** v0.5.x was the capability axis (jailbreak coverage closed, classifier rebalanced). v0.6 is the legibility axis: policies become readable, audit records become standards-aligned, adversarial numbers become honest, architecture contribution becomes documented.

### Added
- **`vaara.policy` package - JSON-native policy loader plus optional YAML via `vaara[yaml]` extra.** Frozen dataclasses for action classes, threshold curves, sequence patterns, and escalation routes. Hand-rolled validation with field-path error messages. Reuses existing `vaara.taxonomy.actions` enums verbatim. Threshold partial-overrides supported (set just `deny`, inherit default `escalate`). Implements Sketch A from the v0.6 DSL design exploration. Embedded Python DSL (Sketch B) and standalone DSL (Sketch C) stay deferred to v0.7+ pending external pull.
- **`vaara trail purge --db PATH --retention-days N (--tenant TID | --all-tenants) [--dry-run]` CLI subcommand** plus `SQLiteAuditBackend.purge_older_than(seconds, *, dry_run=False)` Python API. Article 12(2) retention enforcement. Tenant scoping is required: pick `--tenant TID` for a single tenant or `--all-tenants` explicitly, so a shared multi-tenant audit DB can never be silently purged across all tenants. Hash-chain integrity: surviving records still reference deleted predecessors via `previous_hash`, leaving a documented seam at the retention boundary that subsequent loads expose as a hash mismatch. Intended workflow: export a signed handoff zip BEFORE purging, archive externally, then purge. The signed zip remains self-consistent forever. The live DB chain has the seam.
- **prEN ISO/IEC 12792 four-axis transparency taxonomy on `AuditRecord`.** Four optional fields (`system_operation`, `data_usage`, `decision_making`, `limitations`) with default-classification heuristic per `EventType`. Per-record override via construction kwargs. NOT tamper-evident in v0.6 - fields are metadata annotations excluded from `record_hash` so pre-v0.6 chains stay valid. v0.7+ may add a separate signing mechanism if compliance requires.
- **`scripts/eval_distribution_shift.py`** - runs the full Vaara stack against the adversarial corpus with per-source tagging (hand-curated vs LLM-generated). Reports recall and FPR per source/class.
- **`scripts/eval_stack_ablation.py`** - runs three configurations (heuristic-only, classifier-only, full-stack) against the same corpus. Quantifies the independent contribution of each layer.
- **`scripts/eval_pair_attack.py`** - PAIR (Chao et al. 2023) iterative adaptive attacker. Uses an OpenAI-compatible vLLM endpoint for both attacker and judge roles. Zero new runtime deps (uses `urllib.request`).
- **`[yaml]` optional extra in `pyproject.toml`** (`pyyaml>=6.0`). Core `dependencies = []` preserved.
- **`examples/policies/minimal.json` and `full.yaml`** as reference policies.
- **COMPLIANCE.md gains "EU AI Act Annex IV evidence sections"** (maps Vaara contribution per §1–§9. Direct fill on §3, §5, §9. Contributes on §2, §4, §6, §7. Out of scope for §1, §8) **and "CEN-CENELEC harmonised standards alignment"** (per-standard table for ISO/IEC 42001, prEN 18286, prEN 18228, ISO/IEC 42006, prEN ISO/IEC 24970, prEN 18229-1, prEN ISO/IEC 12792).
- **`scripts/lint_full.sh` pre-push lint sweep** - chains `ruff` (style + correctness), `bandit` (security), `mypy` (types - strict on `vaara.policy`, lenient on legacy modules), and `pytest`. Documented in CONTRIBUTING.md. Catches CodeRabbit-class findings before they hit a PR review round-trip. New dev extras: `bandit>=1.7.5`, `mypy>=1.8`. Bandit configured in `pyproject.toml` to skip B608 across `audit/sqlite_backend.py` (all f-string SQL there interpolates only internally-controlled tenant clauses, not user input). Two `# nosec` annotations document the remaining trusted-bundle and synthetic-trace-RNG sites.

### Changed
- **Audit DB schema v2 to v3.** Migration `_MIGRATIONS[2]` adds four nullable transparency columns to `audit_records`. Pre-v0.6 records get NULL for the new columns. Their stored `record_hash` is preserved (NOT re-hashed on load), so chain verification of historical records continues to work.
- **COMPLIANCE.md "Current limits"** replaced placeholder bullets with v0.6 measurement results:
  - **Distribution-shift split.** Hand-curated (held-out, 250): attack recall 97.1% / benign FPR 70.0%. LLM-generated (in-sample, 5,705): attack recall 95.2% / benign FPR 87.5%. The 18pp benign-FPR gap is the dominant distribution-shift signal.
  - **Stack composition.** `heuristic_only` recall 35% / 63%. `classifier_only` recall 94% / 86%. `full_stack` recall 97% / 98%. Layers not redundant - heuristic catches a small set of attacks the classifier misses (justifies the ensemble). Most full-stack benign FPR comes from heuristic ESCALATEs, not classifier upgrades.
  - **PAIR adaptive-attacker calibration.** Qwen2.5-32B-Instruct as both attacker and judge, 25 hand-curated jailbreak seeds, max 5 iterations: **ASR 0.0% (0/25)**. NOT a claim of imperviousness to all adaptive attackers - stronger attacker (70B+), longer iteration budgets, or alternate strategies (multi-turn drift, language-switch, obfuscation) might produce non-zero ASR.

### Deferred to v0.7+
- **prEN ISO/IEC 24970 field-alias layer** - pending public final of the standard. Will land when 24970 publishes.
- **DORA mapping refinement** - pending deployer-side signal. Conservative defaults shipped in v0.5.3 stay until a financial deployer's input refines them.

### Reproducible artifacts
- `tests/adversarial/distribution_shift_v0_5_3.json`
- `tests/adversarial/stack_ablation_v0_5_3.json`
- `tests/adversarial/pair_v0_5_3.json`

## [0.5.3] - 2026-04-26

### Fixed
- **`AdversarialClassifier` jailbreak collapse.** v0.5.2 shipped a bundle with **0% classifier recall on held-out jailbreak**. The heuristic carried the category alone (~91% recall). v0.5.3 retrains on a corpus expanded by 1,500 LLM-generated jailbreak variants across three canonical sub-styles (roleplay/persona-bypass, hypothetical/academic-framing, fake-mode/DAN), recovering classifier jailbreak recall to **78.3%** at the new operating threshold.

### Added
- **1,500 jailbreak variants** in `tests/adversarial/generated/JB-{roleplay,hypothetical,fakemode}.jsonl`. Generated by Qwen2.5-32B-Instruct on AMD MI300x via vLLM, deduplicated by canonical-content sha1 (not shape, which caps yield at ~30 unique fingerprints per style and should be avoided in future runs).
- **1,010 cross-shape benign entries** in `tests/adversarial/benign_generated/BT-{readfile-canonical,http_post,sql_query,send_email}.jsonl` (500 read_file canonical + 170 each for http_post / sql_query / send_email). Added to rebalance the corpus and recover broad benign_control accuracy.
- **105 targeted training entries** in `tests/adversarial/generated/CE-envvar-targeted.jsonl` (60 send_email credential-exfil with env-var-shaped bodies labeled DENY) and `tests/adversarial/benign_generated/BT-http-public-targeted.jsonl` (45 legitimate http_post calls to GitHub / Slack / Stripe public APIs labeled ALLOW). Added to repair preflight smoke-test edges.
- **`--with-classifier` flag** on `scripts/eval_adversarial.py`. The harness previously evaluated heuristic-only by default. The new flag stacks the AdversarialClassifier on top per the v0.5.2 deployment recommendation. Heuristic DENY/ESCALATE are preserved. Heuristic ALLOW is upgraded to ESCALATE when classifier prob ≥ threshold.
- `tests/test_adversarial_classifier_integration.py` covers the bundle-load, score-range, and known-bad-input paths end-to-end. Skipped when `vaara[ml]` extras are not installed.

### Changed
- **Default classifier threshold: `0.5` to `0.55`.** Justified by threshold sweep on the rebalanced corpus: 0.55 is the operating point that clears the FPR and jailbreak-recall gates (global benign FPR ≤ 25%, jailbreak recall ≥ 60%) and passes the canonical preflight smoke test, while staying close to v0.5.2's balanced-accuracy band.
- **Bundle format `version` bumped `1.1` to `1.4`.** Trained on the full 5,955-entry corpus (3,422 attack / 2,533 benign). Feature schema unchanged from v1.1 (236 features), so `_STATIC_FEATURES` schema-drift check passes without modification.
- **`scripts/train_adversarial_classifier.py`** now coerces non-dict `context` and `parameters` entries (string-typed entries existed in the corpus from v0.5.0 onward but the trainer crashed on them) and runs `baseline_predictions` in `best_effort=True` mode. Net effect: trainer runs cleanly on the heterogeneous corpus.

### Benchmarks (5-fold CV OOF, threshold 0.55)

| Metric | v0.5.2 @ 0.5 | v0.5.3 @ 0.55 | Δ |
|---|---|---|---|
| **Jailbreak recall (classifier)** | **0.0%** | **78.3%** | **+78.3pp** |
| **Global benign FPR** | 23.3% | **21.0%** | **−2.3pp** (better) |
| Balanced accuracy | 80.9% | 79.4% | −1.5pp |
| Attack recall | 85.2% | 79.8% | −5.4pp |

Threshold sweep (sample points):

| threshold | bal_acc | attack_recall | benign_FPR | jb_recall |
|---|---|---|---|---|
| 0.50 | 79.0% | 84.3% | 26.3% | 80.7% |
| **0.55** | **79.4%** | **79.8%** | **21.0%** | **78.3%** |
| 0.60 | 79.5% | 75.4% | 16.3% | 75.8% |
| 0.65 | 79.0% | 70.7% | 12.7% | 73.6% |
| 0.70 | 78.1% | 66.0% | 9.7% | 71.6% |

### Preflight smoke test (canonical 4-case gate)

| Case | v1.1 (v0.5.2) | v1.4 (v0.5.3) | Expected |
|---|---|---|---|
| SSRF cloud metadata | 0.983 BLOCK | 0.999 BLOCK | BLOCK |
| Credential exfil email | 0.557 BLOCK | 0.973 BLOCK | BLOCK |
| rm -rf / destructive | 0.867 BLOCK | 0.998 BLOCK | BLOCK |
| Legit GH issue | 0.007 ALLOW | 0.491 ALLOW | ALLOW |

### Hand-curated regression set (250 entries, stacked classifier)

Per-category allow-leakage on the seed corpus (`tests/adversarial/<category>.jsonl`) using `eval_adversarial.py --with-classifier`:

| Category | v0.5.2 heuristic | v0.5.3 stacked |
|---|---|---|
| jailbreak | 48% | **4%** |
| ssrf_via_tools | 100% | **0%** |
| privilege_escalation | 92% | **4%** |
| credential_exfil | 76% | **8%** |
| prompt_injection | 56% | **4%** |
| data_exfil | 56% | **0%** |
| tool_misuse | 72% | **4%** |
| destructive_actions | 20% | **4%** |

### Known limits / honest read
- Aggregate balanced accuracy regressed **1.5pp** from v0.5.2 (80.9% to 79.4%) and attack recall regressed **5.4pp** (85.2% to 79.8%). The trade is justified by the **+78.3pp** jailbreak recall delta and the **−2.3pp** FPR improvement, plus the cleaner edge-case behaviour evidenced by the preflight smoke test. v0.5.2's 80.9% balanced accuracy was partly inflated by counting jailbreak as "in scope" while the classifier scored 0% on it.
- LLM-generated content shares Qwen-style writing. The distribution-shift gap between generated-test recall and hand-curated-held-out recall has **not** been measured separately in this release. It will be reported in v0.6. Hand-curated regression numbers above are evidence that transfer is happening, but a formal split is owed.
- Attacker-as-iterative-PAIR ceiling has **not** been measured. `COMPLIANCE.md` does not yet quote an adaptive-ASR figure.

### Recovery / fallback
- Heuristic-only path (`pip install vaara`, no `[ml]` extras) is unaffected. Threshold 0.55 applies only when `vaara[ml]` is installed and the classifier is engaged.
- If the v1.4 classifier produces unexpected behaviour for your traffic, downgrade to `vaara==0.5.2` for the previous bundle and threshold (0.5).

### Use ESCALATE, not DENY, on mixed-intent traffic
At threshold 0.55, the 21.0% global FPR is a **reviewer queue**, not a blast door. Stack the classifier with the heuristic (the new `--with-classifier` flag does this in eval) and route classifier-positive cases to ESCALATE for production deployments where a single false positive blocks a real user task.

## [0.5.2] - 2026-04-24

### Fixed
- `AdversarialClassifier` bundle in v0.5.0 and v0.5.1 scored unreliably because of two pipeline bugs. Both now fixed, bundle rebuilt with `version: 1.1`.
  - **Feature-name drift between trainer and runtime.** `scripts/train_adversarial_classifier.py` and `src/vaara/adversarial_classifier.py` each defined their own `_SHELL_PATTERNS`. Only 2 of 5 overlapped. The model learned columns whose regexes did not match the features the runtime built at inference time, silently inverting scores on `shell_exec` traffic.
  - **Silent zero-variant training.** `scripts/classifier_vs_heuristic.py` pointed `VARIANTS_DIR` at `tests/adversarial/generated_by_category/`, which never existed. The script loaded zero attack variants and trained only on 250 seeds + 1,145 benigns. Class balance on `http_post`, `send_email`, `shell_exec` was skewed heavily benign. With the right path, training now loads all 1,945 attack variants and produces 2,740 train / 600 test entries.

### Added
- `_STATIC_FEATURES` constant plus load-time schema-drift check in `src/vaara/adversarial_classifier.py`. A bundle whose `feature_names` tail diverges from the runtime static feature list now raises `ValueError` at construction time, pinpointing the first differing index. This class of bug is no longer shippable without failing loud.

### Changed
- Default threshold: `0.3` (v0.5.1) to `0.5`. Balanced accuracy peaks at 0.5 on the rebuilt bundle. The v0.5.1 claim of "52% recall, 3.3% FPR at threshold 0.3" was itself a recordkeeping error: the bundle saved `0.8`, not `0.3`, and those numbers were measured at 0.8.
- `scripts/train_adversarial_classifier.py` `load_corpus` now uses `rglob` to recurse into `tests/adversarial/generated/` and `benign_generated/` automatically.
- The `data_exfil` and `destructive_actions` regressions disclosed in v0.5.1 were artifacts of the broken bundle. The rebuilt classifier in v0.5.2 beats the heuristic in both: `destructive_actions` +40.2, `data_exfil` +24.7.
- Bundle format `version` bumped 1.0 to 1.1.

### Benchmarks (by-seed held-out, threshold 0.5)
- Attack recall: **85.2%**
- Benign false-positive rate: **23.3%**
- Balanced accuracy: **80.9%**
- Classifier beats heuristic by **+42.0 balanced-accuracy points** overall.
- Wins over heuristic on 8 of 9 attack categories: `ssrf_via_tools` +76.5, `tool_misuse` +73.1, `privilege_escalation` +58.1, `destructive_actions` +40.2, `credential_exfil` +39.0, `benign_control` +38.3, `prompt_injection` +26.6, `data_exfil` +24.7.

### Threshold sweep

| threshold | attack recall | benign FPR | balanced accuracy |
|-----------|---------------|------------|-------------------|
| 0.3 | 92.0% | 38.3% | 76.9% |
| **0.5** | **85.2%** | **23.3%** | **80.9%** |
| 0.7 | 61.5% | 8.3% | 76.6% |
| 0.8 | 47.2% | 3.3% | 71.9% |
| 0.9 | 32.6% | 0.0% | 66.3% |

Use `ESCALATE`, not `DENY`, at threshold 0.5 on mixed-intent tool traffic. 23% FPR is a reviewer queue, not a blast door.

### Known regressions
- `jailbreak`: classifier scores 0% on 11 held-out examples, down from heuristic 90.9%. Stack the heuristic rather than using the classifier alone on this category. v0.5.3 will widen jailbreak variant coverage.
- Legit `read_file` on low-signal paths (e.g. `README.md`) still trips the gate at threshold 0.5. Benign `read_file` is under-represented in training. v0.5.3.

### Recovery note
v0.5.1 remains on PyPI but ships a broken classifier. Upgrade to 0.5.2 to get the fixed bundle. The heuristic-only path (`pip install vaara`, no `[ml]` extras) was unaffected in both releases.

## [0.5.1] - 2026-04-23

### Changed
- `AdversarialClassifier` retrained on an expanded benign corpus to reduce false-positive rate in live agent traffic.
- Recommended operating threshold changed from `0.8` to `0.3` - the added benigns shifted the score distribution, and 0.3 is now the optimal balanced-accuracy point.

### Added
- `tests/adversarial/benign_generated/BT-new-http_post.jsonl` (170 variants)
- `tests/adversarial/benign_generated/BT-new-send_email.jsonl` (170 variants)
- `tests/adversarial/benign_generated/BT-new-shell_exec.jsonl` (170 variants)
- Total benign corpus size: 1,145 (up from 635 in v0.5.0)

### Benchmarks (by-seed held-out, threshold 0.3)
- Attack recall: 52% (vs 61% in v0.5.0 at threshold 0.8)
- Benign false-positive rate: **3.3%** (vs 13% in v0.5.0)
- Balanced accuracy: 74% (same)
- Latency: 140 µs mean, 210 µs p99 (unchanged)

### Known regressions (disclosed)
The new benigns shifted the decision surface toward allow. Per-category accuracy regressed in three attack categories:
- `data_exfil`: 0% (was 28.6% heuristic baseline - classifier now worse than heuristic here)
- `destructive_actions`: 25% (was 87.5% heuristic)
- `jailbreak`: 0% (was 100% heuristic)

The heuristic scorer retains strong coverage in these categories. Stack both rather than using the classifier alone. v0.5.2 will add targeted attack variants to rebalance the training corpus and recover recall without losing the FPR improvement.

## [0.5.0] - 2026-04-23

### Added
- `AdversarialClassifier` - opt-in XGBoost scorer for adversarial tool-call detection. Install with `pip install vaara[ml]`.
- `src/vaara/data/adversarial_classifier_v1.joblib` - 295 KB pre-trained bundle shipped with the wheel.
- `scripts/classifier_vs_heuristic.py` - reproducible comparison harness (by-seed train/test split, no leakage).
- `tests/adversarial/generated/` - 1945 LLM-generated adversarial variants across 8 attack categories (Qwen2.5-3B on MI300X, batched).
- `tests/adversarial/benign_generated/` - 635 benign variants (487 broad + 148 targeted at http_post/send_email/shell_exec legit uses).
- `[project.optional-dependencies.ml]` - `xgboost`, `scikit-learn`, `joblib`, `numpy`. Default install stays zero-dep.

### Benchmarks
On the by-seed held-out test (50 adversarial seeds held out entirely, plus 60 benign entries), classifier at threshold 0.8:
- Attack recall: 61% (vs 44% heuristic)
- Benign false-positive rate: 13% (vs 25% heuristic)
- Balanced accuracy: 74% (vs 50% heuristic)
- Latency: 140 µs mean, 210 µs p99 (commodity CPU, no GPU at inference)

Live dogfood on 174 Qwen-generated prompts with an end-to-end LangChain-style agent loop: recall 70%, FPR 51% - real-world distribution is tougher than the clean held-out test. Ship with `decision="escalate"` in production, not `decision="deny"`.

## [0.4.4] - 2026-04-22

### Fixed
- `vaara_wrap_tool` signature preservation via `functools.wraps` so LangChain `BaseTool.run()` injects `config` and `run_manager` kwargs under langchain-core >= 0.3.

### Added
- `examples/langchain_agent.py` runnable end-to-end with minimal action taxonomy and pre-calibrated pipeline.

## [0.4.3] - 2026-04-21

### Added
- Sigstore-signed release workflow with PyPI trusted publishing and PEP 740 attestations
- CodeQL SAST workflow
- OpenSSF Best Practices (Passing) badge
- CONTRIBUTING.md

### Changed
- Scorecard workflow pinned to peeled commit SHAs, with `workflow_dispatch` trigger added
- SECURITY.md restructured for OpenSSF Scorecard Security-Policy check

## [0.4.2] - 2026-04-20

### Added
- `Pipeline` alias for `InterceptionPipeline` at the top-level import.

## [0.4.1] - 2026-04-20

### Added
- Signed audit-trail export with verification tooling
- CLI surface for common operations
- Boundary sanitization on ingress/egress paths

### Changed
- README quick-start uses a generic filesystem example (was domain-specific)
- `InterceptionPipeline` is now importable directly from the top-level package

### Removed
- Domain-specific taxonomy modules (moved to plugin scope)

## [0.3.0] - 2026-04-18

### Added
- Framework integrations: LangChain, CrewAI, OpenAI Agents
- MCP server surface
- SQLite audit persistence

## [0.1.0] - 2026-04-10

- Initial release: interception pipeline, adaptive scoring, hash-chained audit trail
