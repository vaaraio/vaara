# Vaara Receipt Specification

Status: normative, stable. Version: `vaara.receipt/v1`.
Canonical URL: https://github.com/vaaraio/vaara/blob/main/SPEC.md

This is the parent specification for a Vaara execution receipt: a signed,
independently recomputable record that binds a decision about an agent action to
the evidence it was made on, and optionally to one or more external timestamp
anchors. Any system that emits or consumes Vaara receipts conforms to this
document. Downstream specifications (a payment rail, a compliance regime, a
framework integration) define *profiles* that pin to a version of this document
and add only their own evidence schema; they do not redefine the envelope.

The receipt's trust is root-agnostic. The same record is verifiable with or
without a hardware TEE and re-expressible in IETF RATS EAR claims (AR4SI vector),
whether rooted in a TPM 2.0 host, an AMD SEV-SNP confidential VM, or software
alone. The signature and the optional external time anchor carry the evidence,
not a single trust root.

The key words MUST, MUST NOT, REQUIRED, SHOULD, MAY are to be interpreted as in
RFC 2119.

This document packages a format that already ships and is already recomputed by
independent implementers. It invents nothing new. The executable conformance
fixtures live at `tests/vectors/x402_settlement_v0/` with a dependency-light
checker (`_check_independent.py`) that imports only the standard library,
`cryptography`, and `rfc8785`.

## 1. Canonicalization, digests, numbers and times

Every digest of a JSON value in this document, and every signed payload, is
computed over the JSON Canonicalization Scheme (JCS, RFC 8785). The
canonicalization label for `evidenceRef.canonicalization` (Section 3) is
`jcs-rfc8785`. The values `JCS` and `jcs-json-v1` are accepted aliases for the
same algorithm; producers SHOULD emit `jcs-rfc8785`, consumers MUST accept all
three.

A digest is written `sha256:` followed by 64 lowercase hex characters of a
SHA-256 value. Unless a member's definition says otherwise, the hashed bytes are
the JCS encoding of the referenced JSON value. Where a member is a digest over
bytes that are not a JSON value (a UTF-8 string, a configuration file, a joined
preimage), its definition names those bytes.

No signed block and no evidence record carries a non-integer number. Scores and
thresholds are decimal strings (`"0.12"`). A producer MUST NOT emit a
non-integer JSON number in any of them.

Times are RFC 3339 date-times in UTC with the `Z` designator; fractional
seconds MAY be present. Parse them before comparing: two producers can write the
same instant with a different number of fractional digits.

## 2. The receipt envelope

A receipt is either a *decision receipt* (a decision about an action) or an
*execution receipt* (what followed). Both share this envelope:

| Member | Type | Kind | Presence | Meaning |
|---|---|---|---|---|
| `version` | integer | both | REQUIRED | `1` for this document. |
| `alg` | string | both | REQUIRED | `ES256`, `RS256` or `HS256`. See Section 2.1. |
| `backLink` | object | both | REQUIRED | The predecessor this receipt answers. Section 2.3. |
| `decisionDerived` | object | decision | REQUIRED | The decision and its basis. Section 3. |
| `issuerAsserted` | object | decision | REQUIRED | The issuer block. Section 2.4. |
| `outcomeDerived` | object | execution | REQUIRED | The outcome. Section 2.7. |
| `receiptAsserted` | object | execution | REQUIRED | The issuer block. Section 2.4. |
| `signature` | string | both | REQUIRED | Over the signed payload (Section 2.2). |
| `timestampAnchors` | array | decision | OPTIONAL | External time attestations. Section 4. |
| `pqSignature` | object | execution | OPTIONAL | Post-quantum signature beside the classical one. Section 2.5. |
| `existenceProof` | object | execution | OPTIONAL | A timestamp over the whole signed receipt. Section 2.6. |

A decision receipt carries `decisionDerived` and `issuerAsserted` and neither
`outcomeDerived` nor `receiptAsserted`; an execution receipt the reverse. The
schema is closed: a consumer MUST reject a receipt carrying a member not listed
for its kind, and MUST reject an undefined member inside `backLink`,
`decisionDerived`, `evidenceRef`, `issuerAsserted`, `receiptAsserted`,
`outcomeDerived`, `completeness`, `cryptoPosture`, `pqSignature` or
`existenceProof`. A consumer that rebuilds a signed block from the members it
understands would otherwise leave out signed bytes and report a receipt verified
over content it never checked. Evidence records are defined by profiles
(Section 5), whose own rules govern members they do not define.

### 2.1 Algorithms and signature encoding

`ES256` is ECDSA P-256 with SHA-256, signature the 64-byte `r||s` pair (128 hex
characters). `RS256` is RSASSA-PKCS1-v1_5 with SHA-256 and `HS256` is HMAC-SHA-256,
both as in RFC 7518. `signature` is the lowercase hex of the raw signature or MAC
bytes. No other `alg` is defined in v1, and a consumer MUST reject one.

`HS256` is symmetric: only a holder of the shared secret can verify it, so an
HS256 receipt is not recomputable by an arbitrary third party. Producers whose
receipts leave their own trust domain SHOULD use `ES256` or `RS256`. The
per-profile checkers under `tests/vectors/` accept `ES256` only.

The issuer block repeats the algorithm in its own `alg`, inside the signed bytes.
The two MUST be equal, and a consumer MUST reject a receipt where they differ
before trying any key.

### 2.2 Signed payload

The signature is computed over the JCS encoding of the object containing exactly
these members, with their receipt values:

```
decision receipt:  ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")
execution receipt: ("version", "alg", "backLink", "outcomeDerived", "receiptAsserted")
```

`signature`, `timestampAnchors`, `pqSignature` and `existenceProof` are not part
of the signed payload: attaching one after signing does not invalidate the
signature. A consumer MUST verify by rebuilding the payload for the receipt's kind
from the members as received, canonicalizing it, and checking it under `alg`.

### 2.3 Back link

| Member | Presence | Meaning |
|---|---|---|
| `attestationDigest` | REQUIRED | Digest of the predecessor, over its complete JSON form including its own signature. |
| `attestationNonce` | REQUIRED | Non-empty: the predecessor's nonce, or the identifier the profile names in its place. |
| `fallbackProjection` | OPTIONAL | Present only when no predecessor attestation exists and `attestationDigest` is over a named projection of the originating request; the value names the projection and its version. |

For a decision receipt the predecessor is the attestation of the request the
decision governs, unless the profile names another; the engine decision profile
(Section 5.10) names the previous trail record. An execution receipt carries the
same `backLink` as its decision receipt. A profile that permits
`fallbackProjection` MUST define each value it uses, and a consumer MUST reject a
value it does not implement.

This document defines one value, `tools_call_params_plus_meta_authorization_binding_v1`,
for an MCP `tools/call` request: the projection is `{"projection": <value>,
"name": params.name, "arguments": params.arguments, "authorizationBinding":
params._meta.authorization_binding}` and `attestationDigest` is its digest.
`authorization_binding` is required and is an object with a non-empty string
`nonce`; `name` and `arguments` are required. No other `_meta` member enters, so a
gateway and a provider seeing one call with different sidecars agree. When the
projection cannot be built the binding fails, and a consumer MUST NOT widen it.
Vectors at `tests/vectors/fallback_projection_v0/` and
`tests/vectors/decision_pairing_v0/`.

### 2.4 Issuer block

`issuerAsserted` and `receiptAsserted` have the same members, all inside the
signed payload:

| Member | Type | Presence | Meaning |
|---|---|---|---|
| `iss` | string | REQUIRED | The issuer. |
| `sub` | string | REQUIRED | The agent or principal the action was taken by or for. |
| `iat` | string | REQUIRED | Issuance time, RFC 3339 UTC. A string, not a JWT NumericDate. |
| `nonce` | string | REQUIRED | Unique per receipt; SHOULD carry at least 128 random bits. |
| `alg` | string | REQUIRED | Equal to the envelope `alg`. |
| `secretVersion` | string | REQUIRED | Names the verification key; resolved from `iss` and `secretVersion` out of band. |
| `aud` | string | OPTIONAL | The relying party the receipt was issued for. |
| `taskId` | string | OPTIONAL | The long-running task the action belongs to. |
| `completeness` | object | OPTIONAL | Per-boundary sequence (Section 5.3). |
| `sigSuite` | string | OPTIONAL | Committed hybrid suite (Section 2.5). Execution receipts only. |
| `cryptoPosture` | object | OPTIONAL | Algorithms protecting the receipt (Section 2.5). |

`aud` and `taskId`, when present, are non-empty. Checked against an expected
value they give bound, conflict, or unsupported (absent): absence means the
issuer bound none. `completeness` carries exactly `boundaryId` (non-empty
string), `seq` (integer, 0 or more) and `runningCount` (= `seq + 1`), all
required; a consumer MUST reject a partial or inconsistent block. An execution
receipt's boundary is its decisions' boundary with `#execution` appended, so a
refused decision, which has no execution receipt, does not read as a gap.

### 2.5 Post-quantum signature and crypto posture

There is no post-quantum `alg` in v1. An execution receipt MAY commit to a hybrid
suite with `receiptAsserted.sigSuite`, `"ES256+ML-DSA-65"` or
`"RS256+ML-DSA-65"`, whose classical part MUST equal `alg`. `pqSignature` then
carries `alg` (`"ML-DSA-65"`, FIPS 204), `keyid` (the ML-DSA verification key)
and `sig` (hex ML-DSA-65 over the same signed-payload bytes). A consumer MUST
reject any other `sigSuite`; a consumer that verifies ML-DSA MUST reject a
committed hybrid suite whose `pqSignature` is absent or does not verify (a
stripped signature). A `pqSignature` without `sigSuite` commits nothing. A
decision receipt MUST NOT carry `sigSuite`. Vectors at
`tests/vectors/pq_hybrid_v0/` (needs `dilithium_py`, skips without it).

`cryptoPosture` has `assetType` (`"algorithm"`), `algorithms` (non-empty array of
`{algorithm, primitive, nistQuantumSecurityLevel}`) and `nistQuantumSecurityLevel`
(0 to 5, the highest in `algorithms`). Levels: `HS256` 0 (`mac`), `ES256` and
`RS256` 0 (`signature`), `ML-DSA-65` 3 (`signature`). A consumer recomputes it
from `alg` and `sigSuite`; a posture that does not match, or claims an ML-DSA leg
no `sigSuite` commits, is not backed by the receipt.

### 2.6 Existence proof

An execution receipt MAY carry `existenceProof`: `backend`
(`"rfc3161-eidas-qualified"`), `hashAlgorithm` (`"sha256"`), `recordDigest` (digest
of the receipt with `existenceProof` removed, signature included) and `token`
(base64 DER RFC 3161 TimeStampToken imprinting that digest). It is outside the
signed payload; the token covers the signed receipt. The time is qualified only
when the consumer pins the token signer's issuer from a trusted list it holds.

### 2.7 Execution receipt

| Member | Presence | Meaning |
|---|---|---|
| `status` | REQUIRED | `executed`, `refused` or `errored` (attempted and failed). Anything else is rejected. |
| `completedAt` | REQUIRED | Time the outcome was recorded. |
| `resultCommitment` | OPTIONAL | Commitment to the result, or to the error for `errored`. Absent for `refused`. |
| `decisionDigest` | OPTIONAL | Digest of the decision receipt this outcome answers, over its signed members and signature, without `timestampAnchors`. |

`resultCommitment` is either `{projection, projectionDigest}` (`projection` a
string holding the JCS encoding of the result or of `{"digest": "sha256:..."}`
over it; `projectionDigest` over the UTF-8 bytes of `projection`) or `{ref,
digest, canonicalization}` (`ref` a locator, `digest` over the result,
`canonicalization` `"jcs"`). `decisionDigest` binds an outcome to one decision's
content, so it cannot be reattached to another decision about the same call.
Vectors at `tests/vectors/execution_receipt_v0/` and
`tests/vectors/decision_pairing_v0/`.

## 3. Decision and evidence binding (`decisionDerived`)

| Member | Presence | Meaning |
|---|---|---|
| `decision` | REQUIRED | `allow`, `block` or `escalate`. Anything else is rejected. |
| `decidedAt` | REQUIRED | Time of the decision. |
| `reason` | OPTIONAL | The issuer's reason. |
| `policyId` | OPTIONAL | The policy the decision was made under. |
| `riskScore`, `thresholdAllow`, `thresholdBlock` | OPTIONAL | Decimal strings. |
| `clientTurnId` | OPTIONAL | A turn id the client claimed; recorded, not vouched for. |
| `evidenceRef` | OPTIONAL | Binds the decision to an evidence record. Every profile in Section 5 requires it. |
| `rationale` | OPTIONAL | `rule`, `reason`, `declaredIntent` (strings) and optional `intentSatisfied` (boolean). |
| `binding` | OPTIONAL | `policyDigest` (JCS of the policy), `intentDigest` (UTF-8 of the declared intent), `inputsDigest` (JCS of the inputs) and `bindingDigest` (UTF-8 of the three digests and `decision` joined by byte `0x1F`). |
| `decisionProof` | OPTIONAL | A zero-knowledge proof opened against `bindingDigest`. Its format is not defined here; a consumer that does not verify it MUST NOT read it as evidence. |

`allow` permits the action, `block` refuses it, and `escalate` refers it to a
person or other authority without permitting it; a later decision settles it.
Verdicts such as `deny` or `revise` in the profiles below are checker outputs,
not values of `decision`.

`evidenceRef`:

| Member | Presence | Meaning |
|---|---|---|
| `canonicalization` | REQUIRED | A label from Section 1. |
| `digest` | REQUIRED | Digest of the evidence record. |
| `schema` | REQUIRED | The evidence record's schema id (profile-defined). |
| `ref` | OPTIONAL | An advisory, profile-defined locator. Not an identifier: see below. |

The binding is recomputable: given the receipt and the evidence record, a third
party confirms the record's digest equals `evidenceRef.digest` with no access to
the issuer.

`digest` is the binding; `ref` is advisory. A profile MAY assign the same `ref`
to more than one evidence record, and profiles in use do: where one action
settles to several parties, each party's record is a separate evidence record
under one shared `ref`, and they differ under `digest`. A consumer therefore
MUST NOT resolve an evidence record by `ref` alone, and MUST confirm the digest
before treating the record as the one the receipt decided over.

## 4. Timestamp anchors (`timestampAnchors`)

A timestamp anchor is evidence from outside the issuer that a decision receipt
existed no later than a stated time. Anchors are optional and attach to decision
receipts; an execution receipt uses `existenceProof` (Section 2.6). Every anchor
binds `anchoredDigest`, the digest of the receipt's signed payload (Section 2.2),
so it commits to the exact signed receipt and to no other anchor.

```json
{
  "method": "rfc3161",
  "anchoredDigest": "sha256:…",
  "token": "<base64 DER RFC 3161 TimeStampToken>",
  "authority": "<optional human-readable authority name>"
}
```

`method` and `anchoredDigest` are required; `authority` is optional and
informative.

| `method` | What it is | Further members |
|---|---|---|
| `rfc3161` | An RFC 3161 token imprinting `anchoredDigest`, from any TSA, including one the producer runs (`openssl ts`). | `token` |
| `rfc3161-eidas-qualified` | As `rfc3161`, from a qualified TSA under eIDAS. The qualification adds legal weight and nothing else. | `token` |
| `rfc3161-blinded` | As `rfc3161`, the authority shown a salted digest instead. Section 4.1. | `token`, `anchorSalt` |
| `rfc3161-eidas-qualified-blinded` | As `rfc3161-eidas-qualified`, blinded the same way. | `token`, `anchorSalt` |
| `scitt` | Inclusion of `anchoredDigest` as a leaf of an append-only Merkle log hashed as in RFC 6962. The identifier is historical: this is not registration with an IETF SCITT transparency service, and the entry is not a COSE receipt. | `logId`, `leafIndex`, `treeSize`, `inclusionProof`, `rootHash` |

For every method a consumer MUST first recompute `anchoredDigest` and reject the
anchor if it differs. The method name proves nothing about the token's signer: the
time is independent of the issuer only when the consumer checks the signer against
a certificate or trusted list it holds, and qualified only when that list is a
qualified trust list.

For `scitt`, `logId` is base64 SHA-256 of the log name, `leafIndex` and `treeSize`
are integers, `inclusionProof` is an array of base64 sibling hashes and `rootHash`
the base64 root at append time. The consumer recomputes the root from the leaf
(the 32 raw bytes of `anchoredDigest`) and the proof. `rootHash` is the log
operator's own claim; the anchor witnesses the receipt only when checked against a
tree head held independently of the receipt, directly or through an RFC 9162
consistency proof. Producer: `vaara receipt anchor-scitt`; head: `anchor-scitt-head`;
verify: `verify-scitt --head`.

This document maintains the method registry. A consumer MUST NOT treat an anchor
whose method it does not implement as verified, and MUST NOT reject the receipt
because of it: integrity rests on the signature, and an anchor is extra evidence.

### 4.1 Blinded anchors

An unblinded anchor sends the timestamping authority exactly the value the
receipt then publishes as `anchoredDigest`. An authority keeps a request log,
every entry in it sits behind a customer account, and a log that is sold,
breached or produced under compulsion lets whoever holds it match its entries
against any corpus of published receipts.

A blinded anchor closes that match. The producer draws a fresh 32-byte salt,
sends the authority

```
sha256( "vaara/anchor-blind/v1" || salt || anchoredDigest_bytes )
```

and carries the salt in the anchor entry as `anchorSalt` (64 lowercase hex
characters). `anchoredDigest` still names the Section 2.2 signed payload.

A producer MUST draw the salt from a cryptographic random source and MUST NOT
reuse it. A verifier MUST recompute the imprint, MUST reject a blinded anchor
whose `anchorSalt` is absent or not 32 bytes of hex, and MUST reject an unblinded
method that carries `anchorSalt`, so a blinded anchor is never read as a plain
one.

**What this does and does not buy.** It stops a party holding only the
authority's log from matching it against receipts it was not given. It does not
hide the anchor from anyone holding the receipt, since the salt travels with it,
and it does not hide the fact, timing or volume of anchoring from the authority.

A receipt MAY carry several anchors of different methods. The producer's own time
evidence (`rfc3161` from its own TSA, or `scitt`) and the legal anchor
(`rfc3161-eidas-qualified`) are independent and can be added separately.

## 5. Profiles

A profile is a downstream specification that uses this envelope unchanged and
defines only its own evidence record (the `schema` and contents behind
`evidenceRef`), plus any join keys it needs. A profile MUST state the
`vaara.receipt/vN` version it pins to and SHOULD ship recomputable vectors.

There is one binding mechanism, not one per plane. Each named profile (5.2-5.5)
names an external artifact by content address and binds it through this envelope
unchanged; they differ only in which artifact is hashed and the `evidenceRef.ref`
label. Section 5.6 states that mechanism in schema-agnostic form: a single binding
that does not depend on what is connected to it. The named profiles are instances
of it, kept because a given ecosystem pins to a label it recognizes as its own.

Section 5.7 is the one profile that runs the other way. It does not bind an
artifact into a receipt; it names a receipt as the condition on which something
external happens. It is listed here because it pins to the same envelope and
ships recomputable vectors, not because it is another instance of the binding.

### 5.1 Registry

| Profile | Evidence schema | Pins to | Vectors |
|---|---|---|---|
| x402 settlement binding | `x402.settlement.*/v0` | `vaara.receipt/v1` | `tests/vectors/x402_settlement_v0/` |
| authorization decision | `vaara.authorization/v0` | `vaara.receipt/v1` | `tests/vectors/authorization_v0/`, `tests/vectors/contiguity_v0/` |
| AP2 checkout binding | `vaara.authorization/v0` (names AP2 PEF `frame_id`) | `vaara.receipt/v1` | `tests/vectors/ap2_v0/` |
| TAP request binding | `tap.request/v0` | `vaara.receipt/v1` | `tests/vectors/tap_v0/` |
| generic external execution evidence | `vaara.authorization/v0` (names an `external_execution_evidence` slot) | `vaara.receipt/v1` | `tests/vectors/external_evidence_v0/` |
| release condition | `vaara.release-condition/v0` (consumes `vaara.authorization/v0`) | `vaara.receipt/v1` | `tests/vectors/release_condition_v0/` |
| attribute attestation | `vaara.attribute-attestation/v0` | `vaara.receipt/v1` | `tests/vectors/attribute_attestation_v0/` |
| hidden-value attribute attestation | `vaara.attribute-attestation-zk/v0` (proved by `vaara.attribute-predicate/v0`) | `vaara.receipt/v1` | `tests/vectors/attribute_attestation_zk_v0/` |
| engine decision (the floor) | `vaara.trail-decision/v0` | `vaara.receipt/v1` | `tests/vectors/trail_decision_v0/`, `tests/vectors/cage_v0/` |

The floor of the format is the engine decision profile (5.10): one receipt per
decision an engine records on its hash-chained trail, bound to that trail
record, with no rail, settlement artifact or external evidence record to join.
It is the smallest conforming receipt and what a default install emits for
every decision.

Three further suites in the same repository are related to this format and are
not profiles of it, because the artifact each one verifies is not a
`vaara.receipt/v1` envelope: `tests/vectors/governance_decision_v0/` (signed
governance decision and outcome records in the `{record, signature}` shape
proposed for the CrewAI framework), `tests/vectors/credential_binding_v0/` (the
signed HS256 grant a credential broker issues, the artifact the authorization
profile's `grantFingerprint` names), and `tests/vectors/atlas_threat_v0/` (a
flat HMAC record of a MITRE ATLAS threat detection). Each ships a standalone
checker, and none defines an evidence schema for `evidenceRef`.

### 5.2 Profile example: x402 settlement binding

This profile binds an x402 payment settlement to a Vaara receipt across an action
lifecycle, on a generic rail and on the Sui exact-payment rail. It adds:

- A settlement record (`schema` = `x402.settlement.<rail>/v0`) whose JCS digest
  is the receipt's `evidenceRef.digest`.
- A join key `actionRef` = `sha256(JCS({agentId, actionType, scope, timestampMs,
  seq, terminal}))`, carried on the settlement, so an in-progress receipt
  (`terminal: false`) cannot be presented where the terminal one is required.

A third party recomputes three per-step verdicts (action-ref recomputes,
settlement binding resolves, signature verifies) and one lifecycle verdict, with
only the settlement and the receipt in hand. See `_check_independent.py`.

### 5.3 Profile example: authorization decision

This profile turns an enforcement decision into a receipt. A credential broker
authorizes a tool call against a signed, attestation-bound grant with typed
capability scopes; the gateway's verdict, allow or deny, is minted as a receipt
instead of being discarded. The decision maps onto the envelope verdict
vocabulary: an allowed call is `allow`, a refused call is `block` carrying the
machine reason (`capability_exceeded`, `binding_unknown`, `missing_credential`,
...) as `decisionDerived.reason`. It adds:

- An authorization record (`schema` = `vaara.authorization/v0`) whose JCS digest
  is the receipt's `evidenceRef.digest`. It binds `toolName`, `tenantId`, the
  grant by content address (`grantFingerprint` = `sha256(JCS(grant))` over the
  grant with its `signature` member removed, the bytes the grant's signature
  covers),
  the runtime argument commitment (`argsCommitment` = `sha256(JCS(args))`), the
  evaluated `capabilities`, and the `verdict` / `reason`.
- The raw arguments never enter the record; only their commitment does, so the
  receipt is publishable while the arguments stay private. An auditor holding the
  arguments out of band recomputes the commitment and re-runs the verdict.
- An optional `coverage` block names the observation boundary the decision was
  made under, inside the record and therefore under the signature. It binds the
  `boundary` (the chokepoint identity), the `serverFingerprint` (the exact
  capability surface in scope, `manifest:sha256(JCS(tools))` or the command
  hash), and a `scope` literal stating that only calls routed through the
  chokepoint are observed. A tool reached on an out-of-band path is out of
  coverage. The block is absent when no boundary is asserted, leaving the record
  byte-identical to a coverage-free decision.
- An optional `completeness` block scopes a sequence to that boundary, inside the
  record and therefore under the signature. It binds the `boundaryId` (the same
  boundary the `coverage` block names), a monotonic `seq` starting at 0 with no
  gaps by construction, and a `runningCount` equal to the total receipts issued
  under the boundary up to and including this one (`runningCount` = `seq + 1`).
  The block is absent when no sequence is asserted, leaving the record
  byte-identical to a completeness-free decision.
- An optional sealing record finalizes the boundary: a terminal completeness
  block (`{boundaryId, sealed: true, total: N}`) that pins the boundary's final
  count independently of the per-record sequence. It is additive and emitted once
  the boundary is closed; a boundary that is never sealed verifies exactly as
  before, with the seal absent and the stream byte-identical. The seal may also
  carry `maxClass`, the highest action class the boundary authorized; it bounds a
  gap's worst case (see Section 5.3) and is itself optional.

A verdict is only as meaningful as what the issuer could see. `allow` over an
unbounded surface and `allow` over a stated one are identical bytes with
opposite meaning, so an absent refusal reads as fact only against a declared
scope: "not refused within this boundary", never "not observed". The `coverage`
block carries that boundary in the trace itself, so it is recomputable evidence
rather than a separate trust root. The verdict stays a thin read over it. The
chokepoint remains an observer of what passes through it, not a claim about what
does not.

The deny case is the point. A refused call leaves a signed, content-addressed,
portable proof of the non-action: a third party recomputes the verdict from the
grant and the arguments and confirms the refusal, trusting only the issuer's
public key. A third party recomputes five verdicts per case (grant fingerprint,
argument commitment, capability verdict, evidence binding, signature) with only
the grant, the arguments, the evidence, and the receipt in hand. See
`_check_independent.py`.

Coverage states the boundary; completeness makes a gap inside it provable. With
the per-boundary `seq` contiguous by construction and the `runningCount` signed
into each record, a dropped receipt is a missing sequence number that any holder
detects from the receipts alone: the highest running count names how many exist,
so a short set is self-evidently incomplete and the absent `seq` is named. This
needs no issuer access and no external witness. The `tests/vectors/contiguity_v0/`
vectors and the `vaara verify-contiguity` surface carry that check.

The per-record running count alone cannot tell a pure tail truncation (holding
`0..k` with nothing after) from a complete stream, since the latest held count is
then `k + 1` and reads as whole. The optional sealing record closes that gap: when
a boundary is finalized, the holder expects `max(seq + 1, runningCount, total)`
records, so a dropped tail shows as the missing range up to the sealed `total`. A
boundary that is never sealed verifies exactly as before. One residual remains, and
it is irreducible from the held set alone: a suffix drop that also suppresses the
sealing record leaves nothing to detect. Closing that is the job of an rfc3161
anchor over the running count (Section 4), which attests that at time T, N receipts
existed under the boundary. The layering is `seq` for order, the hash chain for
tamper-evidence, the sealing record for a truncated tail, and the timestamp anchor
for the seal-suppressed residual.

A gap proves that a record is absent but not what it would have authorized. When
worst-case-governs is the reading, the seal's optional `maxClass` bounds it: it
names the highest action class the boundary authorized, so a missing record could
have authorized an action of at most that class. The verifier surfaces this as
`worstCaseClass`, computed from the held set and the seal alone, with no issuer.
The field is optional; absent it, a gap reports only that a record is missing.

Beyond bounding a gap at audit time, the sealed `maxClass` is consumable at
enforcement time. A chain recipient gating its own next unattended action holds a
policy set of action classes it will proceed under and permits iff the sealed
worst-case class is a member of that set, failing closed when no class is sealed.
This is a membership test, not an ordering: Section 5.3 computes no ordering over
class labels, so the recipient asks "is the sealed class one I permit," never "is
it at or below a ceiling." Because the seal bounds a gap's worst case at
`maxClass`, a permitted class permits even when the boundary has a gap: the
recipient consumes the committed bound and does not re-derive the chain or query a
log. The bound is trustworthy under the honest issuer whose seal commits before any
tail is trimmed; a seal that under-states the class is a reconciliation question
against the issuer's log, not one this held-set-alone gate answers.

`maxClass` lives in the unsigned `evidence` block, so a recipient MUST NOT consume
it raw. It rides under signature only through the binding: the seal's signed
`decisionDerived.evidenceRef.digest` is `sha256:` + JCS(`evidence`), so recomputing
that digest proves the class is the class that was signed. Before gating, a
recipient MUST verify each receipt's signature and that its evidence recomputes to
the signed digest; a seal whose binding fails is not trusted, contributes no class,
and the gate fails closed. Without this, an agent loosens the gate by relabeling an
irreversible action's class into a permitted one while the record signature, which
never covered the evidence, still verifies. The conformance vectors are in
`tests/vectors/class_gate_v0/`; the `deny_relabeled` case carries exactly this
attack and the independent checker rejects it.

### 5.4 Profile example: AP2 checkout binding

This profile binds an AP2 checkout to the post-checkout agent actions a
credential broker authorizes, so the actions taken after a payment settles carry
the same recomputable, gap-evident record as the authorization decisions in 5.3.
It reuses the `vaara.authorization/v0` evidence record unchanged and adds a join
to the AP2 Payment Evidence Frame (PEF, AP2 PR #274):

- The AP2 checkout emits a PEF whose `frame_id` = `sha256(JCS(frame))`, with
  `frame_id` and `signature` excluded from the preimage, and whose `receipt_hash`
  = `sha256(JCS(receipt))` content-addresses the wrapped Checkout Receipt.
  Canonicalization is `urn:x402:canonicalisation:jcs-rfc8785-v1` (JCS / RFC 8785),
  the same as this envelope, so the address joins with no re-canonicalization.
- Each post-checkout authorization receipt names the checkout it followed by
  content address: `decisionDerived.evidenceRef.ref` = `ap2:checkout/<frame_id>`,
  under the receipt signature. The AP2 task scope is the `coverage.boundary`
  (5.3), and the `completeness` block sequences the actions under it.

The identity of the checkout is the PEF `frame_id`, a content address the payment
side already computes; the completeness of the actions taken under it is the
`vaara.authorization/v0` contiguity stream. A per-action hash says an action was
recorded; the running count says none inside the AP2 task boundary was dropped.
A third party recomputes the frame address, confirms every receipt names that
checkout, resolves each evidence binding, verifies each signature, and re-runs
the gap check, with only the PEF and the held receipts in hand. See
`tests/vectors/ap2_v0/_check_independent.py`. AP2 can pin from the point the
Checkout Receipt ends rather than define a new post-settlement primitive.

### 5.5 Profile example: TAP request binding

This profile binds a Visa Trusted Agent Protocol (TAP) request to the action a
trusted agent takes under it, across the action lifecycle, so the
post-authorization record is the same recomputable evidence as any other
decision receipt. It adds a TAP request evidence record (`schema` =
`tap.request/v0`) whose JCS digest is the receipt's `evidenceRef.digest`, and the
join key `actionRef` = `sha256(JCS({agentId, actionType, scope, timestampMs,
seq, terminal}))` carried on the request:

- The trusted agent presents the TAP request to the relying party. The decision
  receipt names it by content address: `decisionDerived.evidenceRef.digest` =
  `sha256(JCS(request))`, `decisionDerived.evidenceRef.ref` =
  `tap:request/<actionRef>`, both under the receipt signature. Canonicalization
  is JCS / RFC 8785, the same as this envelope, so the address joins with no
  re-canonicalization.
- The lifecycle lives in the join key. Because the action tuple covers
  `terminal`, the in-progress (`terminal: false`) request has a different
  `actionRef` than the final (`terminal: true`) one, and the in-progress receipt
  does not resolve against the terminal request. A mid-action receipt cannot be
  presented where the final one is required.

The verdict is recomputable offline. A third party recomputes the action ref,
resolves the request binding, and verifies the signature with only the TAP
request, the held receipts, and the issuer's public key, with the TAP service
offline and no live verifier endpoint to trust. See
`tests/vectors/tap_v0/_check_independent.py`. TAP can pin to `vaara.receipt/v1`
for the post-authorization record rather than define a new primitive.

### 5.6 Profile: generic external execution evidence

This is the schema-agnostic binding the named profiles above are instances of. It
takes any external execution-evidence artifact, content-addresses it, and binds it
through this envelope unchanged, with no field names that depend on what produced
it. A verifier carrying an `external_execution_evidence` slot (`linked_call_id` /
`evidence_hash` / `evidence_type`, the shape used by agentrust trace-spec #34 and
cMCP #301) resolves that slot against a `vaara.receipt/v1` authorization receipt as
the recomputable producer:

- `evidence_hash` = `sha256(JCS(evidence_record))`, equal to the receipt's
  `decisionDerived.evidenceRef.digest`, so the slot and the receipt name the same
  recomputable artifact (JCS / RFC 8785, no re-canonicalization).
- `linked_call_id` is the call the receipt names: `decisionDerived.evidenceRef.ref`
  = `mcp:call/<linked_call_id>`, under the receipt signature.
- `evidence_type` is the receipt's evidence schema (`vaara.authorization/v0`).

The trace is the `coverage.boundary`, and each receipt carries a signed
`completeness` block (`seq` + `runningCount`), so the held set proves not only that
each named call's evidence resolves but that none inside the boundary was dropped.
A slot's `evidence_hash` alone proves a given record exists; the completeness block
turns a silent drop into a named gap. The `dropped` vector withholds one record,
slot and receipt both, and the signed running count still proves it existed.

A third party recomputes every verdict offline with only the held slots, the
receipts, and the issuer's public key, with no live verifier endpoint to trust. See
`tests/vectors/external_evidence_v0/_check_independent.py`. Any plane that emits
execution evidence pins here by naming its artifact through this slot, rather than
defining a new primitive or a profile of its own.

### 5.7 Profile: release condition (`vaara.release-condition/v0`)

Every profile above runs one direction: something external happens, and the
receipt records it. Section 5.2 is the clearest case, where a payment gates access
and the settlement lands inside a receipt. In this profile the receipt gates the
payment.

A release condition is a signed, content-addressed statement made by whoever holds
value: what is held, exactly what must be proved before it moves, and when the
offer closes. Unlike the profiles above it does not sit behind an `evidenceRef`;
it *names* a `vaara.receipt/v1` receipt as its release trigger. It adds:

- A condition document (`schema` = `vaara.release-condition/v0`) carrying
  `holds` (amount as a decimal string, asset, network, payee), `requires`, and an
  inclusive `notAfter`. The signature is over `JCS(condition without "signature")`,
  the same rule the receipt envelope uses, so it needs no new cryptography.
- A `requires` block that is matched exactly, never approximately: `actionDigest`
  (the `argsCommitment` of the authorised call), `grantFingerprint` (the
  authorization that governed it), `receiptIssuer`, `receiptKeyFingerprint`
  (`sha256` over the SubjectPublicKeyInfo DER of the one key whose receipts
  count), `decision`, and `evidenceSchema`.
- A decision (`vaara.release-decision/v0`) naming the `conditionDigest` it was
  computed against, so a decision cannot be replayed against a re-issued
  condition.

The document holds no key belonging to a payer, signs no transaction, and reaches
no chain or custodian. It answers one question about bytes, a settlement agent
acts on the answer, and the verifier sits in the settlement path holding nothing.

Evaluation returns one of four states, each carrying a reason from a closed set:

| state | meaning |
|---|---|
| `released` | the authorised action is proved |
| `held` | the evidence is sound and insufficient, or none has been presented |
| `expired` | the window closed |
| `refused` | the presented artifact fails as evidence |

A verifier that proved nothing MUST NOT read as green, and MUST NOT read as the
same false as a genuine failure. `held` because no receipt arrived and `refused`
because a receipt was tampered with are different facts, and one boolean for both
discards the difference between "not yet" and "no". Implementations MUST
partition the reason space so that each reason belongs to exactly one state. A
third boolean beside a pass/fail does not satisfy this: the partition is what
keeps the two negatives from collapsing.

The axis is soundness, then sufficiency. A broken condition signature, a receipt
signed under a key the condition does not pin, a broken receipt signature, or
evidence that does not resolve to the digest the receipt signed are all failures
*as evidence*: `refused`. A missing receipt, a receipt for another action, another
authorization, another issuer, or one that soundly proves a *refusal* are sound
and insufficient: `held`. Checks MUST run soundness before the clock, so an
expired window cannot swallow a tampering finding, and the clock before
sufficiency, so a closed window is reported as the reason the value is not
moving.

A third party recomputes every verdict from the condition, the receipt, the
evidence and the two public keys, with no issuer access. See
`tests/vectors/release_condition_v0/_check_independent.py`; the `vaara
release-check` verb is the same evaluation at the command line.

### 5.8 Profile: attribute attestation (`vaara.attribute-attestation/v0`)

Section 5.7 asks what a receipt is worth when money is waiting. This one asks
what a *value* is worth. An attribute attestation binds a subject to attribute
values, states where each value came from, and says how long it holds.

Any signed record can assert an attribute. Whether the assertion is evidence
depends entirely on its source, so every attribute MUST name its own, drawn from
a closed and totally ordered set:

| standing | meaning |
|---|---|
| `undeclared` | nothing is claimed about where the value came from |
| `operator_declared` | the party being judged supplied it |
| `measured` | the issuer observed it directly |
| `protocol_defined` | the value is fixed by a specification and cannot differ |

`protocol_defined` outranks `measured` because a value fixed by a specification
cannot be wrong, while a measurement can come from a faulty sensor. `undeclared`
is the floor and MUST NOT convert upward. A verifier that encounters a standing
outside this set MUST treat the attestation as malformed and MUST NOT floor it to
`undeclared`, because a verifier that silently downgrades what it does not
recognise lets an issuer introduce a standing of its own.

A relying party states the floor it requires. Evaluation returns one of
`accepted`, `withheld`, `expired` or `refused`, each carrying a reason from a
closed set, with the reason space partitioned exactly as in Section 5.7. A value
below the floor is sound evidence of a claim and no evidence of a fact: it
`withheld`s, and it MUST NOT be reported the same way as a broken signature.

Checks run in the order soundness, clock, sufficiency. Soundness MUST precede the
clock so an expired window cannot swallow a broken signature.

The signature is over the JCS encoding of the document with its own `signature`
member removed, the same rule as Section 5.7 and the data-locality record, so a
verifier that checks one checks all three with no new code. Attributes are
emitted sorted by name so two issuers building the same statement produce the
same bytes. Both ends of the validity window are inclusive.

**What this is not.** A `vaara.attribute-attestation/v0` document is not a
qualified electronic attestation of attributes under Regulation (EU) 910/2014 and
MUST NOT be described as one, or as qualified, in any conforming implementation
or its documentation. Those terms are tied to a supervised, audited entry on a
Member State trusted list, and no cryptographic property substitutes for the
listing. An attestation issued and signed by the party it describes proves
integrity and never independence; implementations SHOULD surface that standing
rather than omit it. A qualified timestamp anchor (Section 4) raises the
confidence in *when* the attestation existed and changes nothing about the
standing of its contents.

Vectors are in `tests/vectors/attribute_attestation_v0/`, whose checker also
asserts that the reason-to-state mapping covers all four states and that the
standing ladder is a total order.

### 5.9 Profile: hidden-value attribute attestation (`vaara.attribute-attestation-zk/v0`)

Section 5.8 asks what a value is worth. This one asks what an issuer has to keep
in order to say it.

An attestation provider that vouches for an attribute has to hold the attribute.
Anything held can be sold, subpoenaed, breached or repurposed, and a policy
statement does not change what the holder is capable of. This profile commits to
the value at issuance and hands the opening to the holder, so what remains on the
issuer's side is a commitment and a signature over it.

One field changes from Section 5.8:

```
5.8:   {name, value,      source, sourceDetail}
5.9:   {name, commitment, source, sourceDetail}
```

`source` and `sourceDetail` stay in the clear and stay on the same closed,
totally ordered ladder, with the same floor rule and the same prohibition on
flooring a standing a verifier does not recognise. A relying party is entitled to
judge how strongly a value was sourced, and is entitled to nothing further.

#### The issuance ritual

An issuer conforming to this profile MUST, for each attribute:

1. draw a fresh blind, uniform over the scalar field, and compute the commitment
2. sign the document containing the commitment
3. hand the value and the blind to the holder
4. retain neither

Step 4 is the property the profile exists for. A blind MUST NOT be reused across
issuances: two commitments to the same value under the same blind are equal, and
a relying party holding both learns that the values match.

#### Commitments and predicates

Commitments are Pedersen commitments `C = v*G + r*H` over NIST P-256. `H` is
derived by hash-to-curve from a fixed public label, so its discrete logarithm to
`G` is unknown by construction, there is no trusted setup, and any party
recomputes `H` from the published label. Commitments are perfectly hiding and
computationally binding.

A relying party asks whether a predicate holds over the hidden value. Three kinds
are defined, and all three reduce to the same range argument over a shifted
commitment, because Pedersen commitments add:

| predicate | prover shows | verifier target |
|---|---|---|
| `at_least` | `value - lower` is in range | `C - lower*G` |
| `at_most` | `upper - value` is in range | `upper*G - C` |
| `in_range` | both, in that order | both, in that order |

The blind follows the shift: it stays as issued for the `at_least` direction and
negates for the `at_most` direction. A witness outside the proved interval has no
valid bit decomposition, so a predicate that does not hold has no proof, and a
conforming prover MUST refuse to emit one rather than emit something that will
not verify.

Each proof's Fiat-Shamir transcript MUST be seeded with the attestation digest,
the attribute name, the JCS encoding of the predicate, and the direction, so a
proof does not transfer to another document, another attribute or another
threshold. A verifier MUST reject a proof whose envelope names an attestation
digest, attribute or predicate other than the one being asked about, and MUST
report that rejection separately from a proof that is bound and fails to verify.

#### Evaluation

Evaluation returns `accepted`, `withheld`, `expired` or `refused`, each carrying a
reason from a closed set, partitioned as in Sections 5.7 and 5.8. Checks run in
the order soundness, clock, sufficiency, and within sufficiency the presented
proof is judged before the standing floor, so a forged proof is reported as
forged rather than as merely weaker than what was asked for.

A presented proof that is absent and a presented proof that is invalid MUST NOT
share a state. Nothing proved is not the same fact as something forged, and one
boolean for both discards the difference between "not yet" and "no".

#### Limits, stated rather than implied

**This is not selective disclosure.** One signature covers every commitment in
the document. A holder cannot present three attributes out of ten from a single
signed credential; that requires a signature scheme built for it and is outside
this profile. Per-attribute commitment covers the model above and nothing wider.

**This is not qualified.** A `vaara.attribute-attestation-zk/v0` document is not a
qualified electronic attestation of attributes under Regulation (EU) 910/2014 and
MUST NOT be described as one, or as qualified, in any conforming implementation
or its documentation. Those terms are tied to a supervised, audited entry on a
Member State trusted list, and no cryptographic property substitutes for the
listing.

**The issuer is still trusted for the value at issuance.** Hiding the value
protects it from the relying party and from anyone the issuer might later sell
to. It says nothing about whether the issuer committed to the truth. This is the
same residual documented in `docs/prove-what-an-ai-agent-did.md`: a record proves
what was recorded and does not prove that the recording was honest.

**Discarding is structural, not physical.** A conforming implementation removes
the reason to retain a value, which is what a subpoena, a breach or a change of
ownership reaches. It does not and cannot guarantee that no copy survives in
process memory, in a backup, or in whatever produced the value upstream.

**Values are bounded integers.** Every committed value and every predicate bound
MUST lie in `[0, 2**32)`, which is the interval the range argument proves
membership of. A value outside it MUST be refused at issuance. String attributes
are not carried by this version; they would require a membership proof against a
committed set.

Vectors are in `tests/vectors/attribute_attestation_zk_v0/`, whose checker rebuilds
the curve arithmetic, the commitments and the range argument from the published
parameters and imports no Vaara. It also asserts, before grading any case, that
`H` recomputes from its label, that commitments are additively homomorphic, that
the same value under two blinds gives two different commitments, and that a
missing proof and a broken proof land in different states.

### 5.10 Profile: engine decision (`vaara.trail-decision/v0`)

The engine writes one of these receipts for every decision it records on its
audit trail, beside the trail at `receipts/<YYYY-MM-DD>/<recordId>.json`. The
file holds the envelope under `receipt` and the evidence record under
`evidence`. The evidence record is the decision as the trail holds it:

| Field | Meaning |
|---|---|
| `schema` | `vaara.trail-decision/v0` |
| `recordId`, `actionId` | The trail record and the action it decided. |
| `eventType` | `decision_made` or `action_blocked`. |
| `agentId`, `toolName`, `tenantId` | As recorded. `tenantId` is `""` when unset. |
| `decision`, `reason` | The trail's words: `allow`, `escalate` or `deny`, and its reason. |
| `riskScore` | Decimal string. |
| `decidedAt` | ISO 8601 UTC, milliseconds. |
| `recordHash` | `sha256:` and the trail record's own hash. |
| `previousHash` | `sha256:` and the hash of the record before it. An empty genesis link is written as the SHA-256 of the empty string. |
| `decisionDetail` | The refinement behind the verdict, present only when the trail record carries one. |
| `approver`, `humanDisposed` | Who disposed of the decision: `approver` is exactly `human` or `policy`, `humanDisposed` is true only when a human acted on this decision, and a producer MUST NOT write `humanDisposed` true with any other approver. Present together, only when the trail record carries an approver; absent means no disposition was asserted, never that a human acted. Vectors: `tests/vectors/decision_disposition_v0/`. |
| `cage` | The cage the deciding process ran in. See "The cage block" below. Absent on records written before the cage layer. |

The envelope writes the trail's `deny` as `block`. `backLink.attestationDigest`
is `previousHash`, `backLink.attestationNonce` is `recordId`, and
`evidenceRef.ref` is `vaara:trail/<recordId>`. The issuer public key sits beside
the receipts as `issuer-es256.pub.pem`, and `issuerAsserted.secretVersion` names
it as `es256:` plus the first 16 hex characters of SHA-256 over its DER
SubjectPublicKeyInfo.

A verifier checks the signature and the evidence digest as in Sections 2.2 and
3. With the trail in hand it also looks up `recordId` and confirms the stored
record hash matches `recordHash`. A receipt whose record is missing from the
trail, or whose hash differs, fails. The evidence record carries no tool
arguments, so a receipt can leave the machine without them.

Vectors are in `tests/vectors/trail_decision_v0/`: receipts written by the
engine's own sink over a SQLite trail, tampered copies, the trail's record
hashes, and `expected.json` with each file's verdict. The macOS app's verifier
checks the same files.

#### The cage block

`cage` says which cage, if any, the process that made the decision ran in,
and whether the issuer confirmed at decision time that the confinement held
on that process. It is a member of the evidence record, so the evidence
digest and the signature bind it like any other member.

| Member | Type | Rule |
|---|---|---|
| `driver` | string | Always present, never empty. The cage's driver name, or `none` when no cage was declared to the deciding process. |
| `confirmed` | boolean | Always present. See below. |
| `basis` | string | Present whenever `driver` is not `none`. What the confirmation rests on: `declared` when only the launcher's declaration stands, otherwise the name of the fact that was read. |
| `configDigest` | string | Optional. `sha256:` and 64 lowercase hex characters over the cage's effective configuration. The driver defines which bytes it covers. |
| `upstream` | string | Optional, informative. The cage's own name and version. |
| `name` | string | Optional, informative. This launch's name inside the cage. |

`driver: none` carries `confirmed: false` and no other member.
`confirmed: true` means exactly this: at decision time the issuer read, on
the deciding process or the platform under it, the fact that `basis` names,
and the fact held. It requires a `basis` other than `none` or `declared`. It
does not mean that the cage enforced the configuration `configDigest` names,
that the cage is free of defects, or that anyone other than the issuer
observed the fact. A verifier ignores members it does not know.

The block names the deciding process, not the agent the decision was about.
One launch can therefore produce blocks that differ by surface: a decision
made inside the caged tree (a harness hook) names the cage and confirms it;
a decision made by a process outside the tree about that tree (the OS-layer
guard on a folder, the egress proxy on a connection) names that process's own
confinement, which may be `none` or the declared block unconfirmed. Each is
true of the process that signed it. A reader collecting one launch's
receipts groups them by agent and `name`, not by the block.

`configDigest` is a comparator, not a recomputation target. Two receipts with
the same `driver` and `configDigest` ran under the same declared
configuration; a verifier holding that configuration and the driver's rule
can confirm the digest, and one without them treats it as an opaque value.

The engine never writes a block that breaks these rules. A block passed in by
custom code is held to them on the receipt: a `configDigest` that is not
`sha256:` hex is left out, and an unsupported `confirmed: true` is written as
`false`. The trail record keeps what it was given.

Known basis values: `apparmor_label` (the process carries the cage's AppArmor
label), `seccomp_filter` (a seccomp filter and `no_new_privs` are on),
`no_new_privs`, `bwrap_init` (pid 1 of the pid namespace is bubblewrap),
`gvisor_kernel_log` (the kernel log is gVisor's), `hypervisor_present` (the
CPU reports a hypervisor underneath). `hypervisor_present` is the weakest:
it shows a virtual machine, not which one. Known drivers: `vaara-cage`,
`openshell`, `codex`, `sandbox-runtime`, `gvisor`, `firecracker`, `kata`,
`agent-sandbox`, `microsandbox`, `nono`, `e2b`, `apple-container`. Both lists
are open.

Vectors are in `tests/vectors/cage_v0/`: an unconfined run, a declared cage
the kernel did not confirm, and a confirmed cage, written by the engine's own
sink; a block changed after signing; and three blocks that are signed and
digest-consistent but break a rule of the block (confirmed on `declared`,
confirmed with `driver: none`, a malformed `configDigest`). The checker gives
each file a `signature`, `evidence` and `cage` verdict.

## 6. The ingest envelope (`vaara.ingest/v0`)

The profiles in Section 5 bind external evidence *into* a `vaara.receipt/v1`
decision: they carry a verdict, or a back-link, or both. Not every foreign
record is a decision. An adjacent log line, an identity assertion, a denial, an
invocation context establishes something narrower, and forcing it into a receipt
or an authorization envelope would fabricate a verdict or a back-link the source
never carried. The ingest envelope is the sink for exactly that case: it wraps
any foreign record, content-addressed, and asserts nothing the source did not
establish.

It is a sibling envelope to `vaara.receipt/v1`, not a profile of it, and reuses
the Section 1 canonicalization and the Section 2.2 signing construction
unchanged. The signed payload is:

- `schema` = `vaara.ingest/v0`, `version`, `alg`.
- `sourceFormat`, the recognized format of the foreign record (or `unknown`).
- `evidenceRef`: `digest` = `sha256(JCS(normalized_evidence))`,
  `canonicalization` = `JCS`, `schema` = `vaara.normalized-evidence/v0`, and an
  optional non-authoritative `ref` locator.
- `ingestAsserted`: `iss` / `sub` / `iat` / `nonce` / `secretVersion` / `alg`.
- `completeness`: a per-stream `seq` and `runningCount`; a lone ingest is `seq 1`
  of a one-record stream. Note the difference from Section 5.3: the ingest
  stream counts from 1, so `runningCount` equals `seq` here, where an
  authorization stream counts from 0 and `runningCount` is `seq + 1`. A
  contiguity checker written for one is wrong by one on the other.

`signature` is appended over the JCS encoding of that payload.

The normalized evidence object pinned by `evidenceRef.digest` carries the SEP-2828
fields the source establishes (`sep2828`), the context it carries that is not on
its own a proof (`advisory`), and the honest gap report (`missing`): what a
complete signed record still needs that this source does not supply. Because the
object is bound by digest under the signature, editing the gap report, a proof
field, or the source format breaks verification. The sink never launders a weak
source into a strong-looking receipt; the `missing` list is the record admitting
what it is not.

## 7. Conformance

An implementation conforms to `vaara.receipt/v1` if, for every receipt it emits:

1. The receipt keeps the rules of Sections 2 and 3: one kind, no undefined
   members, an `alg` from Section 2.1 repeated unchanged in the issuer block, a
   defined `decision` or `status` value, and a well-formed `completeness` block
   where one is present.
2. The Section 2.2 signature verifies against the stated `alg` and key.
3. `evidenceRef.digest` equals `sha256(JCS(evidence_record))` for the referenced
   record, under one of the Section 1 canonicalization labels.
4. Any `timestampAnchors[].anchoredDigest` equals the digest of the signed
   payload of the same receipt, and any `existenceProof.recordDigest` equals the
   digest of the receipt with `existenceProof` removed.
5. For an execution receipt, `backLink` recomputes from its predecessor, and
   when `status` is `executed` the `resultCommitment` recomputes from the
   result.
6. Any cage block in an evidence record keeps the rules of Section 5.10.

The committed vectors plus `_check_independent.py` are the reference conformance
suite; `python tests/vectors/x402_settlement_v0/_check_independent.py` exiting 0
is a passing run for the x402 profile. A `vaara.ingest/v0` envelope conforms when
the evidence object recomputes to `evidenceRef.digest` and the signature verifies,
both reproducible with no Vaara import;
`python tests/vectors/ingest_v0/_check_independent.py` exiting 0 is a passing run.

## 8. Versioning

The envelope version is the integer `version` member and the `vaara.receipt/vN`
schema id. A change to the signed-payload member set, the canonicalization, or
the signature construction bumps `N`.

A new optional member of a block in Sections 2 or 3 is added only by a revision
of this document and does not bump `N`. Because those blocks are closed, a
consumer built to an earlier revision rejects a receipt carrying the new member.
That is intended: it fails closed rather than verifying bytes it does not
understand. New anchor methods and new profiles do not change how the envelope
is parsed.
