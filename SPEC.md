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
without a hardware TEE and re-expressible as an IETF RATS EAR (AR4SI vector),
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

## 1. Canonicalization

All digests and all signed payloads in this specification are computed over the
JSON Canonicalization Scheme (JCS, RFC 8785). The canonicalization label for the
`evidenceRef.canonicalization` field (Section 3) is `jcs-rfc8785`. The values
`JCS` and `jcs-json-v1` are accepted aliases for the same algorithm; producers
SHOULD emit `jcs-rfc8785`, consumers MUST accept all three.

A digest is written `sha256:` followed by the lowercase hex SHA-256 of the
JCS-canonical bytes of the referenced object.

## 2. The receipt envelope

A receipt is a JSON object with these top-level members:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `version` | integer | MUST | Envelope version. `1` for this document. |
| `alg` | string | MUST | Signature algorithm. `ES256` in v1; `ML-DSA-65` MAY be offered as a post-quantum scheme. |
| `backLink` | object | MUST | Binds this receipt to its attestation/predecessor: `attestationDigest`, `attestationNonce`. |
| `decisionDerived` | object | MUST | The decision and the evidence it derives from. See Section 3. |
| `issuerAsserted` | object | MUST | Issuer-asserted identity claims: `iss`, `sub`, `iat`, `nonce`, `alg`, `secretVersion`. |
| `signature` | string | MUST | Detached signature, hex. For `ES256`, the 64-byte `r||s` pair (128 hex chars). |
| `timestampAnchors` | array | MAY | External time attestations over this receipt. See Section 4. |

### 2.1 Signed payload

The signature is computed over the JCS-canonical bytes of the object containing
exactly these members, in this set, with their receipt values:

```
("version", "alg", "backLink", "decisionDerived", "issuerAsserted")
```

`signature` and `timestampAnchors` are NOT part of the signed payload: a receipt
can gain anchors after signing without invalidating the signature. A consumer
MUST verify the signature by reconstructing this payload, canonicalizing it, and
checking it against the public key under `alg`.

## 3. Evidence binding (`decisionDerived.evidenceRef`)

`decisionDerived` carries the decision (`decision`, `decidedAt`, `policyId`,
`reason`, `riskScore`, `thresholdAllow`, `thresholdBlock`) and one
`evidenceRef` object that binds the decision to a recomputable evidence record:

| Field | Meaning |
|---|---|
| `canonicalization` | The label from Section 1 (`jcs-rfc8785` / `JCS` / `jcs-json-v1`). |
| `digest` | `sha256:` of the JCS-canonical evidence record. |
| `ref` | An opaque locator for the evidence record (profile-defined). |
| `schema` | The schema id of the evidence record (profile-defined). |

The binding is recomputable: given the receipt and the evidence record, a third
party confirms `sha256(JCS(evidence_record)) == evidenceRef.digest` with no
access to the issuer. This is the property independent implementers verify today.

## 4. Timestamp anchors (`timestampAnchors`)

A timestamp anchor is an external attestation that this receipt existed no later
than a stated time. Anchors are additive and optional. Each anchor binds the
**anchored digest** = `sha256:` of the JCS-canonical signed payload (Section 2.1),
so an anchor commits to the exact signed receipt without depending on later
anchors.

```json
{
  "method": "rfc3161",
  "anchoredDigest": "sha256:…",
  "token": "<method-specific time token>",
  "authority": "<optional human-readable authority id>"
}
```

Registered methods (the registry is open; a profile MAY register more):

| `method` | What it is | Who can produce it |
|---|---|---|
| `rfc3161` | An RFC 3161 timestamp token from any Time-Stamping Authority. | Self-hostable (e.g. OpenSSL `ts`); needs no third party. |
| `rfc3161-eidas-qualified` | An RFC 3161 token from a *qualified* TSA under eIDAS. | A qualified trust service provider. Adds legal / court-admissible weight; this is the only thing the qualification adds over `rfc3161`. |
| `ledger` | A commitment of the anchored digest to a public ledger; the block time bounds existence. | Self-producible; trust-minimized, no TSA. |

A receipt MAY carry several anchors of different methods. The technical anchor
(`rfc3161`, `ledger`) and the legal anchor (`rfc3161-eidas-qualified`) are
independent: a producer can stand up its own time evidence and add qualified
legal weight as a separate, swappable method. No single anchor method is
load-bearing for the receipt's integrity, which rests on the Section 2.1
signature.

## 5. Profiles

A profile is a downstream specification that uses this envelope unchanged and
defines only its own evidence record (the `schema` and contents behind
`evidenceRef`), plus any join keys it needs. A profile MUST state the
`vaara.receipt/vN` version it pins to and SHOULD ship recomputable vectors.

### 5.1 Registry

| Profile | Evidence schema | Pins to | Vectors |
|---|---|---|---|
| x402 settlement binding | `x402.settlement.*/v0` | `vaara.receipt/v1` | `tests/vectors/x402_settlement_v0/` |
| authorization decision | `vaara.authorization/v0` | `vaara.receipt/v1` | `tests/vectors/authorization_v0/`, `tests/vectors/contiguity_v0/` |

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
  grant by content address (`grantFingerprint` = `sha256(JCS(signed grant))`),
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
vectors and the `vaara verify-contiguity` surface carry that check. One honest
limit remains: a pure tail truncation (holding `0..k` with nothing after) cannot
be told from a complete stream by contiguity alone, since the latest held count
is then `k + 1`. Closing it is the job of an rfc3161 anchor over the running
count (Section 4), which attests that at time T, N receipts existed under the
boundary.

## 6. Conformance

An implementation conforms to `vaara.receipt/v1` if, for every receipt it emits:

1. The Section 2.1 signature verifies against the stated `alg` and key.
2. `evidenceRef.digest` equals `sha256(JCS(evidence_record))` for the referenced
   record, under one of the Section 1 canonicalization labels.
3. Any `timestampAnchors[].anchoredDigest` equals the `sha256:` of the JCS
   signed payload of the same receipt.

The committed vectors plus `_check_independent.py` are the reference conformance
suite; `python tests/vectors/x402_settlement_v0/_check_independent.py` exiting 0
is a passing run for the x402 profile.

## 7. Versioning

The envelope version is the integer `version` field and the `vaara.receipt/vN`
schema id. Additive, backward-compatible changes (new optional fields, new
anchor methods, new profiles) do not bump `N`. A change to the signed-payload
field set, the canonicalization, or the signature construction bumps `N`.
