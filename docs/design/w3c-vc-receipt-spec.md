# Design spec: W3C VC opt-in receipt serialization

Status: draft for v0.5. Priority 3 of 3. Extends
`vaara.attestation.receipt` / `_receipt_emit.py`.

## Goal

Emit execution receipts as W3C Verifiable Credentials **on request**, so a
consumer who standardizes on VCs can ingest Vaara receipts without a custom
parser. This neutralizes the one format edge a clone (Beacon emits receipts
as VCs) has, while keeping JCS / SEP-2787 as Vaara's canonical core.

Hard constraint: **the verification path is unchanged.** A VC is a
presentation wrapper. The signed bytes stay the JCS-canonical receipt; the
VC is recoverable back to that exact canonical form and verifies with the
existing `verify_receipt_signature` stack. No new crypto, no second trust
surface.

## What already exists (reuse, do not rebuild)

- `ExecutionReceipt` (`_receipt_types.py`) with `to_dict()` and
  `receipt_from_dict()`.
- `_signing_payload()` = JCS-canonical encoding of
  `{version, alg, backLink, outcomeDerived, receiptAsserted}`; the
  signature is over exactly these bytes and excludes itself.
- `sign_es256` / `verify_es256` (and HS256 / RS256) in `_sep2787_signing`.

The VC layer is pure serialization over these. It adds no signing.

## Design: VC as a lossless wrapper

The native receipt's signature already covers the canonical receipt blocks.
So the VC must NOT re-sign or re-canonicalize the payload (that would create
a second, divergent signature). Instead:

- The VC `credentialSubject` carries the native receipt blocks verbatim
  (the same dict `to_dict()` produces, minus the detached `signature`).
- The VC `proof` carries the existing SEP-2787 signature, typed so a
  verifier knows it is a detached JCS signature over the receipt blocks,
  not a standard VC Data Integrity proof.
- Verification recovers the native `ExecutionReceipt` from the VC and runs
  `verify_receipt_signature` unchanged. Byte-for-byte the same payload that
  was signed at emit time.

This is the honest framing: the VC is a *view* of the receipt. We do not
claim VC Data Integrity / JWT-VC conformance for the proof; we claim a VC
envelope whose proof is the receipt's own detached signature. (A future
true DataIntegrityProof variant is a separate, larger track; out of scope.)

## Wire shape

```json
{
  "@context": [
    "https://www.w3.org/ns/credentials/v2",
    "https://vaara.io/credentials/execution-receipt/v1"
  ],
  "type": ["VerifiableCredential", "VaaraExecutionReceipt"],
  "issuer": "<receiptAsserted.iss>",
  "validFrom": "<receiptAsserted.iat>",
  "credentialSubject": {
    "version": 1,
    "alg": "ES256",
    "backLink": { "...": "verbatim receipt block" },
    "outcomeDerived": { "...": "verbatim receipt block" },
    "receiptAsserted": { "...": "verbatim receipt block" }
  },
  "proof": {
    "type": "VaaraSep2787DetachedSignature2026",
    "cryptosuite": "jcs-es256",
    "verificationMethod": "<secretVersion or key id>",
    "proofValue": "<the receipt.signature hex>"
  }
}
```

`@context` ships a Vaara term-definition document at the versioned URL; the
context is static and vendored in-repo so verification needs no network.

## API

Opt-in, additive, default behavior unchanged:

- `receipt_to_vc(receipt: ExecutionReceipt) -> dict`: wrap.
- `receipt_from_vc(vc: dict) -> ExecutionReceipt`: unwrap, reconstructing
  the exact `ExecutionReceipt` (inverse of `receipt_to_vc`).
- Convenience on the public `receipt` surface, e.g. an emit helper or a
  `serialization="vc"` flag, that emits the native receipt then wraps it.
  The native path stays the default; VC is requested explicitly.

`receipt_from_vc(receipt_to_vc(r)) == r` is the round-trip invariant the
whole design hangs on.

## Verification path (unchanged, restated)

1. `receipt_from_vc(vc)` -> `ExecutionReceipt`.
2. `verify_receipt_signature(receipt, verifying_material=...)`: the
   existing call, same JCS payload, same ES256/RS256/HS256 check.
3. Back-link and result-commitment checks compose as today.

A consumer who only speaks VC can also read `credentialSubject` directly,
but the *trust* decision always routes through step 2. There is no
VC-native verification that bypasses the receipt signature.

## Test plan

- Round-trip identity: `receipt_from_vc(receipt_to_vc(r)) == r` for
  ES256, RS256, HS256 receipts, with and without a result commitment.
- Signature parity: a receipt verified natively and the same receipt
  unwrapped from its VC verify identically; tampering with any
  `credentialSubject` block fails verification after unwrap.
- Proof tamper: mutating `proof.proofValue` fails; mutating a
  `credentialSubject` field fails; both via the unchanged verifier.
- Context offline: verification does not fetch `@context` (no network).
- Schema presence: emitted VC has the required VCDM 2.0 fields
  (`@context`, `type`, `issuer`, `validFrom`, `credentialSubject`,
  `proof`).

## Out of scope for v0.5

- True VC Data Integrity proof / JWT-VC / SD-JWT conformance (separate
  track; would add a second signing surface).
- Verifiable Presentations, status lists, revocation registries.
- DID-based `issuer` / `verificationMethod` (string identifiers for now;
  DID resolution is a later option).
