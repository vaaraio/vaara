# evidenceRef: binding external drift evidence into the decision basis

Status: draft. Tracks the discussion on
`modelcontextprotocol/modelcontextprotocol#2826` (runtime tool-drift
detection) and the SEP-2828 decision record.

## The gap

A decision record carries the governing server's verdict (`allow` /
`block` / `escalate`) and the basis that drove it (`reason`, `riskScore`,
`thresholdAllow`, `thresholdBlock`, `policyId`). It deliberately does not
fix what *produced* that basis. When the basis is a runtime detection, for
example post-approval tool-surface drift, the evidence lives in a separate
record emitted by the detector. Two ways to join them are both wrong:

- Re-describe the drift inside the decision record. Now there are two
  copies of the evidence and they can disagree; the decision becomes a
  second, competing source of truth.
- Leave the join to an out-of-band monitor. Now "the decision rested on
  this evidence" is a claim a verifier has to take on trust.

`evidenceRef` is the third option: the decision basis names the external
evidence by content address. The detector emits its own record, that
record gets a content address, and the decision cites the address. Two
records stay independent, bound by one hash, and a third party who trusts
neither side can recompute the address from the referenced bytes.

## The field

`evidenceRef` is an optional object inside the signed `decisionDerived`
block:

| key | required | meaning |
|-----|----------|---------|
| `digest` | yes | `sha256:<hex>` over the canonical bytes of the referenced evidence object. The binding. Same digest convention as `backLink.attestationDigest` and `outcomeDerived.decisionDigest`. |
| `canonicalization` | yes | how those bytes were canonicalized before hashing, e.g. `"JCS"`. Names the rule an independent implementation must apply to reproduce the address. |
| `schema` | yes | the referenced object's shape and version, e.g. `"interlock.drift-record/v0"`. Tells the verifier how to read it. |
| `ref` | no | a non-authoritative locator (URI or path) for fetching the bytes. The `digest` binds; the bytes may also travel out of band. |

Because the object sits inside `decisionDerived`, it is covered by the
decision signature. A swapped or stripped citation breaks verification,
so the binding is not advisory: the signed decision commits to that exact
evidence. The slot is additive and optional. A decision with no external
evidence omits it, and the wire bytes are identical to a pre-`evidenceRef`
record.

## Worked example: drift detector to decision

The detector and the governing server are separate parties. Interlock is
the detector here, Vaara is the decision issuer. Nothing in the contract
is specific to either; the same shape holds for any detector that emits a
content-addressable record.

### 1. The drift record (detector's schema)

A tool that gained an external-reach effect after it was approved:

```json
{
  "schema": "interlock.drift-record/v0",
  "tool": "send_invoice",
  "approvedSurfaceHash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "currentSurfaceHash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "classifiedDelta": {
    "kind": "external-reach-added",
    "field": "effects.network",
    "from": [],
    "to": ["https://billing.example.com"]
  },
  "policyId": "policy:tool-surface/2",
  "observedAt": "2026-06-01T10:00:00Z"
}
```

### 2. The content address

Canonicalize the record under RFC 8785 (JCS) and hash. JCS is the same
canonicalization boundary the SEP-2828 records already sign over, so an
implementation that can verify a decision signature already has the bytes
rule it needs here. The canonical bytes are:

```
{"approvedSurfaceHash":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","classifiedDelta":{"field":"effects.network","from":[],"kind":"external-reach-added","to":["https://billing.example.com"]},"currentSurfaceHash":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","observedAt":"2026-06-01T10:00:00Z","policyId":"policy:tool-surface/2","schema":"interlock.drift-record/v0","tool":"send_invoice"}
```

and the content address is:

```
sha256:d303af9242e0d6d6c329c054d1fb2e32bbfde67bbbb7014873f0721174f239ac
```

This digest is deterministic from the record bytes alone. It is the unit
the two implementations must agree on (see the recompute check).

### 3. The decision that cites it

The governing server escalates and cites the drift record in its basis:

```json
{
  "version": 1,
  "alg": "HS256",
  "backLink": { "attestationDigest": "sha256:...", "attestationNonce": "..." },
  "decisionDerived": {
    "decision": "escalate",
    "decidedAt": "2026-06-01T10:00:00Z",
    "reason": "post-approval external-reach drift on send_invoice",
    "riskScore": "0.74",
    "thresholdAllow": "0.30",
    "thresholdBlock": "0.80",
    "policyId": "policy:tool-surface/2",
    "evidenceRef": {
      "digest": "sha256:d303af9242e0d6d6c329c054d1fb2e32bbfde67bbbb7014873f0721174f239ac",
      "canonicalization": "JCS",
      "schema": "interlock.drift-record/v0",
      "ref": "ipfs://<drift-record-cid>"
    }
  },
  "issuerAsserted": { "...": "..." },
  "signature": "..."
}
```

The `policyId` appears in both records: it is the policy that classified
the drift and the policy under which the decision escalated. The
`evidenceRef.digest` equals the address from step 2, so the decision's
basis points at exactly the drift record above and nothing else.

## The two-implementation recompute check

The property worth proving is that two independent implementations agree
on the binding without trusting each other. The check, in order:

1. **Detector emits.** Interlock produces the drift record and computes
   its content address under the named `canonicalization`. That address
   goes into the decision's `evidenceRef.digest`.
2. **Decision issuer signs.** Vaara emits the decision record with the
   `evidenceRef` in the basis and signs the whole `decisionDerived` block.
3. **Verifier recomputes, both sides.** A third party with the drift
   record bytes and the decision record:
   - canonicalizes the drift bytes under `evidenceRef.canonicalization`
     and hashes, and checks the result equals `evidenceRef.digest`
     (the citation resolves);
   - verifies the decision signature (the citation is the one that was
     signed, not substituted).

The check passes only if the detector's address computation and the
verifier's recomputation produce the same bytes and therefore the same
digest. That is the interop contract: both sides apply the same
canonicalization to the same object. `canonicalization` is in the
reference precisely so this is explicit rather than assumed, and a later
canonicalization rule is a named value rather than a silent
reinterpretation.

The two implementations need agree on nothing else. The detector chooses
its own `schema` and record shape; the decision issuer chooses its own
policy and signing key. Only the address has to match.

## Relationship to the public vectors

The SEP-2828 decision/outcome pairing vectors
(`tests/vectors/decision_pairing_v0/`) pin Check A (instance binding, the
shared attestation back-link) and Check B (content binding, the receipt's
`outcomeDerived.decisionDigest` over the signed decision bytes).
`evidenceRef` adds a third, optional binding on the *input* side: the
decision basis to its external evidence. It composes with the existing
two without changing them; a decision record that carries an
`evidenceRef` still pairs with its outcome receipt exactly as before, and
a verifier that does not understand `evidenceRef` ignores an optional
field rather than failing.

## Recompute it

The content address in step 2 is reproducible from this document with the
repository's own canonicalization:

```python
import hashlib
from vaara.attestation._sep2787_canonical import canonical_json
drift = { ... }  # the record from step 1
addr = "sha256:" + hashlib.sha256(canonical_json(drift)).hexdigest()
# sha256:d303af9242e0d6d6c329c054d1fb2e32bbfde67bbbb7014873f0721174f239ac
```
