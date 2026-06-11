# Post-quantum hybrid signing for execution receipts (v0)

Status: v0, additive. The classical receipt envelope, canonicalization, and every
existing conformance vector are untouched. A receipt becomes hybrid only when an
emitter opts in by naming a hybrid signature suite.

## The problem: trust now, forge later

A SEP-2828 execution receipt is a durable record kept across an EU AI Act Article 12
retention window measured in years. Its signature today is classical (ES256 / RS256).
A classical signature is forgeable by an adversary with a cryptographically relevant
quantum computer. Because the record is long-lived, the relevant threat is not "can it
be forged today" but **"trust now, forge later"**: an adversary who records a receipt
in 2026 and gains quantum capability in, say, 2034 can mint a signature that verifies
against the same public key, and backdate it, defeating the retention guarantee. The
defence is to sign, today, with an algorithm that resists that future adversary.

NIST FIPS 204 standardised ML-DSA (the Dilithium lattice scheme) for exactly this. v0
adds ML-DSA-65 as a **second, parallel** signature alongside the classical one. Both
must verify. The classical signature keeps the record verifiable by today's verifiers
and tools; the ML-DSA signature carries the quantum-resistant guarantee.

This is the "hybrid" posture FIPS and the migration guidance recommend during the
transition: do not drop the classical primitive (it is battle-tested and interoperable),
add the post-quantum one beside it, and require both. Blueprint: arXiv:2512.00110. NIST
FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA, not used in v0).

## Envelope: two additive fields, and where they live

The signing preimage is unchanged in shape: the JCS (RFC 8785) canonical encoding of
`{version, alg, backLink, outcomeDerived, receiptAsserted}`, signature excluded. Two
fields are added:

1. **`receiptAsserted.sigSuite`** (optional string, **inside the preimage**). Names the
   full signature suite the issuer committed to, e.g. `"ES256+ML-DSA-65"`. Absent means
   classical-only (the receipt's `alg` alone), so every existing receipt is byte-for-byte
   identical and verifies exactly as before. Allowlisted values only (see below).
2. **`pqSignature`** (optional object, top-level sibling of `signature`, **outside the
   preimage**): `{"alg": "ML-DSA-65", "keyid": "<verification-method-id>", "sig":
   "<hex>"}`. The `sig` is ML-DSA-65 over the **same** preimage bytes the classical
   signature covers.

### Why `sigSuite` is inside the preimage and `pqSignature` is outside

The post-quantum signature cannot be inside its own preimage (it signs the preimage, so
it cannot also be a field of it). That is unavoidable and it opens a **downgrade
attack**: an adversary strips `pqSignature`, leaving a record that still verifies under
the untouched classical signature, and the quantum-resistance silently vanishes.

The fix is to commit the *intent* inside the signed bytes. `sigSuite` is in the
preimage, so it is covered by **both** signatures. A record whose `sigSuite` names a
hybrid suite but is missing a valid `pqSignature` is a detected downgrade and **fails
closed** (it is not bound). An adversary cannot edit `sigSuite` back to classical-only
without breaking the classical signature too. What an adversary still can do is strip a
`pqSignature` that was attached *without* a committed `sigSuite`; such a record was
never claiming hybrid, so its honest verdict is `classical-only`, not a downgrade.

## Allowlisted suites (v0)

- absent / `null` -> classical-only, the receipt's `alg`.
- `"ES256+ML-DSA-65"`, `"RS256+ML-DSA-65"` -> hybrid: the classical member MUST equal
  the receipt's `alg`, and a valid `pqSignature` (alg `ML-DSA-65`) MUST be present.

Any other `sigSuite` value is rejected (unknown suite, fail closed). The allowlist
framing mirrors the #2867 fallback-projection precedent: a closed set of named suites,
not free-form negotiation.

## ML-DSA public keys in the DID document

The classical verification method is unchanged (`publicKeyJwk` with `kty` EC / RSA). The
ML-DSA key rides as a separate verification method whose `publicKeyJwk` uses the
**AKP** (Algorithm Key Pair) key type tracked by the JOSE/COSE post-quantum drafts:

```json
{ "id": "did:web:issuer.example#mldsa-2026",
  "type": "JsonWebKey2020",
  "publicKeyJwk": { "kty": "AKP", "alg": "ML-DSA-65", "pub": "<base64url raw 1952-byte key>" } }
```

v0 honesty note: the `AKP` JWK type is still on the IETF standards track at time of
writing. v0 reads `kty: "AKP"`, `alg: "ML-DSA-65"`, and a base64url `pub` member, and
will align field-for-field when the type ratifies. The raw public-key bytes are what
matter for verification; the JWK is just their envelope.

## Verdict dimension: quantum-resistance tier

A new classifier reports a tier orthogonal to the existing `verifiable` / `corroborated`
retention verdict. It does not change whether a record is `verifiable` today; it states
what the record resists.

- **`classical-only`** — no committed hybrid suite and no valid PQC signature. Still
  `verifiable` today under the classical key. Honestly marked as **not** surviving a
  future quantum adversary. Pre-quantum records (every record emitted before this
  feature) land here and are not failed.
- **`pqc-present`** — a `pqSignature` is attached and verifies, but the issuer did not
  commit `sigSuite` in the preimage, so the PQC protection is strippable and not
  downgrade-resistant. A transitional tier.
- **`hybrid-verified`** — `sigSuite` commits to a hybrid suite, the classical signature
  binds to a document key, **and** the ML-DSA signature binds to an ML-DSA document key
  over the same preimage. This is the only tier that is both quantum-resistant and
  downgrade-resistant. Fail-closed: a committed hybrid suite with an absent or invalid
  `pqSignature` is not `hybrid-verified` and not bound.

The classifier is reported alongside `verify_receipt_retained`: a regulator auditing a
2026 record in 2034 reads both "verifiable under a key valid at issuance" and
"hybrid-verified, quantum-resistant".

## Honest limits (keep visible)

- Hybrid means **both** must verify. The security is `min(classical, PQC)` for forgery
  resistance against a present-day adversary and rests on ML-DSA against a future quantum
  one. It is not a threshold; a single broken member fails the record.
- A `classical-only` record is not quantum-resistant. The verdict says so plainly rather
  than implying every record is protected.
- **`dilithium-py` is a reference implementation, not a production signer.** The
  ML-DSA-65 here is the pure-python `dilithium-py`, whose own documentation states it is an
  educational resource, is **not constant-time**, and is **not hardened against
  side-channel attack**. That is acceptable, and deliberate, for the two jobs it does in
  this design: the independent checker (verification only, no secret key present, so a
  secret cannot leak through timing) and the test vectors (throwaway keys). It is **not**
  acceptable for signing real, long-lived production keys; that path should use a hardened
  or FIPS-validated ML-DSA (for example a `liboqs` binding) behind the same signer
  boundary. The format, the preimage binding, and the verdict are independent of which
  ML-DSA implementation produced the bytes. This is the single named dependency of the
  otherwise standard-library checker, and a regulator can audit or swap it.
- ML-DSA-65 signatures are ~3.3 KB versus ES256's 64 bytes. The record grows; that is the
  cost of the guarantee and is stated, not hidden.
- An unrecognized field under a signed block is rejected at parse, not silently dropped:
  the modeled preimage must be byte-exact to the wire, or a verifier that re-derives the
  preimage from the model would exclude bytes a byte-exact verifier covers, and could call
  a record signed when injected content was never covered. The schema is closed; extending
  it is an explicit version bump.

## What v0 does not do (deferred)

- Wiring the hybrid requirement into the existing pass/fail verdicts. v0 reports the
  quantum-resistance tier through `pq_verdict`, but does **not** yet consult it inside
  `verify_receipt_retained`, `verify_evidence_bundle` / `vaara verify-bundle`, or the
  conformance `conforms` gate. Those surfaces judge the classical signature, so a record
  that commits to a hybrid `sigSuite` yet ships a stripped or invalid `pqSignature` (a
  `hybrid-downgraded` record) still passes them on its still-valid classical signature.
  The downgrade is surfaced today as a conformance **advisory**
  (`committed_suite_has_pq_signature`) and through `pq_verdict`; folding it into the
  pass/fail verdicts across all three surfaces is the E1b follow-on.
- A witnessed transparency-log anchor (Rekor v2 / C2SP) as a second un-forgeable tier
  (E2). Shares this ML-DSA machinery.
- SLH-DSA (FIPS 205) as an alternative PQC member.
