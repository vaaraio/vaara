# Qualified existence-in-time for the execution record

A SEP-2828 execution record proves what an attested call did and pins which
decision it answers. It does not, on its own, prove *when* the record existed
to a party who runs none of the producer's software. The `existenceProof`
block closes that gap with an RFC 3161 trusted timestamp over the record's own
bytes.

## Wire schema

`existenceProof` is an optional top-level field on the execution record:

```json
"existenceProof": {
  "backend": "rfc3161-eidas-qualified",
  "hashAlgorithm": "sha256",
  "recordDigest": "sha256:<hex>",
  "token": "<base64 DER RFC 3161 TimeStampToken>"
}
```

- `backend` names the timestamp regime. `rfc3161-eidas-qualified` means the
  token is expected to be an eIDAS qualified timestamp, checkable against the
  EU trusted list.
- `recordDigest` is `sha256:<hex>` over the JCS-canonical (RFC 8785) encoding
  of the record with the `existenceProof` field removed.
- `token` is the base64 DER RFC 3161 `TimeStampToken` whose message imprint is
  `recordDigest`.

Like `pqSignature`, `existenceProof` is attached after signing and rides
outside the signed preimage (`{version, alg, backLink, outcomeDerived,
receiptAsserted}`). It therefore is not covered by the receipt signature: its
integrity rests on the timestamp token, which imprints the whole signed record
including that signature. Adding the field is an explicit version bump of the
closed record schema, not a silently tolerated extra.

## Verifier obligation

A conformant verifier presented with an `existenceProof`:

1. Recomputes the record digest over the JCS-canonical record with
   `existenceProof` removed, and requires it to equal `recordDigest`. A proof
   stapled to mutated bytes fails here.
2. Requires the token's message imprint to equal that digest, and the token's
   SignedAttributes `message-digest` to equal the hash of the TSTInfo content,
   so the signature covers this token and no other.
3. Verifies the TSA's signature over the SignedAttributes under the certificate
   the token carries.
4. Grades the attested time **qualified** only if the signer certificate is
   directly issued by a CA the verifier pins from a trusted list. For
   `rfc3161-eidas-qualified` that list is the EU trusted list. Absent a pin the
   time is valid but **self-asserted**, and the verifier reports it as such.

The distinction in step 4 is the point. A self-asserted timestamp is signed by
whoever holds the signing certificate and proves nothing to a third party. A
qualified timestamp is signed by an authority on a trusted list that neither
party controls, so the attested time is evidence: the witness sits outside the
party that produced the record. This is checkable offline, with no ledger and
no trust in the emitter.

## Relation to the audit-chain anchor

Vaara already anchors the audit hash chain's head to an RFC 3161 TSA
(`vaara.audit.timeanchor`), which proves the whole chain existed at a time.
`existenceProof` is the same evidence carried by a single record, so a relying
party who holds one execution record and none of the surrounding trail can
still establish its existence-in-time. The two are complementary: the chain
anchor covers a trail, the record proof travels with one record.

## Reference surface

- `vaara.attestation.receipt.attach_existence_proof(record, tsa_url=..., ...)`
  timestamps a record and returns it with the proof attached.
- `vaara.attestation.receipt.verify_existence_proof(record,
  trusted_issuer_cert=...)` returns the graded verdict.
- `vaara verify-record RECORD.json --trusted-issuer-cert CA.pem` surfaces it on
  the command line: qualified, self-asserted, or a failure, with the attested
  time.
- Conformance vectors and a dependency-free checker are in
  `tests/vectors/qualified_time_v0/`.
