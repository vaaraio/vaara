# qualified_time_v0

Conformance vectors for the existence-in-time verifier obligation on a
SEP-2828 execution record.

A record may carry an `existenceProof`: an RFC 3161 trusted timestamp over the
record's own canonical bytes. It rides outside the signed preimage (like
`pqSignature`), so its integrity rests on the timestamp token, which imprints
the whole signed record. The obligation a conformant verifier discharges:

1. Recompute the record digest as SHA-256 over the JCS-canonical record with
   `existenceProof` removed. The proof's `recordDigest` must equal it.
2. Require the token's message imprint to equal that digest, and the
   SignedAttributes message-digest to cover the TSTInfo content.
3. Verify the TSA signature over the SignedAttributes under the certificate the
   token carries.
4. Grade the attested time **qualified** only if the signer certificate is
   directly issued by a CA the verifier pins from a trusted list. For the
   `rfc3161-eidas-qualified` profile that list is the EU trusted list, so a
   match is an eIDAS qualified timestamp, recognised EU-wide and checkable
   offline. Absent a pin, the time is valid but **self-asserted**: an
   unwitnessed timestamp is not evidence, and the property that matters is that
   the witness sits outside the party that produced the record.

## Cases

| case | ok | qualified | what it proves |
|---|---|---|---|
| `qualified_ok` | true | true | signer chains to the pinned trusted-list CA |
| `self_asserted_untrusted_issuer` | true | false | a valid token whose signer is not on the pinned list is not qualified |
| `neg_wrong_digest` | false | false | a token that timestamped a different value does not attest this record |
| `neg_tampered_record` | false | false | a record mutated after the proof was attached no longer matches |
| `neg_malformed_token` | false | false | a corrupt token is rejected |

`trusted_ca.pem` is the CA the checker pins. It is an in-process test CA, not a
real eIDAS QTSA issuer: the vectors prove the verification logic (imprint,
SignedAttributes, signer chain), while the live qualified proof against a real
EU-trusted-list QTSA is exercised separately by
`scripts/qualified_anchor_dss_demo.py`.

## Run

```
python tests/vectors/qualified_time_v0/_check_independent.py
```

The checker imports no Vaara code. It needs `rfc8785`, `asn1crypto`, and
`cryptography` (the `timeanchor` extra); without them it exits 77 (SKIP) so a
base environment grades clean. Regenerate the fixtures with
`_generate.py` (which does import Vaara, only to build and cross-check them).
