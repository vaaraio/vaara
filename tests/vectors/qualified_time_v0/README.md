# qualified_time_v0

Conformance vectors for the existence-in-time verifier obligation on a
SEP-2828 execution record.

A record may carry an `existenceProof`: an RFC 3161 trusted timestamp over the
record's own canonical bytes. It rides outside the signed preimage (like
`pqSignature`), so its integrity rests on the timestamp token, which imprints
the whole signed record. The obligation a conformant verifier discharges:

1. Recompute the record digest as SHA-256 over the JCS-canonical record with
   `existenceProof` removed. The proof's `recordDigest` must equal it.
2. Require the token's message imprint to equal that digest, the
   SignedAttributes message-digest to cover the TSTInfo content, and exactly
   one content-type signed attribute naming id-ct-TSTInfo (RFC 5652 5.3,
   RFC 3161 2.4.2).
3. Verify the TSA signature over the SignedAttributes under the certificate the
   token carries.
4. Grade the attested time **qualified** only if the signer certificate is
   directly issued by a CA the verifier pins from a trusted list and is a TSA
   certificate: RFC 3161 section 2.3 requires one critical extended key usage
   holding id-kp-timeStamping only. A trusted-list CA issues certificates for
   other purposes too, and none of those qualifies a time. For the
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
| `unqualified_signer_without_eku` | true | false | a signer the pinned CA issued with no extended key usage is not a TSA |
| `unqualified_signer_wrong_eku` | true | false | a signer the pinned CA issued for code signing is not a TSA |
| `neg_wrong_digest` | false | false | a token that timestamped a different value does not attest this record |
| `neg_tampered_record` | false | false | a record mutated after the proof was attached no longer matches |
| `neg_malformed_token` | false | false | a corrupt token is rejected |
| `neg_content_type_not_tstinfo` | false | false | a token whose signed content-type attribute names id-data is not a signature over a TSTInfo |
| `neg_content_type_missing` | false | false | a token with no content-type signed attribute is rejected |

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
