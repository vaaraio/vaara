# enforcement_attestation_v0

Conformance vectors for `verify_enforcement` / `vaara verify-enforcement`:
binding an AMD SEV-SNP attestation report to a SEP-2828 execution record, so a
verifier can check that the record was hashed inside an SEV-SNP confidential VM.

Each case in `cases.json` carries:

- `record`: the SEP-2828 execution record (the bound payload).
- `report_b64`: base64 of the binary SEV-SNP attestation report.
- `vcek_jwk`: the report signature is checked against this P-384 public key (a JWK, public verification material; the checker reconstructs the key from it).
- `expected_measurement`: a hex launch measurement to pin, or `null`.
- `strict`: the strict flag.

`expected.json` holds the verdict subset each case must reproduce.

## The binding and the verdict

`REPORT_DATA = SHA-512(jcs(record))` over the **full** record including its
`signature` field. SHA-512 is exactly the 64-byte `REPORT_DATA` slot. Hashing
the whole record (not the five signed blocks alone) is what defeats the
signature-malleable variant: the signed-block subset is byte-identical when only
`signature` changes, so a report bound to a genuinely-signed record must not bind
a stripped variant.

The verdict tier is one of:

- `unverified`: the report did not parse, the version is unsupported, the
  algorithm is not ECDSA-P384-SHA384, the signature did not verify against the
  supplied VCEK, or `REPORT_DATA` did not bind to the record.
- `bound`: the signature verifies and `REPORT_DATA` binds to this record.
- `measurement_pinned`: `bound`, and the report's measurement matches a
  caller-supplied vetted value.

- `attested`: `measurement_pinned`, and the VCEK chains to AMD's pinned root for
  the product and matches the report's chip and TCB. It needs the AMD chain
  (`amd_chain`), which these mock-signed vectors do not carry, so no case here
  reaches it. `tests/test_sev_snp_chain.py` checks the chain on a real AMD Milan
  report.

## What this does not establish

`vcek_chain_basis` is `caller_supplied_unverified` in every case: no AMD chain is
supplied, so the VCEK is trusted as supplied, and a
[`MockSEVSNPAttester`](../../../src/vaara/attestation/tee.py) report with no AMD
provenance (exactly what generates these vectors) is byte-identical and passes
the same check. `enforcement_logic_basis` is always `not_established`: binding a
report to a record does not prove the enforcement decision logic ran in the
enclave. These vectors test the binding and the verdict, not genuine hardware.

## Reproduce

```
python tests/vectors/enforcement_attestation_v0/_generate.py        # Vaara, rewrites cases + expected
python tests/vectors/enforcement_attestation_v0/_check_independent.py  # no Vaara, must exit 0
```

`_check_independent.py` imports only the standard library, `cryptography`, and
`rfc8785`. The ECDSA signatures are randomized, so regenerating overwrites the
cases with fresh but equivalent vectors; commit `cases.json` and `expected.json`
together.
