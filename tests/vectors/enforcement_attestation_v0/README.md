# enforcement_attestation_v0

Conformance vectors for `verify_enforcement` / `vaara verify-enforcement`:
binding an AMD SEV-SNP attestation report to a SEP-2828 execution record, so a
verifier can check that the record was hashed inside an SEV-SNP confidential VM.

Each case in `cases.json` carries:

- `record`: the SEP-2828 execution record (the bound payload).
- `report_b64`: base64 of the binary SEV-SNP attestation report.
- `vcek_pem`: the PEM the report signature is checked against.
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

The tier `attested` is **reserved** for a future release that validates the VCEK
chain to AMD's ARK; v0 never emits it, and `--strict` (which requires it) is
honestly unsatisfiable in v0.

## What this does not establish

`vcek_chain_basis` is always `caller_supplied_unverified`: the VCEK is trusted as
supplied, its chain to AMD's Key Distribution Service is not validated, and a
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
