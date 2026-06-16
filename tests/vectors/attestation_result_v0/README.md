# attestation_result_v0 conformance vectors

Vectors for the Phase 2 "neutral verify" surface: re-expressing a Vaara attestation
verdict as an IETF RATS **EAR** (EAT-based Attestation Result, draft-ietf-rats-ear)
carrying an **AR4SI** trustworthiness vector (draft-ietf-rats-ar4si). They pin the
honesty mapping so a change to it is a visible diff, and they prove an independent
implementation reproduces every EAR.

## Files

- `cases.json` — input verdicts (the `to_dict()` shape of a TPM-binding, TPM-chain,
  or SEV-SNP enforcement verdict) plus the fixed `iat` and `verifier_build` used to
  build the expected EARs.
- `expected.json` — the EAR document each verdict produces, keyed by case name.
- `_check_independent.py` — a Vaara-free re-implementation of the mapping. It derives
  the EAR from each verdict on its own and byte-compares against `expected.json`.
  Imports only the standard library.
- `_generate.py` — regenerates `cases.json` + `expected.json` from the real exporter.
  Run it after changing the exporter or the case matrix.

## The mapping the vectors pin

| AR4SI claim | source | affirming (2) | warning (32) | contraindicated (96) |
|---|---|---|---|---|
| `instance-identity` | attesting key + record binding | validated root, binding holds | root trusted-as-supplied | bad evidence / record not bound |
| `hardware` | attesting key provenance | validated root | trusted-as-supplied | bad evidence |
| `executables` | IMA replay / SEV-SNP measurement | reconciled + pinned | reconciled, unpinned | pin mismatch / not reconciled |
| `configuration` | PCR pin / launch measurement | reconciled + pinned | reconciled, unpinned | pin mismatch / not reconciled |

`ear_status` is set to no higher trust than the worst claim present.

## The honest ceiling

On the shipped capture path the TPM EK chain and the AMD KDS VCEK chain are not
validated (`ak_chain_basis` / `vcek_chain_basis` is `caller_supplied_unverified`), so
`hardware` and `instance-identity` are a warning and the overall `ear_status` cannot
read `affirming`. `affirming` appears in the vectors only for the cases whose basis
reports a validated root (`ek_chain_verified` / `kds_verified`) — the same
un-forgeable-root capability the reserved `attested` tier waits on. The EAR never
claims more than the verdict it was built from. The decision-semantics limit is not
expressed as an AR4SI claim (there is none); it stays a Vaara verifier-claim and the
decision content stays in the signed SEP-2828 record the EAR references by digest.

## Run

```
python tests/vectors/attestation_result_v0/_check_independent.py
python tests/vectors/attestation_result_v0/_generate.py   # to regenerate
```
