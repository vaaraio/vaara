# Design spec: neutral verify, a Vaara verdict in IETF RATS EAR claims

Status: draft for v0.71 (Phase 2 of the hardware-governance layer). Companion to
`docs/design/enforcement-attestation-spec.md` and the TPM specs (the per-root
verdicts this re-expresses), and to `src/vaara/attestation/_attestation_result.py`.

## The problem

Phases 0 and 1 bind a SEP-2828 record to a TPM 2.0 quote, to the kernel's IMA log,
and to a continuous chain of quotes, and a SEV-SNP report binds a record to a
confidential VM. Each verifier returns a verdict in its own Vaara shape: a `tier`
string and a set of `*_basis` honesty fields. A regulator's Relying Party should not
have to learn three bespoke verdict shapes, one per root, to read "is this evidence
trustworthy, and in what respect."

The RATS architecture (RFC 9334) already names the roles: Vaara is the **Verifier**;
its output is **Attestation Results** a **Relying Party** consumes. There is a
standard vocabulary for those results: the **EAR claims** of `draft-ietf-rats-ear`,
carrying the **AR4SI** trustworthiness vector of `draft-ietf-rats-ar4si`. Phase 2
expresses the Vaara verdict in those claims so the output is root-agnostic: a TPM
binding, a TPM chain, and a SEV-SNP report all reduce to one claim set a RATS
consumer reads the same way.

## What this document is not

It is not an EAR, and it is not an EAT. EAR-04 defines exactly two serializations,
JWT (section 3.2) and CWT (section 3.3), and EAT (RFC 9711 section 3) requires that
"An EAT MUST have authenticity and integrity protection". What Vaara emits here is a
keyless JSON map, which satisfies neither condition.

The accurate name is an **unprotected claims set** that reuses the EAR claims. On the
CBOR side the registered analogue is UCCS (RFC 9781). On the JSON side no equivalent
term is registered; the nearest reference is the Unsecured JWT of RFC 7519 section 6,
which EAT cites in section 6.3.6. Signing this claim set as a JWT or CWT would make
it an EAR, and nothing in the shipped path does that.

This distinction is deliberate rather than an oversight. The claim set is the
verifier's appraisal result over evidence that already carries its own signatures,
and it is emitted keyless so that it can be checked without importing Vaara or
holding any Vaara key. The cost of that choice is that it may not be called an EAR,
and this document does not call it one.

## The surface

`vaara export-attestation-result VERDICT.json [--out FILE] [--iat N] [--submod L]`
reads the JSON a `verify-tpm-binding`, `verify-tpm-chain`, or `verify-enforcement`
`--json` run produced and emits a `vaara.attestation-result/v0` document. It is pure:
dict in, dict out. It parses no evidence (the verify command already did), so it needs
neither the attestation extra nor hardware present. The output is an unprotected JSON
claims set, keyless. It is the verifier's appraisal *result*, not a fresh
attestation; the evidence it appraises carries its own signatures, and the result
says `result_is_unsigned: true`.

## The document

The EAR claims of draft-ietf-rats-ear-04, in an unprotected map. The Vaara verifier
annotations ride in the standard `ear_verifier_claims` per-submodule field (the
draft's slot for "claims with Verifier authority, added during appraisal"), not a
custom key. The `eat_profile` value identifies the claims vocabulary in use; it does
not assert EAT conformance, which an unprotected map cannot have:

```
eat_profile      = "tag:ietf.org,2026:rats/ear#04"
iat              = <verifier appraisal time, integer epoch seconds>
ear_status       = <overall tier>
ear_verifier_id  = { developer: "https://vaara.io", build: "vaara <ver>" }
submods          = { <root label>: {
    eat_profile                 = "tag:vaara.io,2026:attestation-result#v0"
    ear_status                  = <tier>
    ear_trustworthiness_vector  = { <AR4SI claim>: <integer> ... }
    ear_verifier_claims         = { schema, source_schema, result_is_unsigned,
                                    honest_limit, native_tier, *_basis ... }
} }
```

The submodule label is the root type (`tpm` or `sev-snp`), overridable with
`--submod`.

## The AR4SI mapping

The vector uses four AR4SI claims (`instance-identity`, `configuration`,
`executables`, `hardware`); the claims this version does not appraise (`file-system`,
`runtime-opaque`, `storage-opaque`, `sourced-data`) are omitted rather than emitted
as a hollow `none`. Only the three canonical AR4SI tier anchors are asserted
(`2` affirming, `32` warning, `96` contraindicated), and `0`/omission for no claim.
Finer per-claim reason codes are deliberately not asserted, to avoid overstating
fidelity.

| Claim | Source | affirming `2` | warning `32` | contraindicated `96` |
|---|---|---|---|---|
| `instance-identity` | attesting key + record binding | validated root, binding holds | root trusted-as-supplied | bad evidence / record not bound |
| `hardware` | attesting-key provenance | validated root | trusted-as-supplied | bad evidence |
| `executables` | IMA replay / SEV-SNP launch measurement | reconciled + pinned | reconciled, unpinned | pin mismatch / not reconciled |
| `configuration` | PCR pin / SEV-SNP launch measurement | reconciled + pinned | reconciled, unpinned | pin mismatch / not reconciled |

`ear_status` is set to no higher trust than the worst claim present (higher AR4SI
integer = lower trust).

## The honest ceiling

On the shipped capture path the TPM EK chain and the AMD KDS VCEK chain are not
validated, so `ak_chain_basis` / `vcek_chain_basis` is `caller_supplied_unverified`.
The mapping then sets `hardware` and `instance-identity` to a warning, and the
overall `ear_status` cannot read `affirming`. `affirming` is reachable only when a
basis reports a validated root (`ek_chain_verified` / `kds_verified`): the same
un-forgeable-root capability the reserved `attested` tier waits on across the rest of
the attestation surface. The claim set never claims more than the verdict it was
built from.

## What it does not claim

IMA and the launch measurement attest the platform (that the code and configuration
present match what was measured), not that the agent decided X for reason Y. There is
no AR4SI claim for decision semantics, so one is not fabricated. The limit is recorded
as the `honest_limit` verifier-claim, and the decision content stays in the signed
SEP-2828 record the claim set references by digest (`bound_record_digest`). The reported
SEV-SNP TCB / policy is carried in the report but not appraised against a reference in
v0; that is recorded as `tcb_appraisal: not_established`, absence of appraisal, not a
found concern, so it is not expressed as a warning.

## Conformance

`tests/vectors/attestation_result_v0` pins the mapping across the tier matrix for all
three roots, with a Vaara-free independent checker that re-derives every claim set
from the input verdict using a second implementation of the mapping.
