# Design spec: bind a signed record to a SEV-SNP confidential VM (verify side)

Status: draft for v0.66. Companion to
`docs/design/cross-org-handoff-spec.md` and
`docs/design/key-rotation-retention-spec.md` (the record-level verdicts), and to
the SEV-SNP primitives in `src/vaara/attestation/tee.py` (the report parser, the
ECDSA-P384 signature check, and `MockSEVSNPAttester`).

## The problem

Every other record in the toolkit answers *who* signed, *when*, and *what*. None
answers *where the enforcement ran*. A signed execution record proves an issuer
asserted an outcome; it does not show that the enforcement point that produced it
ran in hardware the verifier can reason about, rather than on a host the operator
could quietly tamper with.

AMD SEV-SNP lets the enforcement point run inside a confidential VM and obtain a
hardware-signed attestation report whose 64-byte `REPORT_DATA` field the guest
chooses. `vaara verify-enforcement` checks that a report binds to a *specific
signed record*, so a verifier can ask: was this exact record hashed inside an
SEV-SNP confidential VM? It is the verify side; the report arrives pre-captured
(the enforcement point requests it from the chip at runtime).

## The binding

```
REPORT_DATA == SHA-512( canonical_json(record) )
```

over the **full** on-disk record dict, including its top-level `signature` field.
SHA-512 is 64 bytes, exactly the `REPORT_DATA` slot. This is the deliberate
divergence from the handoff anchor imprint, `sha256(jcs(record))`: the same record
bytes, a different digest, because the carriers differ (a 64-byte hardware slot
vs an RFC 3161 imprint). Two consequences fall out of the byte compare over all
64 bytes:

- **Substitution.** A genuine report for record A does not bind record B, because
  `sha512(jcs(B))` differs. A valid report attests only the record it was made for.
- **Signature malleability.** The five signed blocks alone
  (`version, alg, backLink, outcomeDerived, receiptAsserted`) canonicalise
  identically when only `signature` changes, so binding the subset would let a
  report for a genuinely-signed record equally bind a stripped or forged variant.
  Hashing the whole record, signature included, closes that.

## The verdict tiers

`verify_enforcement` returns one `tier`:

- `unverified`: the report did not parse to 1184 bytes, the version is
  unsupported, the algorithm is not ECDSA-P384-SHA384, the signature did not
  verify against the supplied VCEK, or `REPORT_DATA` did not bind to the record.
- `bound`: the signature verifies against the supplied VCEK and `REPORT_DATA`
  binds to this record. The highest tier reachable with an unpinned measurement.
- `measurement_pinned`: `bound`, and the report's launch measurement matches a
  caller-supplied vetted value (`--expected-measurement`).

- `attested`: `measurement_pinned`, and the VCEK chains to AMD's pinned root for
  the product and matches the report's chip and TCB (`--amd-chain`).

## Where trust comes from, stated plainly

This is the load-bearing section, and it is the deliberate contrast with the
cross-org handoff. There, the eIDAS RFC 3161 anchor is signed by a third party
outside both organisations, so the holder cannot forge it. **Without
`--amd-chain`, D1 has no such un-forgeable component.** `MockSEVSNPAttester` builds a byte-valid
1184-byte report signed with a caller-supplied ECDSA-P384 key and no AMD
provenance, and the signature check validates the report only against whatever
VCEK the caller passes. A caller who controls both the report and the VCEK can
mint a green `bound` verdict at will. That is content-addressing-style internal
consistency, not authenticity.

A passing check therefore proves exactly this, and no more: *an ECDSA-P384
SEV-SNP report carrying `sha512(jcs(record))` verifies against the VCEK you
supplied, so this record's bytes were hashed inside some SEV-SNP CVM whose VCEK
you chose to trust.* It does not prove:

1. that the enforcement decision logic ran in the enclave (`REPORT_DATA` only
   shows something inside the measured VM hashed the record and asked for a
   report). `enforcement_logic_basis` is always `not_established`.
2. that the chip is a genuine AMD part, unless the AMD chain is supplied. See
   "The AMD chain" below.
3. which image ran, unless `--expected-measurement` pins it against an
   independently vetted launch measurement.
4. when enforcement happened. A SEV-SNP report has no timestamp or nonce, so a
   captured report can be re-presented against the same record. v0 makes no
   freshness claim.

The one-sentence summary: until `vcek_chain_basis` is `kds_verified` and
`measurement_basis` is `pinned`, this verdict has no component the submitter
cannot forge. AMD's ARK is the analogous un-forgeable root.

## The AMD chain

`--amd-chain` (the `amd_chain` argument) takes the VCEK certificate, the ASK and
the ARK, as one PEM file or three files. `verify_sev_snp_chain` then checks,
offline:

1. The ARK's public key is AMD's root for the product. The SHA-256 of each ARK's
   SubjectPublicKeyInfo (Milan, Genoa, Turin) is pinned in
   `attestation/_sev_snp_chain.py`.
2. The ARK is self-signed, the ASK is signed by the ARK, and the VCEK certificate
   is signed by the ASK (RSA-PSS SHA-384), and all three are within their
   validity period.
3. The VCEK certificate's hwID equals the report's `CHIP_ID` (its first 8 bytes on
   Turin) and its bootloader, TEE, SNP and microcode SPLs equal `REPORTED_TCB`.
4. The report signature verifies under that key, and that key is the one in
   `--vcek`.

A VLEK chain (VLEK, ASVK, ARK) is checked the same way, without the hwID. The
certificates can come from the guest's extended report (`vaara tee emit
--certs-out`) or from AMD's KDS (`vaara tee fetch-chain`); trust comes from the
pinned root, not from where they were fetched. A report from real Milan hardware
and the VCEK AMD issued for it are in `tests/fixtures/sev_snp_milan`.

## The honesty fields

Two `*_basis` fields, modelled on the handoff's `producer_identity_basis`:

- `vcek_chain_basis`: `caller_supplied_unverified` (no chain supplied),
  `kds_verified` (the chain holds and carries the `--vcek` key), or
  `chain_failed` (a chain was supplied and did not hold; `ok` is False).
- `measurement_basis`: `unpinned` (no `--expected-measurement`), `pinned` (a
  constant-time match), or `pin_mismatch` (a value was pinned and differs).

A pinned measurement that does not match is a hard failure: `ok` is False even in
default mode, because passing `--expected-measurement` is an explicit "I require
this image". `report_context` surfaces the raw platform fields (`vmpl`, `policy`,
`guest_svn`, the TCB values, `chip_id`) for inspection without gating on any of
them; pinning those needs a deployment model.

## Strict mode

`--strict` requires the chain-rooted `attested` tier: a VCEK validated to AMD's
ARK plus a pinned measurement.

## Scope and non-goals

In scope: single-record and batch offline verify of a report against a VCEK,
with or without the AMD chain; the four tiers and the honesty fields; the
report-version allowlist `{2, 3, 4, 5}` (versions 3 to 5 only fill bytes that are
reserved in version 2), failing closed on others. The producer side,
`SEVSNPHostAttester`, requests a report inside a guest through configfs-tsm or
the `SNP_GET_REPORT` ioctl on `/dev/sev-guest`.

Not in scope: Intel TDX and SGX; anti-replay (no anchorable timestamp in a SEV-SNP report; a
per-record nonce would break the clean preimage); gating on policy / VMPL / TCB;
a published reference-measurements artifact so a pinned measurement is meaningful
against a third-party reference.

## Conformance vectors

`tests/vectors/enforcement_attestation_v0/` carries ten cases (clean `bound`,
pinned match and mismatch, a report bound to a different record, the
signature-malleable variant, a flipped signature, a wrong VCEK, an unsupported
algorithm, a truncated report, and strict). `_generate.py` builds them with
`MockSEVSNPAttester`; `_check_independent.py` reproduces every verdict importing
only the standard library, `cryptography`, and `rfc8785`.
