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

The tier `attested` is **reserved** for a future release that validates the VCEK
chain to AMD's ARK. It is never emitted in v0.

## Where trust comes from, stated plainly

This is the load-bearing section, and it is the deliberate contrast with the
cross-org handoff. There, the eIDAS RFC 3161 anchor is signed by a third party
outside both organisations, so the holder cannot forge it. **D1 has no such
un-forgeable component in v0.** `MockSEVSNPAttester` builds a byte-valid
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
2. that the chip is a genuine AMD part. The VCEK to ASK to ARK chain is not
   validated (the KDS fetch is deferred, as in `tee.py`). `vcek_chain_basis` is
   always `caller_supplied_unverified` in v0.
3. which image ran, unless `--expected-measurement` pins it against an
   independently vetted launch measurement.
4. when enforcement happened. A SEV-SNP report has no timestamp or nonce, so a
   captured report can be re-presented against the same record. v0 makes no
   freshness claim.

The one-sentence summary, carried in the verdict's `reason`: until
`vcek_chain_basis` is `kds_verified` and `measurement_basis` is `pinned`, this
verdict has no component the submitter cannot forge. AMD's ARK is the analogous
un-forgeable root, and it is exactly the part v0 does not yet check.

## The honesty fields

Two `*_basis` fields, modelled on the handoff's `producer_identity_basis`:

- `vcek_chain_basis`: `caller_supplied_unverified` (always, in v0) or
  `kds_verified` (reserved, never set in v0).
- `measurement_basis`: `unpinned` (no `--expected-measurement`), `pinned` (a
  constant-time match), or `pin_mismatch` (a value was pinned and differs).

A pinned measurement that does not match is a hard failure: `ok` is False even in
default mode, because passing `--expected-measurement` is an explicit "I require
this image". `report_context` surfaces the raw platform fields (`vmpl`, `policy`,
`guest_svn`, the TCB values, `chip_id`) for inspection without gating on any of
them; pinning those needs a deployment model.

## Strict mode

`--strict` requires the chain-rooted `attested` tier: a VCEK validated to AMD's
ARK plus a pinned measurement. Because v0 never validates the chain, a strict
pass is honestly unavailable; strict publishes the bar before the capability
exists rather than pretending to clear it.

## Scope and non-goals

In v0: single-record offline verify of a pre-captured report against a
caller-supplied VCEK; the three reachable tiers and the honesty fields; the
report-version allowlist `{2}` (the AMD ABI rev 1.55 layout the parser reads),
failing closed on others.

Deferred: AMD KDS chain validation (a network fetch; breaks the offline
posture); the `/dev/sev-guest` producer path (`SEVSNPHostAttester` still raises);
Intel TDX and SGX; anti-replay (no anchorable timestamp in a SEV-SNP report; a
per-record nonce would break the clean preimage); gating on policy / VMPL / TCB;
a published reference-measurements artifact so a pinned measurement is meaningful
against a third-party reference; folding an enforcement attestation into the
handoff package and a `verify-enforcements` batch (both set-ready, neither built).

## Conformance vectors

`tests/vectors/enforcement_attestation_v0/` carries ten cases (clean `bound`,
pinned match and mismatch, a report bound to a different record, the
signature-malleable variant, a flipped signature, a wrong VCEK, an unsupported
algorithm, a truncated report, and strict). `_generate.py` builds them with
`MockSEVSNPAttester`; `_check_independent.py` reproduces every verdict importing
only the standard library, `cryptography`, and `rfc8785`.
