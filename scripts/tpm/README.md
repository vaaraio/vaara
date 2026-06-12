# TPM binding (Phase 0)

Bind a signed SEP-2828 record to the local machine's measured, un-tampered state,
so a regulator can check it offline without trusting the operator.

Two halves, deliberately separated:

- **The verifier** (`vaara verify-tpm-binding`, code in
  `src/vaara/attestation/_tpm.py`, `_tpm_binding.py`, `_tpm_bundle.py`) is pure
  and needs no hardware. It is the durable, vendor-neutral core: it parses a TPM
  2.0 quote, verifies the AK signature, checks the record binding, recomputes the
  PCR digest, and replays the IMA log. It runs and is tested with no TPM present.
- **The capture script** (`capture-tpm-binding.sh`, here) is the hardware-touching
  half. It drives a real TPM to produce the evidence bundle the verifier reads.

## What a passing check establishes

For one bundle, four links checked by someone who trusts neither Vaara nor the
operator:

1. the AK signature verifies over the exact quote bytes;
2. the quote's `extraData` equals `SHA-256(jcs(record))`, so the quote was taken
   for *this* record;
3. the supplied PCR values recompute the `pcrDigest` the AK signed;
4. the supplied IMA log replays to the quoted PCR 10.

A signed record proves *what* was decided. The quote proves it ran on measured,
un-tampered hardware. The binding of the two is the point.

## What it does NOT prove (v0)

- **AK provenance.** The AK is trusted as supplied; its endorsement-key
  certificate chain to a TPM vendor root, and the credential-activation binding
  AK to EK, are not validated. A self-generated key passes the same check. This is
  the deferred `attested` tier (the TPM analogue of the deferred AMD KDS chain),
  and why `--strict` is honestly unreachable today.
- **Decision semantics.** IMA measures which files and executables loaded, not why
  the agent decided anything. The decision content is what the signed record
  carries.
- **IMA policy completeness.** v0 does not check which IMA policy was loaded.
- **Freshness.** A single bundle's `extraData` carries the record hash, not a
  verifier challenge, so a lone captured quote can be replayed against the same
  record. The continuous chain (Phase 1, below) moves this to chain-continuity: an
  ordered, single-boot, monotonic-clock window. It is still not a live verifier
  challenge.
- **IMA template-field consistency.** The verifier replays the template-hash
  column and confirms it aggregates to the signed PCR; it does not yet recompute
  each template hash from the `(file-hash, path)` fields, so the human-readable
  path column is reported, not cryptographically checked. Tracked for a later
  revision.

## Requirements (capture only)

- `tpm2-tools` (`tpm2_createek`, `tpm2_createak`, `tpm2_quote`, `tpm2_pcrread`)
- access to `/dev/tpm0` / `/dev/tpmrm0`: the `tss` group, or root
- root/sudo to read the IMA log (`/sys/kernel/security/ima/ascii_runtime_measurements_sha256`, mode 0640)
- a Python with `vaara[attestation]` installed (`VAARA_PYTHON`, else `./.venv/bin/python`, else `python3`)

## Usage

```sh
scripts/tpm/capture-tpm-binding.sh RECORD.json OUT_BUNDLE.json [EXPECTED_IMA_PCR_HEX]
vaara verify-tpm-binding OUT_BUNDLE.json
```

`EXPECTED_IMA_PCR_HEX` is optional: pin the quoted PCR 10 against a vetted
reference state to reach the `pcr_pinned` tier.

## Phase 1: the continuous chain

`vaara verify-tpm-chain` extends Phase 0 from one quote to an ordered sequence,
the Keylime-style loop. The capture script takes a quote on a fixed interval,
reusing one AK, and each tick's `extraData` is the chain-extended nonce
`SHA-256(jcs(record) || prev_digest || seq)` (`prev_digest` is the SHA-256 of the
previous tick's quote; genesis is 32 zero bytes). The links are hash-chained, so a
dropped, reordered, or spliced tick fails its successor's binding.

A `continuous` verdict adds, on top of every link passing the Phase-0 check:

1. the TPM clock strictly increases across the window;
2. `resetCount` / `restartCount` hold constant (no reboot, no unmeasured gap);
3. the IMA log grows append-only, tick to tick.

That moves freshness from unestablished to chain-continuity (an ordered,
single-boot, monotonic-clock window). It is still not a live verifier challenge;
anchoring the chain head and tail to a trusted timestamp is what bounds it to
wall-clock. The Phase-0 limits on AK provenance, decision semantics, and IMA policy
all carry over unchanged.

```sh
scripts/tpm/capture-tpm-chain.sh RECORD.json OUT_CHAIN.json [TICKS] [INTERVAL_SEC]
vaara verify-tpm-chain OUT_CHAIN.json
```

Code in `src/vaara/attestation/_tpm_chain.py`, `_tpm_chain_bundle.py`; schema
`vaara.tpm-evidence-chain/v0`.

## Status

The verifiers and bundle formats are tested end to end (`tests/test_tpm_binding.py`
and `tests/test_tpm_chain.py`, via a software `MockTPMQuoter` that marshals and
signs a real `TPMS_ATTEST`). The capture scripts follow documented tpm2-tools
behaviour but have not been run against live hardware in CI; the one place to
adjust if a tpm2-tools version marshals the quote message or signature differently
is `_assemble_bundle.py` (single) or `_assemble_chain.py` (chain).
