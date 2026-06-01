# Design spec: k-of-n threshold signing for audit exports

Status: draft for v0.5. Author target: Henri's signing-key sharding idea,
scoped as k-of-n multisig (decided 2026-06-01). Companion to
`docs/signing-keys.md`.

## Goal

Remove single-party control of audit-export issuance. Today one private key
signs every export, so one key-holder (including the single maintainer) can
issue or, if compromised, forge a trail. The fix recorded in the trust-model
work is: **no single party holds a key that alone can issue evidence.**

This closes the one attack `docs/signing-keys.md` says signing alone cannot
stop ("The only attack that signing alone cannot stop is key compromise") by
raising the bar from one key to k keys.

## Why multisig, not Shamir or FROST

Three models were on the table:

- **k-of-n multisig (chosen).** n custodians, n independent Ed25519 keys.
  Export carries each available signer's detached signature; verify requires
  >= k valid signatures from the authorized set. No key is ever combined or
  reconstructed. Builds on the existing `Signer` protocol, adds no crypto
  dependency, and reads to a regulator as "5 of 7 named custodians signed
  this export" - which is *more* legible than an opaque single signature.
- **Shamir-shared single key.** Matches the literal "one key, n shards"
  phrasing, but reconstructs the full key in memory at sign time - a
  momentary single point of compromise the multisig model never has.
  Rejected.
- **FROST true threshold.** Cryptographically ideal (one group key, never
  reconstructed, single output signature) but needs a round-based signing
  protocol and a FROST library that is immature in pure Python. Against the
  self-built ethos, heaviest, most fragile. Rejected for v0.5.

Multisig also best satisfies the documented intent: there is never a single
logical issuance key to hold or reconstruct at all.

## Data model

### Manifest additions (`manifest.json`)

When threshold signing is used, the manifest carries:

```json
{
  "signature_algorithm": "threshold-Ed25519",
  "threshold_k": 5,
  "signers_n": 7,
  "member_algorithm": "Ed25519",
  "signer_fingerprints": ["a1b2c3...", "c3d4e5...", "..."],
  "...": "all existing fields unchanged"
}
```

- `signature_algorithm` switches verify-side dispatch.
- `threshold_k` / `signers_n` are the policy. Both are inside the signed
  bytes (the digest is over `trail_bytes || manifest_bytes`), so an attacker
  cannot downgrade k to 1 without invalidating every member signature.
- `member_algorithm` lets members be Ed25519 today and ML-DSA-65 later
  without a new top-level algorithm string.
- `signer_fingerprints` is the **authorized set** (all n), in a fixed
  order. Verify maps a present signature to an authorized fingerprint; an
  unknown signer is ignored, not counted.

### Zip layout

```
trail.jsonl
manifest.json
sigs/<fingerprint>.sig        # >= k present, one per signing custodian
pubkeys/<fingerprint>.pem     # n present, the full authorized set
```

`trail.sig` and `signer_pubkey.pem` are **omitted** in threshold mode (a
single-sig zip and a threshold zip are never ambiguous - dispatch is on
`signature_algorithm`). Older single-sig exports are unchanged.

## Signing flow (`export_signed_threshold`)

A sibling to `export_signed`, sharing the manifest/zip plumbing:

1. Build the manifest first, including `threshold_k`, `signers_n`,
   `member_algorithm`, and the full ordered `signer_fingerprints` list.
   The signing digest depends on the manifest, so k/n/the authorized set
   are bound into what every custodian signs.
2. Compute `digest = sha256(trail_bytes || manifest_bytes)` once.
3. Each available custodian's `Signer` signs the *same* digest. Custodians
   are passed in as a list of `Signer` instances (HSM/KMS-backed in
   production - they need not be co-located; the coordinator collects
   detached signatures).
4. Require at least k signatures collected; else raise (the export is not
   issuable under policy).
5. Write `sigs/<fp>.sig` for each collected signature and
   `pubkeys/<fp>.pem` for all n authorized public keys.

The coordinator is untrusted: it cannot forge a signature it does not hold,
and it cannot lower k (k is signed). The worst a malicious coordinator does
is refuse to assemble an export - a liveness, not an integrity, failure.

## Verify flow (`verify_signed`, threshold branch)

Dispatch when `signature_algorithm == "threshold-Ed25519"`:

1. Read `threshold_k`, `signers_n`, `signer_fingerprints` from the manifest.
2. Load the n authorized public keys from `pubkeys/`. Check each loaded
   key's fingerprint is in `signer_fingerprints` and that exactly n distinct
   fingerprints are present (manifest and zip agree on the authorized set).
3. Recompute `digest = sha256(trail_bytes || manifest_bytes)`.
4. For each file in `sigs/`, verify it against the authorized key whose
   fingerprint matches the filename. Count **distinct authorized signers**
   with a valid signature over the digest.
5. Pass iff `distinct_valid >= k`. An invalid or unauthorized signature is
   ignored (does not subtract); a duplicate fingerprint counts once.
6. All existing checks (trail sha256, hash-chain re-verify, record_count,
   endpoints) run unchanged.

The standalone verifier (`scripts/verify_vaara_trail.py`) gets the same
branch - stdlib + `cryptography` only, no Vaara import.

An out-of-band trusted authorized set may optionally be passed to verify; if
given, the embedded `pubkeys/` set must be a subset (the same "don't trust
the embedded key" discipline as the single-sig path).

## Key lifecycle (the second half - not optional)

The trust-model note frames threshold signing as a *governance* answer to
key-person risk, so the lifecycle is part of the feature, not a follow-up.

### Rotation

- Each custodian rotates independently on the existing 12-month schedule
  (`docs/signing-keys.md`). Rotating one of n does not invalidate prior
  exports - each export names the authorized set that signed it.
- Changing k or the membership set is a **policy event**, recorded (below)
  so a verifier can see when and why the quorum changed.

### Chain-anchored key-compromise markers

This is what makes "issued before compromise" provable rather than asserted,
and it reuses machinery already shipped:

- A compromise/rotation event is written as an **audit record** in the trail
  itself (`event_type: "key_lifecycle"`, data carries the affected
  fingerprint, the action `rotated | revoked | added`, and the new k/n).
  Because it is a record, it inherits the v0.47 hash chain and the v0.48
  external time anchor.
- The external anchor over chain heads is what defeats backdating: a
  compromised key can re-sign, but it cannot re-anchor past chain heads to a
  time source it does not control. So a `revoked` marker that was anchored
  *before* the compromise window pins the revocation in time.
- Verify surfaces any `key_lifecycle` records found in the trail so a
  reviewer sees the custodian set's history inline with the evidence.

This ties directly to the "If a Signing Key Is Compromised" section of the
trust-model doc: with k-of-n plus chain-anchored markers, a single
compromised custodian key is below quorum and is provably marked revoked at
an externally-witnessed time.

## Backward compatibility

- Single-sig exports (`Ed25519`, `ML-DSA-65`) are untouched. No manifest
  field changes for them; `trail.sig` / `signer_pubkey.pem` stay.
- `verify_signed` dispatches on `signature_algorithm`; unknown stays an
  error. Old verifiers reading a threshold zip get
  `unknown signature_algorithm: 'threshold-Ed25519'` - a clean refusal, not
  a false pass.
- `member_algorithm` reserves the PQ upgrade path without a schema break.

## Test plan

- Round-trip: 3-of-5 export, all 5 sign, verify passes; verify still passes
  with only 3 of 5 sigs present; fails with 2.
- Downgrade attack: flip `threshold_k` 3->1 in the manifest, verify fails
  (member signatures no longer match the mutated digest).
- Unauthorized extra: drop a valid signature from a non-authorized key into
  `sigs/`, verify ignores it and still requires k from the authorized set.
- Tamper: mutate one trail record, verify fails on the chain check.
- Standalone verifier parity: same zip through
  `scripts/verify_vaara_trail.py` gives the same result.
- Lifecycle: a `key_lifecycle` revoked record in the trail is surfaced by
  verify and is covered by the hash chain.
- PQ member: one 5-member set with `member_algorithm: ML-DSA-65` round-trips
  (guarded by the `pq` extra).

## Out of scope for v0.5

- FROST / aggregate single-signature output.
- Mixed-algorithm membership in one set (all members share
  `member_algorithm`).
- Automated cross-host signature collection transport (the coordinator
  collecting detached sigs is the operator's integration, as with HSM
  signing today).
