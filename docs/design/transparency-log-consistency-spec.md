# Design spec: append-only consistency proofs for the transparency log

Status: draft for v0.54. Companion to the Merkle transparency log in
`src/vaara/attestation/transparency_log.py` (inclusion proofs, since v0.13)
and the OVERT Phase 3 IAP in `src/vaara/attestation/iap.py`.

## Goal

The transparency log already issues inclusion proofs: given a receipt and a
published root, a verifier confirms the receipt is in the log. An inclusion
proof says nothing about whether the log operator rewrote earlier history.
A log operator could serve one tree to one auditor and a forked tree, with
some past entry quietly altered or dropped, to another. Every inclusion proof
against the matching root still verifies, so inclusion alone does not detect
the fork.

A consistency proof closes that gap. It shows the log at an earlier size is a
verifiable prefix of the log at a later size: every leaf in the smaller tree
is present, in the same position, in the larger one, and nothing earlier
changed. That is the append-only property a transparency log exists to
provide.

## Mechanism

The log is an RFC 6962 binary Merkle tree (leaf hash `SHA-256(0x00 || leaf)`,
node hash `SHA-256(0x01 || left || right)`). Consistency proofs follow
RFC 9162 (RFC 6962-bis) section 2.1.4: a prover emits the minimal set of
subtree hashes that lets a verifier recompute both the old root (at
`first_size`) and the new root (at `second_size`) and confirm the old tree is
a prefix of the new one.

New surface, all additive:

- `consistency_proof(first_size, second_size)` on the log returns a
  `ConsistencyProof` (`first_size`, `second_size`, ordered sibling `hashes`).
  The proof is empty for the trivial cases: an empty prefix (`first_size` is
  0) or identical sizes.
- `root_at(tree_size)` returns the Merkle root over the first `tree_size`
  leaves, so a monitor can pin a historical signed tree head before asking for
  a proof against a later one.
- `verify_consistency(first_size, first_root, second_size, second_root, proof)`
  recomputes both roots from the proof and returns a single `bool`. The two
  roots are supplied by the verifier (the signed tree heads it holds at two
  points in time); the proof hashes alone do not bind to a specific pair of
  roots, exactly as with inclusion proofs.

The receipt envelope, canonicalization, inclusion-proof format, and signature
verification are unchanged. The envelope version stays 1.

## Trust model

A consistency proof is a statement about the log's structure, not about who
signs the tree heads. It detects a rewrite *if* the verifier holds an
authentic earlier root to check against. The earlier root has to come from a
trustworthy channel: a signed tree head from the log operator that the
verifier recorded earlier, an independent monitor's witness cosignature, or
Vaara's own audit-trail hash chain, which already anchors roots over time. The
proof turns "trust the operator not to rewrite history" into "check that the
operator did not rewrite history," conditional on a pinned reference root.

This is the same separation the rest of the evidence plane keeps: durability
from the hash chain and the transparency log, issuance-correctness from
conformance vectors and independent implementations, and semantic-correctness
left to governance and review.

## Production logs

The in-process log is the reference operator for demonstrations. The public
surface (`append`, `inclusion_proof`, `consistency_proof`, `root_at`,
`root_hash`) is shaped to match what a sigstore Rekor-backed adapter would
expose, so a future `RekorTransparencyLog` drops into the same call sites
without changing verifier code.

## Conformance

`tests/vectors/transparency_consistency_v0/` carries nine cases over a
twelve-leaf log: power-of-two and non-power-of-two prefixes, an empty prefix,
identical trees, and two negatives (a flipped proof hash, and a proof checked
against a forked second root). `_check_independent.py` reproduces every
verdict using only the Python standard library and without importing Vaara,
so the append-only guarantee is consumable by a second implementation.
