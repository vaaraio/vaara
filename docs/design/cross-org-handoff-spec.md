# Design spec: hand a record to another organisation's regulator

Status: draft for v0.65. Companion to
`docs/design/key-rotation-retention-spec.md` (the retained-record verdict this
reuses), `docs/design/resolvable-agent-identity-spec.md` (binding), and
`docs/design/evidence-bundle-spec.md` (the within-org capstone). Builds on the
eIDAS RFC 3161 time anchor in `src/vaara/audit/timeanchor.py`.

## The problem

The records so far assume one organisation verifying its own evidence. The EU AI
Act splits the duty across organisations. A *provider* (vendor A) builds and
signs the execution record. A *deployer* (customer B), a different organisation
running A's system, keeps the logs that system generates (Article 26(6),
"to the extent such logs are under their control, for at least six months") and
is audited by B's own regulator. The provider retains its own copies under
Articles 18 and 19. Article 12 is the logging capability all of this rests on.

So the party that must satisfy a regulator (B) is not the party that signed the
evidence (A). At audit time, possibly years later, the regulator:

- has no prior relationship with vendor A and no live channel to A (A may be in
  another jurisdiction, or out of business);
- does not inherently trust customer B, the party relaying the evidence;
- receives the evidence as a single exhibit and must verify it offline.

Nothing in the toolkit packaged a record for that handoff. `verify-bundle`
assembles the within-org lenses but assumes the verifier holds each piece;
`verify-retained` settles one record against an archived document but takes the
document, key history, and anchor as separate inputs the caller already trusts.

## The handoff package

One self-contained JSON document carries everything the regulator needs:

```
{
  "schema": "vaara.cross-org-handoff/v0",
  "evidence": {
    "record":       { ... },   // the SEP-2828 receipt vendor A signed (required)
    "did_document": { ... },   // A's ARCHIVED DID document, lists the retired key (required)
    "key_history":  { ... },   // optional override; else read from the document
    "revocations":  { ... },   // optional override; else read from the document
    "anchor":       { ... }    // optional eIDAS RFC 3161 anchor over the record
  },
  "manifest": {
    "producer": "did:web:vendor-a...",   // MUST equal record.iss and did_document.id
    "holder":   "did:web:customer-b...",  // the relaying deployer (informational)
    "cover":    { ... },                  // optional plain-language block (see below)
    "record_digest", "did_document_digest",
    "key_history_digest", "revocations_digest", "anchor_digest"
  },
  "manifest_digest": "sha256:...",        // the package fingerprint
  "holder_attestation": { ... }           // optional custody signature (see below)
}
```

`verify_handoff` runs four stages.

**Integrity.** Recompute each component digest and compare it to the manifest's
pinned value, recompute the manifest fingerprint, and confirm the producer is
coherent (`manifest.producer == record.iss == did_document.id`). The record,
document, and anchor are pinned by `sha256(jcs(component))`. The key history and
revocations are pinned by *model* digests: `KeyHistory.digest()` and
`RevocationRegistry.digest()` over the canonicalised, re-sorted model (the
override if present, else the projection from the document), the same source the
verdict uses. A raw hash of the supplied bytes would diverge from the verdict's
basis whenever an override arrived in non-canonical order; the model digest does
not. One helper resolves the effective source for both build and verify.

**Record.** Route the record through `verify_receipt_retained` with that
effective key history and revocations, producing the unchanged `verifiable` /
`corroborated` tiers from the retained-record spec. C2 redefines neither word.

**Anchor.** The receipt has no trail `record_hash`, so the C2 anchor is defined
directly: its message imprint is `sha256(jcs(record))`, stored as the anchor's
`chain_head_hash`. Verify requires the RFC 3161 token to attest exactly those
bytes (the cryptographic step, which needs the timeanchor extra) **and** the
stored imprint to equal `sha256(jcs(record))` (a byte compare, always
reproducible). An anchor taken over a different record fails the binding and is
disregarded; it never silently corroborates. The binding lives in
`verify_handoff`; the token's attested time is supplied by the caller, exactly
as `verify-retained` takes an `anchored_time` it verified separately.

**Custody.** An optional `holder_attestation` is an asymmetric signature over
`jcs(manifest)`, verified from the enclosed public JWK and reported in `custody`.
It never affects the record verdict.

## Where trust comes from, stated plainly

The holder assembles the package and controls both the components and the
manifest that pins them. Content addressing therefore proves only that the
package is *internally consistent*: a green `integrity_ok` catches corruption and
accidental drift, not a dishonest holder. The substance rests elsewhere.

- The record's authenticity is vendor A's signature against vendor A's *genuine*
  identity. The package encloses a document *claiming* to be A's; it does not
  prove that claim. `producer_identity_basis` is `self_asserted_unpinned` until
  the caller passes the DID document it independently trusts as A's (its retained
  key archive), which pins it to `pinned` when the bound key matches, or
  `pin_mismatch` otherwise. The match is on key material, so it survives
  rotation: an archive that keeps retired keys still pins an old signing key.
- The eIDAS anchor is the one component a holder cannot forge, because the
  Time-Stamp Authority is outside both A and B. It is what raises a record from
  verifiable to corroborated.
- The holder custody attestation proves only that whoever holds the enclosed key
  signed this exact package for delivery. With a self-supplied key it is
  `holder_attested_selfsupplied` and carries no non-repudiation weight unless the
  holder's key is pinned out of band.

## Strict mode

`--strict` is the regulator-grade gate: it passes only a record that is
corroborated, with a recorded validity window, an affirmative revocation source
(an explicit registry, even an empty one, or revoked markers on the document),
and a pinned producer identity. The default mode passes a verifiable record and
states what is missing for the stronger tier; strict makes the stronger tier
mandatory without changing `verify_receipt_retained`.

## Scope and non-goals

- One record per package. A record is the unit a single anchor binds; a set
  would need a Merkle or per-record anchor, which is the existing trail-head
  anchor's job. The schema is versioned (`/v0`) so a later batch surface
  (`verify-handoffs`, mirroring `verify-bundles`) and a fold into the
  `export-article12` package add cleanly.
- The `cover` block is opaque, caller-supplied plain language (system, action,
  period, provider, deployer, the obligation served). Vaara carries it pinned but
  asserts no legal conclusion about it.
- Out of scope: that the record corresponds to the specific action under audit,
  and that the holder disclosed every record. Selective disclosure and
  suppression are a separate concern; this verifies what is handed over, not what
  is withheld.
- Integrity, both digest kinds, the anchor binding, the custody signature, and
  the producer pin are reproducible by an outside implementation. The one step
  that is not is verifying the RFC 3161 CMS token, which needs the timeanchor
  extra and a trusted TSA chain.

## Conformance vectors

`tests/vectors/cross_org_handoff_v0/` carries the cases (clean verifiable,
corroborated, anchor over a different record, tampered component, tampered
manifest, valid and corrupted holder attestation, non-canonical key-history
override, record signed after retirement, producer pin, strict) and a Vaara-free
checker that reproduces every verdict with only `cryptography` and `rfc8785`,
taking the anchor's attested time pre-verified exactly as `key_rotation_v0` does.
