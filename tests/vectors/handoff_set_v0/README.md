# Handoff-set vectors, v0

Fixtures for the batch handoff check on a directory of cross-org packages: what
`vaara verify-handoffs` produces. Where `cross_org_handoff_v0` asks "does this
one package verify offline under its rotated-out key", these vectors ask the
question that only shows up across a pile of packages, from one provider or
several: how many records verify, how many are anchor-corroborated rather than
resting on the signature alone, and how many had their producer identity pinned?

A set is a list of `cross_org_handoff_v0` case names plus the mode the batch
applies uniformly: one `strict` flag and one optional trusted DID document the
set pins every package against. These vectors add no new crypto: they reuse the
single suite's committed packages, group them into sets, and record the roll-up
`check_handoff_set` produces:

- **Pass count** how many packages are `ok` for the chosen mode.
- **Verifiable / corroborated** how many records reached each tier. The gap
  between them is the part of the set whose authenticity rests on the signature
  alone, with no anchor.
- **Pinned count** how many had their producer pinned against the trusted
  document. `pinningGap` is advisory and fires when no package was pinned: every
  record's authenticity rests on a self-asserted identity. It does not gate.

The set is `ok` iff every document loaded and every loaded package verified for
the chosen mode. A single out-of-band identity reference applies to the whole
set, so only packages whose bound key appears in it pin.

## Layout

```
sets.json              {case: {cases: [name...], strict, trusted_from_case}}
expected.json          {case: {ok, strict, total, loaded, passed, verifiable,
                               corroborated, pinned, pinningGap}}
_generate.py           rebuilds sets.json + expected.json from the single suite
_check_independent.py  reuses the cross_org_handoff_v0 evaluator, no Vaara
                       import, and reproduces the roll-up
```

The packages live in the sibling `cross_org_handoff_v0` suite; these vectors
reference them by name. Each package's anchor time is taken from its case's
pre-verified `anchoredTime`; the RFC 3161 token itself is not re-verified here,
exactly as in the single suite. Like it, they need the attestation extra
(`pip install 'vaara[attestation]'`).
