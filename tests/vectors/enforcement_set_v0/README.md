# Enforcement-set vectors, v0

Fixtures for the batch enforcement check on a directory of (record, report,
VCEK) triples: what `vaara verify-enforcements` produces. Where
`enforcement_attestation_v0` asks "does this one report bind this one record to
a CVM", these vectors ask the question that only shows up across a pile of
enforced records: how many bind, at what tier, and did any pin a vetted launch
image?

A set is a list of `enforcement_attestation_v0` case names plus the mode the
batch applies uniformly (one `expected_measurement`, one `strict` flag). These
vectors add no new crypto: they reuse the single suite's committed cases, group
them into sets, and record the roll-up `check_enforcement_set` produces:

- **Pass count** how many triples are `ok` for the chosen mode.
- **Tier tally** how many landed at each tier (`unverified` / `bound` /
  `measurement_pinned`). The chain-rooted `attested` tier is reserved for the
  future KDS-chained release and never appears in v0.
- **Bound count** how many had `REPORT_DATA` carry `sha512(jcs(record))`,
  independent of whether the signature or measurement checks passed.
- **Pinning coverage** how many pinned a launch measurement. `pinningGap` is
  advisory and fires when no record in the set pinned an image: the set bound to
  a CVM but never to a vetted one. It does not gate.

The set is `ok` iff every triple loaded and every loaded triple verified for the
chosen mode. Strict is unreachable in v0 (no validated VCEK chain), so a strict
set never passes.

## Layout

```
sets.json              {case: {cases: [name...], expected_measurement, strict}}
expected.json          {case: {ok, strict, total, loaded, passed, bound,
                               measurementPinned, tierCounts, pinningGap}}
_generate.py           rebuilds sets.json + expected.json from the single suite
_check_independent.py  reuses the enforcement_attestation_v0 evaluator, no
                       Vaara import, and reproduces the roll-up
```

The cases live in the sibling `enforcement_attestation_v0` suite; these vectors
reference them by name rather than copying the report bytes. Like the single
suite, they need the attestation extra (`pip install 'vaara[attestation]'`).
