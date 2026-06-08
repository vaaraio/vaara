# Evidence-bundle set vectors, v0

Fixtures for the batch full-lens check on a directory of evidence bundles:
what `vaara verify-bundles` produces. Where `bundle_doc_v0` asks "does this
one bundle verify", these vectors ask the question that only shows up across
a pile of bundles, possibly from more than one issuer: how many verify, how
many authenticate, and what does the evidence cover?

Each bundle document is the same on-disk shape `bundle_doc_v0` commits (one
receipt plus whatever identity, signature, back-link, inclusion, consistency,
and revocation evidence accompanies it). Each is loaded and run through the
full lens stack, then the set rolls up:

- **Pass count** — how many bundles are `ok`: signature established and every
  applicable lens passed.
- **Authentication count** — how many had their signature established at all,
  by identity or by the signature lens.
- **Lens coverage** — for each lens, how many bundles carried the evidence it
  needs (`lensApplicable`) and how many passed (`lensPassed`).
- **Coverage gap** (advisory) — `lensGaps` names the lenses no bundle in the
  set exercised: evidence the whole set never carried. Advisory, does not
  gate.

The set is `ok` iff every bundle loaded and every loaded bundle verified.
Unlike the keyless record-set check, this needs the crypto the lenses use, so
the vectors run only with the attestation extra installed.

## Layout

```
sets/<case>/*.json      a directory of evidence-bundle documents (one set)
expected.json           {case: {ok, total, loaded, passed, authenticated,
                                 lensApplicable, lensPassed, lensGaps}}
_check_independent.py    reuses the Vaara-free evidence_bundle_v0 evaluator,
                         no Vaara import
```

`lensGaps` is in lens order (identity, signature, back_link, inclusion,
consistency, revocation), not alphabetical.

## Cases

- `clean` — `all_lenses_pass` + `signature_only`, both verify; every lens is
  exercised somewhere in the set, so no coverage gap.
- `mixed` — `all_lenses_pass` + `tampered_inclusion`; one bundle fails its
  inclusion proof, so the set is not `ok`. No bundle carries signature-lens
  material, so `signature` is a coverage gap.
- `thin` — `signature_only` alone; it verifies, so the set is `ok`, but five
  of the six lenses are never exercised: a thin but valid set.
- `unauthenticated` — `all_lenses_pass` + `unauthenticated_in_log`; the second
  bundle sits in the log but never establishes its signature, so it is not
  `ok` and the set is not `ok`. Authenticated count is 1 of 2.
