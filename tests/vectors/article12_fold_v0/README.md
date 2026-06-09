# article12_fold_v0

Conformance vectors for the EU AI Act Article 12 regulator-package **fold**:
`vaara trail export-article12` attaching verified SEP-2828 evidence as sidecars
under `evidence/`. Cross-org handoff packages (Article 26(6) deployer custody)
and confidential-VM enforcement bindings ("where it ran") are verified at export
and folded alongside the signed trail, the Article 19 time anchor, and the
record-keeping report.

These vectors do not invent new crypto. They reuse the committed corpora next
door, `cross_org_handoff_v0` and `enforcement_attestation_v0`, fold named
subsets into a real package, and pin the roll-up the package carries in
`evidence/attestations_summary.json`.

## Files

- `fold.json`: the scenarios. Each names handoff cases, enforcement cases, and
  the verifier-side inputs the fold applies (a trusted DID document to pin
  handoff producer identity, an expected launch measurement to pin enforcement).
  `fail_closed` names scenarios whose attachment does not verify, which must
  abort the export with no package written.
- `expected.json`: for each producing scenario, the folded `evidence/`
  membership and the handoff / enforcement roll-up the package carries.
- `_generate.py`: builds each package with `export_article12` (reusing each
  package's pre-verified anchor time, exactly as `handoff_set_v0` does) and
  records `expected.json`. No zip is committed: it carries a fresh signature and
  a runtime `.vcek.pem`. The test rebuilds it in a temp directory.
- `_check_independent.py <package.zip>`: the Vaara-free checker. It opens a
  produced package and reproduces every folded verdict from the **same bytes
  folded into the zip**, composing the two single-verb suites' own `_evaluate`
  functions, then asserts each reproduced roll-up equals the one the package
  claims. It never imports Vaara.

## Honesty model

The fold never upgrades a verdict. A handoff is corroborated only with a
verified anchor and pinned only against a trusted DID document; an enforcement
binding is never "attested" (the VCEK chain to AMD's root is not validated). The
eIDAS time anchor stays the only un-forgeable component of the package, and the
report says so. An attachment that does not verify fails the export closed.

Regenerate with `python tests/vectors/article12_fold_v0/_generate.py`.
