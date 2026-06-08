# Conformance-statement vectors, v0

Fixtures for `vaara conformance-statement`: the self-test an emitter runs to
prove SEP-2828 conformance against the published corpus instead of asking to be
trusted. The command checks the corpus bytes against their manifest, reproduces
every recorded verdict with this implementation's keyless check, optionally runs
the emitter's own records through the same set check, and prints one
reproducible statement that names the exact corpus byte set (version plus
corpusDigest).

These goldens pin what the command produces against the real corpus under
`conformance/sep2828`, so they move whenever that corpus does. Regenerate them
with `python scripts/build_conformance_statement_vectors.py`.

## Layout

```
emitter_records/<scenario>/   the emitter's own SEP-2828 records for a scenario
expected.json                 {scenario: statement.to_dict()}
pages/<scenario>.md           the rendered Markdown statement (deterministic)
_check_independent.py         stdlib only, no Vaara import
```

## Scenarios

- `selftest_only`: no emitter records, just the corpus self-test. Conforms.
- `clean`: emitter records that conform (a decision and its outcome). Conforms.
- `flawed`: emitter records with one non-conforming record. NON-CONFORMING,
  even though the corpus self-test passes, because the supplied records do not
  all conform.

The `clean` and `flawed` records are byte copies of the `proper_pair` and
`mixed_nonconforming` sets from the `record_set_v0` vectors, so their set
verdict is already independently fixed there.

## Verifying

```
python tests/vectors/conformance_statement_v0/_check_independent.py
```

The checker re-derives every claim each statement makes (corpus integrity by
recomputing the digests, the self-test by running the corpus's own neutral
runner, the records verdict with a second implementation of the set check) and
asserts both the golden page and `expected.json` state exactly that. Exit 0
means every statement is true under the independent derivation.
