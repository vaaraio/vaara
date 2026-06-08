# Audit-summary golden pages, v0

The one-page Markdown a regulator reads, rendered from a record-set check.
`render_record_set_summary` turns a `RecordSetReport` into a plain page: the
verdict, the record counts, and the findings. These vectors pin the exact
bytes that page produces.

The fixtures are the `record_set_v0` sets, reused: each set is checked with
the keyless `check_record_set`, and the rendered page is committed under
`pages/<case>.md`. The test renders each set and asserts byte-equality with
its committed page.

Two things are verified, on two levels. The golden bytes pin the exact page the
renderer produces (a deterministic render carrying no timestamp and no key, so
the same set always renders the same page). On top of that, `_check_independent.py`
proves the page tells the truth: with no Vaara import it parses each golden page,
pulls out the claims it makes (the verdict word, the counts, the finding ids and
records, the non-conforming count), and asserts they equal what
`record_set_v0/expected.json` independently says for the same case. So the
verdict the regulator reads is confirmed to match the machine verdict by a party
that re-derived it from the records alone. The conformance itself is reproduced
from the SEP-2828 schema by `record_set_v0/_check_independent.py`.

## Layout

```
pages/<case>.md          the exact Markdown render_record_set_summary produces
                         for the record_set_v0 set of the same name
_check_independent.py    stdlib only (re + json), no Vaara import: asserts each
                         page's claims match record_set_v0's independent verdict
```

## Regenerating

The pages are generated from the renderer over the `record_set_v0` sets. If the
renderer changes on purpose, re-render and commit the new pages; an accidental
change shows up as a failing byte comparison.
