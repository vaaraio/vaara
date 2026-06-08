# Audit-summary golden pages, v0

The one-page Markdown a regulator reads, rendered from a record-set check.
`render_record_set_summary` turns a `RecordSetReport` into a plain page: the
verdict, the record counts, and the findings. These vectors pin the exact
bytes that page produces.

The fixtures are the `record_set_v0` sets, reused: each set is checked with
the keyless `check_record_set`, and the rendered page is committed under
`pages/<case>.md`. The test renders each set and asserts byte-equality with
its committed page.

There is no separate Vaara-free re-implementation here, and that is the point.
The summary is a rendering of a verdict, not a second judgement. The verdict it
formats is already verified independently by `record_set_v0/_check_independent.py`
(a Vaara-free checker that reproduces every count and finding from the schema
alone). The golden bytes are this layer's contract: the page is deterministic,
carrying no timestamp and no key, so the same set always renders the same page.

## Layout

```
pages/<case>.md     the exact Markdown render_record_set_summary produces for
                    the record_set_v0 set of the same name
```

## Regenerating

The pages are generated from the renderer over the `record_set_v0` sets. If the
renderer changes on purpose, re-render and commit the new pages; an accidental
change shows up as a failing byte comparison.
