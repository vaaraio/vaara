# capture_digest_only_v0 conformance vectors

Digest-only capture of governed-call arguments. An operator declares argument
fields (patient data, an account number, a document body) in a capture policy;
the trail row, the review queue copy and the approval request then hold each
declared value as a keyed digest and a reference. See
`src/vaara/_capture.py` and `docs/capture-digest-only.md`.

A declared value is stored as

```
{"captured": "digest-only", "digest": "hmac-sha256:<hex>", "reference": "ehr"}

digest = HMAC-SHA256(key, "vaara.capture.v1\n" || path || "\n" || JCS(value))
```

`path` is the concrete position of the value (`rows.2.note`), `JCS` is RFC 8785,
and the key stays outside the trail, so a holder of the trail alone cannot test
a guessed value against a digest. An operator who holds the key shows that a
value is the one that was passed by recomputing its digest.

## Cases (`cases.json`)

- `nested_field_and_reference`: a dotted path, a reference, a non-ASCII value,
  an undeclared field left as sent.
- `every_item_of_a_list`: a `*` segment takes every item of a list; two rows
  with the same value get different digests because their paths differ.
- `a_lone_wildcard_keeps_the_keys`: `*` alone masks every top-level value,
  numbers, booleans, null and nested objects included; the keys stay.
- `a_tool_no_rule_names`: no rule matches, the arguments come back as sent.
- `disclosures`: the right value at the right path matches its digest; another
  value, or the same value at another path, does not.

`key_hex` is a fixed TEST key. A deployment's key is random and never published.

## Honest limits

The vectors cover the masking and the digest. Where it is applied is covered by
`tests/test_capture_digest_only.py`. A digest of a personal identifier can still
be linked back to the person by whoever holds the key. Only arguments are
masked: reason text, tool results, outcome feedback and reviewer free text are
not.

## Reproduce

Independent checker (standard library plus `rfc8785`, no Vaara import):

```
python tests/vectors/capture_digest_only_v0/_check_independent.py
```

Exit code 0 means every case reproduced. Regenerate `cases.json` (stable, the
key is fixed) with:

```
python tests/vectors/capture_digest_only_v0/_generate.py
```
