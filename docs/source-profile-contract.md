# The source-profile contract

`vaara ingest` seals any foreign evidence into one signed, content-addressed
record and reports honestly what a complete signed execution record still
lacks. An unrecognized document is still sealed; it just carries
`recognized: false` and an empty mapping. A **source profile** upgrades that:
it recognizes a format and lifts its fields onto the SEP-2828 evidence model,
so the gap report names what is established and what is missing.

Most profiles are pure field-mapping, and field-mapping is data. This contract
lets you bind a format by dropping a JSON spec into
`src/vaara/attestation/profiles/`, with no Python and no change to the ingest
hot path. You own the mapping; we own the gate.

## What a profile may and may not assert

A profile asserts only what the source document already establishes:

- `advisory` — non-proof context lifted from the source (a builder id, a title).
- `sep2828` — real evidence-model fields the source genuinely carries.
- `missing` — the honest gap: fields a complete signed record needs and this
  source does not provide.

A declarative profile **cannot** fabricate a signature, a back-link digest, or
any other computed value, because those are derived (a canonical hash, a crypto
operation), not copied. A format that needs a computed field stays a Python
profile in `_normalize.py`. The two shipped Python profiles that compute things
(SEP-2787's back-link) are exactly those cases.

## Spec shape

```json
{
  "sourceFormat": "slsa-provenance",
  "sourceTitle": "SLSA v1 in-toto provenance",
  "priority": 50,
  "detect": {
    "all": [
      {"path": "_type", "equals": "https://in-toto.io/Statement/v1"},
      {"path": "predicateType", "startsWith": "https://slsa.dev/provenance/"}
    ]
  },
  "evidencePlane": null,
  "advisory": {
    "subjectName": "subject[0].name",
    "builderId": "predicate.runDetails.builder.id"
  },
  "sep2828": {},
  "missing": ["alg", "signature", "backLink", "receiptAsserted", "outcomeDerived"],
  "notes": ["a SLSA statement attests how an artifact was built, not an execution decision"]
}
```

### Required fields

| Field | Meaning |
| --- | --- |
| `sourceFormat` | Stable id for the format; the registry key. |
| `sourceTitle` | Human-readable title shown in the gap report. |
| `detect` | Match rules (below). A document that matches is recognized. |

### Optional fields

| Field | Default | Meaning |
| --- | --- | --- |
| `priority` | `100` | Lower runs first. Built-ins are 10–30; keep declarative profiles above them unless you mean to pre-empt. |
| `evidencePlane` | `null` | `outcome`, `decision-attested`, `decision-input`, or `null`. |
| `advisory` | `{}` | `{outKey: source}` non-proof context. |
| `sep2828` | `{}` | `{dotted.field: source}` real evidence-model fields. |
| `missing` | `[]` | Field names a complete signed record still needs. |
| `notes` | `[]` | Plain-language caveats for the gap report. |

## Paths

A source is a **path string** or a `{"const": value}` literal.

- Dotted segments walk dict keys: `predicate.runDetails.builder.id`.
- `[n]` indexes a list: `subject[0].name`, `subject[0].digest.sha256`.
- A path that does not resolve yields nothing: the `advisory` key is omitted
  and the `sep2828` field is left unset, rather than failing the ingest. Map a
  field freely even if some documents of the format omit it.

## Detect rules

`detect` holds an `all` group (every rule must match), an `any` group (at least
one must match), or both. Each rule names a `path` and exactly one operator:

| Operator | Matches when |
| --- | --- |
| `equals` | resolved value equals the literal |
| `startsWith` | resolved value is a string with that prefix |
| `in` | resolved value is in the given list |
| `exists` | `true` → path resolves to non-null; `false` → it does not |

A spec with no operator, no `path`, or empty detect groups fails to compile.
`load_builtin_declarative_profiles()` logs and skips a spec that fails to
compile, so one bad file cannot break ingest for the rest.

## Worked example: SLSA provenance

Before the profile, the sink seals a SLSA in-toto statement with
`recognized: false` — sealed and tamper-evident, but unmapped. The spec above
flips it to `recognized: true`, lifts the builder id and subject name into
advisory context, and reports the honest gap: a SLSA statement attests how an
artifact was *built*, so `signature`, `backLink`, and `outcomeDerived` are
genuinely absent from it. Nothing is promoted; an unsigned build statement does
not become a signed execution record.

## Shipping a profile

1. Write the JSON spec into `src/vaara/attestation/profiles/<format>.json`.
2. It registers automatically at import via
   `load_builtin_declarative_profiles()`; `vaara ingest` picks it up.
3. Add an input fixture under `tests/vectors/normalize_v0/inputs/`, regenerate
   `expected.json` and the `ingest_v0` corpus, and the conformance suite covers
   it. The independent checker reads your spec and reproduces the mapping with
   its own code, so a new profile is not self-confirming.

To load a profile from an arbitrary path at runtime, call
`vaara.attestation._declarative.load_profile_file(path)`.
