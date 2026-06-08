# SEP-2828 conformance corpus

A standalone, versioned test corpus for the SEP-2828 server-side signed
execution record. It exists so the record format, not any one
implementation, is the thing a verifier trusts. An independent emitter or
verifier can download this directory, run one command, and check itself
against the same fixtures the reference implementation ships.

The corpus is self-contained. It imports no Vaara code and needs nothing
but the Python standard library, so conforming to it does not mean
conforming to Vaara; it means conforming to the published spec fixtures.

## What is here

```
conformance/sep2828/
  VERSION            the corpus version (semver)
  MANIFEST.json      per-file sha256 plus one corpusDigest over the whole set
  run.py             stdlib runner: run the suites, or verify the bytes
  record_conformance_v0/   per-record conformance fixtures + checker
  record_set_v0/           set-level conformance fixtures + checker
```

Each suite carries its own fixtures, an `expected.json` of the verdicts,
and an independent checker (`_check_independent.py`) written from the
schema alone. The two suites cover the two questions a SEP-2828 verifier
answers without a key:

- **`record_conformance_v0`**: is one record well-formed, and does the
  binding it proves about itself (`resultCommitment.projectionDigest` over
  the projection bytes) recompute from the record alone?
- **`record_set_v0`**: across a directory of records, how many conform,
  and where are the gaps (a call recorded twice, an authorised decision
  with no outcome, an executed action with no committed result)?

## Running it

```
cd conformance/sep2828
python run.py                  # run both suites; exit 0 iff all cases match
python run.py --verify-manifest # confirm the files match the published digests
python run.py --version        # print the corpus version
```

A single suite also runs on its own:

```
python record_conformance_v0/_check_independent.py
python record_set_v0/_check_independent.py
```

## Claiming conformance

Your implementation conforms to a corpus version when, for every fixture,
it reaches the verdict recorded in that suite's `expected.json`: the same
`conforms` decision, the same set of failed required checks, and the same
advisories or findings. Pin the version you tested against (the value in
`VERSION`) and the `corpusDigest` from `MANIFEST.json`, so a claim names an
exact byte set rather than a moving target.

The boundary is deliberate. These checks are keyless: they cover wire
shape, enum values, digest formats, the self-recomputing projection
binding, and the cross-record set properties. They are not signature
verification, not issuer trust, not time-anchor verification, and not a
claim about runtime truth. Those layers need external material and are
checked separately.

## Versioning

`VERSION` is semver. A patch adds fixtures or clarifies docs without
changing any existing verdict. A minor may add new checks or cases that an
already-conforming implementation could newly fail. A major changes the
record contract itself. The `*_v0` suite names track the on-wire record
schema version and move independently of the corpus version.

## Provenance

The fixtures are mirrored from the reference implementation's test vectors
under `tests/vectors/`, assembled by `scripts/build_conformance_corpus.py`.
A test in that repository fails if this corpus drifts from the source
vectors or from its own manifest, so the published bytes always match what
the reference implementation tests against.
