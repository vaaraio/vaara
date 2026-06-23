# vaara.ingest/v0 conformance corpus

The universal sink in vector form. Each case is one foreign evidence record
(a SEP-2643 denial, a SEP-2787 attestation, a SEP-2817 invocation context, an
unrecognized record) mapped onto the SEP-2828 evidence model and sealed into
one signed, content-addressed `vaara.ingest/v0` envelope.

## Layout

- `cases/<name>.json`, a `{record, evidence}` pair. `record` is the signed
  envelope; `evidence` is the normalized evidence object it content-addresses.
- `corpus.json`, the manifest: the case list, the published HS256 test
  secret, the fixed `nonce`/`iat` that make the envelopes reproducible, each
  case's content address and signature, and a `corpusDigest` over the set.
- `_check_independent.py`, a standalone verifier that imports no Vaara code.
- `_generate.py`, regenerates the whole corpus from the registry.

## What passing means

Run the checker:

    python tests/vectors/ingest_v0/_check_independent.py

For every case it recomputes, from the envelope rules alone:

1. the content address: `sha256` over the RFC 8785 (JCS) canonical bytes of
   the evidence object, equal to `record.evidenceRef.digest`;
2. the HS256 signature: `HMAC-SHA256` over the JCS bytes of the envelope with
   the signature field removed;
3. the bind: the envelope `sourceFormat` equals the evidence `sourceFormat`;
4. the `corpusDigest` over the manifest.

Steps 1, 2 and 4 are pure standard library. Only the JCS canonicalizer
(`rfc8785`) is a third-party dependency. The honest gap report (`missing`),
the established proof fields (`sep2828`), and the non-proof context
(`advisory`) all live inside the digested evidence object, so editing any of
them breaks the content address and therefore the signature.

The conformance unit is recompute-determinism over these vectors, not the
authenticity of this publisher. The corpus signs with a published HS256 test
secret to keep the checker dependency-light; production identity is asymmetric
(ES256 / did:web).

## Regenerating

The corpus is never hand-edited. It is a loop over the normalize input corpus
at `../normalize_v0/inputs/`. Add a source format by registering its
`SourceProfile` and dropping an input fixture there, then:

    python tests/vectors/ingest_v0/_generate.py

and commit the result. `test_committed_vectors_match_fresh_emit` fails if the
committed pairs drift from a fresh emit, so a forgotten regen cannot land.
