# Independent second runner: EP-SCITT-STATEMENT-IDENTITY-v0.1

A second party's verifier run over EMILIA Protocol's statement-identity
vectors, published so anyone can fetch the artifacts and run them again.

Iman Schrock asked on the SCITT mailing list for a second runner to consume
`vectors.reference.json` with its own verifier instead of the bundled one,
and to report both positive signatures, the distinct exact-entry digests,
the common signing-input digest and the hostile substitutions. This
directory is that run.

This is the opposite direction from `conformance/reproductions.json`. That
file records other parties reproducing Vaara's vectors. This one records
Vaara's maintainer running somebody else's.

## What is here

```
independent_verify.py     the verifier, human-readable output
emit_report.py            the same checks as deterministic JSON
vectors.reference.json    the author's file, mirrored unmodified
RESULT.txt                the run
report.independent.json   the run, machine-readable
```

The verifier imports nothing from EMILIA and nothing from Vaara. It is the
Python standard library plus `cryptography` for raw P-256, and it
reconstructs every digest from the base64url fields in the vectors.

## Running it

```
python independent_verify.py                 # vectors from this directory
python independent_verify.py path/to/file    # or point it somewhere else
python emit_report.py > report.independent.json
```

`emit_report.py` carries no timestamps and no random values, so a re-run on
the same inputs produces byte-identical output and the same SHA-256. The
one value that moves is `verifier.sha256`, which is a digest of
`independent_verify.py` itself and changes when that file changes.

## The result

Both signatures verify over the Sig_structure. The two exact-entry digests
differ, the signing-input digest is common to both, and the relation is
same `r` with `s_B = n - s_A`, A high-S and B in canonical low-S form. The
classification came out of these checks as
`same_signing_input_different_envelope`.

All seven hostile substitutions were refused: a payload bit flip against
each signature, reversed `s` bytes, an all-zero signature, `s` plus the
group order, a valid signature offered against a different key, and the
envelope bytes used as the signing input.

Totals: 11 of 11 positive assertions matched, 7 of 7 hostile cases refused.

## What the run establishes, and what it does not

It establishes that the published vectors reproduce under an independent
implementation, in a different language on a different platform, and that
the three identities stay separate when a second party computes them.

It does not establish independently derived vectors, EP profile
verification, Transparency Service registration, or anything about the
specification text. None of those were run here.

## Provenance

`vectors.reference.json` is mirrored unmodified from

  https://github.com/emiliaprotocol/emilia-protocol/tree/e507acdf8efbe8951cb4294801d4c440f0b86a5a/conformance/composition/scitt-statement-identity-v0.1

on branch `feat/aic-ccs-partner-artifacts`, sha-256
`889e410cceec75f4c0955ca9a373d4a8375c00300cbe4d2be375a559958de697`, which
matches the pin the author published. The file is the author's work and is
carried here only so this run names an exact byte set. Cite this directory
by commit sha, so a link keeps pointing at the same bytes after the branch
moves.
