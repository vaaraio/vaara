# contiguity_v0 conformance vectors

Gap-evident completeness over a stream of authorization receipts. A dropped
receipt inside a declared coverage boundary becomes a provable gap, detectable
by any third party with only the held receipts and the issuer's public key, with
no issuer access and no external witness. This is the artifact behind
"completeness, not just non-inclusion". See `src/vaara/credential/_contiguity.py`
and the `vaara verify-contiguity` CLI.

Each authorization receipt carries a signed `completeness` block in its evidence
(`boundaryId`, `seq`, `runningCount`). The sequence is contiguous by
construction at issuance; the running count is the total the boundary asserts
exists up to and including that receipt. Because the evidence is bound to the
receipt signature through `decisionDerived.evidenceRef.digest`, the count is
non-repudiable: a short held set cannot honestly claim to be whole.

## Cases

Both cases share the same five byte-identical signed receipts (seq 0..4 under
boundary `vaara-mcp-proxy`); the `dropped` case is the `complete` set minus one
file, so the gap is a genuine omission rather than a re-mint.

- `complete/`: all five receipts held. Expected: every signature verifies, every
  evidence binding resolves, and contiguity is `ok` with `present` 5,
  `expected` 5, no missing seqs.
- `dropped/`: the seq-2 receipt is withheld. The four held receipts still carry
  the signed running count that says five exist, so contiguity is not `ok` with
  `present` 4, `expected` 5, `missingSeqs` `[2]`. Signatures and evidence
  bindings on the held four still verify.
- `grant.json`: the capability grant the receipts authorize against, for
  context.
- `expected.json`: the verdict each case must produce.

## Honest limit

A pure tail truncation (holding seq 0,1,2 with nothing after) is invisible to
contiguity alone, because the latest held running count is then 3 and the held
set cannot tell that later receipts ever existed. Closing that hole is the job
of an rfc3161 anchor over the running count, which attests "at time T, N
receipts existed under this boundary". These vectors cover the gap that
contiguity alone detects.

## Reproduce

Independent checker (standard library plus `cryptography` and `rfc8785`, no
Vaara import):

```
python tests/vectors/contiguity_v0/_check_independent.py
```

Exit code 0 means every case matched its expected verdict. The same vectors are
consumable by the shipped CLI:

```
vaara verify-contiguity tests/vectors/contiguity_v0/complete   # exit 0, contiguous
vaara verify-contiguity tests/vectors/contiguity_v0/dropped    # exit 1, missing seq 2
```

Regenerate the cases (ECDSA signatures are randomized, so signatures change but
verdicts do not) with:

```
python tests/vectors/contiguity_v0/_generate.py
```
