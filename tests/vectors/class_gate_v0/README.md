# class_gate_v0 conformance vectors

Binding profile: a chain recipient gates its own next unattended action on a
boundary's sealed worst-case class (the v1.7.0 seal's `maxClass`). It holds a
policy set of action classes it will proceed under (`permittedClasses`) and
permits iff the sealed class is a member of that set, failing closed when no class
is sealed. Verifiable by any third party with only the held receipts, the signed
seal, and the issuer's public key. No live endpoint, no service to trust, no Vaara
import.

This is the artifact behind "the sealed `maxClass`, beyond bounding a gap at audit
time, is consumable at enforcement time": moving worst-case-across-chain from an
audit property to an enforcement one. See the parent `SPEC.md` Section 5.3 and
`tests/vectors/external_evidence_v0/` for the completeness half this reuses.

## The gate

The gate is **membership**, not an ordering over class labels. SPEC 5.3 computes
no ordering downstream, so the consumer cannot ask "is the sealed class at or
below a ceiling"; it asks "is the sealed class one I permit." It reads the bound
off the boundary and does not re-derive the chain or query a log:

- The seal pins the run length and, optionally, the highest action class the
  boundary authorized (`maxClass`). `maxClass` lives in the unsigned `evidence`
  block; what carries it under signature is the binding: the seal's signed
  `decisionDerived.evidenceRef.digest` is `sha256:` + JCS(evidence), so
  recomputing the digest proves the class is the class that was signed. The gate
  consumes `maxClass` only from a seal whose evidence binds; an unbound (relabeled)
  seal contributes no class and the gate fails closed.
- `permitted` when the sealed class is in `permittedClasses`; `class_not_permitted`
  when it is outside; `unbounded_no_sealed_class` when no class is sealed, so a
  gap's worst case is unbounded and the gate fails closed.

## The wedge

A permitted sealed class permits **even over a gap**. The seal bounds the missing
record's worst case at `maxClass`, so the recipient does not need the dropped
record to act safely; it needs only the committed bound. `_check_independent.py`
imports the standard library plus `cryptography` and `rfc8785`, and nothing else:
a third party reaches the same permit/deny offline from the committed bytes.

## Cases

`permittedClasses` for the vector is `["data.read", "data.write"]`.

- `permit/`: full run (seq 0, 1, 2) plus a seal with `maxClass="data.read"`
  (permitted), so contiguous and permitted.
- `permit_gap_bounded/`: seq 1 withheld, same `data.read` seal. The boundary has a
  provable gap, yet the gate permits, because the seal bounds the gap at the
  permitted class. This is the case that demonstrates the point.
- `deny_class/`: full run plus a seal with `maxClass="tx.transfer"` (outside the
  set), so contiguous but denied.
- `deny_unbounded/`: seq 1 withheld plus a seal that names no class, a gap whose
  worst case is unbounded, so the gate fails closed and denies.
- `deny_relabeled/`: the adversary. A full run plus a seal whose `maxClass` was
  relabeled from `tx.transfer` to the permitted `data.read` after signing. The
  record signature still verifies (`all_signatures_ok` true), but the evidence no
  longer recomputes to the signed digest (`all_evidence_bound` false), so the gate
  refuses to consume the class and fails closed. A gate that read `maxClass` raw
  would be tricked into permit.

## Verdicts

Per case: `all_signatures_ok`, `all_evidence_bound`, `contiguity` (`ok` /
`present` / `expected` / `missingSeqs`), and the gate decision (`permit` /
`reason` / `worstCaseClass`). The expected matrix is `expected.json`.
`all_evidence_bound` is the load-bearing one: it recomputes each receipt's
`evidence` digest against the signed `evidenceRef.digest`, which is what places the
sealed `maxClass` under signature. Note `permit_gap_bounded` permits while its
`contiguity.ok` is false (the gap is real, the seal bounds it), and
`deny_relabeled` denies while `all_signatures_ok` is true (the signature holds, the
binding does not).

## Run

```
python tests/vectors/class_gate_v0/_check_independent.py   # recompute, no Vaara; exit 0 = pass
python tests/vectors/class_gate_v0/_generate.py            # regenerate (imports Vaara to mint)
vaara enforce-by-class tests/vectors/class_gate_v0/permit_gap_bounded --permit data.read --permit data.write
```
