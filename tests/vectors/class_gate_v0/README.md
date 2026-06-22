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
audit property to an enforcement one (C9.2.4 consuming C9.2.10 as data). See the
parent `SPEC.md` Section 5.3 and `tests/vectors/external_evidence_v0/` for the
completeness half this reuses.

## The gate

The gate is **membership**, not an ordering over class labels. SPEC 5.3 computes
no ordering downstream, so the consumer cannot ask "is the sealed class at or
below a ceiling"; it asks "is the sealed class one I permit." It reads the bound
off the boundary and does not re-derive the chain or query a log:

- The seal pins the run length and, optionally, the highest action class the
  boundary authorized (`maxClass`). It is minted as a signed terminal receipt
  here, so the bound is under signature, not asserted loose.
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

## Verdicts

Per case: `all_signatures_ok`, `contiguity` (`ok` / `present` / `expected` /
`missingSeqs`), and the gate decision (`permit` / `reason` / `worstCaseClass`).
The expected matrix is `expected.json`. Note `permit_gap_bounded` permits while its
`contiguity.ok` is false: the gap is real, and the seal bounds it.

## Run

```
python tests/vectors/class_gate_v0/_check_independent.py   # recompute, no Vaara; exit 0 = pass
python tests/vectors/class_gate_v0/_generate.py            # regenerate (imports Vaara to mint)
vaara enforce-by-class tests/vectors/class_gate_v0/permit_gap_bounded --permit data.read --permit data.write
```
