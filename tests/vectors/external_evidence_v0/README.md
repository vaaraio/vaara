# external_evidence_v0 conformance vectors

Binding profile: a verifier that carries an `external_execution_evidence` slot
(`linked_call_id` / `evidence_hash` / `evidence_type`, the shape used by
agentrust trace-spec #34 and cMCP #301) resolves that slot against a
`vaara.receipt/v1` authorization receipt as the recomputable producer. Verifiable
by any third party with only the held slots, the held receipts, and the issuer's
public key. No live endpoint, no service to trust, no Vaara import.

This is the artifact behind "a generic external-evidence slot can pin to
vaara.receipt/v1 as the recomputable producer its verifier checks, rather than
trust an opaque hash a service vouches for." See the parent `SPEC.md` Section 5.6
and `tests/vectors/ap2_v0/` for the completeness half this reuses.

## The mapping

Each agent call in a trace produces an external-execution-evidence slot. The
vaara.receipt/v1 authorization receipt minted for that call is the producer the
slot points at:

- `evidence_hash` = `sha256(JCS(evidence))`, the content address of the receipt's
  evidence record. It equals the receipt's `decisionDerived.evidenceRef.digest`,
  so the slot and the receipt name the same recomputable artifact (JCS / RFC 8785,
  no re-canonicalization).
- `linked_call_id` is the call the receipt names: the receipt carries
  `decisionDerived.evidenceRef.ref` = `mcp:call/<linked_call_id>`, under signature.
- `evidence_type` is the receipt's evidence schema (`vaara.authorization/v0`).

Ours is the six-field action model and it carries a completeness layer the slot
does not have. The trace is the `coverage.boundary`; each receipt carries a signed
`completeness` block (`seq` + `runningCount`), so the held set proves not only that
each named call's evidence resolves but that none inside the boundary was dropped.

## The wedge

The verdict is recomputed offline from the committed bytes and the public key.
`_check_independent.py` imports the standard library plus `cryptography` and
`rfc8785`, and nothing else: no live verifier endpoint, no Vaara. A slot's
`evidence_hash` alone says a record exists somewhere; here the holder recomputes
the whole verdict with no service reachable, and the completeness block turns a
silent drop into a named gap.

## Cases

- `complete/` — the full trace (seq 0, 1, 2): every held slot resolves, and the
  completeness blocks are contiguous.
- `dropped/` — the seq-1 item is withheld, slot and receipt both. A verifier
  holding only the external slots sees {0, 2} with no inherent count and cannot
  tell call 1 existed; the held running count says three exist, so seq 1 is a
  provable gap inside the trace boundary.

## Verdicts

Per case: `all_signatures_ok`, `all_evidence_bindings_resolve`, `all_slots_resolve`,
and `contiguity` (`ok` / `present` / `expected` / `missingSeqs`). The expected
matrix is `expected.json`. Note `all_slots_resolve` is true in both cases: the
held slots resolve cleanly even in `dropped`. Only `contiguity` catches the drop,
which is the point.

## Run

```
python tests/vectors/external_evidence_v0/_check_independent.py   # recompute, no Vaara; exit 0 = pass
python tests/vectors/external_evidence_v0/_generate.py            # regenerate (imports Vaara to mint)
```
