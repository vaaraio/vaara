# Record-conformance vectors, v0

Fixtures for the producer-agnostic conformance check on SEP-2828
execution records. Conformance asks one thing: is this JSON a
well-formed execution record, and is it internally consistent? It needs
no signing key and no matching attestation, so it is the check a neutral
party runs on a record another party produced.

The check covers the wire schema (required fields, types, supported
`alg`, valid `status`, `sha256:<hex>` digest formats) and the one binding
a record proves about *itself*: `resultCommitment.projectionDigest` must
be the SHA-256 of the projection bytes beside it. That digest recomputes
from the record alone, which is what makes a record checkable without
trusting the producer or any key.

The cryptographic checks (signature, back-link to a held attestation,
result commitment against a held runtime result) are out of scope here:
they need external material and live in the `execution_receipt_v0`
vectors and the `vaara receipt verify` command.

## Layout

```
records/<case>.json    a candidate record (sorted keys, two-space indent)
expected.json          {case: {conforms, requiredFailed[], advisories[]}}
_check_independent.py   stdlib only (hashlib + re + json), no Vaara import
```

`requiredFailed` and `advisories` are sorted lists of check ids, so a
case is order-independent.

## Cases

Conforming:

- `conforming_executed_projection`: executed call with a self-consistent
  result projection commitment.
- `conforming_refused_no_commitment`: refused call, no result commitment.

Non-conforming (each isolates one required failure):

- `neg_missing_signature`: no `signature` (`signature_hex`).
- `neg_unsupported_alg`: `alg` outside HS256/ES256/RS256 (`alg_supported`).
- `neg_digest_mismatch`: `projectionDigest` does not match the projection
  bytes (`result_commitment_self_consistent`) — the self-proving check.
- `neg_bad_status`: `status` outside executed/refused/errored
  (`status_valid`).
- `neg_malformed_backlink_digest`: `attestationDigest` is not
  `sha256:<64 hex>` (`back_link_digest_format`).
- `neg_alg_mismatch`: `receiptAsserted.alg` differs from the top-level
  `alg` (`receipt_asserted_alg_matches`).
- `neg_not_an_object`: the record is a JSON array (`top_level_object`).

Advisory (conforms, but a SHOULD is flagged):

- `advisory_refused_with_commitment`: a refused call that carries a
  (self-consistent) result commitment. Conforms, with the advisory
  `refused_has_no_result`.

## Verifying

```
python tests/vectors/record_conformance_v0/_check_independent.py
```

Exit 0 means every case matched its expected verdict under the
independent, Vaara-free implementation.
