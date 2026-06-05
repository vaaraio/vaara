# cross_stack_revocation_v0 conformance vectors

One revocation rule, consulted by every verification lens. See
`docs/design/cross-stack-revocation-spec.md`.

Revocation used to live in exactly one place: the level-3 live identity
check. The same receipt, checked through the receipt verifier, the
transparency log, or an Article-12 export, ignored revocation entirely. A
`RevocationRegistry` lifts the rule out so every lens consults the same
predicate, and these vectors prove the lenses agree.

A receipt issued at `iat` by `iss` (optionally bound to a keyid) is
**revoked-in-time** iff a matching registry entry's `revoked_at` is at or
before `iat`. A key revoked afterwards still binds, because revocation is not
retroactive. An unparseable instant fails closed.

## Files

- `cases.json`: the shared receipt (ES256, `did:web` issuer), and for each
  case the registry, the bound keyid, and one transparency-log inclusion
  proof (the receipt is logged at a non-trivial index so the proof carries
  real siblings).
- `expected.json`: per case, the verdict each lens must produce.

## The three lenses

Each case asserts the same revoked verdict across:

- `receipt_lens_revoked`: the receipt-verifier lens
  (`check_receipt_revocation`), the offline revocation rule.
- `log_lens_included` / `log_lens_ok`: the transparency-log lens
  (`verify_logged_receipt`). The receipt is always included; `ok` is true
  only when it is included *and* not revoked-in-time.
- `registry_digest`: the export lens. A signed Article-12 export pins this
  SHA-256 over the RFC 8785 canonical registry bytes, so a regulator
  recomputes every receipt's verdict against the exact registry the exporter
  used.

## Cases

- `key_revoked_in_time`: a key-scope entry revoking the signing key before
  the receipt's `iat`. Revoked across every lens.
- `key_revoked_after`: the same key revoked after `iat`. Not revoked: the
  receipt still binds.
- `identity_revoked_in_time`: an identity-scope entry revoking the whole
  issuer before `iat`, the source a DID document cannot express (an
  operator's out-of-band revocation list). Revoked across every lens, with
  no keyid supplied.
- `clean`: an empty registry. Not revoked.

## Reproduce

Independent checker (standard library plus `rfc8785`, no Vaara import):

```
python tests/vectors/cross_stack_revocation_v0/_check_independent.py
```

Exit code 0 means every case matched its expected verdict. Regenerate the
cases (ECDSA signatures are randomized, so signatures change but verdicts do
not) with:

```
python tests/vectors/cross_stack_revocation_v0/_generate.py
```
