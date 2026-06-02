# agent_identity_v0 conformance vectors

Resolvable agent identity (did:web) for execution receipts, level-2
pinned-resolvable verification. See
`docs/design/resolvable-agent-identity-spec.md`.

A receipt names its issuer in `receiptAsserted.iss`. When that value is a
`did:web` identifier, a verifier can resolve it to a key set and confirm the
receipt signature was made by a key that identity controls, instead of
trusting an opaque issuer string. These vectors capture the DID document
alongside each receipt, so the check is reproducible offline with no network
call.

## Cases

- `bound.json`: an ES256 receipt whose `iss` is `did:web:agents.example.com:billing`,
  plus a DID document that lists the signing key. Expected: `resolved` and `bound`.
- `unbound.json`: the same receipt against a DID document that lists a
  different key. Expected: `resolved`, not `bound` (the signature matches no
  key the document publishes).
- `expected.json`: the verdict each case must produce
  (`resolved`, `bound`, `keyid`).

The existing `decision_pairing_v0` vectors are unaffected: this family is
additive, the receipt envelope is unchanged, and an opaque-string `iss` is
never failed for lack of a DID.

## Reproduce

Independent checker (standard library plus `cryptography` and `rfc8785`, no
Vaara import):

```
python tests/vectors/agent_identity_v0/_check_independent.py
```

Exit code 0 means every case matched its expected verdict. Regenerate the
cases (ECDSA signatures are randomized, so signatures change but verdicts do
not) with:

```
python tests/vectors/agent_identity_v0/_generate.py
```
