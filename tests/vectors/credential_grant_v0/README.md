# Credential-grant conformance vectors, v0

Fixtures for the receipt-bound credential broker: the authority half of
Vaara's "separate intelligence from authority" move. A grant is a standalone
signed envelope the proxy mints between the allow decision and the upstream
forward, scoped to one tool + args commitment + tenant, bound to a specific
attestation digest, and short-lived. A gateway in front of a protected tool
refuses any call lacking a valid, unexpired, scope-matching, attestation-bound
grant.

Each set under `sets/<name>/` holds the bytes a neutral verifier needs:

- `grant.json`: the brokered-credential wire envelope.
- `attestation.json`: the SEP-2787 attestation the grant is bound to (its
  digest is the `binding.attestationDigest`).
- `inputs.json`: the runtime call a gateway sees: `toolName`, `args`,
  `tenantId`, the `now` epoch, and the HS256 `keyHex`.

`expected.json` maps each set to its verdict `{ok, reason}`. The verdict
precedence is `bad_signature` -> `expired` -> `scope_mismatch` ->
`binding_unknown`, so an expired-but-authentic grant is never mislabeled.

## Reproduce it without Vaara

`_check_independent.py` re-derives every verdict from the committed bytes with
no `import vaara`: the HS256 signature over the JCS-canonical preimage
`{version, alg, scope, binding, asserted}`, the bound attestation digest
(sha256 over the JCS of `attestation.json`), and the args commitment (the
two-step hash-only projection). The one shared primitive is RFC 8785 (JCS);
the rest is the standard library.

```
python conformance/sep2828/credential_grant_v0/_check_independent.py
```

Exit 0 means a second implementation reproduced the format. All vectors are
HS256; ES256 / RS256 verify the identical preimage with a public key. Regenerate
with `python scripts/build_credential_grant_vectors.py`.

## Honest scope

This is detection of a defeated broker, not a mathematical-completeness claim.
Completeness holds only for tools placed behind a gateway that runs the check;
a tool reachable by another path (operator root, an alternate credential
source, a purely-local action) is caught after the fact by reconciliation, not
prevented here. See `docs/design/credential-broker-spec.md`.
