# tap_v0 conformance vectors

Binding profile: a Visa Trusted Agent Protocol (TAP) request becomes the
evidence a `vaara.receipt/v1` decision receipt names, across one action
lifecycle. The pre-action receipt and the terminal one bind to the same logical
TAP request but carry distinct `actionRef` join keys, so a mid-action receipt
cannot be presented where the final one is required. Verifiable by any third
party with only the TAP request, the held receipts, and the issuer's public key.
No TAP service access, no live endpoint, no Vaara import.

This is the artifact behind "TAP can pin to vaara.receipt/v1 for the
post-authorization record, rather than define a new primitive." See the parent
`SPEC.md` Section 5.5 and `tests/vectors/x402_settlement_v0/` for the lifecycle
half this reuses.

## The mapping

The trusted agent presents a TAP request to the relying party. That request is
the evidence record:

- It is content-addressed as `sha256(JCS(request))` (JCS / RFC 8785, the same
  canonicalization `vaara.receipt/v1` uses, so the address joins with no
  re-canonicalization).
- Each decision receipt names it by content address:
  `decisionDerived.evidenceRef.digest` = `sha256(JCS(request))`,
  `decisionDerived.evidenceRef.ref` = `tap:request/<actionRef>`, both carried
  under the receipt signature.

The lifecycle lives in the join key `actionRef` =
`sha256(JCS({agentId, actionType, scope, timestampMs, seq, terminal}))`. Because
the tuple covers `terminal`, the in-progress (`terminal: false`) request has a
different `actionRef` than the final (`terminal: true`) one, and the in-progress
receipt does not resolve against the terminal request.

The TAP request here is representative. The binding rests on the
content-addressing discipline (JCS over the request, the action tuple as the
join key), not on the request's exact field names.

## The wedge

The verdict is recomputed offline from the committed bytes and the public key.
`_check_independent.py` imports the standard library plus `cryptography` and
`rfc8785`, and nothing else: no TAP service, no live verifier endpoint, no Vaara.
An identifier-only or live-endpoint attestation says a record exists somewhere a
service vouches for; this lets the holder recompute the whole verdict with the
service offline.

## Cases

- `step0/` — in-progress (`terminal: false`): the TAP request and the receipt
  that names it.
- `step1/` — terminal (`terminal: true`): the same logical request finalized, a
  distinct `actionRef`, and the receipt that names it.

## Verdicts

Per step: `action_ref_recomputes`, `request_binding_resolves`,
`receipt_signature_ok`. Cross-step: `lifecycle_distinguishes_terminal`. The
expected matrix is `expected.json`.

## Run

```
python tests/vectors/tap_v0/_check_independent.py   # recompute, no Vaara; exit 0 = pass
python tests/vectors/tap_v0/_generate.py            # regenerate (imports Vaara to mint)
```
