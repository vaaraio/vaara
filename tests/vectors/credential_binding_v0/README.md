# credential_binding_v0

Conformance vectors for the Vaara credential gateway enforcement contract
(Track 1, MCP proxy — SEP-2828).

## What this pins

Five cases cover the boundary between the proxy's allow decision and the
upstream forward.  A credential is spendable enforcement material for a single
tool call.  The signed records are audit evidence.  A verifier must be able to
reject a bad credential before upstream execution while still emitting a
concrete outcome record for the denial.

| case | expected verdict |
|------|-----------------|
| `pos_valid_grant` | `ok` — credential matches runtime tool, args, and tenant |
| `neg_args_changed` | `scope_mismatch` — runtime args differ from the committed args |
| `neg_expired` | `expired` — credential is past its expiry window at evaluation time |
| `neg_wrong_tenant` | `scope_mismatch` — tenantId in scope does not match the runtime call |
| `neg_no_credential` | `missing_credential` — `vaara/credential` absent from `_meta` (fail-closed) |

## Fixture format

Each `cases/*.json` file contains:

```
credential               BrokeredCredential.to_dict(), or null
expected_verdict         string verdict the gateway must produce
known_attestation_digests  sha256 digests the gateway accepts as valid bindings
now                      pinned Unix timestamp (seconds) used during verification
runtime_args             tool arguments presented at runtime
runtime_tenant_id        tenant seen by the gateway
runtime_tool_name        tool name seen by the gateway
```

## Recomputation

Any verifier that speaks HS256 over RFC 8785 JCS can reproduce every verdict
from `_check_independent.py` with no Vaara import.  The signing key is
`b"x" * 32` (corpus only; not a deployed key).

The args commitment is a two-step sha256:

```
args_digest  = "sha256:" + sha256(jcs(runtime_args)).hexdigest()
args_commitment = "sha256:" + sha256(jcs({"digest": args_digest})).hexdigest()
```

The signing payload is `jcs({version, alg, scope, binding, asserted})`.

## Regeneration

```
python3 tests/vectors/credential_binding_v0/_generate.py
python3 tests/vectors/credential_binding_v0/_check_independent.py
```
