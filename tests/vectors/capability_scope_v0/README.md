# capability_scope_v0

Conformance vectors for the Vaara capability-mode enforcement contract
(Track 1, Phase C — SEP-2828).

## What this pins

Five cases cover the boundary between typed capability constraints and the
gateway allow/deny decision.  A capability-mode grant authorizes a bounded
class of calls rather than a single exact argument set.  The gateway enforces
closed coverage: every runtime argument must be named by a capability, and
every named capability must hold.

| case | expected verdict |
|------|-----------------|
| `pos_valid_grant` | `ok` — amount le 500, vendor in {acme,globex}, all constraints pass |
| `neg_amount_exceeded` | `capability_exceeded` — amount 600 exceeds the le 500 bound |
| `neg_vendor_not_in_set` | `capability_exceeded` — vendor "evilcorp" not in the allowed set |
| `neg_uncovered_arg` | `capability_uncovered` — runtime arg "memo" is not named by any capability |
| `neg_missing_credential` | `missing_credential` — no credential presented (fail-closed) |

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

The signing payload is `jcs({version, alg, scope, binding, asserted, capabilities})`.
The `capabilities` array uses closed schema per entry: `{arg, op, value}`.

Numeric ops (`le`, `ge`) compare via Decimal to avoid float rounding; bools
are rejected as numeric.  Coverage is closed: an unnamed runtime arg fails
`capability_uncovered` before any constraint is evaluated.

## Regeneration

```
python3 tests/vectors/capability_scope_v0/_generate.py
python3 tests/vectors/capability_scope_v0/_check_independent.py
```
