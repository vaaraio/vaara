# SEP-2828 fallback_projection_v0 conformance vectors

JCS digest vectors for the SEP-2828 fallback binding path — used when no
SEP-2787 attestation exists for the call.

## What is the fallback projection?

When a deployment does not run SEP-2787, the decision and outcome records
must still bind to the originating call. The naive approach — hashing the
full observed `tools/call` envelope including `_meta` — produces
observer-local digests: a gateway and a provider both see the same call
but carry different `_meta` values (progress tokens, trace IDs, injected
correlation headers), so they hash to different values.

The fallback projection fixes this by hashing only the portable subset:

```json
{
  "arguments":   <JCS-normalized params.arguments>,
  "authBinding": <params._meta["authorization_binding"] if present, else absent>,
  "toolName":    "<params.name>"
}
```

`attestationDigest` = `sha256:<hex>` over the UTF-8 JCS encoding of this
object (RFC 8785: keys sorted, no whitespace). The server-chosen
`attestationNonce` in `backLink` still provides instance-binding; the
projection hash provides content-binding.

`authBinding` carries only the authorization-relevant subobject from
`_meta` — scope, policy reference, capability grant. Progress tokens and
transport correlation artifacts are excluded.

## Vectors

| Name | Property tested |
|------|----------------|
| `basic_no_auth_binding` | Projection without authBinding (no authorization_binding in _meta) |
| `with_auth_binding` | Projection with authBinding present |
| `observer_stable_a` | Observer A: gateway sees progressToken + traceId in _meta |
| `observer_stable_b` | Observer B: provider sees different progressToken + x-injected-id |
| `neg_different_tool` | Different toolName → different attestationDigest |

`observer_stable_a` and `observer_stable_b` have **identical projections**
and therefore identical `attestationDigest` values, proving the portability
property: honest observers with different `_meta` sidecars agree on the
digest.

## Running the checker

```
python3 conformance/sep2828/fallback_projection_v0/_check_independent.py
```

Standard library only (`hashlib`, `json`). Exit 0 means all 7 checks pass,
including the cross-vector observer-stability and negative-divergence
assertions.
