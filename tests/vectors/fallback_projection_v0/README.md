# fallback_projection_v0

The SEP-2828 fallback projection, version
`tools_call_params_plus_meta_authorization_binding_v1`.

When no SEP-2787 attestation exists for a call, a decision record and its
execution receipt still have to bind to the call that caused them. Hashing the
whole observed `tools/call` envelope does not work: a gateway and a provider
see the same call with different `_meta` sidecars (progress tokens, trace
context, injected ids) and would compute different digests.

The projection keeps only what binds the call:

```json
{
  "projection": "tools_call_params_plus_meta_authorization_binding_v1",
  "name": "<params.name>",
  "arguments": <params.arguments>,
  "authorizationBinding": <params._meta.authorization_binding>
}
```

`backLink.attestationDigest` is `sha256:` over the RFC 8785 (JCS) encoding of
that object, and `backLink.fallbackProjection` names the version, inside the
signed record, so a verifier rebuilds the same projection instead of guessing
it. `authorization_binding` is required and must be an object with a non-empty
string `nonce` (the server's per-call value); `name` and `arguments` are
required. Anything else under `_meta` never enters the preimage. When the
projection cannot be built, or the version is one the verifier does not know,
the binding fails closed.

## Cases

`envelopes/` holds raw request envelopes, sidecars included.

| Case | What it shows |
|---|---|
| `provider_view`, `gateway_view` | One call, different sidecars, one digest. |
| `binding_with_policy` | Every member of the binding block is bound, not only the nonce. |
| `non_ascii_arguments` | Non-ASCII arguments go through JCS as raw UTF-8. |
| `different_tool`, `different_arguments`, `replayed_binding` | Changing any bound field changes the digest. |
| `no_binding`, `binding_without_nonce`, `binding_not_object`, `missing_arguments` | No projection exists; refused. |
| `unsupported_version` | A version the verifier does not implement; refused. |

The signed receipts bound under this projection are in
`../decision_pairing_v0/normative/fallback_envelope_binding/`.

Earlier contents of this directory hashed `{arguments, authBinding, toolName}`,
a shape no signed record used. They were replaced by these cases.

    python3 tests/vectors/fallback_projection_v0/_check_independent.py
