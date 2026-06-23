# agent_decision_v0 conformance vector

A recomputable `{statement, expected-verdict}` pair for the proposed in-toto
`agent-decision/v0.1` predicate, offered on
[in-toto/attestation#554](https://github.com/in-toto/attestation/issues/554).
Conformance is anchored to the bytes and a public key, not to any one
producer's crate or endpoint.

## What it is

- `statement.json`, an in-toto `Statement/v1` carrying an `agent-decision/v0.1`
  predicate: the agent id, the principal, the policy evaluations, the per-call
  argument commitments with their explicit-omission state (`args_state`), and
  the allow/deny `decision`.
- `envelope.json`, that statement sealed in a **DSSE** envelope (the in-toto
  native signing envelope) under **Ed25519**. The `payload` is the JCS
  (RFC 8785) canonical bytes of the statement, base64-encoded.
- `keys/ed25519_public.pem`, the published verifying key. The signing seed is
  fixed and test-only; Ed25519 is deterministic, so a regenerate is
  byte-identical.
- `expected.json`, the verdict a conformant reader must reach.

## What passing means

Run the checker, which imports no Vaara code (standard library, `cryptography`
for Ed25519, `rfc8785` for JCS):

    python tests/vectors/agent_decision_v0/_check_independent.py

It recomputes, from the bytes alone:

1. `payload_is_jcs_canonical`, the envelope payload re-canonicalizes to itself,
   so any reader recomputes the same signed bytes;
2. `paeSha256`, the DSSE pre-authentication encoding
   (`DSSEv1 SP len(type) SP type SP len(body) SP body`) digests to the
   committed value;
3. `signatureVerifies`, the Ed25519 signature verifies over that PAE under the
   published public key;
4. `normalized`, the SEP-2828 mapping (which evidence plane, which fields
   populated, what is still `missing`) reproduced from the shipped declarative
   profile `src/vaara/attestation/profiles/agent_decision.json` with the
   checker's own code.

Step 4 is why the predicate is not self-confirming: the mapping comes from the
JSON spec the product ships, reproduced by a second implementation, not from a
Vaara function.

## What the mapping says

The predicate is the attested-decision face of an execution event, the closest
foreign analog to a signed decision record. It still fills no SEP-2828 field on
its own:

- the in-toto Statement's signature lives in the DSSE envelope and is a
  separate signing event, so `alg`, `signature`, and `receiptAsserted` are
  absent from the predicate;
- `args_hash` commits the call arguments but is not a `backLink` to an attested
  request (the `attestationDigest` a conformant receipt must answer); the
  commitment is portable only when `args_canonicalization` names a shared
  projection, here JCS;
- `args_state` keeps the explicit-omission state first-class (`present`,
  `args_redacted`, `args_unavailable`, `args_not_recorded`), so a redacted call
  and an unrecorded call stay distinct claims rather than one missing field;
- the predicate records the decision, not `outcomeDerived` (whether the call
  then executed and with what result).

## Regenerating

    python tests/vectors/agent_decision_v0/_generate.py

and commit the result.
