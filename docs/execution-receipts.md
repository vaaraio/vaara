# Execution receipts

The execution receipt is the post-execution sibling of SEP-2787 tool
call attestation.

SEP-2787 attests a `tools/call` **request** before it runs: issuer,
subject, target, intent, nonce, time, and an argument commitment. It
says nothing, by design, about whether the call executed or what came
back. The receipt covers that deferred half. It binds the **outcome** of
one attested request and links back to the attestation it answers.

Together the pair gives end-to-end accountability for one action: the
attestation proves the request was authorised, the receipt proves what
happened to it.

## Shape

A receipt carries three blocks plus a signature, mirroring the SEP-2787
trust-surface layout so both envelopes verify with the same
canonicalization (RFC 8785 JCS) and signing stack (HS256 / ES256 /
RS256):

- `backLink` joins the receipt to its request attestation. It carries
  the attestation's `nonce` for fast correlation and a digest over the
  full SEP-2787 wire envelope (signature included), which pins the exact
  attestation instance.
- `receiptAsserted` is the issuer block, set by whoever observed the
  outcome: the executing server, or an intermediary such as a governance
  proxy.
- `outcomeDerived` carries the status (`executed` / `refused` /
  `errored`), the completion time, and an optional commitment over the
  result. The result commitment reuses the SEP-2787 argument-commitment
  shapes (`ArgsRef` / `ArgsProjection`), because structurally a result
  commitment is the same thing: a commitment over a JSON value.

A receipt is a durable record, not a time-bounded capability, so it
carries no `exp` and the verifier enforces no TTL. The attestation can
expire; the record of what happened does not.

## Emitting

```python
from vaara.attestation.receipt import (
    OutcomeDerived, emit_receipt, make_back_link, make_result_digest,
)

receipt = emit_receipt(
    back_link=make_back_link(attestation),   # the SEP-2787 envelope
    outcome_derived=OutcomeDerived(
        status="executed",
        completed_at="2026-05-29T10:00:00Z",
        result_commitment=make_result_digest(result_obj),  # payload stays local
    ),
    iss="issuer://proxy",
    sub="agent:archiver",
    secret_version="v1",
    alg="HS256",
    signing_material=shared_secret,
)
```

`make_result_digest` keeps the result payload local and commits only to
its hash. `make_result_projection` ships a reviewed projection of the
result with its own digest. Both reuse the SEP-2787 commitment builders.

## Verifying

Three composable checks, none of which require trusting the emitter:

```python
from vaara.attestation.receipt import (
    verify_receipt_signature, verify_back_link,
)
from vaara.attestation.sep2787 import verify_args_commitment

assert verify_receipt_signature(receipt, verifying_material=secret)
assert verify_back_link(receipt, attestation=attestation).ok

commitment = receipt.outcome_derived.result_commitment
if commitment is not None:
    assert verify_args_commitment(commitment, runtime_arguments=result_obj).ok
```

A verifier that already checks SEP-2787 signatures needs no new crypto
to check receipts. The back-link check recomputes the attestation digest
and confirms both the digest and the nonce match. The result-commitment
check recomputes the commitment against the result the server is about
to return, or returned.

## Conformance vectors

Pinned fixtures and a stdlib-only walker live in
[`tests/vectors/execution_receipt_v0/`](../tests/vectors/execution_receipt_v0/).
The walker reproduces canonical bytes, signature verification across all
three algorithms, back-link verification, and result-commitment
verification without importing Vaara, so a second implementation can
consume the format directly.

```
python tests/vectors/execution_receipt_v0/_check_independent.py
```

## Relationship to OVERT 1.0

OVERT Base Envelopes (`vaara.attestation.overt`) and execution receipts
coexist. OVERT is the operator-side per-action attestation kernel
emitting CBOR envelopes for every governed interaction. The execution
receipt is the per-tool-call JSON outcome record that pairs with a
SEP-2787 request attestation and verifies against the SEP-2787 stack.
The outcome half maps to the OVERT 1.0 Part 3 receipt family; see
[OVERT_CONTROLS.md](../OVERT_CONTROLS.md).
