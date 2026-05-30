# SEP-2787 conformance surface

This document states what Vaara's SEP-2787 verifier checks, what it leaves to
the runtime, and which spec revision it tracks. It is the reference for the
`vaara attest verify` and `vaara receipt verify` commands shipped in v0.44.0.

## Spec revision tracked

Vaara aligns to the SEP-2787 draft at commit `dd030d5b` on
`soup-oss/sep-tool-call-attestation`, the revision whose envelope adopted the
three trust-surface blocks (`plannerDeclared`, `issuerAsserted`,
`payloadDerived`) and the four schema points Vaara raised in
`modelcontextprotocol/modelcontextprotocol#2787`
(`issuecomment-4557017068`): tool calls under `payloadDerived`,
`argsProjection` as a JCS-stringified projection, no `kind` discriminator, and
commitment-only audit via a hash-only-identity projection.

The reference implementation is pinned at tag `sep2787-ref-v2`
(`src/vaara/attestation/sep2787.py` and helpers). The draft is community
authored and at the time of writing carries `Sponsor: None`; this is a
first-implementer position, not a maintainer endorsement.

## Signing modes

The signature covers the JCS-canonical encoding (RFC 8785) of the four
envelope blocks `{version, alg, plannerDeclared, issuerAsserted,
payloadDerived}` and is excluded from its own input.

| `alg` | Algorithm | Verify with |
|---|---|---|
| `ES256` | ECDSA P-256, raw `r\|\|s` (not DER) | `--pubkey-file PUB.pem` |
| `RS256` | RSASSA-PKCS1-v1_5 | `--pubkey-file PUB.pem` |
| `HS256` | HMAC-SHA256 | `--hs256-secret-file SECRET` |

ES256 is the recommended default: the public key is publishable, so a relying
party verifies without holding any secret. HS256 envelopes are only verifiable
by a party that already shares the secret.

## Verification steps

The SEP-2787 draft lists five verification steps. Vaara splits them between the
verifier (stateless, no IO) and the runtime (stateful), because nonce replay
and tool-call matching cannot be decided from a saved envelope alone.

| Step | Check | Where | Covered by |
|---|---|---|---|
| 1 | Signature over the canonical envelope | verifier | `vaara attest verify` |
| 2 | Nonce replay (each nonce used once) | runtime | caller's replay cache |
| 3 | TTL (`iat + expSeconds` not past) | verifier | `vaara attest verify` (reported; enforced with `--enforce-ttl`) |
| 4 | Tool-call match (envelope binds the call being executed) | runtime | proxy / host |
| 5 | Argument commitment binds the runtime arguments | verifier | `verify_args_commitment` (composed by the caller, or `vaara receipt verify --result` for result commitments) |

Steps 2 and 4 are stateful and depend on the live request, so they stay in the
runtime. Vaara's MCP proxy performs them inline when it emits a pair; a
standalone verifier reading saved files cannot, and does not pretend to.

### TTL is not enforced at audit time by default

A SEP-2787 attestation carries a short TTL (`expSeconds`, default 300) because
it authorizes a request that is about to run. A *saved* attestation is durable
evidence read long after that window, so `vaara attest verify` reports
`ttl_expired` but does not fail on it. Pass `--enforce-ttl` when you are
verifying an envelope you expect to still be live.

## Execution receipts

An execution receipt is the post-execution complement: it records the outcome
of one attested request and pins the attestation it answers. `vaara receipt
verify` checks three things:

1. the receipt signature (same canonicalization and signing stack);
2. the attestation signature (with TTL ignored, since the attestation is
   expected to be expired by the time the outcome is audited);
3. the `backLink`, which must carry both a SHA-256 digest over the
   attestation's full wire bytes and the attestation's nonce. A receipt with a
   valid signature but a broken back-link proves an outcome that belongs to no
   attested request, and is rejected.

When `outcomeDerived` carries a `resultCommitment`, `--result RESULT.json`
verifies it against the runtime result object using the same
argument-commitment rules (step 5). The proxy does not emit a result
commitment today, so live-proxy receipts report `result_commitment_valid:
null`.

## Conformance vectors

- Execution-receipt vectors live in-repo at
  `tests/vectors/execution_receipt_v0/` (five cases, pinned keys, a
  stdlib-only `_check_independent.py` walker that verifies them without
  importing Vaara).
- SEP-2787 attestation vectors live in-repo at
  `tests/vectors/sep2787_attestation_v0/` (six cases across HS256/ES256/RS256,
  pinned keys, a stdlib-only `_check_independent.py` walker that verifies
  signature, TTL, and the step-5 argument commitment without importing Vaara).
  They mirror the proposed-shape vectors on the fork PR
  `modelcontextprotocol/modelcontextprotocol#2789` so the `vaara attest verify`
  command can be exercised against pinned fixtures the same way the receipt
  vectors are.

## Quick start

```
pip install 'vaara[attestation]'
vaara keygen --attest --out attest_key.pem
# point the proxy at the key, run some tool calls, then:
vaara attest verify  RECEIPTS/0000000001-<nonce>-attest.json  --pubkey-file attest_key.pem.pub
vaara receipt verify RECEIPTS/0000000001-<nonce>-receipt.json --attestation RECEIPTS/0000000001-<nonce>-attest.json --pubkey-file attest_key.pem.pub
```

Both commands print a JSON verdict and exit non-zero on any failed check, so
they compose into CI or an audit script.
