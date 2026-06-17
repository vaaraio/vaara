# Standards and attestation

Vaara's standards-aligned outputs: OVERT 1.0 envelopes, the SEP-2787 execution-receipt pair, and the sovereign inference harness. For the verifier trust model behind each, see [verifying-evidence.md](verifying-evidence.md). For the SEP-2787 request-attestation format and conformance surface, see [execution-receipts.md](execution-receipts.md) and [sep2787-conformance.md](sep2787-conformance.md).

## OVERT 1.0 attestation

OVERT 1.0 is an open standard for runtime trust in AI systems ([overt.is](https://overt.is/), authored by Glacis Technologies, published 25 March 2026): a signed, schema-closed envelope a relying party can verify offline without trusting the emitter. Vaara is the **Arbiter** in OVERT terms and ships Protocol Profile 1.0 Base Envelopes (canonical CBOR per RFC 8949, Ed25519 signatures, HMAC-SHA256 commitments, closed 9-field schema) alongside every audit record when attestation is enabled.

```bash
pip install 'vaara[attestation]'
```

```python
from vaara.attestation.overt import emit_base_envelope, make_request_commitment, encoder_binary_identity

envelope = emit_base_envelope(
    signing_key=key,
    request_commitment=make_request_commitment(payload, operator_key=op_key),
    encoder_binary_identity=encoder_binary_identity(arbiter_version=f"vaara/{vaara.__version__}", policy_hash=ph),
    non_content_metadata={"action_class": "tx.transfer", "decision": "escalate"},
    monotonic_counter=42,
    arbiter_instance_identifier=uuid_bytes,
)
```

`vaara overt verify RECEIPT.cbor --pubkey-file PUB.bin` validates any canonical-CBOR Base Envelope. The verifier reads only the wire format and takes no dependency on Vaara's emitter, so any conformant implementation can route through it. Adjacent surfaces (`vaara.attestation.iap` notary + transparency log, `vaara.attestation.s3p` aggregate intervals, an experimental AMD SEV-SNP TEE hook) and the OVERT 1.0 Part 3 control walk are in [COMPLIANCE.md](COMPLIANCE.md). The OVERT control mapping is in [OVERT_CONTROLS.md](OVERT_CONTROLS.md).

## Sovereign inference harness

The governance proxy binds a `tools/call`. The sovereign inference harness, published in v1.0, binds the model call underneath it: which model answered, on which silicon-resident weights, given what input, and what it returned. It runs a local model behind a signing proxy (OpenAI- and ollama-compatible) and emits a hardware-rooted inference receipt that a second, different local model independently cross-checks. The point is signed evidence that the inference itself is accounted for, not only the tooling around it.

Two envelopes mirror the SEP-2787 attestation and receipt pair and reuse its canonicalization (RFC 8785 JCS) and signing stack (HS256 / ES256 / RS256), so a verifier that already reads Vaara records needs no new crypto:

- `InferenceAttestation` is the pre-call commitment: declared intent, a request commitment, an issuer block with a TTL, and the model facts the proxy derived at call time.
- `InferenceReceipt` is the post-call outcome, back-linked to the exact attestation, carrying status, an output commitment, eval-stat counters, and an honest `tier` self-label.

Tier A (`integrity`) binds model, input, and output with no determinism claim and ships standalone. Tier B (`replay`), the byte-reproducibility claim, is deferred and labeled as such instead of overclaimed.

```python
from vaara.attestation.inference import emit_inference_attestation, emit_inference_receipt
```

The session, chain, cross-check, and determinism verifiers each ship a Vaara-free checker that reproduces its verdict offline, and the governance console renders a live inference chain an outside party can replay. Install with `pip install 'vaara[attestation]'`. Developed privately, published here under AGPL-3.0-or-later.
