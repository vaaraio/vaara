# data_locality_v0

Conformance vectors for a data-locality evidence record: a signed statement that
binds one agent action's cross-border transfer facts to the policy decision that
ran, and optionally carries a region attestation signed by a party other than
the issuer.

## Why this exists

When personal data leaves the EU to a model endpoint, the compliance question is
"where did it go, under what rule, and can I prove it without trusting the party
that sent it." A locality record answers the first two by recompute and the
third only as far as the evidence honestly allows. The suite is deliberately
two-tier, and the verdicts name which tier a record reaches. It does not, and
cannot, establish legal adequacy of a transfer; it establishes what was observed
and what policy was enforced.

## The two tiers

**Tier A — proof, no trusted party.** The record signature verifies, the payload
digest recomputes from the payload bytes, and the recorded allow/block decision
recomputes from the transfer facts under the named policy. An outside party
reproduces all of this from bytes alone.

**Tier B — carried claim.** A `regionAttestation`, signed by an *attester* key
distinct from the issuer, asserts the processing region. The checker verifies
that signature against the attester's public key and checks it agrees with the
claimed endpoint region. A valid attestation upgrades the verdict to
`ok_attested`; its absence yields `ok_asserted` — location asserted by the
client, not independently attested. That distinction is stated in the verdict,
never smoothed over.

A genuine attestation of an EU region still does not prove legal safety from
onward access (an EU endpoint operated by a non-EU provider remains exposed).
Tier B attests the observable region; it is not an adequacy finding.

## Cases

| case | verdict | what it pins |
|------|---------|--------------|
| `pos_attested_eu` | `ok_attested` | PII to an EU endpoint, allow, valid attester signature for the same region |
| `pos_asserted_eu_no_attestation` | `ok_asserted` | PII to an EU endpoint, allow, no attestation — location asserted only |
| `pos_nonpersonal_us` | `ok_asserted` | non-personal data to a US endpoint, allow (policy unconstrained), no attestation |
| `neg_policy_mismatch_pii_us` | `policy_mismatch` | record claims allow for PII to a US endpoint; the independent recompute says block |
| `neg_bad_signature` | `bad_signature` | a covered field mutated after signing |
| `neg_payload_tampered` | `payload_mismatch` | runtime payload differs from the committed digest |
| `neg_attestation_bad_sig` | `attestation_bad_sig` | attestation present, signature does not verify |
| `neg_attestation_region_mismatch` | `attestation_region_mismatch` | attester signs a different region than the endpoint claims |

`neg_policy_mismatch_pii_us` is the load-bearing case: the record carries a valid
issuer signature and a matching payload digest, so it clears every integrity
check, and is caught only because the checker recomputes the decision itself
rather than trusting the recorded one.

## Fixture format

Each `cases/*.json` contains:

```
record              the signed data-locality record (see schema below)
payload             the payload object presented at runtime (its digest is recomputed)
expected_verdict    the verdict the checker must produce
attested            whether a valid Tier-B attestation is expected
```

Record schema (`vaara.data-locality/v0`):

```
version, alg, schema, issuer
transfer            { actionId, dataClass, endpoint, endpointRegion,
                      payloadDigest, tlsCertSha256 }
decision            { decision, policyId }
regionAttestation   { attester, attestedRegion, nonce, sig }   (optional, Tier B)
signature           Ed25519 over JCS of the record minus `signature`
```

## Policy

`eu-inference-only@v1`: personal data may leave only to an EU region
(`eu-central-1`, `eu-north-1`, `eu-west-1`); non-personal data is unconstrained.
The checker recomputes this from `transfer.dataClass` and
`transfer.endpointRegion`.

## Recomputation

Any verifier that speaks Ed25519 over RFC 8785 JCS reproduces every verdict from
`_check_independent.py` with no Vaara import. Both keypairs are Ed25519, derived
by `sha256` over published corpus-only seed labels:

```
issuer seed label    b"vaara-data-locality-issuer/v0"
attester seed label  b"vaara-region-attester/v0"
```

The checker derives the keys and verifies against the **public** half only; a
real deployment ships public keys or trust anchors, never the seed. Public keys
are also recorded in `expected.json` under `keys` for convenience.

Attestation signing payload: `jcs({attestedRegion, attester, nonce})`.
Record signing payload: `jcs(record without "signature")`.

## Regeneration

```
.venv/bin/python tests/vectors/data_locality_v0/_generate.py
.venv/bin/python tests/vectors/data_locality_v0/_check_independent.py
```
