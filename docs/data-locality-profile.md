# Vaara Data-Locality Profile v0

A downstream profile of `vaara.receipt/v1`. It maps a signed data-locality
record to the EU transfer-accountability frame, and it states plainly what the
record proves and what it does not. The profile adds no new primitive: the
record is built from the shipped receipt and decision-record primitives plus a
policy recompute.

The conformance corpus for this profile is `tests/vectors/data_locality_v0/`,
graded by the aggregate runner alongside the rest of Profile v1.

## The problem it addresses

When an agent action sends personal data to a model endpoint outside the EU, the
GDPR Chapter V question is: where did the data go, under what rule, and can that
be checked without trusting the party that sent it. After *Schrems II* the
practical burden is a Transfer Impact Assessment plus supplementary measures,
documented per transfer under the accountability principle (Art. 5(2)). This
profile produces evidence that feeds that assessment. It does not, and cannot,
establish that a transfer is lawful.

## The record

`vaara.data-locality/v0` binds one action's transfer facts (data class,
endpoint, endpoint region, TLS cert fingerprint, payload digest) to the enforced
allow/block decision, and optionally carries a region attestation signed by a
party distinct from the issuer. Canonicalization is RFC 8785 JCS; signing is via
the `vaara.audit.signer` `Signer` protocol (Ed25519 default, ML-DSA-65
optional), the same stack the receipt uses.

## Two tiers, named in the verdict

The corpus checker recomputes each verdict from bytes, and the verdict names how
far the evidence reaches:

- **Tier A — proof, no trusted party.** The issuer signature verifies, the
  payload digest recomputes from the payload bytes, and the recorded allow/block
  decision recomputes from the transfer facts under the named policy. An outside
  party reproduces all of it from bytes alone (`ok_asserted`).
- **Tier B — carried claim.** A region attestation signed by an attester key
  distinct from the issuer is verified against that key and checked to agree with
  the claimed region (`ok_attested`). Its absence is stated, not hidden:
  location asserted by the client, not attested.

## Crosswalk to GDPR Chapter V

| Requirement | What the record supplies | Tier |
|-------------|--------------------------|------|
| Art. 5(2) accountability — demonstrate the transfer control operated | Signed, recomputable record of the enforced decision per action | A |
| Art. 44/46 — transfers only under appropriate safeguards | Policy recompute shows the enforced allow/block for the data class and region | A |
| TIA supplementary-measure evidence (post-*Schrems II*) | Payload-class binding + endpoint region + optional processing-region attestation | A + B |
| Independent verifiability without trusting the exporter | The dependency-free corpus checker reproduces every verdict | A |

## What the profile does NOT establish

- **Not an adequacy finding.** A genuine attestation of an EU processing region
  does not prove legal safety from onward access; an EU endpoint operated by a
  non-EU-controlled provider remains exposed under foreign-access law. Tier B
  attests the observed region, nothing more.
- **Not lawfulness of the transfer.** Whether a transfer is permitted is a legal
  determination for the controller and its DPA. The record is evidence for that
  assessment, not a substitute for it.
- **Not proof of physical location by itself.** Without a Tier-B attestation the
  region is the client's assertion. The verdict says so (`ok_asserted`).

## Why the honesty is the point

The 2026 pressure on the EU-U.S. Data Privacy Framework turns on whether a US
oversight authority is independent enough to be trusted. When trust in an
authority can be revoked, the only locality claim that survives is one the
relying party recomputes itself (Tier A) plus an attestation it verifies against
the attester's own key (Tier B). This profile is built so a record never claims
more than a recipient can check.

## Status

Profile v0, downstream of `vaara.receipt/v1`. The record format and corpus are
open. Acquiring genuine TEE or provider region attestations at runtime is an
operational concern outside this profile.
