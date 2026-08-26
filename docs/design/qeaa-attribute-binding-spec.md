# Design spec: bind an agent action to a qualified attestation of attributes

Status: draft. Companion to `docs/design/cross-org-handoff-spec.md` (the
handoff this feeds), `docs/design/credential-broker-spec.md` (the grant
envelope this extends), and `docs/design/qualified-existence-spec.md` (the
qualified time anchor this reuses the trust model of). Builds on the trusted
list walker in `src/vaara/audit/eu_trusted_list.py`.

## The problem

Every actor claim in the current record is self-asserted.

`GrantAsserted.iss` and `.sub` are strings the issuer writes about itself. A
verifier can check the signature, recompute the args commitment, and confirm
the grant is bound to a specific attestation instance. What no verifier can
check is whether the issuer is the organisation it says it is, or whether the
agent was acting for that organisation with any authority at all.

Inside one organisation this does not matter, because the verifier is the
issuer. Across a boundary it is the whole question. The cross-org handoff spec
gets a regulator to the point of verifying that vendor A signed a record. It
cannot get that regulator to "A was entitled to act for B, and B says so."

The qualified time anchor already solved the same shape of problem for *when*.
An RFC 3161 token from a supervised provider on a national trusted list is not
the issuer's claim about the clock, it is a third party's, and the verifier
resolves that third party from a public register rather than from the issuer.
This spec does the same thing for *on whose authority*.

## What a QEAA is, and why it is the right instrument

A Qualified Electronic Attestation of Attributes is an eIDAS trust service:
`http://uri.etsi.org/TrstSvc/Svctype/EAA/Q`. A supervised, audited provider
attests an attribute of a subject, and the attestation carries a statutory
presumption. An organisation cannot issue one about itself, by construction.

Walking the EU List of Trusted Lists with `eu_trusted_list.py` on **2026-08-26**
finds **two** granted `EAA/Q` services across the 31 national lists it points
to:

| Territory | Provider | Service |
|---|---|---|
| HU | Microsec Micro Software Engineering & Consulting | Issuance of Qualified Electronic Attestations of Attributes |
| SE | IDnow Trust Services AB | IDnow Qualified electronic attestation of attributes |

Two caveats on that count, both of which matter more than the number.
The Portuguese list did not respond during the walk, so two is a floor rather
than a ceiling. And an earlier revision of this document said there was exactly
one, verified the same way on 2026-08-05: the field changed inside three weeks.
**Re-walk the lists before repeating any count, and never publish "the only
provider" from a cached figure.**

The consuming side of this ecosystem is still where the gap is. Vaara is a
relying party, not an issuer, and becoming a QTSP is neither possible nor
necessary here.

## Wire schema

`mandate` is an optional block on the brokered credential, alongside `scope`,
`binding` and `asserted`. It is covered by the existing grant signature, which
is computed over the JCS encoding of `{version, alg, scope, binding, asserted,
mandate}` when present.

```json
"mandate": {
  "format": "eaa-q-jwt",
  "attestationDigest": "sha256:<hex>",
  "attestation": "<base64 attestation as issued>",
  "issuer": {
    "serviceTypeIdentifier": "http://uri.etsi.org/TrstSvc/Svctype/EAA/Q",
    "territory": "HU",
    "trustListRef": "https://.../TL-HU.xml",
    "providerName": "<TSPName as it appears on the national list>"
  },
  "subject": {
    "legalPersonIdentifier": "<identifier as attested>",
    "attributeSet": ["organizationIdentifier", "organizationName"]
  },
  "agentBinding": {
    "agentId": "<the agent id in the attestation instance>",
    "boundVia": "attestationDigest"
  }
}
```

- `attestation` is carried verbatim. The bytes the provider issued are the
  evidence, and a normalised copy is not.
- `attestationDigest` is SHA-256 over those bytes, so the digest survives when
  the attestation is withheld and only the commitment travels.
- `issuer.trustListRef` records which list the provider was resolved from at
  issuance. A verifier resolves the list again at verification time rather
  than trusting this field, and the field exists so a disagreement is
  diagnosable rather than silent.
- `agentBinding.boundVia` names how the agent is tied to the attested legal
  person. `attestationDigest` means the agent id appears inside the attested
  attribute set. Other values are reserved.

## Verification

A verifier holding a grant, a `mandate` block and nothing of the producer's
software performs, in order:

1. Verify the grant signature over the JCS bytes, unchanged from today.
2. Recompute `attestationDigest` over `attestation`. Mismatch is fatal.
3. Verify the attestation's own signature against the provider's certificate.
4. Resolve the provider from the live EU List of Trusted Lists and confirm the
   service is `EAA/Q` with status `granted` at the time the attestation was
   issued. Absence is fatal. Withdrawn-after-issuance is a warning, reported,
   and does not retroactively void a record that carried a qualified time
   anchor proving issuance preceded withdrawal.
5. Confirm the subject in the attestation matches `asserted.iss`.
6. Confirm `agentBinding.agentId` matches the agent id in the bound
   attestation instance.

Steps 1, 2, 3 and 6 are offline. Steps 4 and 5 need the trusted list, which is
public, cacheable, and not controlled by the producer.

## Where trust comes from, stated plainly

| Claim | Rests on |
|---|---|
| the record was not altered | the hash chain, offline |
| the record existed by time T | RFC 3161 token from a supervised TSA |
| the agent was gated before acting | the decision record and the args commitment |
| **the agent acted for this legal person** | **the QEAA, issued by a supervised provider, resolved from a public register** |

The fourth row is the one no operator can grant itself, and it is the reason
this block is worth the work. The failure mode is honest and worth writing
down: if the supervisory regime is captured or a provider is compromised, this
row fails, and it fails for everyone relying on eIDAS rather than for Vaara
specifically.

## Scope and non-goals

- Vaara does not become a trust service provider. It consumes attestations and
  never issues them.
- No attribute content is interpreted. Whether an attested attribute is
  sufficient authority for a given action is a policy question for the relying
  party, and the record carries the attestation so that question can be
  answered later by someone else.
- `mandate` is optional. A record without it is exactly as valid as it is
  today, and verification of everything else is unchanged.
- Selective disclosure and zero-knowledge presentation of attributes are out
  of scope for this revision and are the natural follow-on, because the
  digest-only path already allows the attestation to be withheld.
- No personal-data attributes. The attested subject is a legal person. A
  design that attests attributes about a natural person, and in particular
  about a minor, raises questions this spec does not attempt to answer and
  must not be assumed to cover.

## Conformance vectors

New suite `qeaa_mandate_v0`, in the shape of the existing sets: a `cases.json`
of positive and negative records and a `_check_independent.py` that imports no
Vaara.

- `pos_valid_mandate`: attestation verifies, provider granted on the list,
  subject matches issuer, agent id matches the bound attestation
- `neg_digest_mismatch`: `attestation` bytes altered after signing
- `neg_provider_not_granted`: provider resolves but the service is not
  `EAA/Q` granted
- `neg_provider_absent`: provider not on any national list
- `neg_subject_mismatch`: attested subject is not `asserted.iss`
- `neg_agent_not_bound`: `agentBinding.agentId` absent from the attestation
- `neg_withdrawn_before_issuance`: service withdrawn before the anchor time

The trusted-list responses used by the negative cases are pinned fixtures, so
the suite is deterministic and does not depend on a live national list being
reachable at test time. The Portuguese list being unreachable during the
2026-08-26 walk is exactly why: a suite that needs 31 remote XML documents to
agree is a suite that fails for reasons that have nothing to do with the code.

## State of implementation, 2026-08-26

The design above is settled. **None of it is built yet.** There is no `mandate`
symbol anywhere in the tree, so nothing here has an implementation to drift
from, and the order below is the build order.

1. **`mandate` on the grant envelope.** Optional block, covered by the existing
   grant signature over the JCS encoding. The signature surface changes only
   when the block is present, so every existing grant stays byte-identical and
   every published vector still passes. This is the same constraint the
   decision vocabulary hit on 2026-08-26 and the same answer: add, never
   rename, and keep the absent case identical.
2. **Digest and structural verification**, steps 1, 2, 3 and 6 of the
   verification order. All offline, all testable with fixtures, no network.
3. **Trusted-list resolution**, steps 4 and 5. `eu_trusted_list.py` already
   walks the LOTL and parses national lists, but it filters on the qualified
   timestamp service type. It needs the `EAA/Q` type alongside, plus the
   status-at-a-time check that step 4 describes, which the timestamp path does
   not currently need.
4. **Wire it to `_decision_binding` and `_handoff`.** This is where the value
   lands: a handoff that crosses an organisational boundary carrying an
   attestation the receiving side resolves from a public register.
5. **`qeaa_mandate_v0`**, the seven cases listed above, with pinned list
   fixtures.

Steps 1 to 3 are self-contained and do not need a real attestation. Step 4 is
where a real one from a live provider is needed to prove the path end to end,
and there are now two providers to ask rather than one.
