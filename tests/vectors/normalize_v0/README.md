# normalize_v0 vectors

Fixtures for `vaara normalize`: mapping an adjacent MCP record onto the
SEP-2828 evidence model.

A SEP-2828 execution record is a signed decision+outcome pair. The
surrounding ecosystem emits narrower records that each cover one face of
the same event, and `normalize` files each onto the SEP-2828 model,
reporting which fields it establishes and what is still missing for a
complete signed record. It promotes nothing: an unsigned client claim
stays advisory.

## Inputs (`inputs/`)

Each input is a verbatim or near-verbatim example from the source spec.

| File | Source | Evidence plane | Fills |
| --- | --- | --- | --- |
| `sep2643_url_denial.json` | SEP-2643 denial (URL remediation) | outcome | `outcomeDerived.status = refused` |
| `sep2643_rar_denial.json` | SEP-2643 denial (RAR remediation) | outcome | `outcomeDerived.status = refused` |
| `sep2643_scope_denial.json` | SEP-2643 denial (scope, no hints) | outcome | `outcomeDerived.status = refused` |
| `sep2787_attestation.json` | SEP-2787 tool-call attestation | decision-attested | `backLink` |
| `sep2787_attestation_with_extension.json` | SEP-2787 attestation carrying extension fields | decision-attested | `backLink` (extras dropped) |
| `sep2817_single.json` | SEP-2817 invocation audit context | decision-input | nothing required (advisory) |
| `sep2817_multiturn.json` | SEP-2817 (redacted intent, shared turn) | decision-input | nothing required (advisory) |
| `slsa_provenance.json` | SLSA v1 in-toto provenance (declarative profile) | n/a | nothing required (advisory) |
| `c2pa_manifest.json` | C2PA content provenance manifest (declarative profile) | n/a | nothing required (advisory) |
| `agent_decision.json` | in-toto agent-decision/v0.1 policy decision (declarative profile) | decision-attested | nothing required (advisory) |
| `x402_settlement.json` | x402 settlement response (declarative profile) | outcome | nothing required (advisory) |
| `ap2_payment_receipt.json` | AP2 payment receipt (declarative profile) | outcome | nothing required (advisory) |
| `caep_security_event.json` | OpenID CAEP Security Event Token (declarative profile) | decision-input | nothing required (advisory) |
| `attested_agent_payment.json` | attested agent payment Authorization Scope (declarative profile) | decision-input | nothing required (advisory) |
| `attested_agent_payment_no_expiry.json` | the same scope with `expiry` dropped | n/a | nothing (not recognized) |
| `attested_agent_payment_aggregate_no_executor.json` | an aggregate bound with no `executor` | decision-input | nothing required (advisory) |
| `unknown.json` | unrecognized object | n/a | nothing |

`expected.json` holds the normalized mapping for each input.

## What the attested-agent-payment rows prove

The three payment-scope rows are one positive and two boundary cases against
draft-hawkins-scitt-attested-agent-payment-01 Section 3.

Dropping `expiry`, a required member of the Section 3 CDDL, drops the document
out of detection entirely: it normalizes as unrecognized and is still sealed,
which is the sink's behaviour for any unknown format.

The aggregate case is the more useful one, because it is recognized and should
not be read as a conformance verdict. Section 3 makes `executor` REQUIRED when
`limits` carries an aggregate bound, since an aggregate bound is only an
aggregate if one party serializes spending against it. That is a conditional
requirement, and the declarative detect grammar has no way to express one: rules
are flat predicates over paths. So a scope that violates it is still recognized
and mapped. The profile's notes say so, and this vector is what keeps that
statement honest.

The scope is deterministic CBOR (RFC 8949 Section 4.2.1) and its scope digest is
computed over those bytes. Vaara ingests the decoded map, so these vectors pin
the field mapping and never the scope digest.

The `agent-decision` row covers the closest foreign format to a native signed
decision record: an in-toto predicate that names the policy verdict, the
per-call argument commitments with their explicit-omission state, and the
allow/deny. It maps to the decision-attested plane, yet still fills no SEP-2828
field on its own, because its signature lives in the DSSE envelope (not modeled
here) and its `args_hash` is an argument commitment, not a back-link to an
attested request. The honest gap is the wedge. A DSSE-signed, public-key-
verifiable form of this same statement is the conformance vector in
`../agent_decision_v0/`.

The `slsa-provenance` and `c2pa-manifest` rows come from declarative profiles:
data-only specs under `src/vaara/attestation/profiles/*.json`, compiled by
`_declarative.py`. The independent checker reads those same specs and reproduces
the mapping with its own code, so they are no more self-confirming than the
hand-written profiles. See `docs/source-profile-contract.md`.

## What the back-link case proves

A SEP-2787 attestation is the attested request a SEP-2828 receipt
answers. `normalize` computes the exact `backLink` a conformant receipt
must pin: `attestationDigest` is `sha256` over the JCS canonicalization of
the SEP-2787-modeled fields (parse, then canonicalize), `attestationNonce`
is the issuer nonce. For `sep2787_attestation.json` this is
`sha256:79acdd4bb3c22a688b1c3321b9a26cafb5cb58c990a963874066d04b8497f70b`,
the same value the paired receipt in `execution_receipt_v0` carries. The
attestation fixes the back-link and nothing else: the record's own
`alg`, `signature`, and `receiptAsserted` are a separate signing event by
the recording side.

Fields outside the modeled schema are not covered.
`sep2787_attestation_with_extension.json` carries an extra top-level key
and an `issuerAsserted.kid`, yet yields the same digest, which pins that
the modeled fields, not the raw received bytes, are what gets digested.

## Independent checker

`_check_independent.py` reimplements the normalization from the specs
alone, with no Vaara import, and reproduces every case in `expected.json`.
The SEP-2643 and SEP-2817 maps are pure standard library. The SEP-2787
case reconstructs the modeled envelope (dropping unmodeled fields,
injecting the ArgsRef canonicalization default) and digests that under
`rfc8785`, the same value the receipt verifier pins; it is skipped (not
failed) when `rfc8785` is absent.

```
python tests/vectors/normalize_v0/_check_independent.py
```
