# Vaara Conformance Profile v1

Vaara Conformance Profile v1 is a fixed, versioned target. An outside party runs
against it to decide, on its own machine, whether an implementation agrees with
Vaara on every published vector, with no access to the producer, no producer key,
and none of the producer's software.

This document is the profile. The corpus and its checkers are the normative
content; this page names them, fixes a version, and states what a pass does and
does not establish.

## What the profile is

Each suite under `tests/vectors/<name>/` ships:

- a set of case files (the published vectors),
- an `expected.json` recording the verdict for each case, and
- `_check_independent.py`, a checker that recomputes every verdict from the
  bytes of the case files and **imports no Vaara code**.

An implementation conforms to Profile v1 when, for every suite, its own vectors
reproduce the published verdicts under the same checker. The aggregate runner
`scripts/conformance_runner.py` discovers every suite, runs each checker in a
subprocess, and returns a single pass/fail.

Profile v1 covers the 36 suites listed below, at the `v0` vector format. The
authoritative, always-current enumeration for any tagged release is the runner's
own `--list` output at that tag.

## What you need

The runner itself is standard-library Python. The checkers are not Vaara, but
they are not dependency-free either: they recompute signatures and canonical
bytes using public libraries.

```
pip install rfc8785 cryptography
```

`rfc8785` is JCS canonicalization (RFC 8785); `cryptography` covers the Ed25519
and ECDSA signature paths. The `pq_hybrid_v0` suite additionally needs
`dilithium_py`. None of these is Vaara. The point of the profile is that you run
standard public crypto, never the producer's stack.

## The one command

Grade the reference corpus from a clean checkout:

```
python scripts/conformance_runner.py
```

Grade your own implementation's vectors by pointing the runner at their
directory:

```
python scripts/conformance_runner.py --vectors-dir ./your_vectors
```

Useful flags:

- `--list` prints the discovered suites (the authoritative profile membership).
- `--corpus NAME` runs a single suite; repeatable.
- `--json PATH` writes a dated machine report (Python version, per-suite status,
  totals, `all_passed`).

Exit code is `0` only when no suite failed.

## Required, advisory, and skipped

Inside a checker, two kinds of assertion are distinguished:

- **Required** checks gate conformance. Any required failure fails the suite.
- **Advisory** checks report a deviation without failing the suite (for example,
  a signature that is well-formed but not the expected length for its algorithm).

One suite is reported `SKIP`, not pass, in an aggregate run:

- `article12_fold_v0` validates a bundle handed to it on the command line rather
  than a bare directory of case files. The runner lists it explicitly so the gap
  reads as a gap, not as silent coverage.

## What a pass means, and what it does not

A pass means: an independent party, running standard public libraries and none
of the producer's code, reproduced every published verdict from the case bytes
alone.

A pass does **not** mean the implementation is correct for inputs outside the
corpus, that the corpus is complete, or that any party has endorsed it. It is a
reproducibility result over a fixed, published set, nothing wider. The corpus
grows; conformance is always against a named version.

## Suites in Profile v1

```
agent_identity_v0            enforcement_attestation_v0
ap2_v0                       enforcement_set_v0
article12_fold_v0            evidence_bundle_v0
atlas_threat_v0              evidence_ref_v0
attestation_result_v0        execution_receipt_v0
audit_summary_v0             external_evidence_v0
authorization_v0             governance_decision_v0
build_bundle_v0              handoff_set_v0
bundle_doc_v0                ingest_v0
bundle_set_v0                key_rotation_v0
capability_scope_v0          normalize_v0
class_gate_v0                pq_hybrid_v0
conformance_statement_v0     record_conformance_v0
contiguity_v0                record_set_v0
credential_binding_v0        sep2787_attestation_v0
cross_org_handoff_v0         tap_v0
cross_stack_revocation_v0    transparency_consistency_v0
decision_pairing_v0          x402_settlement_v0
```

## Related

- [docs/verifying-evidence.md](verifying-evidence.md) covers each verifier verb
  and where trust comes from in each case.
- `vaara conformance-statement` is the producer-side keyless self-statement
  against a corpus. It is a different job: the statement is what a producer
  publishes about its own records; this profile is what an outside party runs to
  check an implementation.
