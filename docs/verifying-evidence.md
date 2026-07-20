# Verifying the evidence

Producing a trail is half the job. The other half is letting someone who does not trust you check it, with no key, no access to your system, and none of your software. Every command here reads the wire format, is fail-closed on authenticity, and ships with public conformance vectors plus a standalone checker that imports no Vaara code, so an independent party reproduces every verdict offline. That property is the point of the standards work behind [the Vaara Receipt Internet-Draft](https://datatracker.ietf.org/doc/draft-sirkkavaara-vaara-receipt/): the evidence is verifiable by someone who runs none of your software.

The [README](../README.md#verify-the-evidence) carries the command table. This page carries the full trust model behind each one.

## `verify-bundle`: one bundle, six lenses

```bash
vaara verify-bundle evidence-bundle.json
```

No code to write, and no need to trust the tooling that produced the bundle. The command runs six lenses and is fail-closed on authenticity, so a record that is merely present in a log, with its signature never checked, does not pass:

- **Identity** resolves the signing key to a `did:web` the agent controls, so the receipt names who acted, not just that something signed it.
- **Signature** verifies the receipt under that key.
- **Back-link** checks that the receipt binds to the request attestation it answers and to the prior chain head.
- **Inclusion** checks that the record is in the transparency log.
- **Consistency** checks that the log is append-only, so an earlier verified head stays consistent with the current one and nothing was rewritten behind you.
- **Revocation** checks that no key or receipt in the chain has been revoked, across stacks.

`ok` is true only when the signature is actually established and every applicable lens passes. A bundle that proves inclusion and non-revocation but never verifies a signature is not `ok`.

## `build-bundle`: the issuer side

`build-bundle` produces what `verify-bundle` checks, from the receipt and whatever identity, signature, inclusion, consistency, and revocation material you hold:

```bash
vaara build-bundle --from-dir ./pieces --out evidence-bundle.json
vaara verify-bundle evidence-bundle.json
```

It writes the exact document `verify-bundle` reads, then loads it back and reports the verdict, so producing and checking the evidence is one closed loop over one file.

## `verify-record`: the format itself, keyless

`verify-bundle` checks a bundle you assembled. `verify-record` checks the format: point it at any JSON that claims to be a SEP-2828 execution record, including one Vaara never produced, and it tells you whether the record is well formed and internally consistent.

```bash
vaara verify-record someone-elses-record.json
```

It needs no signing key and no attestation. The check is the wire schema plus the one binding a record proves about itself: the result commitment digest is the SHA-256 of the bytes it sits beside, so a verifier recomputes it with nothing but a hash function. Add `--attestation` to also check the back-link to the request the record answers, still without a key. The signature check, which does need the signer's key, stays in `vaara receipt verify`. This is the check an auditor, or a vendor whose software you do not run, can apply before trusting the producer or any key. The trust rests on the format, not on Vaara.

## The auditor's workbench: set-level forms

When the evidence is a folder of records or bundles rather than one file, each single-file command has a set-level form that runs over a whole directory:

```bash
vaara verify-records      ./records
vaara verify-bundles      ./bundles
vaara verify-handoffs     ./handoffs
vaara verify-enforcements ./enforced
vaara audit-summary       ./records --out summary.md
```

- `verify-records` checks every record for SEP-2828 conformance, then checks the set as a whole: it flags a call recorded twice, an authorised decision with no matching outcome, and an executed action that committed no result. Keyless, like `verify-record`.
- `verify-bundles` runs the full six-lens `verify-bundle` over every bundle and reports per-lens pass counts and how many bundles authenticated.
- `verify-handoffs` runs `verify-handoff` over a directory of cross-org packages and reports how many records verify under their rotated-out keys, how many are anchor-corroborated rather than resting on the signature alone, and how many had their producer pinned.
- `verify-enforcements` runs `verify-enforcement` over a directory of records and their SEV-SNP reports (discovered by stem: `NAME.record.json` with `NAME.report.bin` and `NAME.vcek.pem`), reporting how many bind to a confidential VM, the per-tier tally, and whether any pinned a vetted launch image.
- `audit-summary` renders the conformance verdict for a directory of records as a Markdown page an auditor reads directly. The page states what was checked and every count, and records that any party can reproduce it from the records alone.

Each set form is `ok` only when every item verifies for the chosen mode; a coverage note (no producer pinned, no image pinned) is advisory and does not gate. Vaara-free checkers in `tests/vectors/handoff_set_v0/` and `tests/vectors/enforcement_set_v0/` reproduce every roll-up.

## `conformance-statement`: prove conformance against the corpus

A producer who claims its records are SEP-2828 records can prove it against the published conformance corpus instead of asking to be trusted:

```bash
vaara conformance-statement --corpus conformance/sep2828 --records ./records
```

The command prints one statement. It confirms the corpus bytes match their manifest, re-runs this implementation's keyless conformance check over every corpus fixture to confirm it reproduces the verdict the corpus records, and runs your own records through the same set check. The statement names the exact corpus version and corpusDigest it was checked against, so the claim pins a fixed byte set rather than a moving target. Keyless and deterministic: anyone holding the same corpus re-runs the command and reaches the same verdict.

## `verify-retained`: a record under a rotated-out key

Article 12 records outlive the keys that signed them. A record signed in 2026 is audited years later, after the issuer rotated to a new key and retired the old one. The live DID document no longer lists the key that signed the record, so a plain identity check fails on a record that is perfectly genuine. `verify-retained` checks it against the document you archived at record time:

```bash
vaara verify-retained record.json --did-document archived-did.json --anchor anchor.json
```

It binds the signature to a key the archived document lists, then checks the claimed signing time falls inside that key's validity window (`validFrom` / `validUntil` on the verification method) and that the key was not revoked before issuance. A retired key still verifies a signature it made while it was valid; retirement is graceful end-of-life, not revocation. With a verified time anchor the verdict is corroborated: the record provably existed before the key's end of life, so it cannot be a later forgery made with a stolen retired key. Without an anchor the verdict rests on the record's self-asserted time and says so. The check is offline and reproducible, and a Vaara-free checker in `tests/vectors/key_rotation_v0/` reproduces every verdict with nothing but `cryptography` and a JSON canonicalizer.

## hybrid PQC signing: sign against a future quantum adversary

A receipt kept for years faces "trust now, forge later": its classical ES256 / RS256 signature is forgeable once a quantum computer exists, and the forgery can be backdated into the retention window. A receipt can carry a parallel ML-DSA-65 (FIPS 204) signature over the same preimage; both must verify. The issuer commits the hybrid suite inside the signed bytes (`receiptAsserted.sigSuite`), so stripping the post-quantum signature is a detectable downgrade, not a silent loss of protection. `pq_verdict` (the `vaara.attestation.receipt` API) reports a quantum-resistance tier alongside the rotated-key verdict above: `hybrid-verified` (quantum- and downgrade-resistant), `pqc-present` (a post-quantum signature is attached but not committed, so strippable), `classical-only` (verifiable today, not quantum-resistant), or the fail-closed `hybrid-downgraded`. Pre-quantum records are unchanged and stay verifiable. ML-DSA is the pure-Python `dilithium-py` (`vaara[pq]`), so the base install and every classical path stay standard-library; a Vaara-free checker in `tests/vectors/pq_hybrid_v0/` reproduces every verdict. `dilithium-py` is a reference implementation (not constant-time, not hardened against side channels per its own docs): fine for the independent verifier and for vectors, but production signing of real long-lived keys should use a hardened or FIPS-validated ML-DSA behind the same signer boundary. The verdict tier is `pq_verdict`'s; v0 reports it but does not yet gate `verify-retained`, `verify-bundle`, or the conformance pass/fail on it (an E1b follow-on), so a committed-downgrade record is surfaced as a conformance advisory rather than a hard fail.

## `build-handoff` / `verify-handoff`: hand a record to another org's regulator

A provider signs a record. A deployer who runs that provider's system, a different organisation, has to show it to its own regulator, offline, years later, with no live channel back to the provider. `build-handoff` packs the record, the archived DID document, the key history, revocations, and an optional time anchor into one self-contained file, pinning each piece by content digest. `verify-handoff` checks it:

```bash
vaara build-handoff --record record.json --did-document archived-did.json \
  --anchor anchor.json --holder did:web:deployer.example --out handoff.json
vaara verify-handoff handoff.json --trusted-did-document provider-keys.json --strict
```

It recomputes every pinned digest, routes the record through the same rotated-key lens, and confirms an enclosed anchor's imprint is `sha256` of the record itself, so an anchor taken over a different record never corroborates this one. The verdict is honest about where trust comes from: the digests prove only that the package is internally consistent, since the holder controls both the pieces and the manifest that pins them. The record's authenticity rests on the provider's signature against the provider's identity, which you establish out of band; `--trusted-did-document` pins it against a key set you already trust, and until you do, the verdict says `producer_identity_basis: self_asserted_unpinned`. The eIDAS anchor is the one piece the holder cannot forge. `--strict` passes only a corroborated record with a recorded window, an affirmative revocation source, and a pinned identity. An optional holder custody signature is reported separately and never changes the record verdict. A Vaara-free checker in `tests/vectors/cross_org_handoff_v0/` reproduces every verdict.

## `verify-enforcement`: was the record produced inside a confidential VM

The records prove who signed, when, and what. They do not show *where* the enforcement ran. If the enforcement point runs inside an AMD SEV-SNP confidential VM, it can ask the chip for an attestation report carrying `sha512` of the record it just signed. `verify-enforcement` checks that report binds to that exact record:

```bash
vaara verify-enforcement record.json --report report.bin --vcek vcek.pem \
  --expected-measurement <hex-of-the-vetted-image>
```

A pass means a SEV-SNP report carrying `sha512(jcs(record))` verifies against the VCEK you supplied, so this record's bytes were hashed inside some SEV-SNP confidential VM whose VCEK you chose to trust. The verdict is blunt about the rest. It does not validate the VCEK chain to AMD's ARK (that fetch is deferred), so a mock report with no AMD provenance passes the same check, and `vcek_chain_basis` stays `caller_supplied_unverified`. It does not prove the decision logic ran in the enclave, so `enforcement_logic_basis` is always `not_established`. Pinning the launch measurement with `--expected-measurement` tells you which image ran and lifts the tier to `measurement_pinned`; without it the measurement is reported but unpinned. The binding is over the whole record including its signature, so a report for one record never verifies another, and a signature-stripped variant never rides a genuine report. The word `attested`, and a `--strict` pass, are reserved for a future release that validates the AMD chain; v0 publishes that bar without pretending to clear it. A Vaara-free checker in `tests/vectors/enforcement_attestation_v0/` reproduces every verdict.
