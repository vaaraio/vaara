# Signing keys for Vaara audit exports

Vaara signs every audit-trail export with an Ed25519 detached signature.
The signature lets a third party - auditor, regulator, internal conformity
reviewer - verify that an exported trail came from you and has not been
tampered with.

This document describes how to manage the signing key. It is written for
two audiences:

1. **Engineers running local evaluations or demos.** Skip to
   [Local evaluation keys](#local-evaluation-keys).
2. **Production deployments that submit trails to a regulator or
   auditor.** Read [Production keys](#production-keys) first.

---

## Threat model

A signed trail is a statement of the form:

> "On date *D*, the holder of the private key with public-key
>  fingerprint *F* attests that the exported record set is authentic."

Two failures matter:

- **Forgery.** An attacker who obtains the private key can produce signed
  trails that the verifier will accept as authentic.
- **Tamper.** An attacker who alters the trail file but does *not* have
  the key cannot produce a valid signature. This case is covered by the
  signature scheme itself.

The only attack that signing alone cannot stop is key compromise. Your
operational goal is to make that expensive: keep the private key in
hardware you control, rotate it on a schedule, and keep a public-key
fingerprint ledger so old trails stay verifiable after rotation.

---

## Production keys

Use one of the following, in order of preference:

### 1. Hardware security module (HSM) or cloud KMS

The private key never leaves the HSM. Signing happens by calling a
signing API.

Supported providers:

- **AWS KMS** - create an `ECC_NIST_P256` or `EDDSA` key,
  grant `kms:Sign` to the role your exporter runs as.
- **Google Cloud KMS** - create an `EC_SIGN_ED25519` asymmetric key.
- **Azure Key Vault** - create an EC key with curve `P-256` and the
  `sign` operation enabled.
- **YubiHSM / Nitrokey HSM** - Ed25519 keys with the `sign-eddsa`
  capability.

For any of these, do not export the PEM. In the 0.4.x series
`export_signed` accepts a loaded `Ed25519PrivateKey` instance (or a PEM
path / PEM bytes) - integrate by loading the key material into that
object inside your signing host's memory, never on a developer laptop.
A pluggable signer-adapter interface (call an HSM/KMS `sign` API
without materializing the key) is tracked for a future release. Until
then, HSM integration requires a small wrapper in your deployment code
that fetches the key into a short-lived `Ed25519PrivateKey`.

### 2. Dedicated signing host

A small, locked-down machine (not a developer laptop) that:

- holds the PEM with `chmod 0400`,
- has only the signing user, no shell for other users,
- is network-isolated except for inbound trail-signing requests,
- logs every signing request to an append-only store.

This is a viable fallback if you cannot provision an HSM yet.

### 3. Local PEM (evaluation only)

The `vaara keygen --dev` helper writes a PEM to disk with `0600`
permissions. **Do not use it to sign trails you hand to a regulator.**
It exists for:

- running the example scripts,
- recording a demo,
- writing integration tests.

See [Local evaluation keys](#local-evaluation-keys).

---

## Key rotation

Plan rotation before you need it. A concrete schedule:

- **Every 12 months**, or immediately on any suspicion of compromise.
- Generate the new key pair in the same place as the current one (HSM,
  KMS, etc.).
- Publish the new public-key fingerprint to the same out-of-band channel
  where your current public key lives (company website, signed release
  notes, regulator intake).
- Keep the previous public key available: a regulator who received a
  trail last quarter must still be able to verify it.
- Do **not** re-export old trails with the new key. Signatures are
  historical - each trail belongs to the key that signed it when
  exported.

Keep a fingerprint ledger, versioned in git:

```
# keys/public-key-fingerprints.md

2026-04-19  8f3c29b1a4d25e91...   production, HSM slot 3
2025-04-18  2a1b88f7ce944301...   superseded, rotated on schedule
2024-04-20  01a72dcbe5ff1aab...   superseded, rotated on schedule
```

The fingerprint in this ledger must match the `signer_pubkey_fingerprint`
field that Vaara embeds in every trail manifest.

---

## Distribution

The verifier needs your public key. Two channels:

1. **Out-of-band (recommended).** You hand the auditor the PEM file, or
   the fingerprint, via email, printed handoff, or the regulator's
   intake portal. The verifier passes `--pubkey signer.pem` to
   `vaara trail verify`.

2. **Embedded in the zip.** Every Vaara export embeds
   `signer_pubkey.pem` inside the zip. If no `--pubkey` is passed the
   verifier uses the embedded one. This proves the trail is
   *internally consistent* but not that the signer is who they claim to
   be - the auditor should always verify the fingerprint matches
   something they received out-of-band.

---

## Local evaluation keys

For demos, tests, and local evaluation only:

```bash
pip install 'vaara[export]'
vaara keygen --dev --out ~/.vaara/signer.pem
```

This writes:

- `~/.vaara/signer.pem` (private key, `0600`)
- `~/.vaara/signer.pem.pub` (public key)

and prints the fingerprint.

The `--dev` flag is required. Without it, `keygen` refuses to write a
key so operators do not accidentally treat a development key as
production-grade.

To sign and verify a trail:

```bash
vaara trail export --trail trail.jsonl --out signed.zip --key ~/.vaara/signer.pem
vaara trail verify --zip signed.zip --pubkey ~/.vaara/signer.pem.pub
```

---

## Offline verification

Regulators and auditors may run the standalone verifier with no Vaara
install:

```bash
pip install cryptography
python scripts/verify_vaara_trail.py signed.zip --pubkey signer.pem
```

The script in `scripts/verify_vaara_trail.py` depends only on the
standard library and the `cryptography` package. Copy it to any machine.

A trail that honoured an erasure request (`vaara trail erase`) verifies the
same way. The script lists each erased slot (link verified, content not
checkable) and each erasure record. Pass `--erasure-key eraser.pub.pem`, the
eraser's EC P-256 or RSA public key received out of band, to check the
eraser's signature on every erasure record; with the key given, an unsigned
or wrongly signed erasure fails the run.

A trail that records cage launches (`vaara cage run`, `vaara cage stop`,
`vaara run`) is paired the same way `vaara trail verify` pairs it: each stop
must name the start of the same driver and launch, written earlier in the
chain, or the run fails. The script lists every start and stop; a start with no
stop, and a stop whose start is outside the export, are listed and do not fail
the run.

Both verifiers refuse a trail line or a manifest that repeats a member of one
object, and a number with no finite value (`NaN`, `Infinity`, or a literal such
as `1e999`). A repeated member reads as its last value to one parser and its
first to another, and the hash covers only one of them, so the run fails with
`record N: duplicate member "x"` or `record N: non-finite number NaN` instead of
passing on one reading. A manifest fails as `manifest.json: ...`.

A trail line or manifest that is valid JSON but not an object (`[1]`, `7`,
`"x"`, `null`) also fails the run, as `record N: not a JSON object` or
`manifest.json: not a JSON object`. It used to stop the verifier with a Python
error and no verdict.

## Verifying a Vaara release

This section covers Vaara's own packages. Keys for signing your trails are covered above.

Every release is built by `.github/workflows/release.yml` on the tag push,
which attaches SLSA v1 build provenance to the wheel, the sdist and the
MCP bundle as a GitHub artifact attestation. The wheel on PyPI is the same
file as the one on the GitHub release, so the check works on either
download:

```bash
pip download vaara==<version> --no-deps
gh attestation verify vaara-<version>-py3-none-any.whl --repo vaaraio/vaara
```

A pass means the file's digest matches provenance signed by GitHub's
Sigstore instance for a run of `release.yml` in `vaaraio/vaara`. The
builder is that workflow itself rather than an isolated reusable builder,
so this is SLSA Build Level 2, and `slsa-verifier` rejects it for that
reason.
