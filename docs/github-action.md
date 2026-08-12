# GitHub Action

Vaara ships a composite action so a policy is checked in CI the same way tests are, and a signed
trail can be verified inside a pipeline rather than by hand afterwards.

The action fails the build on a policy that does not parse, on a failing policy case, and on a
trail whose chain or signature does not hold.

## Policy as code, checked on every pull request

```yaml
name: Governance
on: [pull_request]

permissions:
  contents: read

jobs:
  policy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: vaaraio/vaara@v1
        with:
          policy: policies/production.yaml
          cases: policies/production.cases.yaml
```

`policy` is validated and `cases` are run against it. Both accept YAML or JSON. Cases require a
policy, since a case is evaluated through one.

## Verifying a trail a job produced

```yaml
      - uses: vaaraio/vaara@v1
        with:
          trail: artifacts/trail.zip
          pubkey: keys/signer_public.pem
```

Leaving `pubkey` out verifies against the key inside the zip. That shows the trail is internally
intact and its chain unbroken, and it does not tell you who signed it, because the zip supplied
both the claim and the key. Pass a key you obtained separately when you need the signer bound too.
The action says which of the two it did.

## Inputs

| input | default | meaning |
| --- | --- | --- |
| `policy` | | Policy file to validate. |
| `cases` | | Cases file to run against `policy`. |
| `trail` | | Signed trail zip to verify. |
| `pubkey` | | Ed25519 public key (PEM) for trail verification. |
| `fail-on-warnings` | `false` | Treat validation warnings as failures. |
| `version` | latest | Version of the `vaara` package to install. Pin it for reproducible runs. |
| `extras` | `yaml,export` | Extras to install. Vaara has no base dependencies, so `yaml` is what lets a YAML policy be read at all and `export` provides the Ed25519 verification a signed trail needs. |
| `python-version` | `3.12` | Python used to run the check. |
| `working-directory` | `.` | Directory the paths are relative to. |

At least one of `policy` or `trail` is required. Anything else is a usage error and exits 2, which
is distinct from a check that ran and failed.

## Outputs

`errors`, `warnings`, `cases-total`, `cases-passed`, `cases-failed`, and `trail-verified`. Each run
also writes a summary to the job page listing every issue and every failing case by name.

```yaml
      - uses: vaaraio/vaara@v1
        id: gate
        with:
          policy: policies/production.yaml
      - run: echo "policy had ${{ steps.gate.outputs.warnings }} warnings"
```

## Pinning

`version` is unpinned by default, so the action installs the current release. For a build you want
to reproduce later, pin both the action and the package:

```yaml
      - uses: vaaraio/vaara@v1.66.1
        with:
          policy: policies/production.yaml
          version: 1.65.0
```

## What this does not do

It checks artifacts. It does not gate the agent, which is what the runtime does at the moment of
the tool call. A green policy check means the policy parses and behaves as its cases say on the
inputs those cases describe; it is not evidence about any particular production run.
