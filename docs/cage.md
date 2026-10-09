# The cage layer

Vaara decides and records. The cage confines. They are different jobs,
and several good cages exist: NVIDIA OpenShell, Anthropic's
sandbox-runtime, gVisor, Firecracker, Kata, the Kubernetes agent-sandbox.
Vaara does not insist on its own. The cage is a choosable part: Vaara's
own cage (`vaara run`, the Linux OS layer) is one driver, OpenShell is
another, and more follow. One policy, one record, one verifier, whatever
the cage.

What no cage records about itself is whether it was on when a decision
was made. OpenShell logs what it allowed and denied, as unsigned JSONL
that rotates away after three days. None of the others write their own
confinement into evidence. Vaara does: every decision record carries a
`cage` block, and the signed receipt for that decision carries it too.

## The block

```
"cage": {
  "driver": "openshell",
  "confirmed": true,
  "upstream": "openshell 0.1.5",
  "config_digest": "sha256:3f1c...",
  "basis": "seccomp_filter",
  "name": "demo"
}
```

| Key | Meaning |
|---|---|
| `driver` | Which cage: `vaara-cage`, `openshell`, or `none` for a run outside any cage. |
| `confirmed` | Whether the kernel confirmed, on the deciding process at decision time, the confinement that cage imposes. |
| `upstream` | The cage's own name and version. |
| `config_digest` | `sha256:` over the cage's effective configuration. For the Vaara cage, the rendered AppArmor profile. For OpenShell, the sandbox policy YAML as submitted at create. |
| `basis` | What was checked. `apparmor_label`: the deciding process carries the `vaara-agent` label. `seccomp_filter`: it runs under a seccomp filter with `no_new_privs`, which is what OpenShell sets on its main process and everything under it. `declared`: the launcher said so and the kernel check did not pass. |
| `name` | The launch's name in the cage, when set. |

A run outside any cage writes `{"driver": "none", "confirmed": false}`.
A record is never silent about its confinement.

## Two sides, one contract

The launcher side starts the agent inside the cage and hands the governed
tree four environment variables: `VAARA_CAGE`, `VAARA_CAGE_DIGEST`,
`VAARA_CAGE_UPSTREAM`, `VAARA_CAGE_NAME`. `vaara run` sets them itself.
The OpenShell driver passes them as `--env` on `sandbox create`.

The deciding side is `vaara.cage.observe()`, which the trail calls for
every decision it records. It reads the declaration and then asks the
kernel whether the confinement that cage imposes is on this process. The
declaration says which cage; the kernel says whether it holds; `confirmed`
is true only when both agree.

```
  launcher (operator side)                  deciding process (inside)
  +--------------------------+              +---------------------------+
  | driver.start(agent, pol) |  VAARA_CAGE* | observe():                |
  |  vaara run / openshell   | -----------> |  declared + kernel check  |
  |  sandbox create --env    |              |  -> cage block on record  |
  +--------------------------+              +---------------------------+
          |                                            |
  driver.enforcement_state()                 signed receipt carries it
  (guard status / gateway status)
```

`vaara cage observe` prints the block a decision made by the calling
process would carry. Run it inside the sandbox to see what the records
will say.

## Drivers

Five calls, the same for every cage: `start(agent, policy, name=)`,
`stop(name)`, `status(name)`, `enforcement_state(name)` and
`events(name, since)`. Drivers depend on unmodified upstream releases and
shell out to the cage's own tools. No forks; no driver code outside this
repository.

| Driver | Cage | Starts with | Confirmed from inside by | Operator-side state from |
|---|---|---|---|---|
| `vaara-cage` | `vaara-agent` AppArmor profile, a cgroup per launch, the fanotify guard | `vaara run --name N -- agent` | the process's own AppArmor label | the guard's status: profile loaded, profile digest, launches |
| `openshell` | Landlock, seccomp, per-sandbox egress proxy, credentials at the boundary | `openshell sandbox create --name N --policy P --detach --env ... -- agent` | a seccomp filter and `no_new_privs` on the process | `openshell sandbox get N -o json`: phase, policy version, policy hash |

```
vaara cage drivers
vaara cage run --driver openshell --policy policy.yaml --name demo -- claude -p "tidy the README"
vaara cage status --driver openshell demo
vaara cage events --driver openshell demo --since 10m
vaara cage stop --driver openshell demo
vaara cage run --driver vaara-cage -- codex
```

The Vaara cage takes its policy from the OS-layer selection the operator
keeps with `vaara os-layer`, not from a file per launch; `--policy` is
refused there. The OpenShell driver finds the CLI on `PATH` or at
`OPENSHELL_BIN`.

## What the block does and does not establish

- `confirmed: true` with `basis: apparmor_label` means the deciding
  process carried the `vaara-agent` label when it decided. The label is
  set by the kernel at exec and cannot be dropped by the process.
- `confirmed: true` with `basis: seccomp_filter` means a seccomp filter
  and `no_new_privs` were on the deciding process. That is what OpenShell
  sets, and it is also what a container runtime's default profile sets.
  The check confirms the kind of confinement OpenShell imposes; it does
  not identify OpenShell's filter from inside. The operator-side
  `enforcement_state` (the gateway's phase, policy version and hash) is
  the other half, and `vaara cage status` shows it.
- For OpenShell, the record's `config_digest` is over the policy YAML as
  submitted. The gateway may merge a global policy on top; `status`
  reports the digest of the policy the gateway holds as active beside the
  gateway's own `policy_hash`, so the two can be compared.
- A process that is confined but was not started through a driver is not
  guessed at. Without a declaration the block says `none`, since a
  container's default seccomp profile looks the same from inside as a
  cage, and a record must not claim a cage on that evidence.
- The block is written by the deciding process. It is evidence from that
  process, chained and signed with the rest of the record. It is not a
  hardware attestation; TPM and SEV-SNP binding of the record are
  separate (`docs/capabilities.md`, section 3) and apply to this block as
  to the rest of the record.
