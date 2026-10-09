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
| `driver` | Which cage: one of the twelve driver names below, or `none` for a run outside any cage. |
| `confirmed` | Whether the kernel confirmed, on the deciding process at decision time, the confinement that cage imposes. |
| `upstream` | The cage's own name and version. |
| `config_digest` | `sha256:` over the cage's effective configuration. For the Vaara cage, the rendered AppArmor profile. For OpenShell, the sandbox policy YAML as submitted at create. |
| `basis` | What was checked. `apparmor_label`: the deciding process carries the `vaara-agent` label. `seccomp_filter`: it runs under a seccomp filter with `no_new_privs`. `no_new_privs`: that flag alone. `bwrap_init`: pid 1 of its pid namespace is bubblewrap. `gvisor_kernel_log`: the kernel log is gVisor's. `hypervisor_present`: the CPU reports a hypervisor underneath. `declared`: the launcher said so and no kernel check passed. |
| `name` | The launch's name in the cage, when set. |

A run outside any cage writes `{"driver": "none", "confirmed": false}`.
A record is never silent about its confinement.

## Two sides, one contract

The launcher side starts the agent inside the cage and hands the governed
tree four environment variables: `VAARA_CAGE`, `VAARA_CAGE_DIGEST`,
`VAARA_CAGE_UPSTREAM`, `VAARA_CAGE_NAME`. `vaara run` sets them itself;
the other drivers pass them through the cage's own environment option.
A microVM booted from a kernel image has no environment to receive them,
so the Firecracker driver writes the same four as `vaara.cage=`,
`vaara.cage.digest=`, `vaara.cage.upstream=` and `vaara.cage.name=` on the
kernel command line, and the deciding side reads `/proc/cmdline` when the
environment carries nothing.

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

| Driver | Cage (licence) | Starts with | Policy file | Confirmed from inside by | Operator-side state from |
|---|---|---|---|---|---|
| `vaara-cage` | Vaara's own: `vaara-agent` AppArmor profile, a cgroup per launch, the fanotify guard (AGPL-3.0) | `vaara run --name N -- agent` | none; the OS-layer selection | the process's own AppArmor label | the guard's status: profile loaded, profile digest, launches |
| `openshell` | NVIDIA OpenShell: Landlock, seccomp, per-sandbox egress proxy, credentials at the boundary (Apache-2.0) | `openshell sandbox create --name N --policy P --detach --env ... -- agent` | sandbox policy YAML | a seccomp filter and `no_new_privs` on the process | `openshell sandbox get N -o json`: phase, policy version, policy hash |
| `codex` | OpenAI Codex sandbox: bubblewrap, Landlock, seccomp on Linux; Seatbelt on macOS (Apache-2.0) | `codex sandbox --sandbox-state-json ... -- agent` | sandbox state JSON | a seccomp filter and `no_new_privs` (Linux) | the launcher's child process |
| `sandbox-runtime` | Anthropic sandbox-runtime: bubblewrap namespaces and a filtering proxy on Linux; Seatbelt on macOS (Apache-2.0) | `srt --settings S -- agent` | `srt` settings JSON | bubblewrap as pid 1 of the process's pid namespace (Linux) | the launcher's child process |
| `nono` | nono: Landlock first with a seccomp baseline on Linux; Seatbelt on macOS (Apache-2.0) | `nono run --profile P --name N --detached -- agent` | a profile file or catalogue name | a seccomp filter, or `no_new_privs` alone under the Landlock-only policy | `nono ps --json`: session status |
| `gvisor` | gVisor: a user-space kernel, as the `runsc` OCI runtime (Apache-2.0) | `docker run -d --runtime=runsc --name N -e ... IMAGE agent` | none; `--image` and `--security-opt` | the kernel log is gVisor's own | `docker inspect N`: state and runtime |
| `kata` | Kata Containers: a VM per container with its own guest kernel (Apache-2.0) | `docker run -d --runtime=io.containerd.kata.v2 ...` | none; `--image` and `--security-opt` | a hypervisor under the CPU | `docker inspect N`: state and runtime |
| `agent-sandbox` | kubernetes-sigs/agent-sandbox: a `Sandbox` object whose pod runs under gVisor or Kata (Apache-2.0) | `kubectl apply` of the manifest with the agent as the container command | the `Sandbox` manifest, JSON or YAML | gVisor's kernel log, or a hypervisor | `kubectl get sandboxes.agents.x-k8s.io N`: the `Ready` condition |
| `firecracker` | Firecracker: a microVM from a kernel and a root filesystem (Apache-2.0) | `firecracker --api-sock S --id N --config-file C` | the microVM configuration JSON | a hypervisor; the declaration arrives on the kernel command line | the API on the socket: instance state and the active configuration |
| `microsandbox` | microsandbox: a libkrun microVM per sandbox, `msb` (Apache-2.0) | `msb run --conf C --name N --detach --no-tty -e ... -- agent` | the sandbox YAML | a hypervisor | `msb status N --format json` |
| `e2b` | E2B self-hosted infra: Firecracker microVMs behind an HTTP API, envd inside (Apache-2.0) | `POST /sandboxes`, then `process.Process/Start` on envd | the `NewSandbox` request JSON | a hypervisor | `GET /sandboxes/{id}`: `running` or `paused` |
| `apple-container` | Apple's `container`: each Linux container in its own lightweight VM on a Mac with Apple silicon (Apache-2.0) | `container run -d --name N -e ... IMAGE agent` | none; `--image`, `--read-only`, `--cap-drop` | a hypervisor | `container inspect N`: the container's state |

Each driver depends on the release of the cage the operator installed and
shells out to it, or speaks its published API. `docker` is the default
engine for the runtime-backed cages; `VAARA_CAGE_ENGINE=podman` switches.
Every tool's path can be set with its own variable (`OPENSHELL_BIN`,
`CODEX_BIN`, `SRT_BIN`, `NONO_BIN`, `RUNSC_BIN`, `KATA_RUNTIME_BIN`,
`KUBECTL_BIN`, `FIRECRACKER_BIN`, `MSB_BIN`, `APPLE_CONTAINER_BIN`); E2B takes `E2B_API_URL`,
`E2B_API_KEY` and `E2B_DOMAIN`.

## Which cages run where

Vaara ships the drivers. The cages are installed by the operator from
their own releases; Vaara's own cage is the one that comes built in.
`vaara cage drivers` checks the machine it runs on and says, for every
driver, whether it is ready, the version the cage reports, and what is
missing with the install hint. It installs nothing and starts nothing.

| Runs on | Drivers |
|---|---|
| Linux | all but `apple-container` |
| macOS | `openshell` (its sandbox in a Linux VM), `codex`, `sandbox-runtime`, `nono` (Seatbelt), `microsandbox`, `apple-container` (a VM per sandbox) |
| Windows | `microsandbox` (Windows Hypervisor Platform); OpenShell under WSL 2 |
| any OS | `agent-sandbox` (a client of a cluster), `e2b` (a client of a service) |

The Vaara cage needs Linux with AppArmor (Ubuntu, Debian, SUSE). On macOS
the Seatbelt cages are not confirmed from inside yet: their block says
`declared`. The VM cages are confirmed by the hypervisor on every
platform.

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

## The Vaara cage: hardening and egress

`vaara run` confines an agent with the `vaara-agent` AppArmor profile, a
cgroup per launch and the guard's decisions on opens and execs. Two more
layers are the operator's to switch on, for every launch:

```
vaara os-layer harden on
vaara os-layer egress api.anthropic.com '*.github.com' db.internal:5432
vaara os-layer egress --none      # locked, nothing allowed out
vaara os-layer egress --off       # unlocked
```

`harden` applies, in the child before it execs the agent:

- `no_new_privs`, so nothing in the tree gains privileges on exec.
- A seccomp filter (`vaara-seccomp/1`) that refuses with `EPERM` loading
  kernel modules, kexec, eBPF, perf, ptrace and cross-process memory
  access, mounts and the new mount API, swap, reboot, the kernel keyring,
  `userfaultfd`, `open_by_handle_at`, setting the clock, and io_uring,
  whose socket and file operations seccomp would not see. System calls of
  a foreign ABI (32-bit compat, x32) are refused as a whole.

`egress` implies `harden` and locks the network:

- Landlock (ABI 4, Linux 6.7 and later) lets the tree open TCP connections
  only to the egress proxy `vaara run` starts for the launch. The filter
  also refuses UDP and raw IP sockets, so nothing leaves by DNS or ICMP.
  With Landlock ABI 6 the tree cannot signal processes outside itself.
- The proxy (`HTTPS_PROXY`, `HTTP_PROXY`, set for the tree) understands
  `CONNECT` and plain HTTP and resolves names itself. A host passes when
  it matches an entry: `example.com` (ports 443 and 80), `*.example.com`
  (subdomains) or `host:port`. A name that resolves to loopback,
  link-local (cloud metadata), multicast or the unspecified address is
  refused unless the entry names that address literally.
- Every connection, allowed or refused, is a decision on the operator's
  trail (`egress.connect`), with the host, port and the reason, and the
  launch's cage block.
- A launch whose kernel cannot apply a layer that was asked for does not
  start.

With either switch on, the floor renders the move into `//tool` as a stack
on the harness profile (`Px -> &vaara-agent//tool`): AppArmor refuses a plain
transition under no_new_privs and allows a stacked one. A tool is then
confined by both profiles at once, which is no wider than `//tool` alone.
`vaara cage status --driver vaara-cage` reports both switches.

What these layers do not do:

- Landlock rules name ports, not addresses. The proxy's port number on a
  remote host is reachable as well; a server would have to listen there.
- The proxy decides by name and does not look inside TLS. A credential the
  model must not see goes through `vaara llm-proxy`, which holds the key.
- Binding a listening TCP port is left open, so development servers work.

## What the block does and does not establish

- `confirmed: true` with `basis: apparmor_label` means the deciding
  process carried the `vaara-agent` label when it decided. The label is
  set by the kernel at exec and cannot be dropped by the process.
- `confirmed: true` with `basis: seccomp_filter` means a seccomp filter
  and `no_new_privs` were on the deciding process. That is what OpenShell,
  the Codex sandbox and nono set, and it is also what a container
  runtime's default profile sets. The check confirms the kind of
  confinement the cage imposes; it does not identify that cage's filter
  from inside. The operator-side `enforcement_state` is the other half,
  and `vaara cage status` shows it.
- `basis: no_new_privs` (nono under its Landlock-only policy) is weaker
  still: Landlock cannot be queried from inside, only the flag it requires.
- `basis: bwrap_init` means pid 1 of the deciding process's pid namespace
  is bubblewrap, which sandbox-runtime leaves there. A process started by
  bubblewrap without a pid namespace would not show it, and the block
  would stay at `declared`.
- `basis: gvisor_kernel_log` is the strongest of the container checks:
  the kernel log read from inside is gVisor's own text, which a real
  kernel never produces.
- `basis: hypervisor_present` (Kata, Firecracker, microsandbox, E2B, apple/container)
  means the CPU reports a hypervisor underneath. It says the deciding
  process ran in a VM; it does not say which VM, and a Vaara run on any
  cloud VM shows the same flag. The declaration says which cage; the
  flag says only that a VM boundary was there.
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
