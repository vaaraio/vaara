# Supported platforms

Vaara 2.15.0, Helm chart 0.2.0.

Vaara is a Python package with three runtime dependencies (`cryptography`,
`rfc8785` and `cbor2`, which sign and canonicalise receipts) and a container
image built on SUSE's SLE Base Container Image. This page lists what it runs
on and, separately, what it has been run on.

## Package

| Requirement | Supported |
| --- | --- |
| Python | 3.10, 3.11, 3.12, 3.13 |
| Runtime dependencies | `cryptography`, `rfc8785`, `cbor2` |
| Operating system | any platform CPython supports, including Red Hat Enterprise Linux 9 (see below) |

The proxy needs the `proxy` extra (fastapi, uvicorn, httpx). Signed
attestation and receipt pairs use the same three packages the base install
carries; `vaara[attestation]` still installs and names them. The container
image ships the proxy extra as well.

## Red Hat Enterprise Linux

| | |
| --- | --- |
| Supported | Red Hat Enterprise Linux 9 (x86_64), Python 3.12 from Red Hat's packages |
| Tested on every change | the full test suite runs in CI inside Red Hat's Universal Base Image 9 (`ubi9/python-312`), the `tests-rhel` job in [ci.yml](../.github/workflows/ci.yml) |
| Tested on a RHEL 9 system | RHEL 9.8 with SELinux enforcing, installed from PyPI as a non-root user; full test suite passing |

Install on RHEL 9: `sudo dnf install python3.12 python3.12-pip`, then
`python3.12 -m venv .venv && .venv/bin/pip install vaara`.

## Container image

| | |
| --- | --- |
| Image | `ghcr.io/vaaraio/vaara` |
| Base | `registry.suse.com/bci/python:3.12` |
| Architectures | linux/amd64, linux/arm64 |
| Runs as | UID 10001, non-root, read-only root filesystem |
| Exposed port | 8788 |

The base image is SUSE's own BCI, which needs no subscription to pull or
redistribute.

## Red Hat Enterprise Linux

| Requirement | Supported |
| --- | --- |
| Red Hat Enterprise Linux | 9 |
| Python | 3.12, the `python3.12` package from the RHEL 9 repositories |
| Container base | Red Hat Universal Base Image 9 |
| CI | the `tests-rhel` job runs the suite in `registry.access.redhat.com/ubi9/python-312` |

Install from PyPI into a virtual environment made with `python3.12 -m venv`.
For containers, [packaging/ubi9/README.md](../packaging/ubi9/README.md) builds
`vaara proxy` on UBI 9, runs it as a non-root user in group 0, and records a
Red Hat preflight run that passed all 10 checks. That image is built from the
Containerfile; the published `ghcr.io/vaaraio/vaara` image stays on SUSE BCI.

Rerunning the test suite on RHEL 9 needs Node.js 20 or later for the browser
verifier tests (the `nodejs:20` module stream). RHEL 9's default Node.js 16
has no global `crypto`, so those tests fail there for reasons that have
nothing to do with Vaara; without Node installed at all they skip.
OpenShift is not listed: no run on an OpenShift cluster has been recorded.

## Kubernetes

| Requirement | Supported |
| --- | --- |
| Kubernetes | 1.27 and later |
| Distributions | RKE2, K3s |
| Management | SUSE Rancher, or none |
| Helm | 3.8 and later |
| Storage | a StorageClass providing ReadWriteOnce |
| CNI | any; NetworkPolicy support needed only for the model-isolation step |

Deployment instructions are in
[kubernetes-rancher.md](kubernetes-rancher.md).

The chart deploys a StatefulSet with one replica and does not offer a replica
count. The audit trail is a hash chain with exactly one writer, so a second
pod appending to the same volume produces a chain neither pod can verify.
[multi-replica-deployment.md](multi-replica-deployment.md) covers what to do
instead when one proxy is not enough.

## Filesystems for the audit trail

SQLite's WAL journal coordinates readers and writers through a shared-memory
segment, and SQLite's own documentation states that WAL needs coherent shared
memory and working file locking. Several filesystems people mount a home
directory from provide neither. On those the write still goes through and the
database corrupts.

Vaara reads `/proc/mounts` when it opens a trail and, if the file sits on one
of these, uses the DELETE journal instead:

`9p`, `afpfs`, `ceph`, `cifs`, `glusterfs`, `lustre`, `ncpfs`, `nfs`, `nfs4`,
`smb2`, `smb3`, `smbfs`, `vboxsf`, `virtiofs`, `vmhgfs`, and any `fuse*`
filesystem.

That covers the common cases: a container with the host home directory bind
mounted in (Docker Desktop, Rancher Desktop, Colima, Lima, OrbStack all use
virtiofs or FUSE), WSL2 writing to `/mnt/c`, and NFS or SMB network homes.

| | |
| --- | --- |
| Detection | Linux only; `/proc/mounts` is the only portable source of a filesystem type |
| Elsewhere | macOS and Windows keep WAL, since the type is not readable without platform calls |
| Override | `VAARA_TRAIL_JOURNAL_MODE=wal` or `=delete`, in either direction |

Switching an existing trail off WAL needs a brief exclusive lock, and SQLite may
answer without it by leaving the file in the mode it is already in. Vaara reads
the mode back and retries. If the trail is still in WAL afterwards, the Claude
Code session-start hook names the file and the filesystem, because a log line
from a hook process reaches nobody. A trail that stays in WAL on one of these
filesystems is intact and recording, and it corrupts later.

DELETE journalling costs concurrency. Readers block writers where WAL would let
them run together, and durability is unaffected. For a trail written by one
process, which is what the hash chain requires anyway, the difference does not
show.

If the trail lives on a network or virtual filesystem, the better answer is to
move it. In a container, put it on a named volume rather than a bind mount, so
it sits on the container runtime's own filesystem with real POSIX locking.

## Operating systems tested

The full test suite has been run on these hosts, with Python and every
system library taken from the vendor's own repositories. The CI legs on
Ubuntu, macOS, Windows and Red Hat's UBI 9 are defined in
`.github/workflows/ci.yml` and are not repeated here.

| System | Python | Vaara | Date | Result |
| --- | --- | --- | --- | --- |
| SUSE Linux Enterprise Server 15 SP6 | 3.12.12 (`python312` from the SLE BCI repository) | e9de66ee | 2026-10-22 | 6157 passed, 0 failed |
| Red Hat Enterprise Linux 9.8 | 3.12.15 (`python3.12` from Red Hat's UBI 9 repositories) | 0cdb8b66 | 2026-10-18 | 5996 passed, 0 failed |

The SLES 15 SP6 run used SUSE's `registry.suse.com/bci/bci-base:15.6` image,
packages from the public SLE BCI repository only (Python 3.12, SQLite 3.50.2,
Node.js 20 for the browser verifier tests), no registration, the suite run
as a non-root user. `pip install vaara==2.15.0` from PyPI installed and ran on
the same system. The skipped tests need optional extras or tools that were
not installed, and every one is on the skip list CI accepts.

The RHEL 9.8 run used Red Hat's own KVM guest image with SELinux enforcing,
not registered with subscription-manager, the suite run as a non-root user.
Python 3.12 came from Red Hat's signed UBI 9 repositories and SQLite was the
system 3.34.1. The guest had no Node.js for the main run, so the browser
verifier tests skipped; with Node.js 20 from the `nodejs:20` module stream
those two files then ran with 27 passed. `pip install vaara==2.14.0` from
PyPI installed and ran on the same system. Commit 0cdb8b66 is part of
release 2.15.0.

A row here is a test run, not an end-to-end deployment; those go in the next
table.

## Verified releases

A row here means the listed Vaara version was installed on the listed platform
versions and the deployment was exercised end to end: an agent routed through
the proxy, tool calls recorded, the trail exported and the export verified.

| Vaara | Chart | Platform | Verified on |
| --- | --- | --- | --- |
| | | | |

The table is empty. Vaara has run on Rancher in development for years, but no
run has been recorded against a named platform version in the form above, and
this page reports only what has. Rows are added as runs are completed.

## Support

Commercial support for Vaara on the platforms listed here is available from
the maintainer at hello@vaara.io. The software is AGPL-3.0-or-later; see
[LICENSING.md](../LICENSING.md).

Security issues and serious bugs are disclosed in the
[changelog](../CHANGELOG.md) with the release that fixes them.
