# Supported platforms

Vaara 1.67.1, Helm chart 0.1.0.

Vaara is a Python package with no runtime dependencies and a container image
built on SUSE's SLE Base Container Image. This page lists what it runs on and,
separately, what it has been run on.

## Package

| Requirement | Supported |
| --- | --- |
| Python | 3.10, 3.11, 3.12, 3.13 |
| Runtime dependencies | none for the base install |
| Operating system | any platform CPython supports |

The proxy needs the `proxy` extra (fastapi, uvicorn, httpx). Signed
attestation and receipt pairs need the `attestation` extra (cbor2,
cryptography, rfc8785). The container image ships both.

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
