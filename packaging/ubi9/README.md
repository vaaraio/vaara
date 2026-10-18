# Vaara on UBI 9

`Containerfile` builds `vaara proxy` on Red Hat Universal Base Image 9 for RHEL and OpenShift
deployments. The published image (`ghcr.io/vaaraio/vaara`, root `Dockerfile`) is built on SUSE
BCI and is unchanged.

## Build and run

From the repository root:

    podman build -f packaging/ubi9/Containerfile \
      --build-arg VERSION=2.14.0 --build-arg RELEASE=1 \
      --build-arg VCS_REF=$(git rev-parse --short HEAD) \
      -t vaara:2.14.0-ubi9 .
    podman run -p 8788:8788 -v vaara-trail:/var/lib/vaara vaara:2.14.0-ubi9 \
      proxy --listen 0.0.0.0:8788 --upstream http://ollama:11434 \
      --trail /var/lib/vaara/audit.db

The image runs as UID 10001 in group 0. `/var/lib/vaara` is owned by group 0 and group
writable, so OpenShift's restricted-v2 SCC can run it under an arbitrary UID.

## Red Hat preflight

Preflight checks an image in a registry, so push to a local one first:

    podman run -d --rm --name pf-registry -p 127.0.0.1:5000:5000 docker.io/library/registry:2
    podman push --tls-verify=false vaara:2.14.0-ubi9 localhost:5000/vaara:2.14.0-ubi9
    preflight check container localhost:5000/vaara:2.14.0-ubi9 --insecure
    podman stop pf-registry

Result on 2026-10-18 with preflight 1.21.2 (commit 63b376df), image `vaara:2.14.0-ubi9`
(5 layers, 239 MB): PASSED, 10 of 10 checks, no failures, no errors.

| Check | Result |
|---|---|
| HasLicense | passed |
| HasUniqueTag | passed |
| LayerCountAcceptable | passed |
| HasNoProhibitedPackages | passed |
| HasRequiredLabel | passed |
| HasNoProhibitedLabels | passed |
| RunAsNonRoot | passed |
| HasModifiedFiles | passed |
| BasedOnUbi | passed |
| HasProhibitedContainerName | passed |

The same image starts the proxy under an arbitrary UID (`--user 54321:0`), writes its trail to
`/var/lib/vaara` and answers `GET /healthz` with 200.

`tests/test_ubi9_containerfile.py` reads these checks off the Containerfile as text, so CI fails
on a change that would fail preflight without building the image.
