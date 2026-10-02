# Vaara

Vaara is a governance and receipt layer for AI agents. It sits in front of an
in-cluster model endpoint, passes traffic through unchanged, and records every
tool call the model requests into a hash-chained audit trail that anyone can
verify without trusting the machine that wrote it.

## What it deploys

A StatefulSet with one replica, a Service, a ServiceAccount, and a
PersistentVolumeClaim for the trail. The chart offers no replica count. An
audit trail is a hash chain with exactly one writer, so a second pod appending
to the same volume produces a chain neither pod can verify.

## Before you install

Point `upstream.url` at your model service inside the cluster. The default is
`http://ollama.suse-ai.svc.cluster.local:11434`. Vaara decides what leaves, so
it should be the only workload that can reach the model directly.

Put your policy in `proxy.policy`: default thresholds, per-tool overrides and
sequence patterns, the format `vaara policy validate` checks. Without one the
built-in defaults decide, and they hold back very little at first.

Leave `proxy.mode` on `observe` for the first install. Observe records every
tool call, changes nothing, and records what your policy would have blocked;
`vaara trail shadow-report` sums it up. Enforce also gates: denied calls are
rewritten out of the response, and escalations wait on a human decision.
Enforce needs `proxy.policy`, `proxy.allow` or the approvals handshake, and
the chart refuses to render it with none of them.

You need a StorageClass providing ReadWriteOnce. On RKE2 that is usually
`local-path`. The chart keeps the volume on uninstall, because the trail is the
evidence. Set `persistence.retainOnDelete: false` once you have exported it.

## Signing is off by default

Signed attestation and receipt pairs need a key, and the chart will not
generate one. A template-generated key rotates on every upgrade and lands in
the release secret. Create the Secret yourself and name it in
`signing.existingSecret`. It holds two files: `signing_key.pem`, an EC P-256
key from `vaara keygen --attest` that the proxy signs receipts with, and
`trail_key.pem`, an Ed25519 key from `vaara keygen --dev` that signs trail
exports.

## Requirements

Kubernetes 1.27 or later, Helm 3.8 or later. Full detail, including the
filesystems the trail must not live on, is in
[docs/supported-platforms.md](https://github.com/vaaraio/vaara/blob/main/docs/supported-platforms.md).

## Licence and support

AGPL-3.0-or-later. Commercial support is available from the maintainer at
hello@vaara.io.
