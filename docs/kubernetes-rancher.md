# Running Vaara on Kubernetes with Rancher

Vaara's model-endpoint proxy fronts an OpenAI-compatible or ollama server,
passes traffic through unchanged, and records every tool call the model
requests into a hash-chained audit trail. On Kubernetes that means one
workload sitting between your agents and the model service, and one volume
holding the evidence.

This page covers a cluster managed by SUSE Rancher running RKE2 or K3s. The
chart has nothing Rancher-specific in it, so any conformant cluster works, but
the versions in [supported-platforms.md](supported-platforms.md) are the ones
Vaara is documented against.

## What gets deployed

A StatefulSet with one replica, a ClusterIP Service, a ServiceAccount with no
API token mounted, and a PersistentVolumeClaim for the trail.

One replica is not a starting point to scale from. The audit trail is a hash
chain: every record binds to its predecessor, so a chain has exactly one
writer. Two pods appending to one volume interleave records and produce a
chain neither of them can verify. If you need more throughput, run more
proxies, each with its own chain and its own volume, and verify each one
independently. [multi-replica-deployment.md](multi-replica-deployment.md)
explains the reasoning in full.

## Install

```
helm install vaara ./deploy/helm/vaara \
  --namespace vaara --create-namespace \
  --set upstream.url=http://ollama.suse-ai.svc.cluster.local:11434
```

Point your agents at the Service instead of at the model:

```
base_url = http://vaara.vaara.svc.cluster.local:8788
```

Nothing is blocked yet. The default mode is observe: Vaara records what the
model asks for and forwards everything. Run it that way until the trail shows
you what your agents actually call.

Check that the proxy is up without involving the model:

```
kubectl -n vaara port-forward svc/vaara 8788:8788
curl -s localhost:8788/healthz
```

The `/healthz` route is answered by Vaara. Every other path is forwarded, so a
probe against `/` or `/health` reports the model server's state rather than
the proxy's.

## Storage

The trail is a SQLite file on the volume, so the StorageClass needs
ReadWriteOnce and it must not be backed by anything that reorders or delays
writes. On RKE2 the default `local-path` provisioner is fine for a single
node. Longhorn works and gives you replication, which is worth having when
the file is your evidence.

```
--set persistence.storageClass=longhorn --set persistence.size=20Gi
```

The claim is kept when the release is deleted. That is deliberate: uninstalling
a Helm release should not destroy an audit trail. Set
`persistence.retainOnDelete=false` if you have exported the trail and want the
volume to go with the release.

Export before you need it. The export is signed, so it needs a key, which
means this step assumes you have turned on signing (see below):

```
kubectl -n vaara exec vaara-0 -- vaara trail export \
  --db /var/lib/vaara/audit.db \
  --out /var/lib/vaara/handoff.zip \
  --key /etc/vaara/signing/signing_key.pem
```

## Turning on enforcement

Enforce mode gates instead of observing. Denied tool calls are rewritten out
of the model's response and escalations block on the approvals handshake.

An enforce deployment with no allow list gates every tool call, and clients
see their tools disappear. Nothing is damaged and the trail records all of
it, but the session is unusable. The chart refuses to render that
configuration rather than let you find out in a crash loop.

Start wide and tighten:

```
helm upgrade vaara ./deploy/helm/vaara \
  --namespace vaara \
  --set proxy.mode=enforce \
  --set 'proxy.allow[0]=mcp__*'
```

## Signed receipts

Observe and enforce both record. Signing also emits an attestation and receipt
pair per chat call, which is the part a third party can verify without access
to your cluster.

The chart never generates a key. A key minted in a template would rotate on
every upgrade, and every receipt signed by the old one would stop verifying.
Create it yourself and hand the chart a Secret:

```
vaara keygen --dev --out signing_key.pem
kubectl -n vaara create secret generic vaara-signing \
  --from-file=signing_key.pem=signing_key.pem

helm upgrade vaara ./deploy/helm/vaara --namespace vaara \
  --set signing.enabled=true \
  --set signing.existingSecret=vaara-signing
```

`vaara keygen --dev` produces a development key. For anything whose receipts
you intend to show someone, use a key from your own KMS or HSM and mount it
the same way. [signing-keys.md](signing-keys.md) covers the options.

## Keeping the model reachable only through Vaara

A governance layer an agent can route around governs nothing. Once Vaara is
in place, restrict the model service so the proxy is the only client:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-through-vaara-only
  namespace: suse-ai
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: ollama
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: vaara
          podSelector:
            matchLabels:
              app.kubernetes.io/name: vaara
```

RKE2 ships Canal, which enforces NetworkPolicy. Confirm your CNI does before
relying on this.

## Where Vaara sits next to SUSE AI

SUSE AI serves models in-cluster through Ollama, vLLM and LiteLLM, and SUSE
Observability reports on how those workloads are behaving. Vaara answers a
different question: what did the agent ask the model to do, and can someone
outside this cluster check the answer. It runs in front of the model endpoint
and produces a record that verifies on its own.

Both positions are useful in the same cluster. A deployment that governs an
in-cluster model and also governs what that model can reach on the way out
runs the proxy on both sides, each with its own trail.

## Verifying the trail

Verification runs over a signed export rather than over the live database, so
the party checking it never needs access to your cluster. Copy the zip out and
verify it with the auditor CLI, which is the same command a third party would
run:

```
kubectl -n vaara cp vaara/vaara-0:/var/lib/vaara/handoff.zip ./handoff.zip
vaara-audit verify ./handoff.zip
```

Add `--pubkey` if the checker holds the public key separately instead of
taking the copy inside the zip.

Verification is the point of the whole arrangement, so run it on a schedule
and not only when something looks wrong. A chain that fails verification tells
you the file was altered, which is information you want early.

## Uninstalling

```
helm uninstall vaara --namespace vaara
```

The PersistentVolumeClaim stays behind by default and so does the trail.
Delete it explicitly once you have the export:

```
kubectl -n vaara delete pvc data-vaara-0
```
