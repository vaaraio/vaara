# VATE AL2 external-SUT reproducers

Two standalone scripts backing the answers in
[discussion #502](https://github.com/vaaraio/vaara/discussions/502), where
Takao Sato asked for a bounded classification of VATE conformance cases
against shipped Vaara code.

They consume one pinned VATE fixture,
`conformance/al2-vate-v0.3/fixtures/status-stale-just-over-boundary-context.json`
from `Poke-nushi/Verifiable-Agent-Trust-Envelope` at tag `v0.4.0`,
sha-256 `5045e9bcf3711e2de9431d50befb03662c0c868988dcccd616d2392375a545b2`.
Each script verifies that digest before it runs and refuses on a mismatch.

Neither script changes Vaara. Both call the installed package only, and there
is no VATE adapter here: the mapping from a native verdict to a VATE outcome
stays in the discussion thread, in prose, where it can be disagreed with.

## Running them

```
python scripts/vate-al2/ttl_bridge.py            path/to/context.json
python scripts/vate-al2/freshness_admission.py   path/to/context.json
```

The fixture path is required. Fetch it with:

```
curl -sSL -o /tmp/vate-ctx.json \
  https://raw.githubusercontent.com/Poke-nushi/Verifiable-Agent-Trust-Envelope/v0.4.0/conformance/al2-vate-v0.3/fixtures/status-stale-just-over-boundary-context.json
```

## What each one covers

`ttl_bridge.py` runs the bridge exactly as it was proposed in the ask:
`source_issued_at -> asserted.iat`, `max_age_seconds -> expSeconds`,
`checked_at -> now`, and clock skew as a parameter. It calls
`vaara.credential.verify_grant` directly, because the credential TTL and the
VATE status freshness bound are different objects and the bridge is the only
thing joining them.

`freshness_admission.py` runs the surface that is the same object,
`RevocationRegistry.status()`, and then carries it one step further into an
admission decision. That second half is the part the 2 September run did not
do. It shows three layers, and they do not all pin the same thing:

- the registry verdict at a fixed clock, where the 300/301 boundary is exact;
- `verify_grant` at a fixed clock, which reaches `revocation_stale` through
  `establishes_current` when the deployment states a bound;
- `CredentialGateway`, which forwards the bound but exposes no injectable
  clock, so the verdict is reachable there while the boundary is not.

## Scaffolding, and why it is here

Both scripts build a grant whose only reachable failure is the one under test.
HS256 with a literal test secret, `argsCommitment` recomputed from the runtime
arguments, tool name and tenant matching, and the binding digest supplied to
the verifier as known. None of that is VATE. It is stated in the output of
each run so a reader can see what was held constant.

The HS256 secret is a hard-coded test value protecting nothing.
