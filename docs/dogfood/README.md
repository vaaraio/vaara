# Our marketing runs under the same gate we sell

We use AI agents in our own marketing. Instead of putting a disclosure sentence in a footer, we run that pipeline under Vaara, the same gate and trail we sell, and publish the evidence. This directory contains the signed trail from a real run of that pipeline ([marketing-evidence.zip](marketing-evidence.zip)), the exact policy it was governed by ([marketing_policy.yaml](marketing_policy.yaml)), and the public key to verify it against ([vaara-marketing.pub](vaara-marketing.pub)). You do not have to take any sentence on this page on trust; the check is one command, offline.

## The run

The pipeline that drafted our 1.26.0 release posts ran governed. Every tool call was scored and decided against the policy before it ran, and every decision landed in a signed, hash-chained trail:

```
ALLOW    risk=0.113  content.read  CHANGELOG.md
ALLOW    risk=0.111  content.read  docs/logs-vs-evidence.md
ALLOW    risk=0.160  fs.write_file  drafts.manifest.json
ESCALATE risk=0.258  publish.social_post  linkedin
ESCALATE risk=0.258  publish.social_post  blog
ESCALATE risk=0.237  publish.social_post  hackernews
ESCALATE risk=0.257  publish.social_post  reddit
```

The reads are the grounding step: every claim in a draft traces to the changelog or the docs. The write records the SHA-256 of the draft artifact, so the thing a human reviewed and the thing that got recorded are provably the same bytes. And the four publish calls did not run. The policy sets the escalate threshold for `publish.social_post` to 0.0, so no publish action can ever pass the gate without a human. That is not a workflow convention we promise to follow; it is a property of the policy document in this directory, and the trail shows it operating.

## Verify it yourself

```bash
pip install 'vaara[export]'
vaara-audit verify marketing-evidence.zip --pubkey vaara-marketing.pub
```

Verification is offline, from the bundle's bytes and the published key. If we had edited a single byte of the trail after the fact, the check would fail and name the reason. Without `--pubkey` the tool still proves integrity and prints a warning that identity is unauthenticated, rather than implying more than it checked.

Scope, stated honestly: the key is a file key generated for this pipeline, not an HSM identity. What a passing check proves is that the trail was signed by the holder of this key and has not been altered since, record by record. What it cannot prove is that the writer recorded the truth at write time; the mitigations for that, and the limits of tamper-evidence generally, are in [tamper-evident-audit-trail.md](../tamper-evident-audit-trail.md).

## Why publish this

A governance company's marketing is a live test of its own argument. We tell buyers that "we have a human review step" is a claim, and that a signed trail of the review gate operating is evidence. So here is ours. Every post we publish from this pipeline escalated through that gate, and the record of it is yours to check. This also serves as our AI-use disclosure: the drafting is done by AI agents, the publishing is done by a person, and the boundary between the two is enforced by policy and recorded, not asserted.

If you want to run the same loop on your own machine, [examples/prove-it-yourself/](../../examples/prove-it-yourself/) produces a trail, verifies it, and catches a forged byte, in one file.
