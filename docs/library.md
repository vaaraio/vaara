# Library capabilities

Some of what [capabilities.md](capabilities.md) lists has no command of its
own. It is reached from Python, inside the deployer's code, at the point
where the thing being recorded happens: a read leaving a data store, a key
being rotated, a scorer that lives on another host. This page is the door
to those. Every block below runs under `tests/test_docs_library_examples_run.py`
in a throwaway home directory with no network, so what it prints is what
you get.

A block that needs an extra says so on its first line. The default trail is
`~/.vaara/trail/audit.db`; the blocks open a database in the current
directory instead so they are easy to inspect afterwards.

## Access records

A read is an event. `record_access` writes who opened what, on whose
behalf, on what basis, and a digest of what came back. The payload itself
never reaches the record: compute the digest where the result leaves the
store. `record_access_outcome` closes the access with what came of it,
including nothing, and the three query helpers tell the three states
apart: closed with an action, closed with none, never closed.

```python
from vaara.audit.access import (
    accesses_closed_with_no_action,
    accesses_with_no_recorded_outcome,
    digest_returned,
    find_accesses,
    follow_ups,
    record_access,
    record_access_outcome,
)
from vaara.audit.sqlite_backend import SQLiteAuditBackend

trail = SQLiteAuditBackend("audit.db").load_trail()

rows = b'[{"visit": "2026-08-02", "unit": "KYS"}]'  # what the store returned; never stored
access_id = record_access(
    trail,
    subject="patient/4711/kayttoloki",
    accessed_by="agent/log-search@istekki",
    on_behalf_of="person/hetu-pseudonym/9f2c",
    basis="hoitosuhteen tarkistus",
    returned_sha256=digest_returned(rows),
    returned_count=1,
    authorisation_ref="ticket/2026-0812",
    channel="mcp",
)

# Later, by a named person: the read led nowhere.
record_access_outcome(
    trail, access_id, outcome="no clinical need found", action_taken=False,
    by="nurse/anne.k@kys",
)

records = trail.snapshot()
opened = find_accesses(records, on_behalf_of="person/hetu-pseudonym/9f2c")
print(len(opened), opened[0].data["parameters"]["fields_present"])
print(len(follow_ups(records, access_id)))                 # 1: the outcome record
print(len(accesses_closed_with_no_action(records)))        # 1: somebody said so
print(len(accesses_with_no_recorded_outcome(records)))     # 0: nothing is still open
```

## Store queries

`AuditTrail` answers from memory. `SQLiteAuditBackend` answers from the
file, which is what a dashboard or a report process opens without loading
the whole chain.

```python
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline

backend = SQLiteAuditBackend("audit.db")
pipeline = InterceptionPipeline(trail=backend.load_trail())
result = pipeline.intercept(agent_id="ops-bot", tool_name="fs.read_file",
                            parameters={"path": "/var/log/app.log"})

print(len(backend.query_by_action(result.action_id)))        # request, score, decision, ...
print(len(backend.query_by_agent("ops-bot")))
print(len(backend.query_by_event_type(EventType.DECISION_MADE)))
print(len(backend.query_by_regulation("eu_ai_act")))         # records tagged to that domain
print(len(backend.query_blocked()))                           # denied actions, newest first
```

## Store diagnostics

Four readings an operator wants before a backup, in a health check, or on
a dashboard. `head_problem` is the tail-only version of the chain check
`load_trail` runs in full; `corrupt_query_rows` is the read-side counter
that matches `AuditTrail.skeleton_records` on the write side.

```python
from vaara.audit.sqlite_backend import SQLiteAuditBackend

backend = SQLiteAuditBackend("audit.db")
backend.load_trail()

print(backend.journal_mode)              # "wal" on a filesystem that supports it
print(backend.head_problem())            # None: the recorded chain head is stored
print(backend.corrupt_query_rows)        # rows the query path had to skip, so far
total, done = backend.checkpoint_wal()   # before a backup, or a reader that lacks WAL
print(total >= done)
```

## Key lifecycle events

A custodian key rotation, revocation or addition is written as a chained
record, so an anchor placed after it pins the event in time and a
compromised key cannot move it. `vaara verify-retained` and the signed
export verifier show these inline with the evidence they govern.

```python
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType

trail = SQLiteAuditBackend("audit.db").load_trail()

record_id = trail.record_key_lifecycle(
    "revoked", "3f9c2a7b1d4e5f60718293a4b5c6d7e8",
    threshold_k=2, signers_n=3,
    reason="suspected compromise of the laptop it lived on",
    actor="custodian/maria",
)
event = trail.get_records_by_type(EventType.KEY_LIFECYCLE)[0]
print(event.record_id == record_id, event.data["action"], event.data["fingerprint"])
```

## Policy override records

When a person overrides a decision outside the review queue, say by
re-running a denied action under a change ticket, write the override so
the trail says who, why, and what the decision was before and after.

```python
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline

trail = SQLiteAuditBackend("audit.db").load_trail()
pipeline = InterceptionPipeline(trail=trail)
result = pipeline.intercept(agent_id="deploy-bot", tool_name="infra.terminate",
                            parameters={"instance": "i-0abc"})

trail.record_policy_override(
    action_id=result.action_id, agent_id="deploy-bot", tool_name="infra.terminate",
    override_reason="change CHG-4411, approved in the CAB",
    overrider="operator/jussi",
    original_decision=result.decision, new_decision="allow",
)
override = trail.get_records_by_type(EventType.POLICY_OVERRIDE)[0]
print(override.action_id == result.action_id, override.data["overrider"])
```

## Post-quantum signing

Needs `vaara[pq]` (dilithium-py). `MLDSA65Signer` is a drop-in for
`Ed25519Signer` anywhere a signer is taken: the signed export, the
threshold export, the attribute attestation. The export manifest records
`signature_algorithm`, and `verifier_for` dispatches on it so a verifier
written against the manifest does not have to know the algorithm in
advance.

```python
# needs: vaara[pq]
from vaara.audit.export import export_signed
from vaara.audit.signer import MLDSA65Signer, verifier_for
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline

signer, public_key = MLDSA65Signer.generate()
message = b"the bytes a manifest covers"
verifier = verifier_for("ML-DSA-65", public_key)
print(verifier.verify(message, signer.sign(message)))       # True
print(verifier.verify(b"other bytes", signer.sign(message)))  # False
print(len(signer.public_key_fingerprint()))

trail = SQLiteAuditBackend("audit.db").load_trail()
InterceptionPipeline(trail=trail).intercept(agent_id="ops-bot", tool_name="fs.read_file")
export = export_signed(trail, "handoff.zip", signer=signer)
print(export.manifest["signature_algorithm"], export.chain_intact)   # ML-DSA-65 True
```

## Composite scoring

`CompositeScorer` runs Vaara's scorer beside any object with
`evaluate(context) -> dict` and a `name`, and drops into the pipeline in
place of `AdaptiveScorer`. The strongest member decision wins; the
winner's reason and thresholds are what the trail records about the
decision, so the receipt states the operating point that was actually in
force.

```python
from vaara.pipeline import InterceptionPipeline
from vaara.scorer.adaptive import AdaptiveScorer
from vaara.scorer.composite import CompositeScorer


class ChangeFreeze:
    """A peer that denies production changes during a freeze window."""

    name = "change-freeze"

    def evaluate(self, context):
        frozen = context.get("tool_name", "").startswith("infra.")
        return {
            "decision": "deny" if frozen else "allow",
            "point_estimate": 0.95 if frozen else 0.05,
            "reason": "change freeze until 2026-01-07" if frozen else "outside the freeze scope",
            "threshold_allow": 0.3,
            "threshold_deny": 0.7,
            "raw_result": {"point_estimate": 0.95 if frozen else 0.05},
        }


scorer = CompositeScorer([AdaptiveScorer(), ChangeFreeze()], combine="max")
pipeline = InterceptionPipeline(scorer=scorer)

result = pipeline.intercept(agent_id="deploy-bot", tool_name="infra.terminate")
print(result.decision, "|", result.reason)         # deny | change freeze until 2026-01-07
result = pipeline.intercept(agent_id="deploy-bot", tool_name="fs.read_file")
print(result.decision)                             # whatever Vaara's own scorer says
```

## A remote scorer

`ExternalScorer` posts the context to another service's `/v1/score`, in
the wire shape [openapi.yaml](openapi.yaml) publishes, and reads the
answer back into the same dict the composite expects: decision, reason,
thresholds, signals, alpha. Any transport failure is a deny with
`scorer_unavailable`, never an allow by silence, and an answer with no
thresholds is refused the same way because a decision with no stated
operating point cannot be put on a receipt.

```python
from vaara.pipeline import InterceptionPipeline
from vaara.scorer.adaptive import AdaptiveScorer
from vaara.scorer.composite import CompositeScorer
from vaara.scorer.composition import ExternalScorer

peer = ExternalScorer(
    "https://scoring.internal.example/v1/score",
    name="risk-team",
    timeout=2.0,
    headers={"Authorization": "Bearer <token>"},
)
pipeline = InterceptionPipeline(scorer=CompositeScorer([AdaptiveScorer(), peer]))

# The peer is not reachable from here, so this is the fail-closed path.
result = pipeline.intercept(agent_id="ops-bot", tool_name="fs.read_file")
print(result.decision, "|", result.reason)
```

## Trained per-step gates

Needs `vaara[gate]`. `TrainedGateScorer`, `MCDropoutGateScorer` and
`StackedGateScorer` decide per action step from a bundle fitted offline:
a gradient-boosted bootstrap ensemble, an MC-dropout network, and a
logistic stack over both, each deciding by conformal set membership. Vaara
ships no bundle and nothing in the tree trains one; the module docstrings
give the bundle format. Each scorer plugs into the pipeline directly or as
a composite member, and reads `history_pairs` and `proposed_action_text`
off the context.

```python
# needs: vaara[gate]
from vaara.scorer.mc_dropout_gate import MCDropoutGateScorer
from vaara.scorer.stacked_gate import StackedGateScorer
from vaara.scorer.trained_gate import TrainedGateScorer

for scorer in (TrainedGateScorer, MCDropoutGateScorer, StackedGateScorer):
    try:
        gate = scorer("bundles/gate.joblib")
    except FileNotFoundError as exc:
        print(exc)          # no bundle here: the scorer refuses to start without one
```

## Cold-start calibration

`TraceGenerator` writes synthetic benign, careless and adversarial traces
with ground-truth outcomes and feeds them through a pipeline, so the
conformal interval and the expert weights are calibrated before the first
real agent connects. Traces can be saved and reloaded, so one set serves
every replica.

```python
from vaara.pipeline import InterceptionPipeline
from vaara.sandbox.trace_gen import TraceGenerator

generator = TraceGenerator(seed=7)
traces = generator.generate(n_traces=20)
print(max(t.max_outcome for t in traces), min(t.mean_outcome for t in traces))  # the ground truth
generator.save_jsonl(traces, "traces.jsonl")

pipeline = InterceptionPipeline()
stats = generator.pre_calibrate(pipeline, TraceGenerator.load_jsonl("traces.jsonl"))
print(stats["traces_processed"], stats["steps_processed"], stats["is_calibrated"])
```

## Requirements of your own

The compliance engine ships with the EU AI Act, DORA and SOC 2 bundles.
A deployer adds a requirement for its own obligation, in its own
vocabulary, and the report assesses it beside the others.

```python
from vaara.audit.trail import EventType
from vaara.compliance.engine import ComplianceEngine, RegulatoryRequirement
from vaara.pipeline import InterceptionPipeline
from vaara.taxonomy.actions import RegulatoryDomain

engine = ComplianceEngine()
engine.add_requirement(RegulatoryRequirement(
    domain=RegulatoryDomain.SOC2,
    article="CC6.8",
    title="Unauthorized software is detected",
    description="Every install the agent attempted was scored before it ran.",
    evidence_event_types=(EventType.RISK_SCORED, EventType.DECISION_MADE),
    min_evidence_count=5,
    staleness_hours=24 * 30,
))
print([r.article for r in engine.requirements_by_domain(RegulatoryDomain.SOC2)][-1])

pipeline = InterceptionPipeline(compliance=engine)
pipeline.intercept(agent_id="dev-bot", tool_name="pkg.install", parameters={"name": "left-pad"})
report = pipeline.run_compliance_assessment(system_name="build agent", system_version="1.0")
print([a.status.value for a in report.articles if a.requirement.article == "CC6.8"])
```

## The credential gateway from your own code

The MCP proxy builds a `CredentialGateway` for the tools it constrains.
A tool server that is not behind the proxy can hold the same gateway and
ask it the same question: is this call covered by a grant. With a
`ReceiptSigner` it also mints the authorization receipt, for a refusal as
much as for an allow, so the non-action leaves a portable proof too.

```python
from pathlib import Path

from vaara.credential import CredentialGateway, ReceiptSigner
from vaara.integrations._mcp_attest import build_attest_emitter

key = Path("attest.key")
key.write_bytes(b"\x11" * 32)             # the broker's HS256 secret; keep it out of the repo
receipts = Path("receipts")
receipts.mkdir()

# The broker side: an attestation and a grant for one call.
broker = build_attest_emitter(signing_key_path=key, receipts_dir=receipts,
                              upstream_commands={"default": ["echo"]})
attestation, counter = broker.emit_attestation(
    tool_name="read_file", arguments={"path": "/srv/report.csv"},
    upstream_name="default", tenant_id="t1",
)
grant = broker.emit_grant(attestation=attestation, counter=counter, tool_name="read_file",
                          upstream_name="default", tenant_id="t1").to_dict()

# The tool server side: the same key verifies, and a signer mints the receipt.
gateway = CredentialGateway(
    verifying_material=b"\x11" * 32, receipts_dir=receipts, expected_tenant="t1",
    signer=ReceiptSigner(signing_material=b"\x11" * 32, iss="tool-server", sub="t1/default",
                         secret_version="k1", alg="HS256"),
)
params = {"_meta": {"vaara/credential": grant}}
verdict, receipt = gateway.authorize_and_receipt(params, tool_name="read_file",
                                                 arguments={"path": "/srv/report.csv"})
print(verdict.ok, receipt is not None)
verdict, receipt = gateway.authorize_and_receipt(params, tool_name="read_file",
                                                 arguments={"path": "/etc/shadow"})
print(verdict.ok, verdict.reason, receipt is not None)    # False, the reason, True
```

## Transparency log roots and the HCS-27 wire

`InProcessTransparencyLog` is the RFC 6962 Merkle log the trail's
checkpoints are built over. `root_at` gives the root of any earlier tree
size, which is what a consistency proof is about. The `hcs27` helpers
serialise inclusion and consistency proofs in the field names and
encodings the HCS-27 SDK verifies, so a checkpoint written here is checked
there.

```python
from vaara.attestation.transparency_log import InProcessTransparencyLog
from vaara.audit.hcs27 import (
    b64,
    canonical_json,
    consistency_proof_wire,
    entry_for_record,
    inclusion_proof_wire,
    root_for_entries,
)
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline

trail = SQLiteAuditBackend("audit.db").load_trail()
pipeline = InterceptionPipeline(trail=trail)
for path in ("/a", "/b", "/c"):
    pipeline.intercept(agent_id="ops-bot", tool_name="fs.read_file", parameters={"path": path})

entries = [entry_for_record(r) for r in trail.snapshot()]
log = InProcessTransparencyLog()
for entry in entries:
    log.append(canonical_json(entry))

root = log.root_at(len(entries))
print(root == root_for_entries(entries))             # True: the same bytes the SDK computes
earlier = log.root_at(4)
inclusion = inclusion_proof_wire(entry=entries[2], proof=log.inclusion_proof(2), root_hash=root)
consistency = consistency_proof_wire(proof=log.consistency_proof(4, len(entries)),
                                     old_root=earlier, new_root=root)
print(sorted(inclusion), sorted(consistency))
print(b64(root) == inclusion["rootHash"] == consistency["newRootHash"])
```

## From the commit payload to a signed decision record

The trail's commit payload is the hash-chained pre-action decision. The
execution record wire carries a `decisionDerived` block in another
vocabulary (`block` for `deny`, decimal strings, ISO 8601 time).
`decision_derived_from_commit` is the bridge, and it takes the three
fields the commit payload does not hold.

```python
from vaara.audit.receipts import decision_derived_from_commit, extract_receipt
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline

trail = SQLiteAuditBackend("audit.db").load_trail()
result = InterceptionPipeline(trail=trail).intercept(agent_id="ops-bot", tool_name="fs.read_file")

receipt = extract_receipt(trail, result.action_id)
derived = decision_derived_from_commit(receipt.commit, policy_id="prod-2026-10",
                                       reason=result.reason)
print(derived.decision, derived.risk_score, derived.threshold_block, derived.decided_at)
```

## Checking a publication against the log

With `VAARA_AUTO_PUBLISH_EVERY` set, the trail records every published
head under `publications()`. `verify_publication` fetches the entry back
from the log and refuses unless the log holds the digest the local copy
claims. The log's answer wins; that is what publishing there is for.

```python
from vaara.attestation.rekor_log import RekorError, RekorPublication, verify_publication

publication = RekorPublication.from_dict({
    "digest": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
    "uuid": "24296fb24b8ad77a6f3e4ea1b4d6e3c9a0f1b2c3d4e5f60718293a4b5c6d7e8f9",
    "log_index": 1234567,
    "integrated_time": 1760140800,
    "log_url": "https://rekor.sigstore.dev",
})
try:
    integrated = verify_publication(publication, expected_digest=publication.digest)
    print("in the log since", integrated.isoformat())
except RekorError as exc:
    print("not verified:", exc)       # no route to the log, or the log disagrees
```

## Attribute attestations: the openings leave the issuer

`issue` commits to every attribute value, signs the commitments, and
returns the two halves apart. `release_to_holder` hands the openings over
exactly once and the issuer keeps nothing it could sell; afterwards
`retains_openings` is false and a second release raises. The holder proves
a predicate with an opening, and anyone verifies it against the
attestation alone.

```python
from cryptography.hazmat.primitives.asymmetric import ed25519

from vaara.attestation.attribute_zk import (
    AttributeValue,
    Predicate,
    PredicateKind,
    SourceStanding,
    Subject,
    issue,
    open_predicate,
    verify_predicate,
)
from vaara.audit.signer import Ed25519Signer

signer = Ed25519Signer(ed25519.Ed25519PrivateKey.generate())
issued = issue(
    signer=signer, issuer="vakuutus.example", attestation_id="att-7f3c-2026-10",
    subject=Subject(kind="person-pseudonym", id="holder-7f3c"),
    values=[AttributeValue("age", 37, SourceStanding.PROTOCOL_DEFINED, "passport MRZ")],
    not_before="2026-10-01T00:00:00Z", not_after="2026-11-01T00:00:00Z",
)
print(issued.retains_openings)                 # True: nothing has left yet
openings = {o.name: o for o in issued.release_to_holder()}
print(issued.retains_openings)                 # False: the issuer holds nothing now

proof = open_predicate(issued.attestation, openings["age"],
                       Predicate(PredicateKind.AT_LEAST, lower=18))
print(verify_predicate(issued.attestation, proof))   # True, and the value 37 is nowhere in it
```
