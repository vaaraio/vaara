"""Where did the data go? — the data-locality evidence record in action.

Emits two vaara.data-locality/v0 records and recomputes the verdict for each
the way an outside party would: signature, payload digest, and the allow/block
decision under the named policy, plus any carried region attestation. The second
record is recorded as `allow` for personal data leaving to a US region; the
independent recompute catches it, because it never trusts the recorded decision.

Run: python examples/data_locality_demo.py   (needs: pip install 'vaara[attestation]' rich)
"""
import hashlib

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.attestation.data_locality import (
    TransferFacts, emit_data_locality_record, payload_digest, region_attestation,
    verify_record_signature,
)
from vaara.audit.signer import Ed25519Signer, Ed25519Verifier

console = Console()

# Demo keys. The issuer signs the record; the attester (a TEE or the provider)
# signs the region attestation. They are distinct parties on purpose: a relying
# party checks the attestation against the attester's key, never the issuer's.
issuer = Ed25519Signer(Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"demo-issuer").digest()))
attester = Ed25519Signer(Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"demo-tee").digest()))
ISSUER_V = Ed25519Verifier(issuer.public_key_bytes())
ATTESTER_V = Ed25519Verifier(attester.public_key_bytes())

POLICY_ID = "eu-inference-only@v1"
EU_REGIONS = {"eu-central-1", "eu-north-1", "eu-west-1"}


def recompute_policy(data_class: str, region: str) -> str:
    """eu-inference-only@v1: personal data may leave only to an EU region."""
    if data_class != "personal_data":
        return "allow"
    return "allow" if region in EU_REGIONS else "block"


def verdict(record: dict, payload: dict) -> str:
    """Recompute the verdict from the record bytes, trusting no recorded field."""
    if not verify_record_signature(record, verifier=ISSUER_V):
        return "bad_signature"
    t = record["transfer"]
    if t["payloadDigest"] != payload_digest(payload):
        return "payload_mismatch"
    if record["decision"]["decision"] != recompute_policy(t["dataClass"], t["endpointRegion"]):
        return "policy_mismatch"
    att = record.get("regionAttestation")
    if att is None:
        return "ok_asserted"
    body = {"attestedRegion": att["attestedRegion"], "attester": att["attester"], "nonce": att["nonce"]}
    from vaara.attestation._sep2787_canonical import canonical_json
    if not ATTESTER_V.verify(canonical_json(body), bytes.fromhex(att["sig"])):
        return "attestation_bad_sig"
    if att["attestedRegion"] != t["endpointRegion"]:
        return "attestation_region_mismatch"
    return "ok_attested"


def show(title: str, record: dict, payload: dict, note: str) -> None:
    v = verdict(record, payload)
    color = "bold green" if v.startswith("ok") else "bold red"
    t = record["transfer"]
    body = Text()
    body.append("  data class     ", style="dim"); body.append(f"{t['dataClass']}\n", style="white")
    body.append("  endpoint region", style="dim"); body.append(f" {t['endpointRegion']}\n", style="white")
    body.append("  recorded       ", style="dim"); body.append(f"{record['decision']['decision']}", style="white")
    body.append(f"   under {record['decision']['policyId']}\n", style="bright_black")
    body.append("  attestation    ", style="dim")
    att = record.get("regionAttestation")
    body.append(f"{att['attester']} says {att['attestedRegion']}\n" if att else "none (region asserted, not attested)\n", style="white")
    body.append("  recomputed     ", style="dim"); body.append(f"{v}\n", style=color)
    body.append("  ", style="dim"); body.append(note, style="bright_black")
    console.print(Panel(body, title=f"[bold #78a08a]{title}[/]", border_style="#78a08a", padding=(1, 2)))


console.print()
console.print("  [dim]vaara.data-locality/v0 — proving where an agent's data went[/]")
console.print()

# 1. Personal data to an EU endpoint, allowed, with a matching TEE attestation.
p1 = {"prompt": "summarise the customer account note", "subject": "user-42"}
r1 = emit_data_locality_record(
    signer=issuer, issuer="vaara-locality-emitter",
    transfer=TransferFacts("act-eu-01", "personal_data",
                           "https://api.eu-central-1.model.example/v1/infer", "eu-central-1",
                           payload_digest(p1), "sha256:" + hashlib.sha256(b"cert").hexdigest()),
    decision="allow", policy_id=POLICY_ID,
    region_attestation=region_attestation(attester, attester="frankfurt-tee-01",
                                           attested_region="eu-central-1", nonce="n1"),
)
show("attested EU inference", r1, p1, "signature, digest, policy, and the attester all check out.")

# 2. Personal data to a US endpoint, recorded as allow. The issuer's verdict is
#    wrong; recomputing the policy from the bytes catches it.
p2 = {"prompt": "enrich this lead", "subject": "user-99"}
r2 = emit_data_locality_record(
    signer=issuer, issuer="vaara-locality-emitter",
    transfer=TransferFacts("act-us-02", "personal_data",
                           "https://api.us-east-1.model.example/v1/infer", "us-east-1",
                           payload_digest(p2), "sha256:" + hashlib.sha256(b"cert2").hexdigest()),
    decision="allow", policy_id=POLICY_ID,
)
show("US inference, wrongly recorded allow", r2, p2,
     "recorded allow, but the policy recomputes to block. The record does not get the last word.")

console.print("  [dim]Verify any record yourself: tests/vectors/data_locality_v0/ (no vaara import).[/]")
console.print()
