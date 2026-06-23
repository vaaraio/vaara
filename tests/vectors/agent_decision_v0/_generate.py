#!/usr/bin/env python3
"""Generate the agent_decision_v0 conformance vector.

This is the artifact offered on in-toto/attestation#554: a recomputable
``{statement, expected-verdict}`` pair for the proposed ``agent-decision/v0.1``
predicate, anchored to the bytes and a public key rather than to any one
producer's crate or endpoint.

The statement is a normal in-toto ``Statement/v1`` carrying an
``agent-decision/v0.1`` predicate (the agent id, principal, policy evaluations,
the per-tool-call argument commitments with their explicit-omission state, and
the allow/deny decision). It is sealed in a **DSSE** envelope (the in-toto
native signing envelope) under **Ed25519**, so a reader recomputes the DSSE
pre-authentication encoding (PAE) from the payload bytes and verifies the
signature with the published public key. Ed25519 is deterministic (RFC 8032),
so regenerating produces byte-identical output.

The expected verdict is twofold:

1. **byte/crypto:** the PAE digest and that the signature verifies under the
   published key; and
2. **mapping:** the SEP-2828 normalization of the predicate (which evidence
   plane, which fields populated, what is still missing for a complete signed
   execution record), reproduced from the shipped declarative profile spec.

The sibling ``_check_independent.py`` reproduces all of this importing **no
Vaara code**: only the standard library, ``cryptography`` (for Ed25519), and
``rfc8785`` (for JCS). Conformance is recompute-determinism over the bytes.

Run: ``python tests/vectors/agent_decision_v0/_generate.py`` then commit.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.attestation.receipt import normalize

HERE = Path(__file__).resolve().parent
INPUT = HERE.parents[1] / "vectors" / "normalize_v0" / "inputs" / "agent_decision.json"

# Fixed 32-byte Ed25519 seed -> stable public key AND stable signature across
# runs (Ed25519 is deterministic). A published test key, never a production one.
_SEED = bytes.fromhex(
    "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1"
)
PAYLOAD_TYPE = "application/vnd.in-toto+json"
KEYID = "vaara-agent-decision-conformance-ed25519-k1"


def _pae(payload_type: str, body: bytes) -> bytes:
    """DSSE pre-authentication encoding (the signed bytes)."""
    t = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(body), body)


def _b64(raw: bytes) -> str:
    return base64.standard_b64encode(raw).decode("ascii")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    statement = json.loads(INPUT.read_text(encoding="utf-8"))

    priv = Ed25519PrivateKey.from_private_bytes(_SEED)
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    (HERE / "keys").mkdir(parents=True, exist_ok=True)
    (HERE / "keys" / "ed25519_public.pem").write_bytes(pub_pem)

    # JCS payload bytes so an independent reader recomputes the same PAE.
    payload = rfc8785.dumps(statement)
    pae = _pae(PAYLOAD_TYPE, payload)
    sig = priv.sign(pae)

    envelope = {
        "payloadType": PAYLOAD_TYPE,
        "payload": _b64(payload),
        "signatures": [{"keyid": KEYID, "sig": _b64(sig)}],
    }

    _write(HERE / "statement.json", statement)
    _write(HERE / "envelope.json", envelope)

    expected = {
        "predicateType": statement["predicateType"],
        "paeSha256": "sha256:" + hashlib.sha256(pae).hexdigest(),
        "signatureVerifies": True,
        "decision": statement["predicate"]["decision"],
        "normalized": normalize(statement).to_dict(),
    }
    _write(HERE / "expected.json", expected)
    print("wrote agent_decision_v0 vector; paeSha256=" + expected["paeSha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
