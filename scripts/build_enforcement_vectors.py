# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate tests/vectors/enforcement_v0/.

Engine decision receipts (``vaara.trail-decision/v0``) whose evidence record
says whether the party that wrote the decision could have stopped the action,
written by the real sink over a real SQLite trail: a deny that stopped the
call, the same deny from a gate in shadow mode, a call recorded by a party
that could not hold it, and a deny with no mode recorded. The invalid set
holds a shadow deny edited to read as a stop after signing, and a receipt
signed with a word outside the vocabulary. The engine never writes that word
(``record_decision`` drops it), so it is re-signed here with the vector key,
which is what a non-conforming issuer would produce.

The key is fixed so the public key stays put across regenerations; ECDSA
signatures are randomized, so the receipt files change every run.

    .venv/bin/python scripts/build_enforcement_vectors.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "vectors" / "enforcement_v0"
# Not a secret: a published test key, fixed so the vectors' public key is stable.
FIXED_SCALAR = int("7661617261206e66" "6f7263656d656e74" "2f76302074657374" "20766563746f7273", 16)

_SIGNED_KEYS = ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")

_RULE = {"policy_id": "deny_rule:no-pipe-shell", "violation_type": "policy_rule"}

CASES = [
    ("could_block", "deny", "shell_exec", "Deny rule no-pipe-shell", 1.0, _RULE),
    ("shadow", "deny", "shell_exec", "Deny rule no-pipe-shell", 1.0, _RULE),
    ("recorded_only", "allow", "mcp.method.completion/complete",
     "method not modelled by the proxy; forwarded and recorded", 0.0, {}),
    ("not-recorded", "deny", "shell_exec", "Deny rule no-pipe-shell", 1.0, _RULE),
]


def _resign(body: dict, key) -> dict:
    """Recompute the evidence digest and sign the envelope again."""
    receipt = body["receipt"]
    receipt["decisionDerived"]["evidenceRef"]["digest"] = (
        "sha256:" + hashlib.sha256(rfc8785.dumps(body["evidence"])).hexdigest())
    der = key.sign(rfc8785.dumps({k: receipt[k] for k in _SIGNED_KEYS}), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    receipt["signature"] = (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex()
    return body


def _write(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    dr = importlib.import_module("vaara.audit.decision_receipts")
    work = Path(tempfile.mkdtemp())
    db = work / "audit.db"
    key_path = work / "keys" / "receipt-es256.pem"
    key_path.parent.mkdir(mode=0o700)
    key = ec.derive_private_key(FIXED_SCALAR, ec.SECP256R1())
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    key_path.chmod(0o600)

    trail = SQLiteAuditBackend(db).load_trail()
    assert trail._receipt_sink is not None, "signing libraries missing"
    names = []
    for i, (mode, decision, tool, reason, risk, grounds) in enumerate(CASES):
        trail.record_decision(
            action_id=f"vector-action-{i}", agent_id="vector-agent", tool_name=tool,
            decision=decision, reason=reason, risk_score=risk,
            enforcement="" if mode == "not-recorded" else mode, **grounds,
        )
        names.append(f"{i}-{mode}-{decision}.json")

    written = sorted((work / "receipts").rglob("*.json"), key=lambda p: json.loads(
        p.read_text())["evidence"]["actionId"])
    with sqlite3.connect(db) as con:
        hashes = dict(con.execute("SELECT record_id, record_hash FROM audit_records"))

    # Everything here is generated.
    for sub in ("valid", "invalid"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    (OUT / "valid").mkdir(parents=True)
    (OUT / "invalid").mkdir()
    shutil.copy(work / "receipts" / dr.PUBKEY_NAME, OUT / dr.PUBKEY_NAME)
    expected: dict[str, dict] = {}
    for src, name in zip(written, names):
        shutil.copy(src, OUT / "valid" / name)
        expected[f"valid/{name}"] = {
            "signature": True, "evidence": True, "trail": True, "enforcement": True}

    # A shadow deny edited after signing to read as a stop.
    body = json.loads(written[1].read_text())
    body["evidence"]["enforcement"] = "could_block"
    _write(OUT / "invalid" / "shadow-rewritten-as-stop.json", body)
    expected["invalid/shadow-rewritten-as-stop.json"] = {
        "signature": True, "evidence": False, "trail": True, "enforcement": True}

    # Signed by the issuer, but with a word that is not a mode.
    body = json.loads(written[0].read_text())
    body["evidence"]["enforcement"] = "blocked"
    _write(OUT / "invalid" / "word-outside-the-vocabulary.json", _resign(body, key))
    expected["invalid/word-outside-the-vocabulary.json"] = {
        "signature": True, "evidence": True, "trail": True, "enforcement": False}

    (OUT / "trail-hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(work)
    print(f"wrote {len(expected)} vectors to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
