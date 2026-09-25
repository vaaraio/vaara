# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate tests/vectors/trail_decision_v0/.

Engine-emitted decision receipts (``vaara.trail-decision/v0``) written by the
real sink over a real SQLite trail, plus tampered copies, the issuer public
key, the trail's record hashes, and the verdict each file must get. The
Python suite and the macOS app's Swift tests both check this set, so the two
verifiers cannot drift apart.

The key is fixed so the public key stays put across regenerations; ECDSA
signatures are randomized, so the receipt files change every run.

    .venv/bin/python scripts/build_trail_decision_vectors.py
"""

from __future__ import annotations

import importlib
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "vectors" / "trail_decision_v0"
# Not a secret: a published test key, fixed so the vectors' public key is stable.
FIXED_SCALAR = int("7a61617261207472" "61696c2d64656369" "73696f6e2f763020" "7465737420766563", 16)

CASES = [
    ("allow", "read_file", "allow: risk=0.12 (threshold allow<0.55 deny>0.85)", 0.12),
    ("escalate", "send_email", "escalate: risk=0.61", 0.61),
    ("deny", "Bash", "rule: rm -rf on a home path", 0.97),
    # Escapes and non-ASCII: the case where a hand-written JCS goes wrong.
    ("deny", "mcp__fs__write_ä",
     'He said "no"\n\ttab \u0001 back\\slash / ä € \U0001F600  ', 0.5),
]


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
    for i, (decision, tool, reason, risk) in enumerate(CASES):
        trail.record_decision(
            action_id=f"vector-action-{i}", agent_id="vector-agent",
            tool_name=tool, decision=decision, reason=reason, risk_score=risk,
        )
        names.append(f"{i}-{decision}.json")

    written = sorted((work / "receipts").rglob("*.json"), key=lambda p: json.loads(
        p.read_text())["evidence"]["actionId"])
    with sqlite3.connect(db) as con:
        hashes = dict(con.execute("SELECT record_id, record_hash FROM audit_records"))

    # Everything here is generated except the independent checker.
    for sub in ("valid", "invalid"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    (OUT / "valid").mkdir(parents=True)
    (OUT / "invalid").mkdir()
    shutil.copy(work / "receipts" / dr.PUBKEY_NAME, OUT / dr.PUBKEY_NAME)
    expected: dict[str, dict] = {}
    for src, name in zip(written, names):
        shutil.copy(src, OUT / "valid" / name)
        expected[f"valid/{name}"] = {"signature": True, "evidence": True, "trail": True}

    base = json.loads(written[0].read_text())

    tampered = json.loads(json.dumps(base))
    tampered["evidence"]["decision"] = "deny"
    (OUT / "invalid" / "evidence-tampered.json").write_text(json.dumps(tampered, indent=2))
    expected["invalid/evidence-tampered.json"] = {"signature": True, "evidence": False, "trail": True}

    tampered = json.loads(json.dumps(base))
    tampered["receipt"]["decisionDerived"]["decision"] = "block"
    (OUT / "invalid" / "envelope-tampered.json").write_text(json.dumps(tampered, indent=2))
    expected["invalid/envelope-tampered.json"] = {"signature": False, "evidence": True, "trail": True}

    # Signed and self-consistent, but the trail holds no such record.
    orphan = json.loads(written[1].read_text())
    orphan_id = orphan["evidence"]["recordId"]
    (OUT / "invalid" / "not-in-trail.json").write_text(json.dumps(orphan, indent=2))
    expected["invalid/not-in-trail.json"] = {"signature": True, "evidence": True, "trail": False}
    trail_hashes = {k: v for k, v in hashes.items() if k != orphan_id}
    (OUT / "valid" / names[1]).unlink()
    del expected[f"valid/{names[1]}"]

    (OUT / "trail-hashes.json").write_text(json.dumps(trail_hashes, indent=2, sort_keys=True) + "\n")
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(work)
    print(f"wrote {len(expected)} vectors to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
