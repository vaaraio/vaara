# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate tests/vectors/cage_v0/.

Engine decision receipts (``vaara.trail-decision/v0``) whose evidence record
carries a cage block, written by the real sink over a real SQLite trail: a
run outside any cage, a cage the launcher declared that the kernel did not
confirm, and a cage the kernel confirmed. The invalid set holds one tampered
copy and three blocks that are signed and digest-consistent but break a rule
of the block itself. The engine never writes those (``_cage_block`` holds a
receipt to the same rules), so they are re-signed here with the vector key,
which is what a non-conforming issuer would produce.

The key is fixed so the public key stays put across regenerations; ECDSA
signatures are randomized, so the receipt files change every run.

    .venv/bin/python scripts/build_cage_vectors.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "vectors" / "cage_v0"
# Not a secret: a published test key, fixed so the vectors' public key is stable.
FIXED_SCALAR = int("7661617261206361" "67652d626c6f636b" "2f76302074657374" "20766563746f7273", 16)
_SIGNED_KEYS = ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")


def _jcs_digest(obj) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


# The effective configurations the two named cages declare. A driver defines
# what its digest covers; a verifier compares it and never recomputes it.
OPENSHELL_CONFIG = {"policy": "default", "network": "deny", "filesystem": {"read_write": ["/sandbox"]}}
VAARA_CAGE_PROFILE = "profile vaara-agent flags=(attach_disconnected) { deny network raw, }\n"

CASES = [
    ("unconfined", {"driver": "none", "confirmed": False}),
    ("declared", {
        "driver": "openshell", "confirmed": False, "upstream": "openshell 0.1.5",
        "config_digest": _jcs_digest(OPENSHELL_CONFIG), "basis": "declared",
        "name": "vector-launch",
    }),
    ("confirmed", {
        "driver": "vaara-cage", "confirmed": True, "upstream": "apparmor 4.0.1",
        "config_digest": "sha256:" + hashlib.sha256(VAARA_CAGE_PROFILE.encode()).hexdigest(),
        "basis": "apparmor_label",
    }),
]


def _resign(body: dict, key) -> dict:
    """Recompute the evidence digest and sign the envelope again."""
    receipt = body["receipt"]
    receipt["decisionDerived"]["evidenceRef"]["digest"] = _jcs_digest(body["evidence"])
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
    for i, (label, block) in enumerate(CASES):
        trail.record_decision(
            action_id=f"vector-action-{i}", agent_id="vector-agent", tool_name="read_file",
            decision="allow", reason="allow: risk=0.12", risk_score=0.12, cage=block,
        )
        names.append(f"{i}-{label}.json")

    written = sorted((work / "receipts").rglob("*.json"), key=lambda p: json.loads(
        p.read_text())["evidence"]["actionId"])

    # Everything here is generated except the independent checker.
    for sub in ("valid", "invalid"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    (OUT / "valid").mkdir(parents=True)
    (OUT / "invalid").mkdir()
    shutil.copy(work / "receipts" / dr.PUBKEY_NAME, OUT / dr.PUBKEY_NAME)
    expected: dict[str, dict] = {}
    bodies = {}
    for src, name in zip(written, names):
        shutil.copy(src, OUT / "valid" / name)
        bodies[name] = json.loads(src.read_text())
        expected[f"valid/{name}"] = {"signature": True, "evidence": True, "cage": True}

    def copy(name: str) -> dict:
        return json.loads(json.dumps(bodies[name]))

    # Confirmed after signing: the evidence no longer matches its digest.
    body = copy(names[1])
    body["evidence"]["cage"].update(confirmed=True, basis="seccomp_filter")
    _write(OUT / "invalid" / "confirmed-after-signing.json", body)
    expected["invalid/confirmed-after-signing.json"] = {"signature": True, "evidence": False, "cage": True}

    # Signed by the issuer, but confirmed on the launcher's word alone.
    body = copy(names[1])
    body["evidence"]["cage"]["confirmed"] = True
    _write(OUT / "invalid" / "confirmed-on-declared-basis.json", _resign(body, key))
    expected["invalid/confirmed-on-declared-basis.json"] = {"signature": True, "evidence": True, "cage": False}

    # Signed, but a confirmed cage on a run that declared none.
    body = copy(names[0])
    body["evidence"]["cage"]["confirmed"] = True
    _write(OUT / "invalid" / "confirmed-without-a-cage.json", _resign(body, key))
    expected["invalid/confirmed-without-a-cage.json"] = {"signature": True, "evidence": True, "cage": False}

    # Signed, but the configuration digest is not sha256 hex.
    body = copy(names[2])
    body["evidence"]["cage"]["configDigest"] = "sha256:ff"
    _write(OUT / "invalid" / "malformed-config-digest.json", _resign(body, key))
    expected["invalid/malformed-config-digest.json"] = {"signature": True, "evidence": True, "cage": False}

    (OUT / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(work)
    print(f"wrote {len(expected)} vectors to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
