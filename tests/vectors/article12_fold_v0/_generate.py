#!/usr/bin/env python3
"""Generate the article12_fold_v0 vectors from the single-verb corpora.

The Article 12 fold attaches verified SEP-2828 evidence as sidecars inside the
regulator package: cross-org handoff packages (Article 26(6)) and
confidential-VM enforcement bindings, each verified at export and folded under
``evidence/``. This script does not invent new crypto. It reuses the committed
``cross_org_handoff_v0`` and ``enforcement_attestation_v0`` cases, folds named
subsets into a real package with :func:`export_article12`, and records the
roll-up the package carries in ``evidence/attestations_summary.json`` plus the
``evidence/`` membership. ``expected.json`` is the committed truth the
in-process test (Vaara) and ``_check_independent.py`` (Vaara-free, reading the
folded bytes back out of the zip) both reproduce.

No zip is committed: it carries a fresh signature and a runtime ``.vcek.pem``.
The test rebuilds it in a temp dir and runs the Vaara-free checker over it.

Run: ``python tests/vectors/article12_fold_v0/_generate.py``.
"""

from __future__ import annotations

import base64
import json
import tempfile
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.audit.article12_export import export_article12
from vaara.audit.trail import AuditTrail
from vaara.taxonomy.actions import ActionRequest, create_default_registry

HERE = Path(__file__).resolve().parent
HANDOFF = HERE.parent / "cross_org_handoff_v0"
ENFORCEMENT = HERE.parent / "enforcement_attestation_v0"

# Scenarios that produce a package, named by the evidence they fold and the
# verifier-side inputs the fold applies. ``trusted_from_case`` pins handoff
# producer identity against that case's archived DID document.
SCENARIOS: dict[str, dict] = {
    # A handoff with no anchor beside an anchor-corroborated one, plus two
    # enforcement binds, one pinned to its launch measurement. The handoff set
    # pins no producer (pinning gap); the enforcement set pins one image.
    "full": {
        "handoff": {"cases": ["clean_no_anchor", "corroborated"],
                    "trusted_from_case": None},
        "enforcement": {"cases": ["clean_bound", "pinned_measurement_match"],
                        "expected_measurement": (
                            "0102030405060708090a0b0c0d0e0f10"
                            "1112131415161718191a1b1c1d1e1f20"
                            "2122232425262728292a2b2c2d2e2f30")},
    },
    # A single handoff pinned against a trusted DID document: producer pinned,
    # no pinning gap, anchor-corroborated.
    "pinned_handoff": {
        "handoff": {"cases": ["pinned_corroborated"],
                    "trusted_from_case": "pinned_corroborated"},
        "enforcement": None,
    },
    # Enforcement only, unpinned: binds to a CVM but to no vetted image, so the
    # enforcement pinning gap fires (advisory, does not gate).
    "enforcement_only": {
        "handoff": None,
        "enforcement": {"cases": ["clean_bound"], "expected_measurement": None},
    },
}

# Scenarios that must FAIL the export (fail closed): an attachment that does not
# verify aborts the whole package, no zip written. Exercised by the test.
FAIL_CLOSED: dict[str, dict] = {
    "bad_handoff": {
        "handoff": {"cases": ["clean_no_anchor", "signed_after_retirement"],
                    "trusted_from_case": None},
        "enforcement": None,
    },
    "bad_enforcement": {
        "handoff": None,
        "enforcement": {"cases": ["clean_bound", "bad_signature"],
                        "expected_measurement": None},
    },
}

HANDOFF_KEYS = ("ok", "strict", "total", "loaded", "passed", "verifiable",
                "corroborated", "pinned", "pinningGap")
ENFORCEMENT_KEYS = ("ok", "strict", "total", "loaded", "passed", "bound",
                    "measurementPinned", "tierCounts", "pinningGap")

_REGISTRY = create_default_registry()
_TX = _REGISTRY.get("tx.transfer")


def _jwk_to_pem(jwk: dict) -> bytes:
    def _i(v: str) -> int:
        return int.from_bytes(base64.urlsafe_b64decode(v + "=" * (-len(v) % 4)), "big")
    pub = ec.EllipticCurvePublicNumbers(
        _i(jwk["x"]), _i(jwk["y"]), ec.SECP384R1()).public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _trail() -> AuditTrail:
    trail = AuditTrail()
    for i in range(2):
        req = ActionRequest(agent_id=f"agent-{i}", tool_name="send_funds",
                            action_type=_TX, parameters={"to": f"0x{i}", "amount": i})
        action_id = trail.record_action_requested(req)
        trail.record_decision(action_id, f"agent-{i}", "send_funds", "allow", "t", 0.5)
        trail.record_execution(action_id, f"agent-{i}", "send_funds", {"ok": True})
        trail.record_outcome(action_id, f"agent-{i}", "send_funds", "success")
    return trail


def _key(d: Path) -> Path:
    k = Ed25519PrivateKey.generate()
    p = d / "signer.pem"
    p.write_bytes(k.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    return p


def _build(spec: dict, hcases: dict, ecases: dict, work: Path) -> Path:
    """Fold one scenario into a real package and return the zip path."""
    handoffs = None
    enforcements = None
    trusted = None
    expected_measurement = None

    if spec["handoff"]:
        h = spec["handoff"]
        handoffs = []
        for name in h["cases"]:
            case = hcases[name]
            package = case["package"]
            anchored = package.get("evidence", {}).get("anchor") is not None
            handoffs.append(
                (name, package, case.get("anchoredTime") if anchored else None))
        if h["trusted_from_case"] is not None:
            trusted = hcases[h["trusted_from_case"]]["trustedDidDocument"]

    if spec["enforcement"]:
        e = spec["enforcement"]
        enforcements = []
        for name in e["cases"]:
            case = ecases[name]
            enforcements.append((
                name, case["record"], base64.b64decode(case["report_b64"]),
                _jwk_to_pem(case["vcek_jwk"])))
        expected_measurement = e["expected_measurement"]

    out = work / "pack.zip"
    if out.exists():
        out.unlink()
    export_article12(
        _trail(), out, signer_key=_key(work),
        handoffs=handoffs, enforcements=enforcements,
        trusted_did_document=trusted, expected_measurement=expected_measurement)
    return out


def main() -> int:
    hcases = {c["name"]: c
              for c in json.loads((HANDOFF / "cases.json").read_text())["cases"]}
    ecases = {c["name"]: c
              for c in json.loads((ENFORCEMENT / "cases.json").read_text())["cases"]}

    expected: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, spec in SCENARIOS.items():
            zip_path = _build(spec, hcases, ecases, work)
            with zipfile.ZipFile(zip_path) as zf:
                members = sorted(m for m in zf.namelist() if m.startswith("evidence/"))
                summary = json.loads(zf.read("evidence/attestations_summary.json"))
            entry: dict = {"evidence_members": members}
            if summary["handoff"]["present"]:
                rep = summary["handoff"]["report"]
                entry["handoff"] = {k: rep[k] for k in HANDOFF_KEYS}
            if summary["enforcement"]["present"]:
                rep = summary["enforcement"]["report"]
                entry["enforcement"] = {k: rep[k] for k in ENFORCEMENT_KEYS}
            expected[name] = entry

    fold_doc = {"scenarios": SCENARIOS, "fail_closed": FAIL_CLOSED}
    (HERE / "fold.json").write_text(json.dumps(fold_doc, indent=2, sort_keys=True) + "\n")
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(SCENARIOS)} scenarios to fold.json and expected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
