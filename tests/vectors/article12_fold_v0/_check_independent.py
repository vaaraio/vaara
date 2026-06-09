#!/usr/bin/env python3
"""Independent checker for the article12_fold_v0 vectors.

Reads a produced Article 12 package zip and reproduces every folded attestation
verdict WITHOUT importing Vaara, from the same bytes folded into the zip (not a
re-snapshot of the source fixtures). It imports only the standard library,
``cryptography``, and the two single-verb suites' own Vaara-free checkers
(``cross_org_handoff_v0`` and ``enforcement_attestation_v0``). For the package:

1. Read ``evidence/attestations_summary.json`` (the roll-up the package claims).
2. For each ``evidence/handoff/<name>.json``, rebuild the single-verb case from
   the folded package bytes plus the summary's recorded anchor time, trusted DID
   document, and strict flag; run the handoff checker; reproduce the handoff
   roll-up the same way ``check_handoff_set`` does.
3. For each ``evidence/enforcement/<name>`` triple, read the folded
   ``.record.json`` / ``.report.bin`` / ``.vcek.pem``, convert the VCEK PEM back
   to the JWK the checker takes, run the enforcement checker under the summary's
   expected measurement and strict flag, and reproduce that roll-up.
4. Assert each reproduced roll-up equals the one folded into the package.

The chain from folded bytes to set verdict never touches Vaara. Run:
``python tests/vectors/article12_fold_v0/_check_independent.py <package.zip>``.
Exit code 0 means every folded verdict reproduced and matched.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.serialization import load_pem_public_key

HERE = Path(__file__).resolve().parent
HANDOFF = HERE.parent / "cross_org_handoff_v0"
ENFORCEMENT = HERE.parent / "enforcement_attestation_v0"

HANDOFF_KEYS = ("ok", "strict", "total", "loaded", "passed", "verifiable",
                "corroborated", "pinned", "pinningGap")
ENFORCEMENT_KEYS = ("ok", "strict", "total", "loaded", "passed", "bound",
                    "measurementPinned", "tierCounts", "pinningGap")
TIER_NAMES = ("unverified", "bound", "measurement_pinned")


def _load_single_checker(path: Path, mod_name: str):
    """Load a single-verb suite's Vaara-free ``_evaluate`` from its checker."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "_evaluate")


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _pem_to_jwk(pem: bytes) -> dict:
    """Recover the P-384 JWK the enforcement checker takes from a VCEK PEM."""
    pub = load_pem_public_key(pem)
    nums = pub.public_numbers()  # type: ignore[attr-defined]
    size = (nums.curve.key_size + 7) // 8
    return {
        "kty": "EC", "crv": "P-384",
        "x": _b64url(nums.x.to_bytes(size, "big")),
        "y": _b64url(nums.y.to_bytes(size, "big")),
    }


def _handoff_rollup(verdicts: list, *, strict: bool) -> dict:
    total = len(verdicts)
    passed = sum(1 for v in verdicts if v["ok"])
    pinned = sum(1 for v in verdicts if v["producer_identity_basis"] == "pinned")
    return {
        "ok": passed == total, "strict": strict, "total": total, "loaded": total,
        "passed": passed,
        "verifiable": sum(1 for v in verdicts if v["verifiable"]),
        "corroborated": sum(1 for v in verdicts if v["corroborated"]),
        "pinned": pinned, "pinningGap": total > 0 and pinned == 0,
    }


def _enforcement_rollup(verdicts: list, *, strict: bool) -> dict:
    total = len(verdicts)
    passed = sum(1 for v in verdicts if v["ok"])
    pinned = sum(1 for v in verdicts if v["measurement_basis"] == "pinned")
    tier_counts = {name: 0 for name in TIER_NAMES}
    for v in verdicts:
        tier_counts[v["tier"]] = tier_counts.get(v["tier"], 0) + 1
    return {
        "ok": passed == total, "strict": strict, "total": total, "loaded": total,
        "passed": passed, "bound": sum(1 for v in verdicts if v["bound"]),
        "measurementPinned": pinned, "tierCounts": tier_counts,
        "pinningGap": total > 0 and pinned == 0,
    }


def _check_zip(zip_path: Path) -> list:
    failures: list = []
    judge_handoff = _load_single_checker(
        HANDOFF / "_check_independent.py", "_h_single")
    judge_enforcement = _load_single_checker(
        ENFORCEMENT / "_check_independent.py", "_e_single")

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        summary = json.loads(zf.read("evidence/attestations_summary.json"))

        h = summary["handoff"]
        if h["present"]:
            strict = h["strict"]
            verdicts = []
            members = sorted(
                n for n in names
                if n.startswith("evidence/handoff/") and n.endswith(".json"))
            for member in members:
                base = member[len("evidence/handoff/"):-len(".json")]
                doc = json.loads(zf.read(member))
                case = {"package": doc, "strict": strict,
                        "anchoredTime": h["anchoredTimes"].get(base)}
                if h.get("trustedDidDocument") is not None:
                    case["trustedDidDocument"] = h["trustedDidDocument"]
                verdicts.append(judge_handoff(case))
            got = _handoff_rollup(verdicts, strict=strict)
            want = {k: h["report"][k] for k in HANDOFF_KEYS}
            if {k: got[k] for k in HANDOFF_KEYS} != want:
                failures.append(f"handoff:\n    folded {want}\n    reproduced {got}")
            else:
                print(f"handoff: OK ok={got['ok']} verifiable={got['verifiable']} "
                      f"corroborated={got['corroborated']} pinned={got['pinned']}")

        e = summary["enforcement"]
        if e["present"]:
            strict = e["strict"]
            verdicts = []
            records = sorted(
                n for n in names
                if n.startswith("evidence/enforcement/")
                and n.endswith(".record.json"))
            for member in records:
                base = member[len("evidence/enforcement/"):-len(".record.json")]
                record = json.loads(zf.read(member))
                report_bytes = zf.read(f"evidence/enforcement/{base}.report.bin")
                vcek_pem = zf.read(f"evidence/enforcement/{base}.vcek.pem")
                case = {"record": record,
                        "report_b64": base64.b64encode(report_bytes).decode("ascii"),
                        "vcek_jwk": _pem_to_jwk(vcek_pem),
                        "expected_measurement": e["expectedMeasurement"],
                        "strict": strict}
                verdicts.append(judge_enforcement(case))
            got = _enforcement_rollup(verdicts, strict=strict)
            want = {k: e["report"][k] for k in ENFORCEMENT_KEYS}
            if {k: got[k] for k in ENFORCEMENT_KEYS} != want:
                failures.append(
                    f"enforcement:\n    folded {want}\n    reproduced {got}")
            else:
                print(f"enforcement: OK ok={got['ok']} bound={got['bound']} "
                      f"pinned={got['measurementPinned']}")
    return failures


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _check_independent.py <article12_package.zip>", file=sys.stderr)
        return 2
    failures = _check_zip(Path(sys.argv[1]))
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall folded attestations reproduced Vaara-free from the zip bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
