#!/usr/bin/env python3
"""Generate the enforcement_set_v0 vectors from the single-verb corpus.

A set is a list of ``enforcement_attestation_v0`` case names. This script does
not invent new crypto: it reuses that suite's committed cases (each a record,
a SEV-SNP report, and a VCEK JWK), groups them into sets, and records the
roll-up ``check_enforcement_set`` produces. ``expected.json`` is the committed
truth the in-process test (Vaara) and ``_check_independent.py`` (Vaara-free)
both reproduce.

Run: ``python tests/vectors/enforcement_set_v0/_generate.py``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation.receipt import check_enforcement_set

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "enforcement_attestation_v0"

# Each set names cases from the single-verb suite, plus the set-level mode the
# batch applies uniformly (expected_measurement, strict). Cases that pin a
# measurement only do so when the set pins the matching value, mirroring how the
# single verb takes --expected-measurement as a verifier-side input.
SETS: dict[str, dict] = {
    # Two clean binds, one of them pinned when the set pins its measurement:
    # passes, and the pinned one lifts the pinning-coverage gap.
    "all_bound": {
        "cases": ["clean_bound", "pinned_measurement_match"],
        "expected_measurement": (
            "0102030405060708090a0b0c0d0e0f10"
            "1112131415161718191a1b1c1d1e1f20"
            "2122232425262728292a2b2c2d2e2f30"
        ),
        "strict": False,
    },
    # One clean bind, no pin: passes, but every record bound to a CVM without a
    # vetted image, so the pinning gap fires (advisory, does not gate).
    "unpinned_only": {
        "cases": ["clean_bound"],
        "expected_measurement": None,
        "strict": False,
    },
    # A clean bind alongside a bad signature and a report bound to another
    # record: the set fails, one entry passes.
    "mixed_failure": {
        "cases": ["clean_bound", "bad_signature", "bound_to_different_record"],
        "expected_measurement": None,
        "strict": False,
    },
    # Nothing binds: a truncated report and a wrong signature algorithm.
    "all_unverified": {
        "cases": ["truncated_report", "wrong_signature_algo"],
        "expected_measurement": None,
        "strict": False,
    },
    # Strict is unreachable in v0 (no validated KDS chain), so even a clean
    # pinned bind does not pass strict.
    "strict_unreachable": {
        "cases": ["pinned_measurement_match"],
        "expected_measurement": (
            "0102030405060708090a0b0c0d0e0f10"
            "1112131415161718191a1b1c1d1e1f20"
            "2122232425262728292a2b2c2d2e2f30"
        ),
        "strict": True,
    },
}

SUMMARY_KEYS = (
    "ok", "strict", "total", "loaded", "passed", "bound",
    "measurementPinned", "tierCounts", "pinningGap",
)


def _jwk_to_pem(jwk: dict) -> bytes:
    def _i(v: str) -> int:
        return int.from_bytes(base64.urlsafe_b64decode(v + "=" * (-len(v) % 4)), "big")

    pub = ec.EllipticCurvePublicNumbers(
        _i(jwk["x"]), _i(jwk["y"]), ec.SECP384R1()).public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def main() -> int:
    cases = {c["name"]: c
             for c in json.loads((SIBLING / "cases.json").read_text())["cases"]}
    expected: dict[str, dict] = {}
    for set_name, spec in SETS.items():
        triples = []
        for case_name in spec["cases"]:
            case = cases[case_name]
            triples.append((
                case_name,
                case["record"],
                base64.b64decode(case["report_b64"]),
                _jwk_to_pem(case["vcek_jwk"]),
            ))
        report = check_enforcement_set(
            triples,
            expected_measurement=spec["expected_measurement"],
            strict=spec["strict"],
        )
        d = report.to_dict()
        expected[set_name] = {k: d[k] for k in SUMMARY_KEYS}

    sets_doc = {
        "sets": {
            name: {
                "cases": spec["cases"],
                "expected_measurement": spec["expected_measurement"],
                "strict": spec["strict"],
            }
            for name, spec in SETS.items()
        }
    }
    (HERE / "sets.json").write_text(json.dumps(sets_doc, indent=2, sort_keys=True) + "\n")
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(SETS)} sets to sets.json and expected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
