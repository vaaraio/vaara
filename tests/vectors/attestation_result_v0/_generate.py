"""Generate the attestation_result_v0 conformance vectors.

Writes ``cases.json`` (input verdicts) and ``expected.json`` (the EAR each one
produces). Run after changing the exporter or the case matrix::

    python tests/vectors/attestation_result_v0/_generate.py

The verdicts here are the ``to_dict()`` shape of TPM-binding, TPM-chain, and
SEV-SNP enforcement verdicts. The exporter is pure (dict in, dict out), so the
cases are authored directly rather than captured from hardware.
"""

from __future__ import annotations

import json
from pathlib import Path

from vaara.attestation._attestation_result import build_attestation_result

IAT = 1750000000
BUILD = "vaara 0.0.0-test"
HERE = Path(__file__).resolve().parent

_DIGEST = "ab" * 32  # 64-hex placeholder record-binding digest


def _chain(tier, links_bound, ak_basis, pcr_basis, append_only):
    return {
        "schema": "vaara.tpm-evidence-chain/v0",
        "tier": tier,
        "ok": tier != "unverified",
        "strict": False,
        "n_links": 15,
        "links_bound": links_bound,
        "clock_monotonic": True,
        "reboot_free": True,
        "ima_append_only": append_only,
        "ak_stable": True,
        "reset_count": 1139,
        "restart_count": 0,
        "window": {},
        "pcr_pin_basis": pcr_basis,
        "ak_chain_basis": ak_basis,
        "ima_policy_basis": "not_established",
        "decision_logic_basis": "not_established",
        "freshness_basis": "chain_continuity" if tier == "continuous" else "not_established",
        "links": [],
        "record": {},
        "reason": tier,
    }


def _binding(tier, parsed, sig, bound, ak_basis, pcr_basis, ima_ok, pcr_recomputed):
    return {
        "schema": "vaara.tpm-binding-attestation/v0",
        "tier": tier,
        "ok": tier != "unverified",
        "strict": False,
        "parsed": parsed,
        "magic_ok": parsed,
        "signature_algo_ok": parsed,
        "signature_valid": sig,
        "bound": bound,
        "extra_data_expected": _DIGEST,
        "pcr_digest_recomputed": pcr_recomputed,
        "pcr_digest_quoted": "cd" * 32,
        "ima_pcr_index": 10,
        "ima_replayed": ima_ok,
        "ima_log_entries": 42,
        "pcr_pin_basis": pcr_basis,
        "ak_chain_basis": ak_basis,
        "ima_policy_basis": "not_established",
        "decision_logic_basis": "not_established",
        "freshness_basis": "not_established",
        "pcr_context": {},
        "record": {},
        "reason": tier,
    }


def _enforcement(tier, parsed, sig, bound, vcek_basis, meas_basis, measurement):
    return {
        "schema": "vaara.enforcement-attestation/v0",
        "tier": tier,
        "ok": tier != "unverified",
        "strict": False,
        "parsed": parsed,
        "report_version": 2,
        "signature_algo_ok": parsed,
        "signature_valid": sig,
        "bound": bound,
        "report_data_expected": "ab" * 64,
        "report_data_actual": "ab" * 64 if bound else "cd" * 64,
        "measurement": measurement,
        "expected_measurement": measurement if meas_basis == "pinned" else None,
        "measurement_basis": meas_basis,
        "vcek_chain_basis": vcek_basis,
        "enforcement_logic_basis": "not_established",
        "report_context": {},
        "record": {},
        "reason": tier,
    }


CASES = [
    ("tpm_chain_continuous_unverified_root",
     _chain("continuous", True, "caller_supplied_unverified", "unpinned", True)),
    ("tpm_chain_continuous_pinned_validated_root",
     _chain("continuous", True, "ek_chain_verified", "pinned", True)),
    ("tpm_chain_linked_unpinned",
     _chain("linked", True, "caller_supplied_unverified", "unpinned", True)),
    ("tpm_chain_unverified",
     _chain("unverified", False, "caller_supplied_unverified", "unpinned", False)),
    ("tpm_binding_bound_unpinned",
     _binding("bound", True, True, True, "caller_supplied_unverified", "unpinned", True, True)),
    ("tpm_binding_pcr_pinned_validated_root",
     _binding("pcr_pinned", True, True, True, "ek_chain_verified", "pinned", True, True)),
    ("tpm_binding_pin_mismatch",
     _binding("bound", True, True, True, "caller_supplied_unverified", "pin_mismatch", True, True)),
    ("tpm_binding_unverified",
     _binding("unverified", False, False, False, "caller_supplied_unverified", "unpinned", False, False)),
    ("enforcement_measurement_pinned_unverified_root",
     _enforcement("measurement_pinned", True, True, True, "caller_supplied_unverified", "pinned", "01" * 48)),
    ("enforcement_measurement_pinned_kds_verified",
     _enforcement("measurement_pinned", True, True, True, "kds_verified", "pinned", "01" * 48)),
    ("enforcement_bound_unpinned",
     _enforcement("bound", True, True, True, "caller_supplied_unverified", "unpinned", "01" * 48)),
    ("enforcement_unverified",
     _enforcement("unverified", False, False, False, "caller_supplied_unverified", "unpinned", None)),
]


def main() -> None:
    cases = [{"name": name, "verdict": verdict} for name, verdict in CASES]
    expected = {
        name: build_attestation_result(verdict, issued_at=IAT, verifier_build=BUILD)
        for name, verdict in CASES
    }
    (HERE / "cases.json").write_text(
        json.dumps({"iat": IAT, "verifier_build": BUILD, "cases": cases}, indent=2) + "\n"
    )
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {len(cases)} cases + expected EARs")


if __name__ == "__main__":
    main()
