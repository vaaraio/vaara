"""Phase 2 neutral verify: express a Vaara verdict as an IETF RATS EAR.

Covers ``build_attestation_result`` -- the AR4SI trustworthiness mapping, the
honest ``warning`` ceiling while the root is trusted as supplied, root-agnostic
shape across TPM and SEV-SNP, input validation -- plus the
``attestation_result_v0`` conformance vectors and the Vaara-free independent
checker that reproduces every EAR.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation._attestation_result import (
    AFFIRMING,
    CONTRAINDICATED,
    EAR_PROFILE,
    VAARA_PROFILE,
    VERIFIER_CLAIMS_KEY,
    WARNING,
    build_attestation_result,
)

VECTORS = Path(__file__).resolve().parent / "vectors" / "attestation_result_v0"
IAT = 1750000000
BUILD = "vaara 0.0.0-test"


def _chain(tier="continuous", ak="caller_supplied_unverified", pcr="unpinned"):
    return {
        "schema": "vaara.tpm-evidence-chain/v0", "tier": tier, "ok": tier != "unverified",
        "n_links": 15, "links_bound": tier != "unverified", "ima_append_only": True,
        "pcr_pin_basis": pcr, "ak_chain_basis": ak, "ima_policy_basis": "not_established",
        "decision_logic_basis": "not_established", "freshness_basis": "chain_continuity",
    }


def _enf(tier="bound", vcek="caller_supplied_unverified", meas="unpinned", measurement="01" * 48):
    return {
        "schema": "vaara.enforcement-attestation/v0", "tier": tier, "parsed": tier != "unverified",
        "signature_valid": tier != "unverified", "bound": tier != "unverified",
        "measurement": measurement, "measurement_basis": meas, "vcek_chain_basis": vcek,
        "enforcement_logic_basis": "not_established", "report_data_expected": "ab" * 64,
    }


def _build(verdict):
    return build_attestation_result(verdict, issued_at=IAT, verifier_build=BUILD)


def _vector(ear):
    return next(iter(ear["submods"].values()))["ear_trustworthiness_vector"]


# ── honesty model ────────────────────────────────────────────────────────────

def test_trusted_as_supplied_root_tops_out_at_warning():
    ear = _build(_chain())
    assert ear["ear_status"] == "warning"
    assert all(v == WARNING for v in _vector(ear).values())


def test_affirming_needs_a_validated_root_and_a_pin():
    ear = _build(_chain(ak="ek_chain_verified", pcr="pinned"))
    assert ear["ear_status"] == "affirming"
    assert all(v == AFFIRMING for v in _vector(ear).values())


def test_validated_root_without_a_pin_still_warns_on_measured_state():
    ear = _build(_chain(ak="ek_chain_verified", pcr="unpinned"))
    vec = _vector(ear)
    assert vec["hardware"] == AFFIRMING and vec["instance-identity"] == AFFIRMING
    assert vec["executables"] == WARNING and vec["configuration"] == WARNING
    assert ear["ear_status"] == "warning"


def test_unverified_chain_is_contraindicated():
    ear = _build(_chain(tier="unverified"))
    assert ear["ear_status"] == "contraindicated"
    assert all(v == CONTRAINDICATED for v in _vector(ear).values())


def test_pcr_pin_mismatch_contraindicates_state_not_identity():
    binding = {
        "schema": "vaara.tpm-binding-attestation/v0", "tier": "bound", "parsed": True,
        "signature_valid": True, "bound": True, "ak_chain_basis": "caller_supplied_unverified",
        "pcr_pin_basis": "pin_mismatch", "ima_replayed": True, "ima_log_entries": 42,
        "pcr_digest_recomputed": True, "extra_data_expected": "ab" * 32,
    }
    vec = _vector(_build(binding))
    assert vec["executables"] == CONTRAINDICATED and vec["configuration"] == CONTRAINDICATED
    assert vec["instance-identity"] == WARNING and vec["hardware"] == WARNING


def test_sev_snp_affirming_reachable_with_kds_and_pin():
    assert _build(_enf(tier="measurement_pinned", vcek="kds_verified", meas="pinned"))[
        "ear_status"] == "affirming"


def test_sev_snp_unverified_makes_no_measured_claim():
    vec = _vector(_build(_enf(tier="unverified", measurement=None)))
    assert "executables" not in vec and "configuration" not in vec
    assert vec["instance-identity"] == CONTRAINDICATED


# ── root-agnostic shape ──────────────────────────────────────────────────────

def test_same_ear_shape_across_roots():
    tpm = _build(_chain())
    sev = _build(_enf())
    assert set(tpm) == set(sev) == {
        "eat_profile", "iat", "ear_status", "ear_verifier_id", "submods"}
    assert tpm["eat_profile"] == sev["eat_profile"] == EAR_PROFILE
    assert list(tpm["submods"]) == ["tpm"] and list(sev["submods"]) == ["sev-snp"]
    for ear in (tpm, sev):
        sm = next(iter(ear["submods"].values()))
        assert sm["eat_profile"] == VAARA_PROFILE
        assert sm[VERIFIER_CLAIMS_KEY]["result_is_unsigned"] is True


def test_submod_label_override():
    ear = build_attestation_result(
        _chain(), issued_at=IAT, verifier_build=BUILD, submod_label="node-7")
    assert list(ear["submods"]) == ["node-7"]


def test_verifier_id_and_iat_carried():
    ear = _build(_chain())
    assert ear["iat"] == IAT
    assert ear["ear_verifier_id"] == {"developer": "https://vaara.io", "build": BUILD}


# ── input validation ─────────────────────────────────────────────────────────

def test_unknown_schema_raises():
    with pytest.raises(ValueError, match="unrecognised verdict schema"):
        _build({"schema": "vaara.something/v9"})


def test_non_dict_verdict_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        build_attestation_result([1, 2], issued_at=IAT, verifier_build=BUILD)


@pytest.mark.parametrize("bad", [1750000000.0, True, "now", None])
def test_non_int_iat_raises(bad):
    with pytest.raises(ValueError, match="issued_at must be an integer"):
        build_attestation_result(_chain(), issued_at=bad, verifier_build=BUILD)


@pytest.mark.parametrize("bad", ["", None, 7])
def test_bad_verifier_build_raises(bad):
    with pytest.raises(ValueError, match="verifier_build"):
        build_attestation_result(_chain(), issued_at=IAT, verifier_build=bad)


def test_result_is_json_serializable_and_deterministic():
    a = json.dumps(_build(_chain()), sort_keys=True)
    b = json.dumps(_build(_chain()), sort_keys=True)
    assert a == b


# ── conformance vectors ──────────────────────────────────────────────────────

def _cases():
    return json.loads((VECTORS / "cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_vaara_reproduces_vector_ear(case):
    doc = json.loads((VECTORS / "cases.json").read_text())
    expected = json.loads((VECTORS / "expected.json").read_text())[case["name"]]
    got = build_attestation_result(
        case["verdict"], issued_at=doc["iat"], verifier_build=doc["verifier_build"])
    assert got == expected


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_independent_checker_is_vaara_free():
    source = (VECTORS / "_check_independent.py").read_text()
    assert "import vaara" not in source and "from vaara" not in source
