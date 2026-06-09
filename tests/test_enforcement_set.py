"""Batch enforcement verification and the `vaara verify-enforcements` command.

The batch twin of `verify-enforcement`: an auditor points the tool at a
directory of (record, report, VCEK) triples and gets the roll-up a single-file
check cannot give. Covers the `enforcement_set_v0` vectors through both Vaara
and the standalone Vaara-free checker, the module behaviour (the per-tier tally,
the pinning-coverage gap, gating on a non-binding triple), and the CLI end to
end including stem-based triple discovery. Runs the binding crypto, so the suite
skips when the attestation extra is absent.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation.receipt import (  # noqa: E402
    ENFORCEMENT_SET_SCHEMA_NAME,
    check_enforcement_set,
)
from vaara.cli import main  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors" / "enforcement_set_v0"
SIBLING = Path(__file__).resolve().parent / "vectors" / "enforcement_attestation_v0"

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


def _cases() -> dict:
    return {c["name"]: c
            for c in json.loads((SIBLING / "cases.json").read_text())["cases"]}


def _sets() -> dict:
    return json.loads((VECTORS / "sets.json").read_text())["sets"]


def _expected() -> dict:
    return json.loads((VECTORS / "expected.json").read_text())


def _triples_for(spec: dict, cases: dict):
    out = []
    for case_name in spec["cases"]:
        case = cases[case_name]
        out.append((
            case_name,
            case["record"],
            base64.b64decode(case["report_b64"]),
            _jwk_to_pem(case["vcek_jwk"]),
        ))
    return out


# ── Vectors ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_expected().keys()))
def test_module_reproduces_vector_rollup(name):
    cases, sets, want = _cases(), _sets(), _expected()[name]
    spec = sets[name]
    report = check_enforcement_set(
        _triples_for(spec, cases),
        expected_measurement=spec["expected_measurement"],
        strict=spec["strict"],
    )
    got = {k: report.to_dict()[k] for k in SUMMARY_KEYS}
    assert got == want


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_public_reexport_is_wired():
    from vaara.attestation.receipt import check_enforcement_set as public
    assert public is check_enforcement_set


# ── Unit behaviour ────────────────────────────────────────────────────────────


def test_clean_set_passes_without_pinning_gap_when_measurement_pinned():
    cases, sets = _cases(), _sets()
    spec = sets["all_bound"]
    report = check_enforcement_set(
        _triples_for(spec, cases),
        expected_measurement=spec["expected_measurement"], strict=False)
    assert report.ok
    assert report.passed == report.total == 2
    assert report.pinning_gap is False


def test_unpinned_set_passes_but_flags_the_pinning_gap():
    cases = _cases()
    report = check_enforcement_set(_triples_for(_sets()["unpinned_only"], cases))
    assert report.ok  # bound is enough for a default-mode pass
    assert report.measurement_pinned == 0
    assert report.pinning_gap is True  # advisory, did not gate


def test_a_non_binding_triple_gates_the_set():
    cases = _cases()
    report = check_enforcement_set(_triples_for(_sets()["mixed_failure"], cases))
    assert not report.ok
    assert report.passed == 1
    failing = [e for e in report.entries if not e.ok]
    assert len(failing) == 2  # bad signature + report bound to another record


def test_strict_is_unreachable_even_for_a_clean_pinned_set():
    cases, sets = _cases(), _sets()
    spec = sets["strict_unreachable"]
    report = check_enforcement_set(
        _triples_for(spec, cases),
        expected_measurement=spec["expected_measurement"], strict=True)
    assert not report.ok  # no validated VCEK chain in v0
    assert report.measurement_pinned == 1  # pinned, yet strict still fails


def test_malformed_triple_is_a_failing_entry_not_a_raise():
    cases = _cases()
    good = _triples_for(_sets()["unpinned_only"], cases)[0]
    # A report that will not parse yields an unverified verdict (no raise), so
    # this is a failing entry that gates rather than an exception.
    bad = ("broken", {"version": 1}, b"too short", good[3])
    report = check_enforcement_set([good, bad])
    assert not report.ok
    broken = next(e for e in report.entries if e.name == "broken")
    assert broken.ok is False


def test_attested_tier_never_appears_in_v0():
    cases, sets = _cases(), _sets()
    report = check_enforcement_set(
        _triples_for(sets["all_bound"], cases),
        expected_measurement=sets["all_bound"]["expected_measurement"])
    assert "attested" not in report.tier_counts


def test_empty_set_is_ok_without_a_pinning_gap():
    report = check_enforcement_set([])
    assert report.ok and report.total == 0
    assert report.pinning_gap is False  # no loaded record means no gap to flag


# ── CLI ───────────────────────────────────────────────────────────────────────


def _materialize(tmp_path: Path, case_names: list[str]) -> Path:
    """Write NAME.record.json / NAME.report.bin / NAME.vcek.pem triples."""
    cases = _cases()
    for name in case_names:
        case = cases[name]
        (tmp_path / f"{name}.record.json").write_text(json.dumps(case["record"]))
        (tmp_path / f"{name}.report.bin").write_bytes(
            base64.b64decode(case["report_b64"]))
        (tmp_path / f"{name}.vcek.pem").write_bytes(_jwk_to_pem(case["vcek_jwk"]))
    return tmp_path


def test_cli_clean_set_exit_0(tmp_path, capsys):
    _materialize(tmp_path, ["clean_bound"])
    rc = main(["verify-enforcements", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out and "1/1 bind" in out


def test_cli_failing_set_exit_1(tmp_path, capsys):
    _materialize(tmp_path, ["clean_bound", "bad_signature"])
    rc = main(["verify-enforcements", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAILED" in out
    assert "bad_signature" in out


def test_cli_json_output(tmp_path, capsys):
    _materialize(tmp_path, ["clean_bound"])
    rc = main(["verify-enforcements", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["schema"] == ENFORCEMENT_SET_SCHEMA_NAME
    assert payload["incomplete"] == []


def test_cli_missing_companion_gates(tmp_path, capsys):
    _materialize(tmp_path, ["clean_bound"])
    (tmp_path / "clean_bound.vcek.pem").unlink()  # drop one companion
    rc = main(["verify-enforcements", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert payload["incomplete"][0]["name"] == "clean_bound.record.json"
    assert "vcek.pem" in payload["incomplete"][0]["error"]


def test_cli_not_a_directory(tmp_path, capsys):
    rc = main(["verify-enforcements", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_no_matching_files(tmp_path, capsys):
    rc = main(["verify-enforcements", str(tmp_path)])
    assert rc == 2
    assert "no files matched" in capsys.readouterr().err


def test_cli_bad_expected_measurement(tmp_path, capsys):
    _materialize(tmp_path, ["clean_bound"])
    rc = main(["verify-enforcements", str(tmp_path),
               "--expected-measurement", "xyz"])
    assert rc == 1
    assert "not valid hex" in capsys.readouterr().err
