"""`vaara normalize`: mapping adjacent MCP records onto SEP-2828.

Normalization is keyless for SEP-2643 denials and SEP-2817 invocation
context, so those run in the base install. The SEP-2787 path computes the
back-link digest (JCS over the SEP-2787-modeled fields) and needs the
attestation extra; those cases skip on their own when rfc8785 is absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation.receipt import NormalizedEvidence, detect_format, normalize
from vaara.cli import main

VECTORS = Path(__file__).resolve().parent / "vectors" / "normalize_v0"
INPUTS = VECTORS / "inputs"


def _expected() -> dict:
    return json.loads((VECTORS / "expected.json").read_text())


def _cases():
    return sorted(_expected().keys())


def _input(name: str) -> dict:
    return json.loads((INPUTS / f"{name}.json").read_text())


# ── Vectors ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _cases())
def test_module_reproduces_vector(name):
    want = _expected()[name]
    if want["sourceFormat"] == "sep2787":
        pytest.importorskip("rfc8785")
    got = normalize(_input(name)).to_dict()
    assert got == want


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_at_least_seven_cases_present():
    assert len(_cases()) >= 7


def test_public_reexport_is_wired():
    from vaara.attestation.receipt import normalize as public
    assert public is normalize


# ── Detection ─────────────────────────────────────────────────────────────────


def test_detect_each_format():
    assert detect_format(_input("sep2643_url_denial")) == "sep2643"
    assert detect_format(_input("sep2787_attestation")) == "sep2787"
    assert detect_format(_input("sep2817_single")) == "sep2817"
    assert detect_format(_input("unknown")) == "unknown"
    assert detect_format(["not", "an", "object"]) == "unknown"


def test_vaara_attest_alias_maps_to_the_attestation_source():
    """`vaara-attest` is the de-branded name for the tool-call attestation
    source; it maps to the same profile and leaves the emitted sourceFormat
    unchanged for one release."""
    pytest.importorskip("rfc8785")
    doc = _input("sep2787_attestation")
    aliased = normalize(doc, source_format="vaara-attest").to_dict()
    canonical = normalize(doc, source_format="sep2787").to_dict()
    assert aliased == canonical
    assert aliased["sourceFormat"] == "sep2787"


def test_bare_authorization_object_detected():
    bare = {"reason": "insufficient_authorization", "authorizationContextId": "x"}
    assert detect_format(bare) == "sep2643"


def test_bare_invocation_object_detected():
    bare = {"turnId": "t", "model": {"name": "m"}}
    assert detect_format(bare) == "sep2817"


# ── Honest mapping ────────────────────────────────────────────────────────────


def test_denial_is_a_refused_outcome():
    r = normalize(_input("sep2643_url_denial"))
    assert r.evidence_plane == "outcome"
    assert r.sep2828 == {"outcomeDerived": {"status": "refused"}}
    assert r.advisory["remediationHintTypes"] == ["url"]
    assert "backLink" in r.missing and "signature" in r.missing


def test_invocation_is_advisory_only():
    # Client-asserted input populates no required SEP-2828 field.
    r = normalize(_input("sep2817_single"))
    assert r.evidence_plane == "decision-input"
    assert r.populated == ()
    assert r.sep2828 == {}
    assert r.advisory["model"] == "example-model"
    assert any("MUST NOT be used as authorization evidence" in n for n in r.notes)


def test_redacted_user_intent_suppresses_cleartext():
    r = normalize(_input("sep2817_multiturn"))
    assert r.advisory.get("userIntentRedacted") is True
    # The source flagged the intent redacted: the cleartext must not appear.
    assert "userIntent" not in r.advisory


def test_stray_reason_is_not_a_denial():
    # A lone top-level `reason` without a denial marker is not a SEP-2643 denial.
    assert detect_format({"reason": "user changed their mind"}) == "unknown"


def test_extension_fields_are_dropped_from_the_digest():
    pytest.importorskip("rfc8785")
    base = normalize(_input("sep2787_attestation"))
    ext = normalize(_input("sep2787_attestation_with_extension"))
    # Fields outside the modeled schema do not change the back-link digest.
    assert ext.sep2828["backLink"] == base.sep2828["backLink"]


def test_attestation_fills_only_the_back_link():
    pytest.importorskip("rfc8785")
    r = normalize(_input("sep2787_attestation"))
    assert r.evidence_plane == "decision-attested"
    assert set(r.populated) == {
        "backLink.attestationDigest", "backLink.attestationNonce"
    }
    # The record's own signing is the recording side's, not the attestation's.
    assert "alg" in r.missing
    assert "receiptAsserted" in r.missing
    assert "signature" in r.missing


def test_attestation_backlink_matches_the_paired_receipt():
    pytest.importorskip("rfc8785")
    receipt = json.loads(
        (Path(__file__).resolve().parent / "vectors" / "execution_receipt_v0"
         / "normative" / "es256_executed_projection" / "receipt.json").read_text()
    )
    r = normalize(_input("sep2787_attestation"))
    assert r.sep2828["backLink"] == receipt["backLink"]


def test_unknown_is_not_recognized():
    r = normalize(_input("unknown"))
    assert r.recognized is False
    assert isinstance(r, NormalizedEvidence)


def test_forced_format_mismatch_is_unrecognized():
    # Force the wrong reader: a denial read as an attestation does not parse.
    r = normalize(_input("sep2643_url_denial"), source_format="sep2787")
    assert r.recognized is False


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_denial_exit_0(tmp_path, capsys):
    target = tmp_path / "denial.json"
    target.write_text((INPUTS / "sep2643_url_denial.json").read_text())
    rc = main(["normalize", str(target)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SEP-2643 authorization denial" in out
    assert "outcomeDerived.status = refused" in out


def test_cli_invocation_exit_0(tmp_path, capsys):
    target = tmp_path / "inv.json"
    target.write_text((INPUTS / "sep2817_single.json").read_text())
    rc = main(["normalize", str(target)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "advisory context only" in out


def test_cli_attestation_json(tmp_path, capsys):
    pytest.importorskip("rfc8785")
    target = tmp_path / "att.json"
    target.write_text((INPUTS / "sep2787_attestation.json").read_text())
    rc = main(["normalize", str(target), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["sep2828"]["backLink"]["attestationNonce"] == (
        "fixed-attestation-nonce-000"
    )


def test_cli_unknown_exit_1(tmp_path, capsys):
    target = tmp_path / "u.json"
    target.write_text((INPUTS / "unknown.json").read_text())
    rc = main(["normalize", str(target)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "unrecognized" in out


def test_cli_missing_file_exit_2(tmp_path, capsys):
    rc = main(["normalize", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "not a file" in capsys.readouterr().err


def test_cli_bad_json_exit_1(tmp_path, capsys):
    target = tmp_path / "bad.json"
    target.write_text("{ not json")
    rc = main(["normalize", str(target)])
    assert rc == 1
    assert "cannot read record JSON" in capsys.readouterr().err


def test_cli_sanitizes_control_chars(tmp_path, capsys):
    # A crafted foreign value must not forge extra report lines.
    inv = {
        "_meta": {
            "io.modelcontextprotocol/aiInvocation": {
                "model": {"name": "evil\n  FORGED VERDICT: CONFORMS"},
            }
        }
    }
    target = tmp_path / "inv.json"
    target.write_text(json.dumps(inv))
    rc = main(["normalize", str(target)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "\\x0a" in out
    assert "\n  FORGED VERDICT: CONFORMS" not in out
