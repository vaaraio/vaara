# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The verification commands, run as a user runs them: files in, exit code out.

Every verifier here had its library function tested and its command
never run. ``vaara attribute-check``, ``verify-retained``, ``verify-handoff``,
``build-handoff``, ``verify-tpm-binding``, ``verify-tpm-chain`` and
``export-attestation-result`` are the doors a relying party or a regulator
actually opens, so each one is driven through ``vaara.cli.main`` on the
published vectors where there are any, and on the same builders the
library tests use where there are none. The exit code is the contract a
CI step reads, so it is asserted beside the printed verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.cli import main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests" / "vectors"


def _write(path: Path, obj) -> str:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def _json_out(capsys) -> dict:
    out = capsys.readouterr().out
    return json.loads(out)


# ── attribute-check ────────────────────────────────────────────────────────

def _attribute_cases():
    folder = VECTORS / "attribute_attestation_zk_v0"
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))["cases"]
    for case in sorted((folder / "cases").glob("*.json")):
        yield pytest.param(case, expected[case.stem], id=case.stem)


@pytest.mark.parametrize("case,expected", list(_attribute_cases()))
def test_attribute_check_reproduces_every_vector_verdict(case, expected, capsys):
    bundle = json.loads(case.read_text(encoding="utf-8"))
    query = bundle["query"]
    argv = ["attribute-check", str(case), "--name", query["name"],
            "--min-source", query["minimum_source"],
            "--key", str(case.parent.parent / bundle["issuer_key"]), "--json"]
    predicate = query["predicate"]
    if predicate.get("lower") is not None:
        argv += ["--at-least", str(predicate["lower"])]
    if predicate.get("upper") is not None:
        argv += ["--at-most", str(predicate["upper"])]
    if query.get("subject_id"):
        argv += ["--subject", query["subject_id"]]
    for issuer in query.get("accepted_issuers") or []:
        argv += ["--issuer", issuer]

    rc = main(argv)
    decision = _json_out(capsys)

    assert decision["state"] == expected["expected_state"]
    assert decision["reason"] == expected["expected_reason"]
    assert rc == (0 if decision["state"] == "accepted" else 1)


def test_attribute_check_without_a_key_accepts_nothing_and_says_why(capsys):
    case = VECTORS / "attribute_attestation_zk_v0" / "cases" / "pos_at_least_holds.json"
    rc = main(["attribute-check", str(case), "--name", "age", "--at-least", "18",
               "--min-source", "measured"])
    out, err = capsys.readouterr()
    assert rc == 1
    assert "no --key given" in err
    assert out.split()[0] != "ACCEPTED"


def test_attribute_check_needs_a_predicate(capsys):
    case = VECTORS / "attribute_attestation_zk_v0" / "cases" / "pos_at_least_holds.json"
    rc = main(["attribute-check", str(case), "--name", "age", "--min-source", "measured"])
    assert rc == 2
    assert "--at-least" in capsys.readouterr().err


# ── verify-retained ────────────────────────────────────────────────────────

def _retained_cases():
    folder = VECTORS / "key_rotation_v0"
    doc = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    for case in doc["cases"]:
        yield pytest.param(case, expected[case["name"]], id=case["name"])


@pytest.mark.parametrize("case,expected", list(_retained_cases()))
def test_verify_retained_reproduces_every_vector_verdict(case, expected, tmp_path, capsys):
    argv = ["verify-retained", _write(tmp_path / "record.json", case["receipt"]),
            "--did-document", _write(tmp_path / "did.json", case["didDocument"]), "--json"]
    if case.get("anchoredTime"):
        argv += ["--anchored-time", case["anchoredTime"]]

    rc = main(argv)
    got = _json_out(capsys)

    assert {k: got[k] for k in expected} == expected
    assert rc == (0 if expected["verifiable"] else 1)


def test_verify_retained_refuses_a_record_that_is_not_one(tmp_path, capsys):
    folder = VECTORS / "key_rotation_v0"
    case = json.loads((folder / "cases.json").read_text(encoding="utf-8"))["cases"][0]
    rc = main(["verify-retained", _write(tmp_path / "record.json", {"not": "a record"}),
               "--did-document", _write(tmp_path / "did.json", case["didDocument"])])
    assert rc == 1
    assert "not a parseable record" in capsys.readouterr().err


# ── verify-handoff and build-handoff ───────────────────────────────────────

def _handoff_cases():
    folder = VECTORS / "cross_org_handoff_v0"
    doc = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    for case in doc["cases"]:
        yield pytest.param(case, expected[case["name"]], id=case["name"])


@pytest.mark.parametrize("case,expected", list(_handoff_cases()))
def test_verify_handoff_reproduces_every_vector_verdict(case, expected, tmp_path, capsys):
    argv = ["verify-handoff", _write(tmp_path / "package.json", case["package"]), "--json"]
    if case.get("anchoredTime"):
        argv += ["--anchored-time", case["anchoredTime"]]
    else:
        argv += ["--no-anchor"]
    if case.get("trustedDidDocument"):
        argv += ["--trusted-did-document",
                 _write(tmp_path / "trusted.json", case["trustedDidDocument"])]
    if case.get("strict"):
        argv += ["--strict"]

    rc = main(argv)
    got = _json_out(capsys)

    assert {k: got[k] for k in expected} == expected
    assert rc == (0 if expected["ok"] else 1)


def test_build_handoff_writes_a_package_verify_handoff_accepts(tmp_path, capsys):
    from tests.test_cross_org_handoff import DID, _in_window_doc, _receipt

    key = ec.generate_private_key(ec.SECP256R1())
    record = _write(tmp_path / "record.json", _receipt(key))
    did = _write(tmp_path / "did.json", _in_window_doc(key))
    package = tmp_path / "handoff.json"

    rc = main(["build-handoff", "--record", record, "--did-document", did,
               "--producer", DID, "--out", str(package)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "record verifiable: True" in out
    assert "anchor present:    False" in out

    rc = main(["verify-handoff", str(package), "--no-anchor", "--json"])
    verdict = _json_out(capsys)
    assert rc == 0
    assert verdict["ok"] and verdict["verifiable"] and verdict["integrity_ok"]
    assert verdict["corroborated"] is False


def test_build_handoff_without_out_prints_the_package(tmp_path, capsys):
    from tests.test_cross_org_handoff import _in_window_doc, _receipt

    key = ec.generate_private_key(ec.SECP256R1())
    rc = main(["build-handoff", "--record", _write(tmp_path / "r.json", _receipt(key)),
               "--did-document", _write(tmp_path / "d.json", _in_window_doc(key))])
    assert rc == 0
    package = _json_out(capsys)
    assert package["schema"].startswith("vaara.") and "manifest" in package


def test_build_handoff_refuses_a_producer_that_is_not_the_issuer(tmp_path, capsys):
    from tests.test_cross_org_handoff import _in_window_doc, _receipt

    key = ec.generate_private_key(ec.SECP256R1())
    rc = main(["build-handoff", "--record", _write(tmp_path / "r.json", _receipt(key)),
               "--did-document", _write(tmp_path / "d.json", _in_window_doc(key)),
               "--producer", "did:web:somebody-else.example", "--out", str(tmp_path / "out.json")])
    assert rc == 1
    assert "cannot assemble handoff" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_build_handoff_refuses_a_record_it_cannot_pin(tmp_path, capsys):
    from tests.test_cross_org_handoff import _in_window_doc

    key = ec.generate_private_key(ec.SECP256R1())
    rc = main(["build-handoff", "--record", _write(tmp_path / "r.json", {"schema": "nope"}),
               "--did-document", _write(tmp_path / "d.json", _in_window_doc(key)),
               "--out", str(tmp_path / "out.json")])
    assert rc == 1
    assert "cannot assemble handoff" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


# ── verify-tpm-binding, verify-tpm-chain, export-attestation-result ────────

def _ak():
    from cryptography.hazmat.primitives import serialization

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return key, pem


def _tpm_bundle() -> dict:
    from tests.test_tpm_binding import _ima, _quote, _record
    from vaara.attestation._tpm import IMA_PCR
    from vaara.attestation.receipt import build_tpm_bundle_document

    key, pem = _ak()
    record = _record()
    log, pcr10 = _ima()
    attest, sig = _quote(key, record, pcr10)
    return build_tpm_bundle_document(record, attest, sig, pem, {IMA_PCR: pcr10}, log,
                                     bank="sha256", expected_ima_pcr=pcr10.hex())


def _tpm_chain() -> dict:
    from tests.test_tpm_chain import _build_links, _record
    from vaara.attestation.receipt import build_tpm_chain_document

    key, pem = _ak()
    record = _record()
    return build_tpm_chain_document(record, _build_links(record, key, pem))


def test_verify_tpm_binding_passes_a_pinned_bundle(tmp_path, capsys):
    bundle = _write(tmp_path / "bundle.json", _tpm_bundle())
    rc = main(["verify-tpm-binding", bundle, "--json"])
    verdict = _json_out(capsys)
    assert rc == 0
    assert verdict["ok"] is True and verdict["tier"] == "pcr_pinned"
    assert verdict["ak_chain_basis"] == "caller_supplied_unverified"

    rc = main(["verify-tpm-binding", bundle])
    out = capsys.readouterr().out
    assert rc == 0 and "pcr_pinned" in out and "verdict: OK" in out


def test_verify_tpm_binding_strict_refuses_an_ak_nobody_validated(tmp_path, capsys):
    """Strict wants an AK validated to a vendor root, which v0 cannot give."""
    bundle = _write(tmp_path / "bundle.json", _tpm_bundle())
    rc = main(["verify-tpm-binding", bundle, "--strict", "--json"])
    verdict = _json_out(capsys)
    assert rc == 1
    assert verdict["ok"] is False and verdict["strict"] is True
    assert "strict mode requires an AK validated" in verdict["reason"]


def test_verify_tpm_binding_fails_a_bundle_whose_pin_does_not_match(tmp_path, capsys):
    doc = _tpm_bundle()
    doc["expectedImaPcr"] = "ab" * 32
    rc = main(["verify-tpm-binding", _write(tmp_path / "bundle.json", doc), "--json"])
    verdict = _json_out(capsys)
    assert rc == 1
    assert verdict["ok"] is False


def test_verify_tpm_binding_refuses_the_wrong_schema(tmp_path, capsys):
    doc = _tpm_bundle()
    doc["schema"] = "vaara.something-else/v0"
    rc = main(["verify-tpm-binding", _write(tmp_path / "bundle.json", doc)])
    assert rc == 1
    assert "vaara verify-tpm-binding:" in capsys.readouterr().err


def test_verify_tpm_chain_passes_a_continuous_chain(tmp_path, capsys):
    chain = _write(tmp_path / "chain.json", _tpm_chain())
    rc = main(["verify-tpm-chain", chain, "--json"])
    verdict = _json_out(capsys)
    assert rc == 0
    assert verdict["ok"] is True and verdict["tier"] == "continuous"

    rc = main(["verify-tpm-chain", chain])
    assert rc == 0 and "continuous" in capsys.readouterr().out


def test_verify_tpm_chain_refuses_an_empty_chain(tmp_path, capsys):
    doc = _tpm_chain()
    doc["links"] = []
    rc = main(["verify-tpm-chain", _write(tmp_path / "chain.json", doc)])
    assert rc == 1
    assert "vaara verify-tpm-chain:" in capsys.readouterr().err


def test_a_verdict_json_feeds_export_attestation_result(tmp_path, capsys):
    """The documented pipeline: verify --json, then re-express the verdict as EAR."""
    main(["verify-tpm-chain", _write(tmp_path / "chain.json", _tpm_chain()), "--json"])
    verdict = _write(tmp_path / "verdict.json", _json_out(capsys))

    out = tmp_path / "ear.json"
    rc = main(["export-attestation-result", verdict, "--iat", "1760140800", "--out", str(out)])
    err = capsys.readouterr().err
    assert rc == 0
    assert "wrote" in err and "ear_status=" in err
    ear = json.loads(out.read_text(encoding="utf-8"))
    assert ear["ear_status"] in ("warning", "contraindicated", "none")
    assert ear["iat"] == 1760140800


def _ear_cases():
    folder = VECTORS / "attestation_result_v0"
    doc = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    for case in doc["cases"]:
        yield pytest.param(case, doc["iat"], expected[case["name"]], id=case["name"])


@pytest.mark.parametrize("case,iat,expected", list(_ear_cases()))
def test_export_attestation_result_reproduces_every_vector(case, iat, expected, tmp_path, capsys):
    rc = main(["export-attestation-result", _write(tmp_path / "v.json", case["verdict"]),
               "--iat", str(iat)])
    got = _json_out(capsys)
    assert rc == 0
    # The verifier build names the version that ran; everything else is the vector's.
    assert got["ear_verifier_id"]["build"].startswith("vaara ")
    got["ear_verifier_id"]["build"] = expected["ear_verifier_id"]["build"]
    assert got == expected


def test_export_attestation_result_refuses_a_verdict_it_cannot_map(tmp_path, capsys):
    rc = main(["export-attestation-result", _write(tmp_path / "v.json", {"ok": True})])
    assert rc == 1
    assert "vaara export-attestation-result:" in capsys.readouterr().err
