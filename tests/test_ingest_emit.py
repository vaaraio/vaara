"""vaara.ingest/v0: the universal sink envelope.

Every registered foreign source normalizes, then seals into one signed,
content-addressed ingest envelope. These tests run generatively over the
existing normalize input corpus, so a new SourceProfile that drops an input
fixture is covered here with no new test code: the per-source vector grind
becomes a loop over the registry.

Ingest always canonicalizes to sign, so the attestation extra (rfc8785) is
the baseline; the whole module skips without it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
import sys
from pathlib import Path

import pytest

rfc8785 = pytest.importorskip("rfc8785")  # the only third-party dep recompute needs

from vaara.attestation.receipt import (  # noqa: E402
    INGEST_SCHEMA,
    NORMALIZED_EVIDENCE_SCHEMA,
    emit_ingest_receipt,
    normalize,
    verify_ingest_signature,
)

INPUTS = Path(__file__).resolve().parent / "vectors" / "normalize_v0" / "inputs"
KEY = b"ingest-conformance-shared-secret-0001"

# Fixed nonce + iat make the signed envelope reproducible: a golden vector,
# not a fresh record each run.
FIXED = dict(
    iss="did:web:vaara.io",
    sub="sink",
    secret_version="k1",
    alg="HS256",
    signing_material=KEY,
    nonce="ingest-fixed-nonce-000000",
    iat="2026-06-23T00:00:00Z",
)


def _corpus():
    return sorted(p.stem for p in INPUTS.glob("*.json"))


def _doc(name: str) -> dict:
    return json.loads((INPUTS / f"{name}.json").read_text())


def _emit(name: str, **over):
    return emit_ingest_receipt(normalized=normalize(_doc(name)), **{**FIXED, **over})


# ── Every source seals and round-trips ────────────────────────────────────────


@pytest.mark.parametrize("name", _corpus())
def test_every_source_seals_and_verifies(name):
    r = _emit(name)
    assert r.record["schema"] == INGEST_SCHEMA
    assert r.record["sourceFormat"] == r.evidence["sourceFormat"]
    ref = r.record["evidenceRef"]
    assert ref["schema"] == NORMALIZED_EVIDENCE_SCHEMA
    assert ref["canonicalization"] == "JCS"
    assert ref["digest"].startswith("sha256:")
    assert verify_ingest_signature(r.record, r.evidence, KEY)
    # The evidence object is the normalized source, unaltered.
    assert r.evidence == normalize(_doc(name)).to_dict()


@pytest.mark.parametrize("name", _corpus())
def test_digest_recomputes_with_zero_vaara_import(name):
    # rfc8785 + sha256 only: a third party reproduces the content address.
    r = _emit(name)
    recomputed = "sha256:" + hashlib.sha256(rfc8785.dumps(r.evidence)).hexdigest()
    assert recomputed == r.record["evidenceRef"]["digest"]


@pytest.mark.parametrize("name", _corpus())
def test_hs256_signature_recomputes_with_zero_vaara_import(name):
    # HS256 is pure stdlib hmac: the full signature verifies without Vaara.
    r = _emit(name)
    body = {k: v for k, v in r.record.items() if k != "signature"}
    expected = hmac.new(KEY, rfc8785.dumps(body), hashlib.sha256).hexdigest()
    assert expected == r.record["signature"]


def test_emit_is_deterministic_under_fixed_nonce_and_iat():
    assert _emit("sep2817_single").record == _emit("sep2817_single").record


# ── The honest gap report travels under the signature ─────────────────────────


def test_tampering_the_gap_report_breaks_verification():
    r = _emit("sep2817_single")
    assert r.evidence["missing"]  # there is a real gap to forge away
    forged = {**r.evidence, "missing": []}
    assert not verify_ingest_signature(r.record, forged, KEY)


def test_tampering_a_proof_field_breaks_verification():
    r = _emit("sep2643_url_denial")
    # Flip the refused outcome to allowed: must not verify.
    forged = {**r.evidence, "sep2828": {"outcomeDerived": {"status": "allowed"}}}
    assert not verify_ingest_signature(r.record, forged, KEY)


def test_sourceformat_mismatch_breaks_verification():
    r = _emit("sep2817_single")
    forged = {**r.evidence, "sourceFormat": "sep2643"}
    assert not verify_ingest_signature(r.record, forged, KEY)


def test_unknown_source_still_seals_honestly():
    r = _emit("unknown")
    assert r.record["sourceFormat"] == "unknown"
    assert r.evidence["recognized"] is False
    assert verify_ingest_signature(r.record, r.evidence, KEY)


# ── Completeness ──────────────────────────────────────────────────────────────


def test_single_record_completeness_is_the_default():
    assert _emit("sep2817_single").record["completeness"] == {
        "seq": 1,
        "runningCount": 1,
    }


def test_stream_completeness_is_carried_and_signed():
    r = _emit("sep2817_single", completeness={"seq": 3, "runningCount": 7})
    assert r.record["completeness"] == {"seq": 3, "runningCount": 7}
    assert verify_ingest_signature(r.record, r.evidence, KEY)
    forged = {**r.record, "completeness": {"seq": 1, "runningCount": 1}}
    assert not verify_ingest_signature(forged, r.evidence, KEY)


# ── Signing modes, surface, and rejects ───────────────────────────────────────


def test_es256_round_trip():
    from cryptography.hazmat.primitives.asymmetric import ec

    sk = ec.generate_private_key(ec.SECP256R1())
    r = emit_ingest_receipt(
        normalized=normalize(_doc("sep2817_single")),
        iss="i", sub="s", secret_version="k1", alg="ES256", signing_material=sk,
    )
    assert verify_ingest_signature(r.record, r.evidence, sk.public_key())
    assert not verify_ingest_signature(
        r.record, {**r.evidence, "missing": []}, sk.public_key()
    )


def test_optional_evidence_ref_locator_is_bound():
    r = _emit("sep2817_single", evidence_ref="https://logs.example/abc")
    assert r.record["evidenceRef"]["ref"] == "https://logs.example/abc"
    assert verify_ingest_signature(r.record, r.evidence, KEY)


def test_unsupported_alg_is_rejected():
    from vaara.attestation.sep2787 import AttestationError

    with pytest.raises(AttestationError):
        emit_ingest_receipt(
            normalized=normalize(_doc("unknown")),
            iss="i", sub="s", secret_version="k1", alg="NONE", signing_material=b"x",
        )


def test_public_reexport_is_wired():
    from vaara.attestation import _ingest_emit

    assert emit_ingest_receipt is _ingest_emit.emit_ingest_receipt
    assert verify_ingest_signature is _ingest_emit.verify_ingest_signature


# ── Published conformance corpus (generated, not hand-authored) ───────────────

VECTORS = Path(__file__).resolve().parent / "vectors" / "ingest_v0"


def test_committed_vectors_match_fresh_emit():
    # Drift guard: if emit logic changes without regenerating the corpus, the
    # committed pairs stop matching and this fails, forcing a conscious regen.
    meta = json.loads((VECTORS / "corpus.json").read_text())
    secret = bytes.fromhex(meta["sharedSecretHex"])
    f = meta["fixed"]
    for name in meta["cases"]:
        committed = json.loads((VECTORS / "cases" / f"{name}.json").read_text())
        r = emit_ingest_receipt(
            normalized=normalize(_doc(name)),
            iss=f["iss"], sub=f["sub"], secret_version=f["secretVersion"],
            alg="HS256", signing_material=secret, nonce=f["nonce"], iat=f["iat"],
        )
        assert r.record == committed["record"], f"{name}: record drift; regenerate"
        assert r.evidence == committed["evidence"], f"{name}: evidence drift; regen"


def test_corpus_tracks_the_full_input_registry():
    # The corpus is a loop over the registry's input fixtures: every one is in.
    meta = json.loads((VECTORS / "corpus.json").read_text())
    assert sorted(meta["cases"]) == _corpus()


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ── CLI: vaara ingest ─────────────────────────────────────────────────────────


def test_cli_ingest_hs256_seals_and_verifies(tmp_path, capsys):
    from vaara.cli import main

    sec = tmp_path / "sec.bin"
    sec.write_bytes(KEY)
    rc = main([
        "ingest", str(INPUTS / "sep2817_single.json"),
        "--hs256-secret-file", str(sec),
        "--iss", "did:web:vaara.io", "--sub", "sink", "--secret-version", "k1",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["record"]["schema"] == INGEST_SCHEMA
    assert verify_ingest_signature(out["record"], out["evidence"], KEY)


def test_cli_ingest_writes_out_file(tmp_path):
    from vaara.cli import main

    sec = tmp_path / "sec.bin"
    sec.write_bytes(KEY)
    out = tmp_path / "receipt.json"
    rc = main([
        "ingest", str(INPUTS / "sep2643_url_denial.json"),
        "--hs256-secret-file", str(sec), "--out", str(out),
    ])
    assert rc == 0
    d = json.loads(out.read_text())
    assert verify_ingest_signature(d["record"], d["evidence"], KEY)


def test_cli_ingest_requires_a_signing_key(capsys):
    from vaara.cli import main

    rc = main(["ingest", str(INPUTS / "unknown.json")])
    assert rc == 2
    assert "required" in capsys.readouterr().err
