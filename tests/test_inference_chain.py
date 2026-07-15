# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Inference -> fTPM weld: session manifest + the composite chain verdict.

Covers ``_inference_session`` (build / closed-schema parse / tamper / receipt
match) and ``_inference_chain_verify`` end to end: a TPM evidence chain bound to a
``vaara.inference-session/v0`` manifest is checked against the signed receipts it
claims to bind. The whole wire path runs through ``MockTPMQuoter`` (real signed
``TPMS_ATTEST`` quotes), so no TPM is needed. Receipts are HS256-signed; the chain
AK is a separate ES256 key, mirroring the real split.
"""

from __future__ import annotations

import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._inference_chain_verify import verify_inference_chain  # noqa: E402
from vaara.attestation._inference_session import (  # noqa: E402
    build_session_manifest,
    parse_session_manifest,
    session_manifest_covers_prefix,
    session_manifest_matches_receipts,
)
from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.attestation._tpm import (  # noqa: E402
    IMA_PCR,
    TPM_ALG_SHA256,
    MockTPMQuoter,
    replay_ima_pcr,
)
from vaara.attestation._tpm_binding import bind_record_to_chain_extra_data  # noqa: E402
from vaara.attestation._tpm_chain import (  # noqa: E402
    GENESIS_PREV_DIGEST,
    TPMChainLink,
    link_digest,
)
from vaara.attestation.inference import (  # noqa: E402
    InferenceOutcome,
    ModelDerived,
    RequestDeclared,
    emit_inference_attestation,
    emit_inference_receipt,
    make_inference_back_link,
    make_output_commitment,
    make_request_commitment,
)
from vaara.attestation.receipt import build_tpm_chain_document  # noqa: E402

SECRET = b"k" * 32


def _ima_line(a: str, b: str, path: str) -> str:
    return f"10 {a * 64} ima-ng sha256:{b * 64} {path}\n"


_L0 = _ima_line("a", "b", "/usr/bin/a")
_L1 = _L0 + _ima_line("c", "d", "/usr/bin/b")
_L2 = _L1 + _ima_line("e", "f", "/usr/bin/c")
_LOGS = [_L0, _L1, _L2]


@pytest.fixture
def ak_key():
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture
def ak_pem(ak_key):
    return ak_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _pair(model_ref: str, content: str):
    """A signed (attestation, receipt) inference pair, HS256."""
    md = ModelDerived(
        model_ref=model_ref,
        manifest_digest="sha256:" + "a" * 64,
        gguf_metadata_hash="sha256:" + "b" * 64,
    )
    rd = RequestDeclared(
        intent=f"inference/chat/{model_ref}",
        request_commitment=make_request_commitment(
            messages=[{"role": "user", "content": content}],
            sampling={"temperature": 0.7},
        ),
    )
    att = emit_inference_attestation(
        request_declared=rd, model_derived=md, iss="vaara-infer-proxy",
        sub=model_ref, secret_version="v1", alg="HS256", signing_material=SECRET,
    )
    outcome = InferenceOutcome(
        status="completed", completed_at="2026-06-16T00:00:00Z", tier="integrity",
        output_commitment=make_output_commitment({"content": content + "-out"}),
        eval_stats={"evalCount": 10},
    )
    rec = emit_inference_receipt(
        back_link=make_inference_back_link(att), outcome_derived=outcome,
        iss="vaara-infer-proxy", sub=model_ref, secret_version="v1", alg="HS256",
        signing_material=SECRET,
    )
    return att, rec


def _chain_over(record, ak_key, ak_pem, n=3):
    q = MockTPMQuoter(ak_key)
    links = []
    prev = GENESIS_PREV_DIGEST
    for seq, log in enumerate(_LOGS[:n]):
        pcr10 = replay_ima_pcr(log, TPM_ALG_SHA256)
        extra = bind_record_to_chain_extra_data(record, prev, seq)
        attest, sig = q.quote(
            extra, {IMA_PCR: pcr10}, clock=1000 * (seq + 1), reset_count=3,
            restart_count=0,
        )
        links.append(TPMChainLink(attest, sig, ak_pem, {IMA_PCR: pcr10}, log))
        prev = link_digest(attest)
    return build_tpm_chain_document(record, links)


# --- session manifest -------------------------------------------------------


def test_manifest_roundtrip():
    receipts = [_pair("m", "p1")[1], _pair("m", "p2")[1]]
    manifest = build_session_manifest(receipts)
    assert manifest["count"] == 2
    assert parse_session_manifest(manifest) is manifest
    assert session_manifest_matches_receipts(manifest, receipts)


def test_parse_rejects_tampered_root():
    manifest = build_session_manifest([_pair("m", "p1")[1]])
    manifest["root"] = "sha256:" + "0" * 64
    with pytest.raises(AttestationError, match="does not recompute"):
        parse_session_manifest(manifest)


def test_parse_rejects_dropped_link():
    receipts = [_pair("m", "p1")[1], _pair("m", "p2")[1]]
    manifest = build_session_manifest(receipts)
    manifest["links"].pop()  # count now disagrees with links
    with pytest.raises(AttestationError):
        parse_session_manifest(manifest)


def test_matches_is_false_on_reorder():
    r1, r2 = _pair("m", "p1")[1], _pair("m", "p2")[1]
    manifest = build_session_manifest([r1, r2])
    assert not session_manifest_matches_receipts(manifest, [r2, r1])


# --- composite weld verdict -------------------------------------------------


def test_composite_valid(ak_key, ak_pem):
    pairs = [_pair("m", "p1"), _pair("m", "p2"), _pair("m", "p3")]
    receipts = [r for _a, r in pairs]
    chain = _chain_over(build_session_manifest(receipts), ak_key, ak_pem, n=3)
    docs = [(r.to_dict(), a.to_dict()) for a, r in pairs]

    v = verify_inference_chain(chain, receipts=docs, verifying_material=SECRET)
    assert v["ok"]
    assert v["chainContinuous"]
    assert v["recordIsSession"]
    assert v["manifestMatchesReceipts"]
    assert v["receiptsAllValid"]
    assert v["inferenceTiers"] == ["integrity"]
    assert v["basis"]["weldProves"].endswith("not_inference_determinism")


def test_composite_fails_on_mutated_receipt(ak_key, ak_pem):
    pairs = [_pair("m", "p1"), _pair("m", "p2")]
    chain = _chain_over(
        build_session_manifest([r for _a, r in pairs]), ak_key, ak_pem, n=2
    )
    docs = [(r.to_dict(), a.to_dict()) for a, r in pairs]
    # Flip a status in the wire receipt after the manifest was bound.
    docs[0][0]["outcomeDerived"]["status"] = "refused"

    v = verify_inference_chain(chain, receipts=docs, verifying_material=SECRET)
    assert not v["ok"]
    assert not v["manifestMatchesReceipts"]


def test_composite_fails_on_wrong_bound_record(ak_key, ak_pem):
    receipts_a = [_pair("m", "p1")[1], _pair("m", "p2")[1]]
    pairs_b = [_pair("m", "q1"), _pair("m", "q2")]
    # Chain binds session B; we hand the verifier session A's receipts.
    chain = _chain_over(
        build_session_manifest([r for _a, r in pairs_b]), ak_key, ak_pem, n=2
    )
    docs_a = [(r.to_dict(), None) for r in receipts_a]

    v = verify_inference_chain(chain, receipts=docs_a, verifying_material=SECRET)
    assert not v["ok"]
    assert v["recordIsSession"]
    assert not v["manifestMatchesReceipts"]


def test_composite_rejects_non_session_record(ak_key, ak_pem):
    not_a_session = {"schema": "sep2828/v0", "decisionId": "d1", "signature": "x"}
    chain = _chain_over(not_a_session, ak_key, ak_pem, n=2)
    rec = _pair("m", "p1")[1]
    v = verify_inference_chain(
        chain, receipts=[(rec.to_dict(), None)], verifying_material=SECRET
    )
    assert not v["ok"]
    assert not v["recordIsSession"]


# --- prefix coverage: chatting past the last capture ------------------------


def test_covers_prefix_true_on_appended_tail():
    r1, r2, r3 = _pair("m", "p1")[1], _pair("m", "p2")[1], _pair("m", "p3")[1]
    manifest = build_session_manifest([r1, r2])  # bind only the first two
    # Exact match is false (a third receipt arrived), but the prefix is intact.
    assert not session_manifest_matches_receipts(manifest, [r1, r2, r3])
    assert session_manifest_covers_prefix(manifest, [r1, r2, r3])


def test_covers_prefix_false_when_fewer_than_bound():
    r1, r2 = _pair("m", "p1")[1], _pair("m", "p2")[1]
    manifest = build_session_manifest([r1, r2])
    # A receipt the manifest commits to is missing: not a covered prefix.
    assert not session_manifest_covers_prefix(manifest, [r1])


def test_covers_prefix_false_on_prefix_tamper():
    r1, r2, r3 = _pair("m", "p1")[1], _pair("m", "p2")[1], _pair("m", "p3")[1]
    manifest = build_session_manifest([r1, r2])
    # Reorder *within* the bound prefix: the prefix no longer rebuilds the root.
    assert not session_manifest_covers_prefix(manifest, [r2, r1, r3])


def test_composite_reports_prefix_coverage(ak_key, ak_pem):
    pairs = [_pair("m", "p1"), _pair("m", "p2"), _pair("m", "p3")]
    # Bind the first two, then hand the verifier all three (operator chatted on).
    chain = _chain_over(
        build_session_manifest([r for _a, r in pairs[:2]]), ak_key, ak_pem, n=2
    )
    docs = [(r.to_dict(), a.to_dict()) for a, r in pairs]

    v = verify_inference_chain(chain, receipts=docs, verifying_material=SECRET)
    assert not v["ok"]                       # strict verdict is unchanged
    assert not v["manifestMatchesReceipts"]  # exact match is false
    assert v["manifestCoversPrefix"]         # but the bound prefix is intact
    assert v["boundCount"] == 2
    assert v["unboundTail"] == 1
    assert v["chainContinuous"]              # the hardware chain is still good
    assert "first 2 of 3" in v["reason"]
