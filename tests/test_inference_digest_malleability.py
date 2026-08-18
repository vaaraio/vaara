# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Receipt and attestation identity must survive ECDSA signature malleability.

An ES256 signature is not byte-unique for one signing act. Given a valid
``(r, s)`` over P-256, ``(r, n - s)`` is also valid over the same payload and
the same key, and ``verify_es256`` accepts it because it decodes the pair and
re-encodes to DER. Nothing is forged: an observer in transit can produce the
twin without holding the key, and both copies carry identical authority.

That is harmless until something takes identity from bytes that include the
signature. ``inference_receipt_digest`` did, by its own docstring, and its
value is used as ``receiptDigest`` in the session manifest and as
``subject_receipt_digest`` in the crosscheck. One signing act could therefore
produce two receipts that both verify and disagree about which receipt they
are. The stated intent, pinning every receipt byte-for-byte, is the thing
malleability defeats, because the bytes are not unique for one act.

The rest of the tree already had this right. ``_signed_payload_digest`` in
``vaara.audit.timeanchor`` hashes only the signed blocks under JCS, which is
why ``anchoredDigest`` was never exposed. These digests now use the module's
own ``_receipt_signing_payload`` and ``_attestation_signing_payload``, so
identity comes from what was signed rather than from one encoding of the
signature over it.

Raised on the SCITT list by Anton Sokolov, 2026-08-18, and measured against
this implementation the same day: 200 of 200 signatures produced a verifying
twin with a different digest.
"""
from __future__ import annotations

import importlib.util
from dataclasses import replace

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._attest_signing import sign_es256, verify_es256  # noqa: E402
from vaara.attestation.inference import (  # noqa: E402
    InferenceOutcome,
    ModelDerived,
    RequestDeclared,
    emit_inference_attestation,
    emit_inference_receipt,
    inference_attestation_digest,
    inference_receipt_digest,
    make_inference_back_link,
    make_output_commitment,
    make_request_commitment,
)

# Order of the P-256 base point (SEC 2, secp256r1).
P256_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

MESSAGES = [{"role": "user", "content": "Summarize the Q3 report."}]
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 42}
OUTPUT = {"content": "The Q3 report covers three regions.", "toolCalls": []}
MODEL = ModelDerived(
    model_ref="qwen3:30b-a3b",
    manifest_digest="sha256:" + "a" * 64,
    gguf_metadata_hash="sha256:" + "b" * 64,
    quantization="Q4_K_M",
    param_count="30B",
)


def malleate(signature_hex: str) -> str:
    """The other valid encoding of the same ECDSA signature: (r, n - s)."""
    r = int(signature_hex[:64], 16)
    s = int(signature_hex[64:], 16)
    return format(r, "064x") + format(P256_N - s, "064x")


@pytest.fixture
def es256_key():
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture
def es256_attestation(es256_key):
    return emit_inference_attestation(
        request_declared=RequestDeclared(
            intent="inference/chat/qwen3:30b-a3b",
            request_commitment=make_request_commitment(
                messages=MESSAGES, sampling=SAMPLING
            ),
        ),
        model_derived=MODEL,
        iss="vaara-infer-proxy",
        sub="vaara/homeserver",
        secret_version="v1",
        alg="ES256",
        signing_material=es256_key,
    )


@pytest.fixture
def es256_receipt(es256_key, es256_attestation):
    return emit_inference_receipt(
        back_link=make_inference_back_link(es256_attestation),
        outcome_derived=InferenceOutcome(
            status="completed",
            completed_at="2026-06-14T22:00:00Z",
            tier="integrity",
            output_commitment=make_output_commitment(OUTPUT),
            eval_stats={"promptEvalCount": 11, "evalCount": 256},
        ),
        iss="vaara-infer-proxy",
        sub="vaara/homeserver",
        secret_version="v1",
        alg="ES256",
        signing_material=es256_key,
    )


def test_the_twin_verifies_so_the_premise_holds(es256_key):
    """Guard the assumption the rest of this file rests on.

    If this fails, malleability stopped applying here and the tests below are
    checking a condition that cannot arise.
    """
    payload = b'{"one":"act"}'
    sig = sign_es256(payload, private_key=es256_key)
    twin = malleate(sig)
    assert sig != twin
    assert verify_es256(payload, signature_hex=sig, public_key=es256_key.public_key())
    assert verify_es256(payload, signature_hex=twin, public_key=es256_key.public_key())


def test_receipt_digest_is_unchanged_by_malleation(es256_receipt):
    """One signing act, one identity, whichever encoding of s arrives."""
    twin = replace(es256_receipt, signature=malleate(es256_receipt.signature))
    assert twin.signature != es256_receipt.signature
    assert inference_receipt_digest(twin) == inference_receipt_digest(es256_receipt)


def test_attestation_digest_is_unchanged_by_malleation(es256_attestation):
    twin = replace(es256_attestation, signature=malleate(es256_attestation.signature))
    assert twin.signature != es256_attestation.signature
    assert (
        inference_attestation_digest(twin)
        == inference_attestation_digest(es256_attestation)
    )


def test_digest_still_moves_when_signed_content_changes(es256_receipt):
    """Immunity to malleation must not become indifference to content.

    Identity comes from the signed blocks, so changing one has to change the
    digest. Otherwise the fix trades one collision for a worse one.
    """
    other = replace(
        es256_receipt,
        outcome_derived=replace(es256_receipt.outcome_derived, status="failed"),
    )
    assert inference_receipt_digest(other) != inference_receipt_digest(es256_receipt)
