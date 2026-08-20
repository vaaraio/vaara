# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grant identity must survive ECDSA signature malleability.

``grant_fingerprint`` content-addressed ``credential.to_dict()``, and that dict
carries a ``signature`` member. ES256 is an accepted grant algorithm, and
neither ``sign_es256`` nor ``verify_es256`` constrains ``s``, so one signing act
has two valid encodings and produced two fingerprints for the same grant.

The exposure runs opposite to the one repaired in
``test_inference_digest_malleability``. There, two byte-different receipts could
claim one signing act. Here, one act produced two identities: a verifier
recomputing the fingerprint from the grant it holds could fail to match a
receipt for an authorization that did govern, and a false miss on identity reads
as a tampering signal.

The old behaviour was deliberate. The docstring said signature included, so that
a re-minted grant produced a different fingerprint. That intent survives the
repair. ``_signing_payload`` covers ``alg``, ``scope``, ``binding`` and
``asserted``, and ``asserted`` carries ``iss``, ``sub`` and ``secretVersion``, so
a re-mint under a different key or a changed scope still moves the value while a
malleated twin no longer does.

Found by applying Joel Hillier's generalisation of Henri's own rule, posted to
the SCITT list 2026-08-20: fixing the identity rules you find is not enough,
because a digest can carry a signature it never names, through an object someone
else signed. This is the third instance of Anton Sokolov's class in this tree,
after the two repaired on 18 August.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._attest_canonical import make_args_digest  # noqa: E402
from vaara.attestation._attest_signing import (  # noqa: E402
    sign_es256,
    verify_es256,
)
from vaara.credential import GrantBinding, GrantScope, emit_grant  # noqa: E402
from vaara.credential._authorization_receipt import (  # noqa: E402
    grant_fingerprint,
)

# Order of the P-256 base point (SEC 2, secp256r1).
P256_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

DIGEST = "sha256:" + "ab" * 32
NONCE = "att-nonce-xyz"
IAT = "2026-06-18T12:00:00Z"
ARGS = {"path": "/tmp/report.txt"}
COMMIT = make_args_digest(ARGS).projection_digest


def malleate(signature_hex: str) -> str:
    """The other valid encoding of the same ECDSA signature: (r, n - s)."""
    r = int(signature_hex[:64], 16)
    s = int(signature_hex[64:], 16)
    return format(r, "064x") + format(P256_N - s, "064x")


@pytest.fixture
def es256_key():
    return ec.generate_private_key(ec.SECP256R1())


def _mint(key, *, tool="fs.read", tenant="tenant-a", secret_version="key-v1"):
    return emit_grant(
        scope=GrantScope(
            tool_name=tool, args_commitment=COMMIT, tenant_id=tenant
        ),
        binding=GrantBinding(attestation_digest=DIGEST, attestation_nonce=NONCE),
        iss="vaara-mcp-proxy",
        sub="tenant-a/upstream",
        secret_version=secret_version,
        alg="ES256",
        signing_material=key,
        iat=IAT,
    )


def test_the_twin_verifies_so_the_premise_holds(es256_key):
    """Guard the assumption the rest of this file rests on."""
    payload = b'{"one":"act"}'
    sig = sign_es256(payload, private_key=es256_key)
    twin = malleate(sig)
    assert sig != twin
    assert verify_es256(payload, signature_hex=sig, public_key=es256_key.public_key())
    assert verify_es256(payload, signature_hex=twin, public_key=es256_key.public_key())


def test_fingerprint_is_unchanged_by_malleation(es256_key):
    """One signing act, one grant identity, whichever encoding of s arrives."""
    grant = _mint(es256_key)
    twin = replace(grant, signature=malleate(grant.signature))

    assert twin.signature != grant.signature
    assert grant_fingerprint(twin) == grant_fingerprint(grant)


def test_fingerprint_still_moves_on_a_changed_scope(es256_key):
    """The property the old construction existed to provide, preserved."""
    grant = _mint(es256_key)
    other = _mint(es256_key, tool="fs.write")

    assert grant_fingerprint(other) != grant_fingerprint(grant)


def test_fingerprint_still_moves_on_a_re_mint_under_a_different_key(es256_key):
    """A re-minted grant remains distinguishable from the one that governed."""
    grant = _mint(es256_key)
    other = _mint(
        ec.generate_private_key(ec.SECP256R1()), secret_version="key-v2"
    )

    assert grant_fingerprint(other) != grant_fingerprint(grant)
