"""Capability-mode grant tests (Phase C): typed, enforceable arg limits."""

from __future__ import annotations

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._sep2787_canonical import (  # noqa: E402
    iso8601_to_epoch,
    make_args_digest,
)
from vaara.attestation._sep2787_types import AttestationError  # noqa: E402
from vaara.credential import (  # noqa: E402
    Capability,
    GrantBinding,
    GrantScope,
    emit_grant,
    grant_from_dict,
    verify_grant,
)

SECRET = b"s" * 32
DIGEST = "sha256:" + "ab" * 32
NONCE = "att-nonce-xyz"
IAT = "2026-06-18T12:00:00Z"
IAT_EPOCH = iso8601_to_epoch(IAT)

# Mint-time attested args; runtime args vary within the capability bounds.
MINT_ARGS = {"amount": 100, "vendor": "acme", "destination": "0xABC"}
COMMIT = make_args_digest(MINT_ARGS).projection_digest
CAPS = (
    Capability("amount", "le", "500"),
    Capability("vendor", "in", ("acme", "globex")),
    Capability("destination", "eq", "0xABC"),
)
RUNTIME_OK = {"amount": 400, "vendor": "globex", "destination": "0xABC"}


def _mint(*, alg="HS256", material=SECRET, capabilities=CAPS):
    return emit_grant(
        scope=GrantScope(tool_name="pay.send", args_commitment=COMMIT, tenant_id="tenant-a"),
        binding=GrantBinding(attestation_digest=DIGEST, attestation_nonce=NONCE),
        iss="vaara-mcp-proxy",
        sub="tenant-a/upstream",
        secret_version="key-v1",
        alg=alg,
        signing_material=material,
        exp_seconds=60,
        iat=IAT,
        capabilities=capabilities,
    )


def _verify(cred, *, material=SECRET, runtime_args=None, **over):
    kw = dict(
        verifying_material=material,
        runtime_tool_name="pay.send",
        runtime_args=RUNTIME_OK if runtime_args is None else runtime_args,
        runtime_tenant_id="tenant-a",
        known_attestation_digests=frozenset({DIGEST}),
        now=IAT_EPOCH + 5,
    )
    kw.update(over)
    return verify_grant(cred, **kw)


def test_round_trip_preserves_capabilities():
    cred = _mint()
    again = grant_from_dict(cred.to_dict())
    assert again == cred
    assert again.capabilities == CAPS


def test_within_bounds_hs256_ok():
    v = _verify(_mint())
    assert v.ok and v.reason == "ok"


def test_within_bounds_es256_ok():
    priv = ec.generate_private_key(ec.SECP256R1())
    cred = _mint(alg="ES256", material=priv)
    v = _verify(cred, material=priv.public_key())
    assert v.ok and v.reason == "ok"


def test_commitment_not_enforced_in_capability_mode():
    # Runtime args differ from MINT_ARGS but stay within bounds -> still ok.
    assert _verify(_mint(), runtime_args={"amount": 499, "vendor": "acme", "destination": "0xABC"}).ok


def test_over_numeric_bound_exceeded():
    args = {"amount": 600, "vendor": "acme", "destination": "0xABC"}
    assert _verify(_mint(), runtime_args=args).reason == "capability_exceeded"


def test_value_not_in_set_exceeded():
    args = {"amount": 100, "vendor": "evilcorp", "destination": "0xABC"}
    assert _verify(_mint(), runtime_args=args).reason == "capability_exceeded"


def test_pinned_arg_mismatch_exceeded():
    args = {"amount": 100, "vendor": "acme", "destination": "0xDEAD"}
    assert _verify(_mint(), runtime_args=args).reason == "capability_exceeded"


def test_unnamed_runtime_arg_uncovered():
    args = {**RUNTIME_OK, "memo": "drain"}
    assert _verify(_mint(), runtime_args=args).reason == "capability_uncovered"


def test_missing_named_arg_exceeded():
    args = {"amount": 100, "vendor": "acme"}  # destination dropped
    assert _verify(_mint(), runtime_args=args).reason == "capability_exceeded"


def test_capabilities_are_signed():
    cred = _mint()
    d = cred.to_dict()
    d["capabilities"][0]["value"] = "999"  # raise the bound after signing
    forged = grant_from_dict(d)
    assert _verify(forged).reason == "bad_signature"


def test_unknown_capability_key_malformed():
    d = _mint().to_dict()
    d["capabilities"][0]["rogue"] = 1
    with pytest.raises(AttestationError):
        grant_from_dict(d)


def test_float_capability_value_rejected():
    d = _mint().to_dict()
    d["capabilities"][0]["value"] = 500.0  # float in a signed numeric bound
    with pytest.raises(AttestationError):
        grant_from_dict(d)


def test_empty_capabilities_list_rejected():
    d = _mint().to_dict()
    d["capabilities"] = []
    with pytest.raises(AttestationError):
        grant_from_dict(d)
