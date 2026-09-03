"""Unit tests for the brokered-credential mint/verify path."""

from __future__ import annotations

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation import RevocationEntry, RevocationRegistry  # noqa: E402
from vaara.attestation._attest_canonical import (  # noqa: E402
    iso8601_to_epoch,
    make_args_digest,
)
from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.credential import (  # noqa: E402
    GrantBinding,
    GrantScope,
    emit_grant,
    grant_from_dict,
    verify_grant,
    verify_grant_signature,
)

SECRET = b"s" * 32
DIGEST = "sha256:" + "ab" * 32
NONCE = "att-nonce-xyz"
IAT = "2026-06-18T12:00:00Z"
IAT_EPOCH = iso8601_to_epoch(IAT)
ARGS = {"path": "/tmp/report.txt"}
COMMIT = make_args_digest(ARGS).projection_digest


def _scope(tool="fs.read", tenant="tenant-a", commitment=COMMIT):
    return GrantScope(tool_name=tool, args_commitment=commitment, tenant_id=tenant)


def _mint(*, alg="HS256", material=SECRET, scope=None, exp_seconds=60, iat=IAT):
    return emit_grant(
        scope=scope or _scope(),
        binding=GrantBinding(attestation_digest=DIGEST, attestation_nonce=NONCE),
        iss="vaara-mcp-proxy",
        sub="tenant-a/upstream",
        secret_version="key-v1",
        alg=alg,
        signing_material=material,
        exp_seconds=exp_seconds,
        iat=iat,
    )


def _ok(cred, material=SECRET, **over):
    kw = dict(
        verifying_material=material,
        runtime_tool_name="fs.read",
        runtime_args=ARGS,
        runtime_tenant_id="tenant-a",
        known_attestation_digests=frozenset({DIGEST}),
        now=IAT_EPOCH + 5,
    )
    kw.update(over)
    return verify_grant(cred, **kw)


def test_round_trip_to_from_dict():
    cred = _mint()
    again = grant_from_dict(cred.to_dict())
    assert again == cred


def test_happy_path_hs256():
    v = _ok(_mint())
    assert v.ok and v.reason == "ok"


def test_happy_path_es256():
    priv = ec.generate_private_key(ec.SECP256R1())
    cred = _mint(alg="ES256", material=priv)
    v = _ok(cred, material=priv.public_key())
    assert v.ok and v.reason == "ok"


def test_signature_verifies_standalone():
    assert verify_grant_signature(_mint(), verifying_material=SECRET) is True
    assert verify_grant_signature(_mint(), verifying_material=b"wrong" * 8) is False


def test_tamper_signature_bad_signature():
    cred = _mint()
    forged = grant_from_dict({**cred.to_dict(), "signature": "00" * 32})
    assert _ok(forged).reason == "bad_signature"


def test_tamper_tool_name_scope_mismatch():
    assert _ok(_mint(), runtime_tool_name="fs.write").reason == "scope_mismatch"


def test_tamper_tenant_scope_mismatch():
    assert _ok(_mint(), runtime_tenant_id="tenant-b").reason == "scope_mismatch"


def test_mutated_args_scope_mismatch():
    assert _ok(_mint(), runtime_args={"path": "/etc/shadow"}).reason == "scope_mismatch"


def test_expired_past_deadline():
    # iat + exp(60) + skew(30) = +90s; now at +200s is expired.
    assert _ok(_mint(exp_seconds=60), now=IAT_EPOCH + 200).reason == "expired"


def test_future_dated_expired():
    # now well before iat (beyond skew) is future-dated -> expired.
    assert _ok(_mint(), now=IAT_EPOCH - 200).reason == "expired"


def test_unparseable_iat_expired():
    cred = _mint(iat="not-a-timestamp")
    assert _ok(cred).reason == "expired"


def test_revocation_before_iat_revoked():
    reg = RevocationRegistry(
        (RevocationEntry(scope="key", subject="key-v1", revoked_at="2026-06-18T11:00:00Z"),)
    )
    assert _ok(_mint(), revocation=reg).reason == "revoked"


def test_revocation_after_iat_not_revoked():
    reg = RevocationRegistry(
        (RevocationEntry(scope="key", subject="key-v1", revoked_at="2026-06-18T13:00:00Z"),)
    )
    assert _ok(_mint(), revocation=reg).ok


def test_revocation_identity_scope_revoked():
    reg = RevocationRegistry(
        (
            RevocationEntry(
                scope="identity",
                subject="vaara-mcp-proxy",
                revoked_at="2026-06-18T11:00:00Z",
            ),
        )
    )
    assert _ok(_mint(), revocation=reg).reason == "revoked"


# --- revocation freshness ------------------------------------------------
#
# v1.80.0 gave RevocationStatus a freshness verdict, but no shipped path read
# it: verify_grant called status() with no bound and consumed only .revoked,
# so a registry that said in its own words it could establish nothing about
# the present still produced ok=True. These pin the pass-through that closes
# that, and pin that omitting the bound changes nothing.

STALE_REG = RevocationRegistry((), as_of="2026-06-18T00:00:00Z")  # 12h before iat
FRESH_REG = RevocationRegistry((), as_of="2026-06-18T11:59:00Z")  # 60s before iat
REVOKED_STALE_REG = RevocationRegistry(
    (RevocationEntry(scope="key", subject="key-v1", revoked_at="2026-06-18T11:00:00Z"),),
    as_of="2026-06-18T00:00:00Z",
)
CHECK_AT = "2026-06-18T12:00:05Z"  # matches now=IAT_EPOCH + 5


def test_stale_registry_admits_when_no_bound_is_stated():
    # Backward compatibility. Section 10 puts the bound on the deployment, so
    # a caller that states none gets exactly the pre-existing behaviour.
    assert _ok(_mint(), revocation=STALE_REG).ok


def test_stale_registry_refuses_once_a_bound_is_stated():
    v = _ok(
        _mint(),
        revocation=STALE_REG,
        revocation_now=CHECK_AT,
        max_staleness_seconds=300,
    )
    assert not v.ok and v.reason == "revocation_stale"


def test_fresh_registry_inside_the_bound_admits():
    v = _ok(
        _mint(),
        revocation=FRESH_REG,
        revocation_now=CHECK_AT,
        max_staleness_seconds=300,
    )
    assert v.ok and v.reason == "ok"


def test_registry_without_as_of_refuses_under_a_bound():
    # freshness is "unknown", not "fresh". Absence of an observation instant
    # is never read as permission to treat the answer as current.
    v = _ok(
        _mint(),
        revocation=RevocationRegistry(()),
        revocation_now=CHECK_AT,
        max_staleness_seconds=300,
    )
    assert not v.ok and v.reason == "revocation_stale"


def test_future_as_of_refuses_under_a_bound():
    # Observation instant after now means the clocks disagree, so the bound
    # cannot be evaluated honestly and is not quietly treated as met.
    v = _ok(
        _mint(),
        revocation=RevocationRegistry((), as_of="2026-06-19T00:00:00Z"),
        revocation_now=CHECK_AT,
        max_staleness_seconds=300,
    )
    assert not v.ok and v.reason == "revocation_stale"


def test_revocation_binds_however_stale_the_registry_is():
    # THE ASYMMETRY, and the case most likely to regress. A revocation fact
    # does not expire, so staleness must not downgrade "revoked" to
    # "revocation_stale" and must never turn a refusal into an admit.
    v = _ok(
        _mint(),
        revocation=REVOKED_STALE_REG,
        revocation_now=CHECK_AT,
        max_staleness_seconds=300,
    )
    assert not v.ok and v.reason == "revoked"


def test_binding_absent_unknown():
    assert _ok(_mint(), known_attestation_digests=frozenset()).reason == "binding_unknown"


def test_binding_none_fails_closed():
    assert _ok(_mint(), known_attestation_digests=None).reason == "binding_unknown"


def test_extra_wire_key_malformed():
    bad = {**_mint().to_dict(), "rogue": 1}
    with pytest.raises(AttestationError):
        grant_from_dict(bad)


def test_extra_scope_key_malformed():
    d = _mint().to_dict()
    d["scope"]["rogue"] = "x"
    with pytest.raises(AttestationError):
        grant_from_dict(d)


def test_non_positive_exp_rejected_at_mint():
    with pytest.raises(AttestationError):
        _mint(exp_seconds=0)
