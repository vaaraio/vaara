"""The revocation check and the staleness bound are reachable from the proxy.

Before this, ``verify_grant`` could refuse with ``revoked`` and (since the
staleness bound landed) ``revocation_stale``, but neither verdict could be
produced through the shipped enforcement path. ``CredentialGateway`` never
forwarded ``max_staleness_seconds``, and ``AttestPairEmitter`` built its
gateway with only a verifying key and a receipts directory, so the registry
argument was always ``None`` and the revocation branch never ran.

These tests pin the reachability, and pin the default that made the gap
invisible: with no registry configured the check does not run at all, which
is not the same as running and passing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from vaara.attestation._revocation import RevocationRegistry  # noqa: E402
from vaara.credential import CredentialGateway  # noqa: E402
from vaara.integrations._mcp_attest import (  # noqa: E402
    AttestConfigError,
    build_attest_emitter,
)

KEY = b"x" * 32
TOOL = "read_file"
ARGS = {"path": "/tmp/x"}
TENANT = "t1"

# The grant is minted now, so a registry observed long ago cannot speak to it.
OLD_AS_OF = "2020-01-01T00:00:00Z"


def _keyfile(d: Path) -> Path:
    p = d.parent / "attest.key"
    p.write_bytes(KEY)
    return p


def _emitter(receipts_dir: Path, **kwargs: Any) -> Any:
    return build_attest_emitter(
        signing_key_path=_keyfile(receipts_dir),
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
        **kwargs,
    )


def _mint(receipts_dir: Path) -> dict:
    em = _emitter(receipts_dir)
    att, counter = em.emit_attestation(
        tool_name=TOOL, arguments=ARGS, upstream_name="default", tenant_id=TENANT
    )
    cred = em.emit_grant(
        attestation=att, counter=counter, tool_name=TOOL,
        upstream_name="default", tenant_id=TENANT,
    )
    return cred.to_dict()


def _params(cred: dict) -> dict:
    return {"_meta": {"vaara/credential": cred}}


def _registry_file(path: Path, *, as_of: str | None = None) -> Path:
    doc: dict[str, Any] = {"version": 1, "entries": []}
    if as_of is not None:
        doc["as_of"] = as_of
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# ── The gateway forwards the bound at all ───────────────────────────────────

def test_gateway_without_bound_admits_a_registry_of_any_age(tmp_path: Path):
    cred = _mint(tmp_path)
    gw = CredentialGateway(
        verifying_material=KEY,
        receipts_dir=tmp_path,
        expected_tenant=TENANT,
        revocation=RevocationRegistry((), as_of=OLD_AS_OF),
    )
    verdict = gw.authorize(_params(cred), tool_name=TOOL, arguments=ARGS)
    assert verdict.ok, verdict.reason


def test_gateway_with_bound_refuses_a_registry_that_cannot_speak_to_now(
    tmp_path: Path,
):
    """The verdict that was unreachable through this class."""
    cred = _mint(tmp_path)
    gw = CredentialGateway(
        verifying_material=KEY,
        receipts_dir=tmp_path,
        expected_tenant=TENANT,
        revocation=RevocationRegistry((), as_of=OLD_AS_OF),
        max_staleness_seconds=300,
    )
    verdict = gw.authorize(_params(cred), tool_name=TOOL, arguments=ARGS)
    assert not verdict.ok
    assert verdict.reason == "revocation_stale"


# ── The emitter can be given a registry at all ──────────────────────────────

def test_emitter_without_registry_does_not_check_revocation(tmp_path: Path):
    """The shipped default, pinned so the gap cannot reappear silently.

    The point is not that the credential passes a revocation check. It is
    that no revocation check runs, because there is no registry to run it
    against.
    """
    em = _emitter(
        tmp_path, tool_constraints_path=None,
    )
    assert em._revocation is None
    assert em._max_staleness_seconds is None


def test_emitter_loads_a_registry_from_disk(tmp_path: Path):
    reg = _registry_file(tmp_path / "rev.json", as_of=OLD_AS_OF)
    em = _emitter(tmp_path, revocation_registry_path=reg)
    assert em._revocation is not None
    assert em._revocation.as_of == OLD_AS_OF


def test_emitter_forwards_skew_and_tenant(tmp_path: Path):
    em = _emitter(tmp_path, clock_skew_seconds=0, expected_tenant=TENANT)
    assert em._clock_skew_seconds == 0
    assert em._expected_tenant == TENANT


# ── Configuration that cannot mean anything is refused, not ignored ─────────

def test_bound_without_a_registry_refuses_to_start(tmp_path: Path):
    with pytest.raises(AttestConfigError, match="no registry"):
        _emitter(tmp_path, max_staleness_seconds=300)


def test_unparseable_registry_refuses_to_start(tmp_path: Path):
    bad = tmp_path / "rev.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(AttestConfigError, match="not a usable registry"):
        _emitter(tmp_path, revocation_registry_path=bad)


def test_missing_registry_file_refuses_to_start(tmp_path: Path):
    with pytest.raises(AttestConfigError, match="file not found"):
        _emitter(tmp_path, revocation_registry_path=tmp_path / "nope.json")


# ── Clock skew widens a stated lifetime, and now it can be turned off ───────

def test_default_skew_admits_a_credential_one_second_past_its_bound(tmp_path: Path):
    """The finding reported against VATE's just-over-boundary case.

    ``clock_skew_seconds`` defaults to 30, so a grant one second past a 300
    second lifetime is still admitted. A freshness margin smaller than the
    skew cannot separate a verifier with no expiry check from a verifier
    that has one plus any tolerance at all.
    """
    from vaara.credential import verify_grant
    from vaara.credential._grant_verify import iso8601_to_epoch

    em = _emitter(tmp_path)
    att, counter = em.emit_attestation(
        tool_name=TOOL, arguments=ARGS, upstream_name="default", tenant_id=TENANT
    )
    cred = em.emit_grant(
        attestation=att, counter=counter, tool_name=TOOL,
        upstream_name="default", tenant_id=TENANT,
    )
    iat = iso8601_to_epoch(cred.asserted.iat)
    assert iat is not None
    one_second_over = iat + cred.asserted.exp_seconds + 1

    common = dict(
        verifying_material=KEY,
        runtime_tool_name=TOOL,
        runtime_args=ARGS,
        runtime_tenant_id=TENANT,
        known_attestation_digests=frozenset(
            {cred.binding.attestation_digest}
        ),
        now=one_second_over,
    )

    assert verify_grant(cred, **common).ok, "default skew absorbs the margin"
    strict = verify_grant(cred, clock_skew_seconds=0, **common)
    assert not strict.ok
    assert strict.reason == "expired"


# ── End to end: the emitter's own gateway produces the verdict ──────────────

def test_emitter_gateway_refuses_on_a_stale_registry(tmp_path: Path):
    """Reachable from the object the proxy actually holds."""
    constraints = tmp_path / "constraints.json"
    constraints.write_text(
        json.dumps({"tools": {TOOL: [{"arg": "path", "op": "eq", "value": "/tmp/x"}]}}),
        encoding="utf-8",
    )
    reg = _registry_file(tmp_path / "rev.json", as_of=OLD_AS_OF)
    em = _emitter(
        tmp_path,
        tool_constraints_path=constraints,
        revocation_registry_path=reg,
        max_staleness_seconds=300,
        expected_tenant=TENANT,
    )
    assert em.gateway is not None

    att, counter = em.emit_attestation(
        tool_name=TOOL, arguments=ARGS, upstream_name="default", tenant_id=TENANT
    )
    cred = em.emit_grant(
        attestation=att, counter=counter, tool_name=TOOL,
        upstream_name="default", tenant_id=TENANT,
    )
    verdict = em.gateway.authorize(
        _params(cred.to_dict()), tool_name=TOOL, arguments=ARGS
    )
    assert not verdict.ok
    assert verdict.reason == "revocation_stale"
