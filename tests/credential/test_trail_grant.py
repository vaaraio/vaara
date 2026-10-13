"""Offline check of a recorded trail against a capability grant."""

from __future__ import annotations

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from vaara.attestation._attest_canonical import (  # noqa: E402
    iso8601_to_epoch,
    make_args_digest,
)
from vaara.audit.trail import AuditTrail  # noqa: E402
from vaara.credential import (  # noqa: E402
    Capability,
    GrantBinding,
    GrantScope,
    check_trail_against_grant,
    emit_grant,
)
from vaara.taxonomy.actions import ActionRequest, create_default_registry  # noqa: E402

SECRET = b"s" * 32
IAT = "2026-06-18T12:00:00Z"
IAT_EPOCH = iso8601_to_epoch(IAT)
MINT_ARGS = {"amount": 100, "vendor": "acme"}
CAPS = (
    Capability("amount", "le", "500"),
    Capability("vendor", "in", ("acme", "globex")),
)


def _grant(*, capabilities=CAPS, exp_seconds=3600, tool="pay.send", tenant=""):
    return emit_grant(
        scope=GrantScope(
            tool_name=tool,
            args_commitment=make_args_digest(MINT_ARGS).projection_digest,
            tenant_id=tenant,
        ),
        binding=GrantBinding(attestation_digest="sha256:" + "ab" * 32, attestation_nonce="n"),
        iss="principal:alice",
        sub="agent:buyer",
        secret_version="key-v1",
        alg="HS256",
        signing_material=SECRET,
        exp_seconds=exp_seconds,
        iat=IAT,
        capabilities=capabilities,
    )


def _call(trail, args, decision, *, tool="pay.send", at=IAT_EPOCH + 10):
    registry = create_default_registry()
    req = ActionRequest(
        agent_id="agent:buyer",
        tool_name=tool,
        action_type=registry.classify(tool, args),
        parameters=args,
    )
    action_id = trail.record_action_requested(req)
    if decision is not None:
        trail.record_decision(
            action_id=action_id, agent_id="agent:buyer", tool_name=tool,
            decision=decision, reason="test", risk_score=0.0,
        )
    for rec in trail.get_action_trail(action_id):
        object.__setattr__(rec, "timestamp", at)
    return action_id


def _check(trail, grant=None, material=SECRET):
    return check_trail_against_grant(
        trail, grant or _grant(), verifying_material=material, verify_chain=False,
    )


def test_calls_inside_the_grant_are_within():
    trail = AuditTrail()
    _call(trail, {"amount": 400, "vendor": "globex"}, "allow")
    report = _check(trail)
    assert report.counts == {"within": 1}
    assert report.ok


def test_an_allowed_call_outside_the_grant_is_a_violation():
    trail = AuditTrail()
    aid = _call(trail, {"amount": 900, "vendor": "acme"}, "allow")
    report = _check(trail)
    assert report.counts == {"violated": 1}
    assert not report.ok
    assert report.calls[0].action_id == aid
    assert report.calls[0].reason == "capability_exceeded"


def test_a_refused_call_outside_the_grant_shows_the_gate_held():
    trail = AuditTrail()
    _call(trail, {"amount": 900, "vendor": "acme"}, "deny")
    _call(trail, {"amount": 10, "vendor": "evil"}, "escalate")
    report = _check(trail)
    assert report.counts == {"refused": 1, "held": 1}
    assert report.ok


def test_an_argument_the_grant_never_named_is_a_violation_when_allowed():
    trail = AuditTrail()
    _call(trail, {"amount": 10, "vendor": "acme", "memo": "x"}, "allow")
    report = _check(trail)
    assert report.counts == {"violated": 1}
    assert report.calls[0].reason == "capability_uncovered"


def test_a_call_with_no_decision_is_undecided():
    trail = AuditTrail()
    _call(trail, {"amount": 900, "vendor": "acme"}, None)
    report = _check(trail)
    assert report.counts == {"undecided": 1}
    assert not report.ok


def test_other_tools_are_out_of_scope_and_not_counted():
    trail = AuditTrail()
    _call(trail, {"path": "/etc/passwd"}, "allow", tool="fs.read")
    report = _check(trail)
    assert report.counts == {}
    assert report.ok


def test_calls_outside_the_validity_window_are_reported_apart():
    trail = AuditTrail()
    _call(trail, {"amount": 900, "vendor": "acme"}, "allow", at=IAT_EPOCH + 7200)
    _call(trail, {"amount": 900, "vendor": "acme"}, "allow", at=IAT_EPOCH - 600)
    report = _check(trail)
    assert report.counts == {"outside_window": 2}
    assert report.ok


def test_truncated_arguments_cannot_be_judged():
    trail = AuditTrail()
    aid = _call(trail, {"amount": 1, "vendor": "acme"}, "allow")
    req = trail.get_action_trail(aid)[0]
    req.data["parameters"] = {"_truncated": True, "_original_bytes": 99999}
    report = _check(trail)
    assert report.counts == {"unverifiable": 1}
    assert not report.ok


def test_a_grant_with_a_bad_signature_judges_nothing():
    trail = AuditTrail()
    _call(trail, {"amount": 400, "vendor": "globex"}, "allow")
    report = _check(trail, material=b"w" * 32)
    assert report.grant_problem == "bad_signature"
    assert report.calls == ()
    assert not report.ok


def test_an_exact_args_grant_matches_on_the_commitment():
    trail = AuditTrail()
    _call(trail, dict(MINT_ARGS), "allow")
    _call(trail, {"amount": 101, "vendor": "acme"}, "allow")
    report = _check(trail, grant=_grant(capabilities=()))
    assert report.counts == {"within": 1, "violated": 1}
    assert report.calls[1].reason == "scope_mismatch"


def test_a_broken_chain_stops_the_check():
    trail = AuditTrail()
    _call(trail, {"amount": 400, "vendor": "globex"}, "allow")
    trail._records[0].data["parameters"]["amount"] = 1
    report = check_trail_against_grant(trail, _grant(), verifying_material=SECRET)
    assert report.chain_problem
    assert report.calls == ()
    assert not report.ok


def test_cli_reports_a_violation_from_a_sqlite_trail(tmp_path, capsys):
    import json

    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.cli import main

    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    trail = backend.load_trail()
    grant = _grant(exp_seconds=10**9)
    _call(trail, {"amount": 400, "vendor": "acme"}, "allow", at=IAT_EPOCH + 5)
    _call(trail, {"amount": 900, "vendor": "acme"}, "deny", at=IAT_EPOCH + 5)
    _call(trail, {"amount": 900, "vendor": "acme"}, "allow", at=IAT_EPOCH + 5)
    backend.close()
    grant_file = tmp_path / "grant.json"
    grant_file.write_text(json.dumps(grant.to_dict()))
    secret_file = tmp_path / "secret"
    secret_file.write_bytes(SECRET)

    code = main(["trail", "check-grant", "--db", str(db), "--grant", str(grant_file),
                 "--hs256-secret-file", str(secret_file), "--format", "json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["counts"] == {"within": 1, "refused": 1, "violated": 1}
