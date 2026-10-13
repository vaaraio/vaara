"""The standing permission: signed by the principal, bounding calls one by one
and taken together."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._attest_canonical import iso8601_to_epoch  # noqa: E402
from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.credential import (  # noqa: E402
    Capability,
    Limit,
    evaluate_call,
    permission_digest,
    permission_from_dict,
    sign_permission,
    verify_permission_signature,
)

KEY = ec.generate_private_key(ec.SECP256R1())
START = "2026-10-01T00:00:00Z"
END = "2026-11-01T00:00:00Z"
T0 = iso8601_to_epoch(START)
DAY = 86400


def _perm(**over):
    kw = dict(
        principal="alice@example.org", key_id="alice-2026", agent="agent:buyer",
        tool_name="pay.send", not_before=START, not_after=END, alg="ES256",
        signing_material=KEY,
        capabilities=(Capability("amount", "le", "500"),
                      Capability("vendor", "in", ("acme", "globex"))),
        limits=(Limit(kind="sum", arg="amount", max="1000", window_seconds=7 * DAY),
                Limit(kind="count", max="3", window_seconds=DAY)),
    )
    kw.update(over)
    return sign_permission(**kw)


def test_the_principal_signature_verifies_and_survives_the_wire():
    p = _perm()
    assert verify_permission_signature(p, verifying_material=KEY.public_key())
    again = permission_from_dict(json.loads(json.dumps(p.to_dict())))
    assert again == p
    assert permission_digest(again) == permission_digest(p)
    assert verify_permission_signature(again, verifying_material=KEY.public_key())


def test_raising_a_limit_after_signing_breaks_the_signature():
    d = _perm().to_dict()
    d["limits"][0]["max"] = "100000"
    tampered = permission_from_dict(d)
    assert not verify_permission_signature(tampered, verifying_material=KEY.public_key())


def test_another_key_does_not_verify():
    other = ec.generate_private_key(ec.SECP256R1())
    assert not verify_permission_signature(_perm(), verifying_material=other.public_key())


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(extra=1), "unrecognized"),
    (lambda d: d.update(type="vaara.permission/v2"), "type"),
    (lambda d: d["limits"].append({"kind": "avg", "max": "1"}), "kind"),
    (lambda d: d["limits"].append({"kind": "sum", "max": "1"}), "argument"),
    (lambda d: d["limits"].append({"kind": "count", "max": "Infinity"}), "max"),
    (lambda d: d["limits"].append({"kind": "count", "max": "1", "windowSeconds": 0}), "windowSeconds"),
    (lambda d: d.update(notAfter="2026-09-01T00:00:00Z"), "notBefore"),
])
def test_a_malformed_permission_is_refused(mutate, needle):
    d = _perm().to_dict()
    mutate(d)
    with pytest.raises(AttestationError, match=needle):
        permission_from_dict(d)


def test_a_call_inside_every_bound_is_allowed():
    assert evaluate_call(_perm(), {"amount": 400, "vendor": "acme"}, T0 + 10) == (True, "ok")


def test_a_call_over_the_per_call_bound_is_refused():
    ok, reason = evaluate_call(_perm(), {"amount": 600, "vendor": "acme"}, T0 + 10)
    assert not ok and reason == "capability_exceeded"


def test_the_weekly_sum_counts_earlier_calls_in_the_window():
    p = _perm()
    prior = [(T0 + 1 * DAY, {"amount": 400, "vendor": "acme"}),
             (T0 + 2 * DAY, {"amount": 400, "vendor": "acme"})]
    ok, reason = evaluate_call(p, {"amount": 300, "vendor": "acme"}, T0 + 3 * DAY, prior)
    assert not ok and reason.startswith("limit_exceeded: sum amount 1100 > 1000")
    # Eight days on, the first two calls have left the rolling window.
    assert evaluate_call(p, {"amount": 300, "vendor": "acme"}, T0 + 9 * DAY + 1, prior)[0]


def test_the_daily_count_is_a_rolling_window():
    p = _perm()
    prior = [(T0 + 100 + i, {"amount": 1, "vendor": "acme"}) for i in range(3)]
    ok, reason = evaluate_call(p, {"amount": 1, "vendor": "acme"}, T0 + 200, prior)
    assert not ok and "count 4 > 3" in reason
    assert evaluate_call(p, {"amount": 1, "vendor": "acme"}, T0 + DAY + 200, prior)[0]


def test_a_limit_without_a_window_spans_the_validity_period():
    p = _perm(limits=(Limit(kind="sum", arg="amount", max="500"),))
    prior = [(T0 + 20 * DAY, {"amount": 450, "vendor": "acme"})]
    ok, _ = evaluate_call(p, {"amount": 100, "vendor": "acme"}, T0 + 29 * DAY, prior)
    assert not ok


def test_outside_the_validity_period_nothing_is_allowed():
    ok, reason = evaluate_call(_perm(), {"amount": 1, "vendor": "acme"}, T0 - 1)
    assert not ok and reason == "outside_validity"


def test_an_unreadable_summed_argument_is_refused():
    p = _perm(capabilities=())
    ok, reason = evaluate_call(p, {"amount": "lots"}, T0 + 10)
    assert not ok and reason == "limit_argument_unreadable"


def test_floats_and_decimal_strings_sum_exactly():
    p = _perm(capabilities=(), limits=(Limit(kind="sum", arg="amount", max="0.3"),))
    prior = [(T0 + 1, {"amount": 0.1}), (T0 + 2, {"amount": "0.1"})]
    assert evaluate_call(p, {"amount": 0.1}, T0 + 3, prior) == (True, "ok")


def _record(trail, args, decision, at, *, agent="agent:buyer", tool="pay.send"):
    from vaara.taxonomy.actions import ActionRequest, create_default_registry

    req = ActionRequest(agent_id=agent, tool_name=tool,
                        action_type=create_default_registry().classify(tool, args),
                        parameters=args)
    aid = trail.record_action_requested(req)
    if decision:
        trail.record_decision(action_id=aid, agent_id=agent, tool_name=tool,
                              decision=decision, reason="t", risk_score=0.0)
    for rec in trail.get_action_trail(aid):
        object.__setattr__(rec, "timestamp", at)
    return aid


def _replay(trail, p=None):
    from vaara.credential import check_trail_against_permission

    return check_trail_against_permission(
        trail, p or _perm(), verifying_material=KEY.public_key(), verify_chain=False)


def test_the_replay_finds_the_allowed_call_that_crossed_the_weekly_sum():
    from vaara.audit.trail import AuditTrail

    trail = AuditTrail()
    _record(trail, {"amount": 400, "vendor": "acme"}, "allow", T0 + 1 * DAY)
    _record(trail, {"amount": 400, "vendor": "acme"}, "allow", T0 + 2 * DAY)
    over = _record(trail, {"amount": 300, "vendor": "acme"}, "allow", T0 + 3 * DAY)
    report = _replay(trail)
    assert report.counts == {"within": 2, "violated": 1}
    assert not report.ok
    bad = [c for c in report.calls if c.state == "violated"][0]
    assert bad.action_id == over and bad.reason.startswith("limit_exceeded: sum amount")


def test_a_refusal_at_the_limit_shows_the_gate_held_and_adds_nothing():
    from vaara.audit.trail import AuditTrail

    trail = AuditTrail()
    _record(trail, {"amount": 400, "vendor": "acme"}, "allow", T0 + 1 * DAY)
    _record(trail, {"amount": 400, "vendor": "acme"}, "allow", T0 + 2 * DAY)
    _record(trail, {"amount": 300, "vendor": "acme"}, "deny", T0 + 3 * DAY)
    _record(trail, {"amount": 150, "vendor": "acme"}, "allow", T0 + 3 * DAY + 60)
    report = _replay(trail)
    assert report.counts == {"within": 3, "refused": 1}
    assert report.ok


def test_other_agents_and_calls_outside_validity_are_kept_apart():
    from vaara.audit.trail import AuditTrail

    trail = AuditTrail()
    _record(trail, {"amount": 9999, "vendor": "evil"}, "allow", T0 + 10, agent="agent:other")
    _record(trail, {"amount": 9999, "vendor": "evil"}, "allow", T0 - DAY)
    report = _replay(trail)
    assert report.counts == {"outside_window": 1}
    assert report.ok


def test_a_permission_signed_by_someone_else_judges_nothing():
    from vaara.audit.trail import AuditTrail

    from vaara.credential import check_trail_against_permission

    trail = AuditTrail()
    _record(trail, {"amount": 1, "vendor": "acme"}, "allow", T0 + 10)
    other = ec.generate_private_key(ec.SECP256R1())
    report = check_trail_against_permission(
        trail, _perm(), verifying_material=other.public_key(), verify_chain=False)
    assert report.grant_problem == "bad_signature" and report.calls == ()


def test_cli_signs_a_permission_and_replays_a_sqlite_trail(tmp_path, capsys):
    from cryptography.hazmat.primitives import serialization

    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.cli import main

    priv = tmp_path / "alice.pem"
    priv.write_bytes(KEY.private_bytes(serialization.Encoding.PEM,
                                       serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    pub = tmp_path / "alice.pub.pem"
    pub.write_bytes(KEY.public_key().public_bytes(serialization.Encoding.PEM,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "principal": "alice@example.org", "keyId": "alice-2026", "agent": "agent:buyer",
        "toolName": "pay.send", "notBefore": "2026-01-01T00:00:00Z",
        "notAfter": "2099-01-01T00:00:00Z",
        "capabilities": [{"arg": "amount", "op": "le", "value": "500"}],
        "limits": [{"kind": "sum", "arg": "amount", "max": "600", "windowSeconds": 86400}],
    }))
    out = tmp_path / "permission.json"
    assert main(["permission", "sign", "--spec", str(spec), "--key", str(priv),
                 "--out", str(out)]) in (None, 0)
    assert json.loads(out.read_text())["alg"] == "ES256"

    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    trail = backend.load_trail()
    import time
    now = time.time()
    _record(trail, {"amount": 400}, "allow", now)
    _record(trail, {"amount": 300}, "allow", now)
    backend.close()

    code = main(["trail", "check-permission", "--db", str(db), "--permission", str(out),
                 "--pubkey-file", str(pub), "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["counts"] == {"within": 1, "violated": 1}
