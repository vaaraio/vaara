"""Decision-record round-trip, back-link, pairing, and commit-bridge tests.

The decision record is the pre-execution sibling of the SEP-2787 request
attestation and the execution receipt: it binds the governing server's
policy verdict (``allow`` / ``block`` / ``escalate``) and its risk basis
before the side effect runs, and links back to the attestation it governs.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402

from vaara.attestation.decision import (  # noqa: E402
    DecisionDerived,
    emit_decision_record,
    make_back_link,
    parse_decision_record,
    records_paired,
    verify_decision_back_link,
    verify_decision_signature,
)
from vaara.attestation.receipt import (  # noqa: E402
    OutcomeDerived,
    emit_receipt,
    make_result_digest,
)
from vaara.attestation.sep2787 import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HS_SECRET = b"\x42" * 32


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="delete_file",
        server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"path": "/archive/2024-Q3.md"}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="archive obsolete report"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )


def _decision(decision="allow"):
    return DecisionDerived(
        decision=decision,
        decided_at="2026-05-31T09:30:00Z",
        reason="risk below allow threshold",
        risk_score="0.21",
        threshold_allow="0.40",
        threshold_block="0.70",
        policy_id="sha256:3c9d4b8a",
    )


def _emit(att, **overrides):
    kwargs = dict(
        back_link=make_back_link(att),
        decision_derived=_decision(),
        iss="vaara-proxy://acme-eu",
        sub="tenant:acme/agent:billing-bot",
        secret_version="2026-05",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_decision_record(**kwargs)


def test_hs256_round_trip():
    r = _emit(_attestation())
    assert r.alg == "HS256"
    assert r.version == 1
    assert r.signature
    assert verify_decision_signature(r, verifying_material=HS_SECRET) is True


def test_es256_round_trip():
    priv = ec.generate_private_key(ec.SECP256R1())
    r = _emit(_attestation(), alg="ES256", signing_material=priv)
    assert len(r.signature) == 128
    assert verify_decision_signature(
        r, verifying_material=priv.public_key()
    ) is True


def test_rs256_round_trip():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    r = _emit(_attestation(), alg="RS256", signing_material=priv)
    assert verify_decision_signature(
        r, verifying_material=priv.public_key()
    ) is True


def test_wrong_secret_fails_signature():
    r = _emit(_attestation())
    assert verify_decision_signature(r, verifying_material=b"\x00" * 32) is False


def test_tampered_decision_fails_signature():
    att = _attestation()
    r = _emit(att)
    tampered = dataclasses.replace(r, decision_derived=_decision("block"))
    assert verify_decision_signature(tampered, verifying_material=HS_SECRET) is False


def test_escalate_round_trips_and_verifies():
    att = _attestation()
    r = _emit(att, decision_derived=_decision("escalate"))
    assert r.decision_derived.decision == "escalate"
    assert verify_decision_signature(r, verifying_material=HS_SECRET) is True


def test_back_link_valid():
    att = _attestation()
    r = _emit(att)
    assert verify_decision_back_link(r, attestation=att).ok is True


def test_back_link_rejects_other_attestation():
    att = _attestation()
    r = _emit(att)
    other = _attestation()  # fresh nonce, different signature
    res = verify_decision_back_link(r, attestation=other)
    assert res.ok is False
    assert res.reason == "back_link_mismatch"


def test_back_link_tampered_digest():
    att = _attestation()
    r = _emit(att)
    bad = dataclasses.replace(
        r.back_link, attestation_digest="sha256:" + "0" * 64
    )
    r = dataclasses.replace(r, back_link=bad)
    assert verify_decision_back_link(r, attestation=att).ok is False


def test_optional_fields_omitted_when_none():
    att = _attestation()
    minimal = DecisionDerived(decision="block", decided_at="2026-05-31T09:30:00Z")
    r = _emit(att, decision_derived=minimal)
    dd = r.to_dict()["decisionDerived"]
    assert dd == {"decision": "block", "decidedAt": "2026-05-31T09:30:00Z"}
    assert verify_decision_signature(r, verifying_material=HS_SECRET) is True


def test_emit_rejects_float_risk_score():
    att = _attestation()
    # A float on the wire is the most common signature-drift source and is
    # banned by the JCS boundary; the decision record must inherit that ban.
    bad = dataclasses.replace(_decision(), risk_score=0.21)  # type: ignore[arg-type]
    with pytest.raises(Exception):
        _emit(att, decision_derived=bad)


def test_parse_rejects_invalid_verdict():
    att = _attestation()
    d = _emit(att).to_dict()
    d["decisionDerived"]["decision"] = "maybe"
    with pytest.raises(Exception):
        parse_decision_record(d)


def test_emit_rejects_bad_digest_prefix():
    att = _attestation()
    bl = dataclasses.replace(make_back_link(att), attestation_digest="deadbeef")
    with pytest.raises(Exception):
        _emit(att, back_link=bl)


def test_wire_round_trip():
    att = _attestation()
    r = _emit(att)
    reparsed = parse_decision_record(r.to_dict())
    assert reparsed == r
    assert verify_decision_signature(reparsed, verifying_material=HS_SECRET) is True
    assert verify_decision_back_link(reparsed, attestation=att).ok is True


def test_decision_and_outcome_pair_on_shared_attestation():
    att = _attestation()
    decision = _emit(att)
    receipt = emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status="executed",
            completed_at="2026-05-31T09:30:02Z",
            result_commitment=make_result_digest({"ok": True}),
        ),
        iss="vaara-proxy://acme-eu",
        sub="tenant:acme/agent:billing-bot",
        secret_version="2026-05",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    assert records_paired(decision, receipt) is True


def test_records_from_different_calls_do_not_pair():
    decision = _emit(_attestation())
    other_att = _attestation()  # fresh nonce
    receipt = emit_receipt(
        back_link=make_back_link(other_att),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at="2026-05-31T09:30:02Z",
        ),
        iss="vaara-proxy://acme-eu",
        sub="tenant:acme/agent:billing-bot",
        secret_version="2026-05",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    assert records_paired(decision, receipt) is False


def test_commit_payload_bridges_to_decision_derived():
    from vaara.audit.receipts import CommitPayload, decision_derived_from_commit

    commit = CommitPayload(
        action_id="act-1",
        decision="deny",
        risk_score=0.82,
        threshold_allow=0.4,
        threshold_deny=0.7,
        decided_at=1700000000.0,
    )
    dd = decision_derived_from_commit(commit, policy_id="ruleset:v3")
    # deny in the audit vocabulary maps to block on the SEP wire.
    assert dd.decision == "block"
    # Floats become decimal strings; epoch becomes ISO 8601 UTC.
    assert dd.risk_score == "0.82"
    assert dd.threshold_allow == "0.4"
    assert dd.threshold_block == "0.7"
    assert dd.decided_at == "2023-11-14T22:13:20Z"
    assert dd.policy_id == "ruleset:v3"


def test_commit_payload_allow_maps_through():
    from vaara.audit.receipts import CommitPayload, decision_derived_from_commit

    commit = CommitPayload(
        action_id="act-2",
        decision="allow",
        risk_score=0.1,
        threshold_allow=0.4,
        threshold_deny=0.7,
        decided_at=1700000000.0,
    )
    dd = decision_derived_from_commit(commit)
    assert dd.decision == "allow"
    # A bridged decision is a valid signing input under the float ban.
    att = _attestation()
    r = _emit(att, decision_derived=dd)
    assert verify_decision_signature(r, verifying_material=HS_SECRET) is True
