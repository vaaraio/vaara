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
    AmbiguousSupersessionError,
    DecisionDerived,
    EvidenceRef,
    decision_digest,
    emit_decision_record,
    make_back_link,
    parse_decision_record,
    records_paired,
    superseding_decision,
    verify_decision_back_link,
    verify_decision_signature,
)
from vaara.attestation.receipt import (  # noqa: E402
    OutcomeDerived,
    check_decision_conformance,
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


def _drift_evidence_ref(digest_hex="a" * 64, ref="ipfs://bafy-drift-record"):
    """A content-addressed pointer to an external drift-detection record."""
    return EvidenceRef(
        digest="sha256:" + digest_hex,
        canonicalization="JCS",
        schema="interlock.drift-record/v0",
        ref=ref,
    )


def test_evidence_ref_round_trips_and_signs():
    att = _attestation()
    dd = dataclasses.replace(_decision(), evidence_ref=_drift_evidence_ref())
    r = _emit(att, decision_derived=dd)
    # The reference rides inside the signed basis, so it is covered.
    assert verify_decision_signature(r, verifying_material=HS_SECRET) is True
    reparsed = parse_decision_record(r.to_dict())
    assert reparsed == r
    assert reparsed.decision_derived.evidence_ref == _drift_evidence_ref()
    assert verify_decision_signature(reparsed, verifying_material=HS_SECRET) is True


def test_evidence_ref_omitted_when_none():
    att = _attestation()
    r = _emit(att)  # _decision() has no evidence_ref
    assert "evidenceRef" not in r.to_dict()["decisionDerived"]


def test_evidence_ref_minimal_ref_optional():
    att = _attestation()
    minimal = EvidenceRef(
        digest="sha256:" + "b" * 64,
        canonicalization="JCS",
        schema="interlock.drift-record/v0",
    )
    dd = dataclasses.replace(_decision(), evidence_ref=minimal)
    r = _emit(att, decision_derived=dd)
    wire = r.to_dict()["decisionDerived"]["evidenceRef"]
    assert "ref" not in wire
    assert parse_decision_record(r.to_dict()) == r


def test_swapping_evidence_ref_breaks_signature():
    # The binding property: cite a different evidence digest after signing
    # and the decision no longer verifies.
    att = _attestation()
    dd = dataclasses.replace(_decision(), evidence_ref=_drift_evidence_ref())
    r = _emit(att, decision_derived=dd)
    swapped = dataclasses.replace(
        r,
        decision_derived=dataclasses.replace(
            r.decision_derived,
            evidence_ref=_drift_evidence_ref(digest_hex="c" * 64),
        ),
    )
    assert verify_decision_signature(swapped, verifying_material=HS_SECRET) is False


def test_stripping_evidence_ref_breaks_signature():
    # Dropping the cited evidence after signing is a detectable downgrade.
    att = _attestation()
    dd = dataclasses.replace(_decision(), evidence_ref=_drift_evidence_ref())
    r = _emit(att, decision_derived=dd)
    stripped = dataclasses.replace(
        r,
        decision_derived=dataclasses.replace(r.decision_derived, evidence_ref=None),
    )
    assert verify_decision_signature(stripped, verifying_material=HS_SECRET) is False


def test_parse_rejects_evidence_ref_bad_digest():
    att = _attestation()
    dd = dataclasses.replace(_decision(), evidence_ref=_drift_evidence_ref())
    d = _emit(att, decision_derived=dd).to_dict()
    d["decisionDerived"]["evidenceRef"]["digest"] = "deadbeef"  # missing sha256: prefix
    with pytest.raises(Exception):
        parse_decision_record(d)


def test_parse_rejects_evidence_ref_missing_canonicalization():
    att = _attestation()
    dd = dataclasses.replace(_decision(), evidence_ref=_drift_evidence_ref())
    d = _emit(att, decision_derived=dd).to_dict()
    del d["decisionDerived"]["evidenceRef"]["canonicalization"]
    with pytest.raises(Exception):
        parse_decision_record(d)


def test_parse_rejects_evidence_ref_unknown_key():
    # The signed sub-block is closed: an extra key would not survive a
    # byte-exact re-derivation, so parsing must fail closed.
    att = _attestation()
    dd = dataclasses.replace(_decision(), evidence_ref=_drift_evidence_ref())
    d = _emit(att, decision_derived=dd).to_dict()
    d["decisionDerived"]["evidenceRef"]["surprise"] = "x"
    with pytest.raises(Exception):
        parse_decision_record(d)


def test_emit_rejects_evidence_ref_bad_digest():
    att = _attestation()
    bad = EvidenceRef(
        digest="deadbeef",
        canonicalization="JCS",
        schema="interlock.drift-record/v0",
    )
    dd = dataclasses.replace(_decision(), evidence_ref=bad)
    with pytest.raises(Exception):
        _emit(att, decision_derived=dd)


def test_conformance_passes_with_evidence_ref():
    att = _attestation()
    dd = dataclasses.replace(_decision(), evidence_ref=_drift_evidence_ref())
    r = _emit(att, decision_derived=dd)
    report = check_decision_conformance(r.to_dict())
    assert report.conforms is True
    assert any(c.id == "evidence_ref_digest_format" and c.ok for c in report.checks)


def test_conformance_fails_on_malformed_evidence_ref():
    att = _attestation()
    dd = dataclasses.replace(_decision(), evidence_ref=_drift_evidence_ref())
    d = _emit(att, decision_derived=dd).to_dict()
    d["decisionDerived"]["evidenceRef"]["digest"] = "not-a-digest"
    report = check_decision_conformance(d)
    assert report.conforms is False


def _outcome_receipt(att, decision, *, status="executed", dec_digest=...):
    """Emit a receipt back-linked to ``att``. ``dec_digest`` defaults to
    the digest of ``decision`` (Check B satisfied); pass ``None`` to omit
    it or a string to forge it."""
    if dec_digest is ...:
        dec_digest = decision_digest(decision)
    return emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status=status,
            completed_at="2026-05-31T09:30:02Z",
            result_commitment=make_result_digest({"ok": True}),
            decision_digest=dec_digest,
        ),
        iss="vaara-proxy://acme-eu",
        sub="tenant:acme/agent:billing-bot",
        secret_version="2026-05",
        alg="HS256",
        signing_material=HS_SECRET,
    )


def test_decision_and_outcome_pair_on_shared_attestation():
    att = _attestation()
    decision = _emit(att)
    receipt = _outcome_receipt(att, decision)
    assert records_paired(decision, receipt) is True


def test_receipt_without_decision_digest_does_not_pair():
    # Check B is mandatory: shared attestation (Check A) is not enough.
    att = _attestation()
    decision = _emit(att)
    receipt = _outcome_receipt(att, decision, dec_digest=None)
    assert records_paired(decision, receipt) is False


def test_receipt_bound_to_other_decision_does_not_pair():
    # Same attestation, but the outcome commits to a different decision's
    # content. Check A passes, Check B rejects.
    att = _attestation()
    bound = _emit(att)
    presented = emit_decision_record(
        back_link=make_back_link(att),
        decision_derived=DecisionDerived(
            decision="block", decided_at="2026-05-31T09:30:01Z"),
        iss="vaara-proxy://acme-eu",
        sub="tenant:acme/agent:billing-bot",
        secret_version="2026-05",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    receipt = _outcome_receipt(att, bound)  # commits to `bound`, not `presented`
    assert decision_digest(bound) != decision_digest(presented)
    assert records_paired(bound, receipt) is True
    assert records_paired(presented, receipt) is False


def test_decision_digest_is_deterministic_and_instance_bound():
    att = _attestation()
    d1 = _emit(att)
    assert decision_digest(d1) == decision_digest(parse_decision_record(d1.to_dict()))
    assert decision_digest(d1).startswith("sha256:")
    d2 = _emit(_attestation())  # fresh attestation nonce -> distinct instance
    assert decision_digest(d1) != decision_digest(d2)


def test_superseding_decision_latest_wins_and_ties_are_ambiguous():
    att = _attestation()
    bl = make_back_link(att)

    def _dec(decided_at, nonce, decision="allow"):
        return emit_decision_record(
            back_link=bl,
            decision_derived=DecisionDerived(
                decision=decision, decided_at=decided_at),
            iss="vaara-proxy://acme-eu", sub="tenant:acme/agent:billing-bot",
            secret_version="2026-05", alg="HS256", signing_material=HS_SECRET,
            nonce=nonce)

    earlier = _dec("2026-05-31T09:00:00Z", "n-zzz")
    later = _dec("2026-05-31T10:00:00Z", "n-mmm")
    assert superseding_decision([earlier, later]) is later

    # Equal decidedAt, distinct records, no ordering field: ambiguous, not
    # resolved by nonce/file/arrival order.
    tie_a = _dec("2026-05-31T11:00:00Z", "n-aaa", decision="block")
    tie_b = _dec("2026-05-31T11:00:00Z", "n-bbb", decision="allow")
    with pytest.raises(AmbiguousSupersessionError):
        superseding_decision([tie_b, tie_a])

    # Byte-identical records are one decision, not a tie.
    dup = _dec("2026-05-31T12:00:00Z", "n-dup")
    assert superseding_decision([dup, dup]) is dup

    with pytest.raises(ValueError):
        superseding_decision([])


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
